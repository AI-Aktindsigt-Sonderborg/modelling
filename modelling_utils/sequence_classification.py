# pylint: disable=protected-access
import argparse
import json
import os
import shutil
import sys
from datetime import datetime
from typing import List

import numpy as np
import torch
from datasets import load_dataset, ClassLabel
from opacus import PrivacyEngine, GradSampleModule
from opacus.data_loader import DPDataLoader
from opacus.optimizers import DPOptimizer
from opacus.utils.batch_memory_manager import BatchMemoryManager
from sklearn.metrics import f1_score, accuracy_score
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, TrainingArguments, Trainer, \
    BertForSequenceClassification, \
    DataCollatorWithPadding

from data_utils.helpers import DatasetWrapper
from local_constants import DATA_DIR, MODEL_DIR
from modelling_utils.helpers import create_scheduler, get_lr, validate_model, \
    get_metrics, save_key_metrics_sc
from utils.helpers import TimeCode, append_json
from utils.visualization import plot_running_results, plot_confusion_matrix


class SequenceClassification:
    """
    Class to train an MLM unsupervised model

    Attributes
    ----------
    args: parsed args from MLMArgParser - see utils.input_args.MLMArgParser for
    possible arguments

    Methods
    -------
    train_model()
        load data, set up training and train model
    train_epoch(model, train_loader, optimizer, epoch, val_loader, step)
        train one epoch
    evaluate(model, val_loader)
        evaluate model at given step
    set_up_training()
        Load data and set up for training an MLM unsupervised model
    create_dummy_trainer(train_data_wrapped):
        Create dummy trainer, such that we get optimizer and can save model object
    tokenize_and_wrap_data(data):
        Tokenize dataset with tokenize_function and wrap with DatasetWrapper
    load_data(train, test):
        Load data using datasets.load_dataset for training and evaluation
    Static methods
    --------------
    get_data_collator
        Based on word mask load corresponding data collator
    freeze_layers(model)
        Freeze all bert layers in model, such that we can train only the head
    unfreeze_layers(model)
        Un-freeze all bert layers in model, such that we can train full model
    save_model(model, output_dir, data_collator, tokenizer, step):
        Wrap model in trainer class and save to pytorch object
    save_config(output_dir: str, args: argparse.Namespace)
        Save config file with input arguments
    """

    def __init__(self, args: argparse.Namespace):
        self.total_steps = None
        self.args = args
        self.output_name = f'{self.args.model_name.replace("/", "_")}-' \
                           f'{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}'
        self.args.output_name = self.output_name
        self.output_dir = os.path.join(MODEL_DIR, self.output_name)
        self.metrics_dir = os.path.join(self.output_dir, 'metrics')
        self.train_data = None
        self.eval_data = None
        self.test_data = None
        self.model = None
        self.local_alvenir_model_path = None
        self.label2id, self.id2label = self.label2id2label()
        self.class_labels = ClassLabel(
            num_classes=len(self.args.labels),
            names=self.args.labels)

        if self.args.load_alvenir_pretrained:
            self.model_path = os.path.join(MODEL_DIR, self.args.model_name)
        else:
            self.model_path = self.args.model_name

        self.tokenizer = self.get_tokenizer
        self.data_collator = self.get_data_collator

        self.scheduler = None
        # ToDo: Experiment with freezing layers - see SPRIN-159
        if not self.args.freeze_layers:
            self.args.freeze_layers_n_steps = 0
            self.args.lr_freezed_warmup_steps = 0
            self.args.lr_freezed = 0

    def train_model(self):
        """
        load data, set up training and train model
        """
        model, optimizer, train_loader = self.set_up_training()

        code_timer = TimeCode()
        all_lrs = []
        step = 0

        if self.eval_data:
            _, eval_loader = self.create_data_loader(
                data=self.eval_data,
                batch_size=self.args.eval_batch_size,
                shuffle=False)

            losses = []
            accuracies = []
            f1s = []

            for epoch in tqdm(range(self.args.epochs), desc="Epoch", unit="epoch"):
                model, step, lrs, losses, accuracies, f1s = self.train_epoch(
                    model=model,
                    train_loader=train_loader,
                    optimizer=optimizer,
                    val_loader=eval_loader,
                    epoch=epoch + 1,
                    step=step,
                    eval_losses=losses,
                    eval_accuracies=accuracies,
                    eval_f1s=f1s)
                all_lrs.extend(lrs)

            if step > self.args.freeze_layers_n_steps:
                best_metrics = get_metrics(
                    losses=losses,
                    accuracies=accuracies,
                    f1s=f1s,
                    freeze_layers_n_steps=self.args.freeze_layers_n_steps)

                save_key_metrics_sc(output_dir=self.metrics_dir, args=self.args,
                                    metrics=best_metrics,
                                    total_steps=self.total_steps)

            if self.args.make_plots:
                plot_running_results(
                    output_dir=self.metrics_dir,
                    epochs=self.args.epochs, f1=f1s,
                    lrs=all_lrs, accs=accuracies, loss=losses)

        else:
            for epoch in tqdm(range(self.args.epochs), desc="Epoch", unit="epoch"):
                model, step, lrs = self.train_epoch(model=model,
                                                    train_loader=train_loader,
                                                    optimizer=optimizer,
                                                    step=step)
                all_lrs.append(lrs)
        code_timer.how_long_since_start()
        if self.args.save_model_at_end:
            self.save_model(
                model=model,
                output_dir=self.output_dir,
                data_collator=self.data_collator,
                tokenizer=self.tokenizer
            )
        self.model = model

    def train_epoch(self,
                    model: BertForSequenceClassification,
                    train_loader: DataLoader,
                    optimizer, epoch: int = None,
                    val_loader: DataLoader = None,
                    step: int = 0,
                    eval_losses: List[dict] = None,
                    eval_accuracies: List[dict] = None,
                    eval_f1s: List[dict] = None):
        """
        Train one epoch
        :param model: Model of type BertForMaskedLM
        :param train_loader: Data loader of type DataLoader
        :param optimizer: Default is AdamW optimizer
        :param epoch: Given epoch: int
        :param val_loader: If evaluate_during_training: DataLoader containing validation data
        :param step: Given step
        :return: if self.eval_data:
            return model, eval_accuracies, eval_f1s, eval_losses, step, lrs
        else: return model, step, lrs
        """
        model.train()
        model = model.to(self.args.device)
        train_losses = []
        lrs = []

        for batch in tqdm(train_loader,
                          desc=f'Train epoch {epoch} of {self.args.epochs}',
                          unit="batch"):

            if self.args.freeze_layers and step == 0:
                model = self.freeze_layers(model)

            if self.args.freeze_layers and step == self.args.freeze_layers_n_steps:
                model = self.unfreeze_layers(model)
                # ToDo: Below operation only works if lr is lower than lr_freezed: fix this
                self.scheduler = create_scheduler(
                    optimizer,
                    start_factor=(self.args.learning_rate / self.args.lr_freezed)
                                 / self.args.lr_warmup_steps,
                    end_factor=self.args.learning_rate / self.args.lr_freezed,
                    total_iters=self.args.lr_warmup_steps)

            if step == self.args.lr_start_decay:
                self.scheduler = create_scheduler(
                    optimizer,
                    start_factor=1,
                    end_factor=self.args.learning_rate / (self.total_steps - step),
                    total_iters=self.total_steps - step)

            append_json(output_dir=self.metrics_dir,
                        filename='learning_rates',
                        data={'epoch': epoch, 'step': step, 'lr': get_lr(optimizer)[0]})
            lrs.append({'epoch': epoch, 'step': step, 'lr': get_lr(optimizer)[0]})

            optimizer.zero_grad()

            # compute model output
            output = model(input_ids=batch["input_ids"].to(self.args.device),
                           attention_mask=batch["attention_mask"].to(self.args.device),
                           labels=batch["labels"].to(self.args.device))

            loss = output.loss
            loss.backward()
            train_losses.append(loss.item())
            optimizer.step()
            self.scheduler.step()

            if step % self.args.logging_steps == 0 \
                and not step % self.args.evaluate_steps == 0:
                print(
                    f"\n\tTrain Epoch: {epoch} \t"
                    f"Step: {step} \t LR: {get_lr(optimizer)[0]}\t"
                    f"Loss: {np.mean(train_losses):.6f} "
                )

            if val_loader and (step > 0 and (step % self.args.evaluate_steps == 0)):
                print(
                    f"\tTrain Epoch: {epoch} \t"
                    f"Step: {step} \t LR: {get_lr(optimizer)[0]}\t"
                    f"Loss: {np.mean(train_losses):.6f} "
                )

                eval_accuracy, eval_f1, eval_loss = self.evaluate(
                    model, val_loader)

                print(
                    f"\n"
                    f"eval loss: {eval_loss} \t"
                    f"eval acc: {eval_accuracy}\t"
                    f"eval f1: {eval_f1}"
                )

                current_metrics = {'loss': eval_loss,
                                   'acc': eval_accuracy,
                                   'f1': eval_f1}
                append_json(output_dir=self.metrics_dir,
                            filename='eval_losses',
                            data={'epoch': epoch, 'step': step, 'score': eval_loss})
                append_json(output_dir=self.metrics_dir,
                            filename='accuracies',
                            data={'epoch': epoch, 'step': step, 'score': eval_accuracy})
                append_json(output_dir=self.metrics_dir,
                            filename='f1s',
                            data={'epoch': epoch, 'step': step, 'score': eval_f1})

                eval_losses.append(
                    {'epoch': epoch, 'step': step, 'score': eval_loss})
                eval_accuracies.append(
                    {'epoch': epoch, 'step': step, 'score': eval_accuracy})
                eval_f1s.append(
                    {'epoch': epoch, 'step': step, 'score': eval_f1})

                if self.args.save_steps is not None and (
                    step > 0 and (step % self.args.save_steps == 0)):
                    best_metrics = get_metrics(
                        freeze_layers_n_steps=self.args.freeze_layers_n_steps,
                        losses=eval_losses,
                        accuracies=eval_accuracies,
                        f1s=eval_f1s
                    )
                    self.save_model_at_step(
                        model=model,
                        epoch=epoch,
                        step=step,
                        current_metrics=current_metrics,
                        best_metrics=best_metrics)
                model.train()
            step += 1
        if self.eval_data:
            return model, step, lrs, eval_losses, eval_accuracies, eval_f1s
        return model, step, lrs

    def save_model_at_step(self, model, epoch, step,
                           current_metrics: dict,
                           best_metrics: dict):
        """
        Save model at step and overwrite best_model if the model
        have improved evaluation performance.
        :param best_metrics: at which step does model have best metrics
        :param current_metrics: Current eval metrics of model: f1, acc and loss
        :param model: Current model
        :param epoch: Current epoch
        :param step: Current step
        """
        if not self.args.save_only_best_model:
            self.save_model(model, output_dir=self.output_dir,
                            data_collator=self.data_collator,
                            tokenizer=self.tokenizer,
                            step=f'/epoch-{epoch}_step-{step}')
        if step > self.args.freeze_layers_n_steps:
            save_best_model = True
            for metric in self.args.eval_metrics:
                if best_metrics[metric]['score'] != current_metrics[metric]:
                    save_best_model = False
                    break

            if save_best_model:
                self.save_model(model, output_dir=self.output_dir,
                                data_collator=self.data_collator,
                                tokenizer=self.tokenizer,
                                step='/best_model')

    def evaluate(self, model, val_loader: DataLoader,
                 conf_plot: bool = False) -> tuple:
        """
        :param model:
        :param val_loader:
        :return: mean loss, accuracy-score, f1-score
        """
        # ToDo: Change return to EvalScore like in MLM
        # set up preliminaries
        model.eval()
        model.to(self.args.device)
        y_true, y_pred, loss = [], [], []

        # get model predictions and labels
        for batch in tqdm(val_loader, unit="batch", desc="val"):
            output = model(
                input_ids=batch["input_ids"].to(self.args.device),
                attention_mask=batch["attention_mask"].to(self.args.device),
                labels=batch["labels"].to(self.args.device)
            )

            batch_preds = np.argmax(
                output.logits.detach().cpu().numpy(), axis=-1
            )
            batch_labels = batch["labels"].cpu().numpy()
            batch_loss = output.loss.item()

            y_true.extend(list(batch_labels))
            y_pred.extend(list(batch_preds))
            loss.append(batch_loss)

        # calculate metrics of interest
        acc = accuracy_score(y_true, y_pred)
        f_1 = f1_score(y_true, y_pred, average='macro')
        loss = np.mean(loss)

        if conf_plot:
            y_true_plot = list(map(lambda x: self.id2label[int(x)], y_true))
            y_pred_plot = list(map(lambda x: self.id2label[int(x)], y_pred))

            plot_confusion_matrix(
                y_true=y_true_plot,
                y_pred=y_pred_plot,
                labels=self.args.labels,
                model_name=self.args.model_name)

        return acc, f_1, loss

    def set_up_training(self):
        """
        Load data and set up for training an Sequence classification model
        :return: model, optimizer and train_loader for training
        """
        self.load_data()

        if self.args.auto_lr_scheduling:
            self.compute_lr_automatically()

        if self.args.save_config:
            self.save_config(output_dir=self.output_dir,
                             metrics_dir=self.metrics_dir,
                             args=self.args)
        train_data_wrapped, train_loader = self.create_data_loader(
            data=self.train_data,
            batch_size=self.args.train_batch_size)

        model = self.get_model

        if self.args.freeze_embeddings:
            for param in model.bert.embeddings.parameters():
                param.requires_grad = False

        dummy_trainer = self.create_dummy_trainer(train_data_wrapped=train_data_wrapped,
                                                  model=model)

        # ToDo: Notice change of optimizer here
        optimizer = dummy_trainer.create_optimizer()
        # optimizer = torch.optim.AdamW(model.parameters(), lr=self.args.learning_rate)

        if self.args.freeze_layers:
            self.scheduler = create_scheduler(
                optimizer,
                start_factor=self.args.lr_freezed / self.args.lr_freezed_warmup_steps,
                end_factor=1,
                total_iters=self.args.lr_freezed_warmup_steps)
        else:
            self.scheduler = create_scheduler(
                optimizer,
                start_factor=self.args.learning_rate / self.args.lr_warmup_steps,
                end_factor=1,
                total_iters=self.args.lr_warmup_steps)

        return model, optimizer, train_loader

    def create_data_loader(self, data: Dataset, batch_size: int,
                           shuffle: bool = True) -> tuple:
        data_wrapped = self.tokenize_and_wrap_data(data=data)
        data_loader = DataLoader(dataset=data_wrapped,
                                 collate_fn=self.data_collator,
                                 batch_size=batch_size,
                                 shuffle=shuffle)
        return data_wrapped, data_loader

    def tokenize_and_wrap_data(self, data: Dataset):
        """
        Method to tokenize and wrap data needed for DataLoader
        @param data: Dataset
        @return: Wrapped, tokenized dataset
        """
        tokenized = data.map(
            self.tokenize,
            batched=True,
            remove_columns=data.column_names,
            load_from_cache_file=False,
            desc="Running tokenizer on dataset line_by_line",
        )

        tokenized.set_format('torch')
        wrapped = DatasetWrapper(tokenized)

        return wrapped

    def tokenize(self, batch):
        """
        Tokenize each batch in dataset
        @param batch: batch of data
        @return: tokens for each batch
        """
        batch['text'] = [
            line for line in batch['text'] if len(line) > 0 and not line.isspace()
        ]

        tokens = self.tokenizer(batch['text'],
                                padding='max_length',
                                max_length=self.args.max_length,
                                truncation=True)
        tokens['labels'] = self.class_labels.str2int(batch['label'])
        return tokens

    def load_data(self, train: bool = True, test: bool = False):
        """
        Load data using datasets.load_dataset for training and evaluation
        :param train: Whether to train model
        :param test: Whether to load test data
        """
        if train:
            self.train_data = load_dataset(
                'json',
                data_files=os.path.join(DATA_DIR, self.args.train_data),
                split='train')
            self.total_steps = int(
                len(self.train_data) / self.args.train_batch_size * self.args.epochs)
            self.args.total_steps = self.total_steps

            if self.args.compute_delta:
                self.args.delta = 1 / (2*len(self.train_data))

        if self.args.evaluate_during_training:
            self.eval_data = load_dataset(
                'json',
                data_files=os.path.join(DATA_DIR, self.args.eval_data),
                split='train')

        if test:
            self.test_data = load_dataset(
                'json',
                data_files=os.path.join(DATA_DIR, self.args.test_data),
                split='train')

    def label2id2label(self):
        label2id, id2label = {}, {}

        for i, label in enumerate(self.args.labels):
            label2id[label] = str(i)
            id2label[i] = label
        return label2id, id2label

    def compute_lr_automatically(self):

        if self.args.freeze_layers:
            self.args.lr_freezed_warmup_steps = \
                int(np.ceil(0.1 * self.args.freeze_layers_n_steps))

        self.args.lr_warmup_steps = \
            int(np.ceil(0.1 * (self.total_steps - self.args.freeze_layers_n_steps)))
        self.args.lr_start_decay = int(
            np.ceil((self.total_steps - self.args.freeze_layers_n_steps) *
                    0.5 + self.args.freeze_layers_n_steps))

    def create_dummy_trainer(self, train_data_wrapped: DatasetWrapper, model):
        """
        Create dummy trainer, such that we get optimizer and can save model object
        :param train_data_wrapped: Train data of type DatasetWrapper
        :param model: initial model
        :return: Trainer object
        """
        if self.args.freeze_layers:
            learning_rate_init: float = self.args.lr_freezed
        else:
            learning_rate_init: float = self.args.learning_rate

        training_args = TrainingArguments(
            output_dir=self.output_dir,
            learning_rate=learning_rate_init,
            weight_decay=self.args.weight_decay,
            fp16=self.args.use_fp16,
        )

        # Dummy training for optimizer
        return Trainer(
            model=model,
            args=training_args,
            train_dataset=train_data_wrapped,
            data_collator=self.data_collator,
            tokenizer=self.tokenizer
        )

    @property
    def get_tokenizer(self):
        return AutoTokenizer.from_pretrained(
            self.model_path,
            local_files_only=self.args.load_alvenir_pretrained)

    @property
    def get_data_collator(self):
        return DataCollatorWithPadding(tokenizer=self.tokenizer)

    @property
    def get_model(self):
        return BertForSequenceClassification.from_pretrained(
            self.model_path,
            num_labels=len(self.args.labels),
            label2id=self.label2id,
            id2label=self.id2label,
            local_files_only=self.args.load_alvenir_pretrained)

    @staticmethod
    def freeze_layers(model):
        """
        Freeze all bert layers in model, such that we can train only the head
        :param model: Model of type BertForSequenceClassification
        :return: freezed model
        """
        for name, param in model.named_parameters():
            if not (name.startswith("bert.pooler") or name.startswith("classifier")):
                param.requires_grad = False
        model.train()
        return model

    @staticmethod
    def unfreeze_layers(model):
        """
        Un-freeze all layers exept for embeddings in model, such that we can train full model
        :param model: Model of type BertForSequenceClassification
        :return: un-freezed model
        """
        for name, param in model.named_parameters():
            if not name.startswith("bert.embeddings"):
                param.requires_grad = True
        print("bert layers unfreezed")
        model.train()
        return model

    @staticmethod
    def save_model(model: BertForSequenceClassification, output_dir: str,
                   data_collator,
                   tokenizer,
                   step: str = ""):
        """
        Wrap model in trainer class and save to pytorch object
        :param model: BertForSequenceClassification to save
        :param output_dir: model directory
        :param data_collator:
        :param tokenizer:
        :param step: if saving during training step should be '/epoch-{epoch}_step-{step}'
        """
        output_dir = output_dir + step
        trainer_test = Trainer(
            model=model,
            args=TrainingArguments(output_dir=output_dir),
            data_collator=data_collator,
            tokenizer=tokenizer
        )
        trainer_test.save_model(output_dir=output_dir)
        torch.save(model.state_dict(), os.path.join(output_dir, 'model_weights.json'))

    @staticmethod
    def save_config(output_dir: str, metrics_dir: str, args: argparse.Namespace):
        """
        Save config file with input arguments
        :param output_dir: model directory
        :param args: input args
        """

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        if not os.path.exists(metrics_dir):
            os.makedirs(metrics_dir)

        with open(os.path.join(metrics_dir, 'training_run_config.json'), 'w',
                  encoding='utf-8') as outfile:
            json.dump(args.__dict__, outfile, indent=2)

        if args.print_only_args:
            metrics_path = os.path.dirname(os.path.abspath(metrics_dir))
            print('Training arguments:\n')
            for arg in args.__dict__:
                print(f'{arg}: {args.__dict__[arg]}')

            print('\nDeleting model path...')
            shutil.rmtree(metrics_path)

            print('\nExiting.')
            sys.exit(0)


class SequenceClassificationDP(SequenceClassification):
    """
        Class inherited from SequenceClassification to train an unsupervised MLM model with
        differential privacy

        Attributes
        ----------
        args: parsed args from MLMArgParser - see utils.input_args.MLMArgParser for possible
        arguments
        Methods
        -------
        train_model()
            Load data, set up private training and train with differential privacy -
            modification of superclass train_epoch
        train_epoch(model, train_loader, optimizer, epoch, val_loader, step)
            Train one epoch with DP - modification of superclass train_epoch
        set_up_training()
            Load data and set up for training an unsupervised MLM model with differential privacy -
            modification of superclass set_up_training
        set_up_privacy(train_loader)
            Set up privacy engine for private training

        Static methods
        --------------
        validate_model(model, strict)
            Validate whether model is compatible with opacus
        freeze_layers(model)
            Freeze all bert layers in model, such that we can train only the head -
            modification of superclass freeze_layers
        unfreeze_layers(model)
            Un-freeze all bert layers in model, such that we can train full model -
            modification of superclass freeze_layers
        save_model(model, output_dir, data_collator, tokenizer, step)
            Wrap model in trainer class and save to pytorch object -
            modification of superclass save_model
        save_config(output_dir: str, args: argparse.Namespace)
            Save config file with input arguments
        get_lr(optimizer: DPOptimizer)
            Get current learning rate from optimizer
        save_key_metrics(output_dir, args, best_acc, best_loss, filename)
            Save important args and performance for benchmarking
        create_scheduler(optimizer, start_factor, end_factor, total_iters)
            Create scheduler for learning rate warmup and decay
        """

    def __init__(self, args: argparse.Namespace):
        super().__init__(args)

        self.privacy_engine = None
        self.output_name = f'DP-eps-{int(self.args.epsilon)}-' \
                           f'{self.args.model_name.replace("/", "_")}-' \
                           f'{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}'
        self.args.output_name = self.output_name
        self.output_dir = os.path.join(MODEL_DIR, self.output_name)
        self.metrics_dir = os.path.join(self.output_dir, 'metrics')
        if not self.args.freeze_layers:
            self.args.freeze_layers_n_steps = 0

    def train_model(self):
        """
        Load data, set up private training and train with differential privacy
        """
        model, dp_optimizer, dp_train_loader = self.set_up_training()

        code_timer = TimeCode()
        all_lrs = []
        step = 0

        if self.eval_data:
            _, eval_loader = self.create_data_loader(
                data=self.eval_data,
                batch_size=self.args.eval_batch_size,
                shuffle=False)

            losses = []
            accuracies = []
            f1s = []
            for epoch in tqdm(range(self.args.epochs), desc="Epoch", unit="epoch"):
                model, losses, accuracies, step, lrs = self.train_epoch(
                    model=model,
                    train_loader=dp_train_loader,
                    optimizer=dp_optimizer,
                    val_loader=eval_loader,
                    epoch=epoch + 1,
                    step=step,
                    eval_losses=losses,
                    eval_accuracies=accuracies,
                    eval_f1s=f1s)

                all_lrs.extend(lrs)

            if step > self.args.freeze_layers_n_steps:
                best_metrics = get_metrics(
                    freeze_layers_n_steps=self.args.freeze_layers_n_steps,
                    losses=losses,
                    accuracies=accuracies,
                    f1s=f1s
                )
                save_key_metrics_sc(output_dir=self.metrics_dir,
                                    args=self.args,
                                    metrics=best_metrics,
                                    total_steps=self.total_steps)

            if self.args.make_plots:
                plot_running_results(
                    output_dir=self.metrics_dir,
                    epsilon=str(self.args.epsilon),
                    delta=str(self.args.delta),
                    epochs=self.args.epochs,
                    lrs=all_lrs, accs=accuracies, loss=losses, f1=f1s)

        else:
            for epoch in tqdm(range(self.args.epochs), desc="Epoch", unit="epoch"):
                model, step, lrs = self.train_epoch(model=model,
                                                    train_loader=dp_train_loader,
                                                    optimizer=dp_optimizer,
                                                    step=step)
                all_lrs.append(lrs)
        code_timer.how_long_since_start()
        if self.args.save_model_at_end:
            self.save_model(
                model=model._module,
                output_dir=self.output_dir,
                data_collator=self.data_collator,
                tokenizer=self.tokenizer
            )
        self.model = model

    def train_epoch(self, model: GradSampleModule, train_loader: DPDataLoader,
                    optimizer: DPOptimizer, epoch: int = None,
                    val_loader: DataLoader = None,
                    step: int = 0, eval_losses: List[dict] = None,
                    eval_accuracies: List[dict] = None,
                    eval_f1s: List[dict] = None):
        """
        Train one epoch with DP - modification of superclass train_epoch
        :param eval_accuracies: list of evaluation accuracies
        :param eval_losses: list of evaluation losses
        :param model: Differentially private model wrapped in GradSampleModule
        :param train_loader: Differentially private data loader of type DPDataLoader
        :param optimizer: Differentially private optimizer of type DPOptimizer
        :param epoch: Given epoch: int
        :param val_loader: If evaluate_during_training: DataLoader containing validation data
        :param step: Given step
        :return: if self.eval_data: return model, eval_losses, eval_accuracies, step, lrs
        else: return model, step, lrs
        """
        model.train()
        model = model.to(self.args.device)
        train_losses = []
        lrs = []
        with BatchMemoryManager(
            data_loader=train_loader,
            max_physical_batch_size=self.args.train_batch_size,
            optimizer=optimizer
        ) as memory_safe_data_loader:

            for batch in tqdm(memory_safe_data_loader,
                              desc=f'Epoch {epoch} of {self.args.epochs}',
                              unit="batch"):

                if self.args.freeze_layers and step == 0:
                    model = self.freeze_layers(model)

                if self.args.freeze_layers and step == self.args.freeze_layers_n_steps:
                    model = self.unfreeze_layers(model)
                    # ToDo: Below operation only works if lr is lower than lr_freezed: fix this
                    self.scheduler = create_scheduler(
                        optimizer,
                        start_factor=(self.args.learning_rate / self.args.lr_freezed)
                                     / self.args.lr_warmup_steps,
                        end_factor=self.args.learning_rate / self.args.lr_freezed,
                        total_iters=self.args.lr_warmup_steps)

                if step == self.args.lr_start_decay:
                    self.scheduler = create_scheduler(
                        optimizer,
                        start_factor=1,
                        end_factor=self.args.learning_rate / (self.total_steps - step),
                        total_iters=self.total_steps - self.args.lr_start_decay)

                append_json(output_dir=self.metrics_dir,
                            filename='learning_rates',
                            data={'epoch': epoch, 'step': step,
                                  'lr': get_lr(optimizer)[0]})
                lrs.append({'epoch': epoch, 'step': step, 'lr': get_lr(optimizer)[0]})

                # compute models
                output = model(
                    input_ids=batch["input_ids"].to(self.args.device),
                    attention_mask=batch["attention_mask"].to(self.args.device),
                    labels=batch["labels"].to(self.args.device))

                loss = output.loss
                loss.backward()
                train_losses.append(loss.item())
                optimizer.step()
                optimizer.zero_grad()
                self.scheduler.step()

                if step % self.args.logging_steps == 0 \
                    and not step % self.args.evaluate_steps == 0:
                    print(
                        f"\tTrain Epoch: {epoch} \t"
                        f"Step: {step} \t LR: {get_lr(optimizer)[0]}\t"
                        f"Loss: {np.mean(train_losses):.6f} "
                    )

                if val_loader and (step > 0 and (step % self.args.evaluate_steps == 0)):

                    print(
                        f"\tTrain Epoch: {epoch} \t"
                        f"Step: {step} \t LR: {get_lr(optimizer)[0]}\t"
                        f"Loss: {np.mean(train_losses):.6f} "
                        f"(ε = {self.privacy_engine.get_epsilon(self.args.delta):.2f}, "
                        f"δ = {self.args.delta})"
                    )
                    eval_accuracy, eval_f1, eval_loss = self.evaluate(model, val_loader)
                    print(
                        f"\n"
                        f"eval loss: {eval_loss} \t"
                        f"eval acc: {eval_accuracy}\t"
                        f"eval f1: {eval_f1}"
                    )

                    current_metrics = {
                        'loss': eval_loss,
                        'acc': eval_accuracy,
                        'f1': eval_f1}
                    append_json(
                        output_dir=self.metrics_dir,
                        filename='eval_losses',
                        data={'epoch': epoch, 'step': step, 'score': eval_loss})
                    append_json(
                        output_dir=self.metrics_dir,
                        filename='accuracies',
                        data={'epoch': epoch, 'step': step, 'score': eval_accuracy})
                    append_json(
                        output_dir=self.metrics_dir,
                        filename='f1s',
                        data={'epoch': epoch, 'step': step, 'score': eval_f1})

                    eval_losses.append(
                        {'epoch': epoch, 'step': step, 'score': eval_loss})
                    eval_accuracies.append(
                        {'epoch': epoch, 'step': step, 'score': eval_accuracy})
                    eval_f1s.append(
                        {'epoch': epoch, 'step': step, 'score': eval_f1})

                    if self.args.save_steps is not None and (
                        step > 0 and (step % self.args.save_steps == 0)):
                        best_metrics = get_metrics(
                            freeze_layers_n_steps=self.args.freeze_layers_n_steps,
                            losses=eval_losses,
                            accuracies=eval_accuracies,
                            f1s=eval_f1s
                        )

                        self.save_model_at_step(
                            model=model._module,
                            epoch=epoch,
                            step=step,
                            current_metrics=current_metrics,
                            best_metrics=best_metrics)
                    model.train()
                step += 1

        if self.eval_data:
            return model, eval_losses, eval_accuracies, step, lrs
        return model, step, lrs

    def set_up_training(self):
        """
        Load data and set up for training an unsupervised MLM model with differential privacy
        :return: model, optimizer and train_loader for DP training
        """
        self.load_data()

        if self.args.auto_lr_scheduling:
            self.compute_lr_automatically()

        if self.args.save_config:
            self.save_config(output_dir=self.output_dir, metrics_dir=self.metrics_dir,
                             args=self.args)

        train_data_wrapped, train_loader = self.create_data_loader(
            data=self.train_data,
            batch_size=self.args.lot_size)

        model = self.get_model

        if self.args.freeze_embeddings:
            for param in model.bert.embeddings.parameters():
                param.requires_grad = False

        dp_model, dp_optimizer, dp_train_loader = self.set_up_privacy(
            train_loader=train_loader,
            model=model)

        if self.args.freeze_layers:
            self.scheduler = create_scheduler(
                dp_optimizer,
                start_factor=self.args.lr_freezed / self.args.lr_freezed_warmup_steps,
                end_factor=1,
                total_iters=self.args.lr_freezed_warmup_steps)
        else:
            self.scheduler = create_scheduler(
                dp_optimizer,
                start_factor=self.args.learning_rate / self.args.lr_warmup_steps,
                end_factor=1,
                total_iters=self.args.lr_warmup_steps)

        return dp_model, dp_optimizer, dp_train_loader

    def set_up_privacy(self, train_loader: DataLoader, model):
        """
        Set up privacy engine for private training
        :param train_loader: DataLoader for training
        :return: model, optimizer and train loader for private training
        """
        self.privacy_engine = PrivacyEngine()

        model = model.train()

        # validate if model works with opacus
        validate_model(model, strict_validation=True)
        # new opacus version requires to generate optimizer from model.parameters()
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=self.args.learning_rate)

        dp_model, dp_optimizer, dp_train_loader = \
            self.privacy_engine.make_private_with_epsilon(
                module=model,
                optimizer=optimizer,
                data_loader=train_loader,
                epochs=self.args.epochs,
                target_epsilon=self.args.epsilon,
                target_delta=self.args.delta,
                max_grad_norm=self.args.max_grad_norm
            )

        return dp_model, dp_optimizer, dp_train_loader

    @staticmethod
    def freeze_layers(model):
        """
        Freeze all bert encoder layers in model, such that we can train only the head
        :param model: Model of type GradSampleModule
        :return: freezed model
        """
        for name, param in model.named_parameters():
            if not name.startswith("_module.cls"):
                param.requires_grad = False
        model.train()
        return model

    @staticmethod
    def unfreeze_layers(model):
        """
        Un-freeze all bert layers in model, such that we can train full model
        :param model: Model of type GradSampleModule
        :return: un-freezed model
        """
        for name, param in model.named_parameters():
            if not name.startswith("_module.bert.embeddings"):
                param.requires_grad = True
        model.train()
        return model
