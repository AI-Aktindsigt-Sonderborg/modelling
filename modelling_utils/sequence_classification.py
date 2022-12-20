# pylint: disable=protected-access
import argparse
import json
import os
import shutil
import sys
from datetime import datetime
from time import sleep
from typing import List

import numpy as np
import torch
from datasets import load_dataset, ClassLabel
from opacus import PrivacyEngine, GradSampleModule
from opacus.data_loader import DPDataLoader
from opacus.optimizers import DPOptimizer
from opacus.utils.batch_memory_manager import BatchMemoryManager
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, TrainingArguments, Trainer, \
    BertForSequenceClassification, \
    DataCollatorWithPadding

from data_utils.helpers import DatasetWrapper
from local_constants import DATA_DIR, MODEL_DIR
from modelling_utils.helpers import create_scheduler, get_lr, validate_model, get_max_acc_min_loss, \
    save_key_metrics_sc
from utils.helpers import TimeCode, append_json, accuracy
from utils.visualization import plot_running_results


class SequenceClassification:
    """
    Class to train an MLM unsupervised model

    Attributes
    ----------
    args: parsed args from MLMArgParser - see utils.input_args.MLMArgParser for possible arguments

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
    load_data(train):
        Load data using datasets.load_dataset for training and evaluation
    load_model_and_replace_bert_head():
        Load BertForMaskedLM, replace head and freeze all params in embeddings layer

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
        self.model = None
        self.local_alvenir_model_path = None
        self.label2id, self.id2label = self.label2id2label()
        self.class_labels = ClassLabel(
            num_classes=len(self.args.labels),
            names=self.args.labels)

        if self.args.load_alvenir_pretrained:
            self.local_alvenir_model_path = os.path.join(MODEL_DIR, self.args.model_name)
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.local_alvenir_model_path,
                local_files_only=self.args.load_alvenir_pretrained)
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.args.model_name,
                local_files_only=self.args.load_alvenir_pretrained)

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
            eval_data_wrapped = self.tokenize_and_wrap_data(data=self.eval_data)
            # self.eval_data_test = self.tokenize_eval_data(self.eval_data)

            eval_loader = DataLoader(dataset=eval_data_wrapped,
                                     collate_fn=self.data_collator,
                                     batch_size=self.args.eval_batch_size)
            losses = []
            accuracies = []

            for epoch in tqdm(range(self.args.epochs), desc="Epoch", unit="epoch"):
                model, losses, accuracies, step, lrs = self.train_epoch(
                    model=model,
                    train_loader=train_loader,
                    optimizer=optimizer,
                    val_loader=eval_loader,
                    epoch=epoch + 1,
                    step=step,
                    eval_losses=losses,
                    eval_accuracies=accuracies)
                all_lrs.extend(lrs)

            if step > self.args.freeze_layers_n_steps:
                min_loss, max_acc = get_max_acc_min_loss(losses, accuracies,
                                                         self.args.freeze_layers_n_steps)

                save_key_metrics_sc(output_dir=self.metrics_dir, args=self.args,
                                    best_acc=max_acc, best_loss=min_loss,
                                    total_steps=self.total_steps)

            if self.args.make_plots:
                plot_running_results(
                    output_dir=self.metrics_dir,
                    epochs=self.args.epochs,
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

    def train_epoch(self, model: BertForSequenceClassification, train_loader: DataLoader,
                    optimizer, epoch: int = None, val_loader: DataLoader = None,
                    step: int = 0, eval_losses: List[dict] = None,
                    eval_accuracies: List[dict] = None):
        """
        Train one epoch
        :param eval_accuracies:
        :param eval_losses:
        :param model: Model of type BertForMaskedLM
        :param train_loader: Data loader of type DataLoader
        :param optimizer: Default is AdamW optimizer
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

        for batch in tqdm(train_loader, desc=f'Train epoch {epoch} of {self.args.epochs}',
                          unit="batch"):

            if self.args.freeze_layers and step == 0:
                model = self.freeze_layers(model)

            if self.args.freeze_layers and step == self.args.freeze_layers_n_steps:
                model = self.unfreeze_layers(model)
                # ToDo: Below operation only works if lr is lower than lr_freezed: fix this
                self.scheduler = create_scheduler(optimizer,
                                                  start_factor=(self.args.learning_rate
                                                                / self.args.lr_freezed)
                                                               / self.args.lr_warmup_steps,
                                                  end_factor=self.args.learning_rate
                                                             / self.args.lr_freezed,
                                                  total_iters=self.args.lr_warmup_steps)

            if step == self.args.lr_start_decay:
                self.scheduler = create_scheduler(optimizer,
                                                  start_factor=1,
                                                  end_factor=self.args.learning_rate / (
                                                      self.total_steps - step),
                                                  total_iters=self.total_steps - step
                                                  )
            append_json(output_dir=self.metrics_dir, filename='learning_rates',
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

            if step % self.args.logging_steps == 0 and not step % self.args.evaluate_steps == 0:
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

                eval_loss, eval_accuracy = self.evaluate(model, val_loader)

                print(
                    f"\n"
                    f"eval loss: {eval_loss} \t"
                    f"eval acc: {eval_accuracy}"
                )
                append_json(output_dir=self.metrics_dir, filename='eval_losses',
                            data={'epoch': epoch, 'step': step, 'loss': eval_loss})
                append_json(output_dir=self.metrics_dir, filename='accuracies',
                            data={'epoch': epoch, 'step': step, 'acc': eval_accuracy})
                eval_losses.append({'epoch': epoch, 'step': step, 'loss': eval_loss})
                eval_accuracies.append({'epoch': epoch, 'step': step, 'acc': eval_accuracy})

                if self.args.save_steps is not None and (
                    step > 0 and (step % self.args.save_steps == 0)):
                    self.save_model_at_step(model, epoch, step, eval_losses, eval_accuracies)
                model.train()
            step += 1
        if self.eval_data:
            return model, eval_losses, eval_accuracies, step, lrs
        return model, step, lrs

    def save_model_at_step(self, model, epoch, step, eval_losses, eval_accuracies):
        """
        Save model at step and overwrite best_model if loss is lowest and acc is highest
        :param model: Current model
        :param epoch: Current epoch
        :param step: Current step
        :param eval_losses: Current list of losses
        :param eval_accuracies: Current list of accuracies
        """
        if not self.args.save_only_best_model:
            self.save_model(model, output_dir=self.output_dir,
                            data_collator=self.data_collator,
                            tokenizer=self.tokenizer,
                            step=f'/epoch-{epoch}_step-{step}')
        if step > self.args.freeze_layers_n_steps:
            min_loss, max_acc = get_max_acc_min_loss(eval_losses, eval_accuracies,
                                                     self.args.freeze_layers_n_steps)
            if min_loss['loss'] == eval_losses[-1]['loss'] \
                and max_acc['acc'] == eval_accuracies[-1]['acc']:
                self.save_model(model, output_dir=self.output_dir,
                                data_collator=self.data_collator,
                                tokenizer=self.tokenizer,
                                step='/best_model')

    def evaluate(self, model, val_loader: DataLoader):
        """
        Evaluate model at given step
        :param model:
        :param val_loader:
        :return: mean eval loss and mean accuracy
        """
        model.eval()
        if not next(model.parameters()).is_cuda:
            model = model.to(self.args.device)
        loss_arr = []
        accuracy_arr = []

        # with tqdm(val_loader, unit="batch", desc="Batch") as batches:
        for batch in tqdm(val_loader, unit="batch", desc="Eval"):
            # for batch in val_loader:
            output = model(input_ids=batch["input_ids"].to(self.args.device),
                           attention_mask=batch["attention_mask"].to(self.args.device),
                           labels=batch["labels"].to(self.args.device))

            preds = np.argmax(output.logits.detach().cpu().numpy(), axis=-1)
            labels = batch["labels"].cpu().numpy()

            eval_acc = accuracy(preds, labels)
            eval_loss = output.loss.item()

            if not np.isnan(eval_loss):
                loss_arr.append(eval_loss)
            if not np.isnan(eval_acc):
                accuracy_arr.append(eval_acc)

            sleep(0.001)

        return np.mean(loss_arr), np.mean(accuracy_arr)

    def set_up_training(self):
        """
        Load data and set up for training an Sequence classification model
        :return: model, optimizer and train_loader for training
        """
        self.load_data()

        if self.args.auto_lr_scheduling:
            self.compute_lr_automatically()

        if self.args.save_config:
            self.save_config(output_dir=self.output_dir, metrics_dir=self.metrics_dir,
                             args=self.args)

        train_data_wrapped = self.tokenize_and_wrap_data(data=self.train_data)

        if self.args.load_alvenir_pretrained:
            model = BertForSequenceClassification.from_pretrained(
                self.local_alvenir_model_path,
                num_labels=len(self.args.labels),
                label2id=self.label2id,
                id2label=self.id2label,
                local_files_only=self.args.load_alvenir_pretrained)
        else:
            model = BertForSequenceClassification.from_pretrained(
                self.args.model_name,
                num_labels=len(self.args.labels),
                label2id=self.label2id,
                id2label=self.id2label,
                local_files_only=self.args.load_alvenir_pretrained)

        if self.args.freeze_embeddings:
            # ToDo: For now we are freezing embedding layer until (maybe) we have implemented
            #  grad sampler - as this is not implemented in opacus
            for param in model.bert.embeddings.parameters():
                param.requires_grad = False

        dummy_trainer = self.create_dummy_trainer(train_data_wrapped=train_data_wrapped,
                                                  model=model)

        train_loader = DataLoader(dataset=train_data_wrapped, batch_size=self.args.train_batch_size,
                                  collate_fn=self.data_collator,
                                  shuffle=True)

        # ToDo: Notice change of optimizer here
        optimizer = dummy_trainer.create_optimizer()
        # optimizer = torch.optim.AdamW(model.parameters(), lr=self.args.learning_rate)

        if self.args.freeze_layers:
            self.scheduler = create_scheduler(optimizer,
                                              start_factor=self.args.lr_freezed /
                                                           self.args.lr_freezed_warmup_steps,
                                              end_factor=1,
                                              total_iters=self.args.lr_freezed_warmup_steps)
        else:
            self.scheduler = create_scheduler(optimizer,
                                              start_factor=self.args.learning_rate /
                                                           self.args.lr_warmup_steps,
                                              end_factor=1,
                                              total_iters=self.args.lr_warmup_steps)

        return model, optimizer, train_loader

    def tokenize_and_wrap_data(self, data: Dataset):
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
        batch['text'] = [
            line for line in batch['text'] if len(line) > 0 and not line.isspace()
        ]

        tokens = self.tokenizer(batch['text'],
                                padding='max_length',
                                max_length=self.args.max_length,
                                truncation=True)
        tokens['labels'] = self.class_labels.str2int(batch['label'])
        return tokens

    def tokenize_and_wrap_data_old(self, data: Dataset):
        """
        Tokenize dataset with tokenize_function and wrap with DatasetWrapper
        :param data: Dataset
        :return: DatasetWrapper(Dataset)
        """

        def tokenize_function(examples):
            """
            Using .map to tokenize each sentence examples['text'] in dataset
            :param examples: data
            :return: tokenized examples
            """
            # Remove empty lines
            examples['text'] = [
                line for line in examples['text'] if len(line) > 0 and not line.isspace()
            ]

            examples['labels'] = self.class_labels.str2int(examples['label'])

            return self.tokenizer(
                examples['text'],
                padding='max_length',
                truncation=True,
                max_length=self.args.max_length,
                # We use this option because DataCollatorForLanguageModeling (see below)
                # is more efficient when it receives the `special_tokens_mask`.
                return_special_tokens_mask=True,
            )

        tokenized = data.map(
            tokenize_function,
            batched=True,
            num_proc=1,
            remove_columns=['text'],  # , 'label'],
            load_from_cache_file=False,
            desc="Running tokenizer on dataset line_by_line",
        )

        wrapped = DatasetWrapper(tokenized)

        return wrapped

    def load_data(self, train: bool = True):
        """
        Load data using datasets.load_dataset for training and evaluation
        :param train: Whether to train model
        """
        if train:
            self.train_data = load_dataset('json',
                                           data_files=os.path.join(DATA_DIR, self.args.train_data),
                                           split='train')
            self.total_steps = int(
                len(self.train_data) / self.args.train_batch_size * self.args.epochs)
            self.args.total_steps = self.total_steps

            if self.args.compute_delta:
                self.args.delta = 1 / len(self.train_data)

        if self.args.evaluate_during_training:
            self.eval_data = load_dataset('json',
                                          data_files=os.path.join(DATA_DIR, self.args.eval_data),
                                          split='train')

    def label2id2label(self):
        label2id, id2label = dict(), dict()

        for i, label in enumerate(self.args.labels):
            label2id[label] = str(i)
            id2label[i] = label
        return label2id, id2label

    def compute_lr_automatically(self):

        if self.args.freeze_layers:
            self.args.lr_freezed_warmup_steps = int(np.ceil(0.1 * self.args.freeze_layers_n_steps))

        self.args.lr_warmup_steps = int(
            np.ceil(0.1 * (self.total_steps - self.args.freeze_layers_n_steps)))
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
    def get_data_collator(self):
        """
        Based on word mask load corresponding data collator
        :return: DataCollator
        """

        return DataCollatorWithPadding(tokenizer=self.tokenizer)

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
        Class inherited from MLMUnsupervisedModelling to train an unsupervised MLM model with
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
        save_json(output_dir, data, filename)
            Save list of dicts to json dump
        accuracy(preds, labels)
            Compute accuracy on predictions and labels
        create_scheduler(optimizer, start_factor, end_factor, total_iters)
            Create scheduler for learning rate warmup and decay
        """

    def __init__(self, args: argparse.Namespace):
        super().__init__(args)

        self.privacy_engine = None
        self.output_name = f'DP-eps-{int(self.args.epsilon)}-{self.args.model_name.replace("/", "_")}-' \
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
            eval_data_wrapped = self.tokenize_and_wrap_data(data=self.eval_data)
            eval_loader = DataLoader(dataset=eval_data_wrapped,
                                     collate_fn=self.data_collator,
                                     batch_size=self.args.eval_batch_size)
            losses = []
            accuracies = []
            for epoch in tqdm(range(self.args.epochs), desc="Epoch", unit="epoch"):
                model, losses, accuracies, step, lrs = \
                    self.train_epoch(model=model,
                                     train_loader=dp_train_loader,
                                     optimizer=dp_optimizer,
                                     val_loader=eval_loader,
                                     epoch=epoch + 1,
                                     step=step,
                                     eval_losses=losses,
                                     eval_accuracies=accuracies)

                all_lrs.extend(lrs)

            if step > self.args.freeze_layers_n_steps:
                min_loss, max_acc = get_max_acc_min_loss(losses, accuracies,
                                                         self.args.freeze_layers_n_steps)
                save_key_metrics_sc(output_dir=self.metrics_dir, args=self.args,
                                    best_acc=max_acc, best_loss=min_loss,
                                    total_steps=self.total_steps)

            if self.args.make_plots:
                plot_running_results(
                    output_dir=self.metrics_dir,
                    epsilon=str(self.args.epsilon),
                    delta=str(self.args.delta),
                    epochs=self.args.epochs,
                    lrs=all_lrs, accs=accuracies, loss=losses)

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
                model=model,
                output_dir=self.output_dir,
                data_collator=self.data_collator,
                tokenizer=self.tokenizer
            )
        self.model = model

    def train_epoch(self, model: GradSampleModule, train_loader: DPDataLoader,
                    optimizer: DPOptimizer, epoch: int = None, val_loader: DataLoader = None,
                    step: int = 0, eval_losses: List[dict] = None,
                    eval_accuracies: List[dict] = None):
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

            for batch in tqdm(memory_safe_data_loader, desc=f'Epoch {epoch} of {self.args.epochs}',
                              unit="batch"):

                if self.args.freeze_layers and step == 0:
                    model = self.freeze_layers(model)

                if self.args.freeze_layers and step == self.args.freeze_layers_n_steps:
                    model = self.unfreeze_layers(model)
                    # ToDo: Below operation only works if lr is lower than lr_freezed: fix this
                    self.scheduler = create_scheduler(optimizer,
                                                      start_factor=(self.args.learning_rate
                                                                    / self.args.lr_freezed)
                                                                   / self.args.lr_warmup_steps,
                                                      end_factor=self.args.learning_rate
                                                                 / self.args.lr_freezed,
                                                      total_iters=self.args.lr_warmup_steps)

                if step == self.args.lr_start_decay:
                    self.scheduler = create_scheduler(optimizer,
                                                      start_factor=1,
                                                      end_factor=self.args.learning_rate /
                                                                 (self.total_steps - step),
                                                      total_iters=self.total_steps -
                                                                  self.args.lr_start_decay)

                append_json(output_dir=self.metrics_dir, filename='learning_rates',
                            data={'epoch': epoch, 'step': step,
                                  'lr': get_lr(optimizer)[0]})
                lrs.append({'epoch': epoch, 'step': step, 'lr': get_lr(optimizer)[0]})

                # compute models
                output = model(input_ids=batch["input_ids"].to(self.args.device),
                               attention_mask=batch["attention_mask"].to(self.args.device),
                               labels=batch["labels"].to(self.args.device))

                loss = output.loss
                loss.backward()
                train_losses.append(loss.item())
                optimizer.step()
                # ToDo: Consider zero grad after optimizer.step? see test_mlm line 647
                optimizer.zero_grad()
                self.scheduler.step()

                if step % self.args.logging_steps == 0 and not step % self.args.evaluate_steps == 0:
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
                    eval_loss, eval_accuracy = self.evaluate(model, val_loader)
                    print(
                        f"\n"
                        f"eval loss: {eval_loss} \t"
                        f"eval acc: {eval_accuracy}"
                    )
                    append_json(output_dir=self.metrics_dir, filename='eval_losses',
                                data={'epoch': epoch, 'step': step, 'loss': eval_loss})
                    append_json(output_dir=self.metrics_dir, filename='accuracies',
                                data={'epoch': epoch, 'step': step, 'acc': eval_accuracy})

                    eval_losses.append({'epoch': epoch, 'step': step, 'loss': eval_loss})
                    eval_accuracies.append({'epoch': epoch, 'step': step, 'acc': eval_accuracy})

                    if self.args.save_steps is not None and (
                        step > 0 and (step % self.args.save_steps == 0)):
                        self.save_model_at_step(model, epoch, step, eval_losses, eval_accuracies)
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

        train_data_wrapped = self.tokenize_and_wrap_data(data=self.train_data)

        if self.args.load_alvenir_pretrained:
            model = BertForSequenceClassification.from_pretrained(
                self.local_alvenir_model_path,
                num_labels=len(self.args.labels),
                label2id=self.label2id,
                id2label=self.id2label,
                local_files_only=self.args.load_alvenir_pretrained)
        else:
            model = BertForSequenceClassification.from_pretrained(
                self.args.model_name,
                num_labels=len(self.args.labels),
                label2id=self.label2id,
                id2label=self.id2label,
                local_files_only=self.args.load_alvenir_pretrained)
        # ToDo: For now we are freezing embedding layer until (maybe)
        #  we have implemented grad sampler -
        #  as this is not implemented in opacus
        if self.args.freeze_embeddings:
            for param in model.bert.embeddings.parameters():
                param.requires_grad = False

        train_loader = DataLoader(dataset=train_data_wrapped,
                                  batch_size=self.args.lot_size,
                                  collate_fn=self.data_collator,
                                  shuffle=True)

        dummy_trainer = self.create_dummy_trainer(
            train_data_wrapped=train_data_wrapped,
            model=model)


        optimizer = dummy_trainer.create_optimizer()

        dp_model, dp_optimizer, dp_train_loader = self.set_up_privacy(
            train_loader=train_loader,
            model=model,
            optimizer=optimizer)

        # ToDo: finish head warmup lr
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

    def set_up_privacy(self, train_loader: DataLoader, model, optimizer):
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

        dp_model, dp_optimizer, dp_train_loader = self.privacy_engine.make_private_with_epsilon(
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
            # ToDo: Fix this
            if not name.startswith("_module.bert.embeddings"):
                param.requires_grad = True
        model.train()
        return model

    @staticmethod
    def save_model(model: GradSampleModule, output_dir: str, data_collator, tokenizer,
                   step: str = ""):
        """
        Wrap model in trainer class and save to pytorch object
        :param model: GradSampleModule to save
        :param output_dir: model directory
        :param data_collator:
        :param tokenizer:
        :param step: if saving during training step should be '/epoch-{epoch}_step-{step}'
        """
        output_dir = output_dir + step
        trainer_test = Trainer(
            model=model._module,
            args=TrainingArguments(output_dir=output_dir),
            data_collator=data_collator,
            tokenizer=tokenizer
        )
        trainer_test.save_model(output_dir=output_dir)
        torch.save(model._module.state_dict(), os.path.join(output_dir, 'model_weights.json'))
