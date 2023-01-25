import argparse
import datetime
import json
import os
import shutil
import sys
from datetime import datetime
from time import sleep
from typing import List

import numpy as np
import torch
from datasets import ClassLabel
from opacus import PrivacyEngine, GradSampleModule
from opacus.data_loader import DPDataLoader
from opacus.optimizers import DPOptimizer
from opacus.utils.batch_memory_manager import BatchMemoryManager
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, DataCollatorForTokenClassification, \
    AutoModelForTokenClassification, Trainer, TrainingArguments, BertConfig

from ner.data_utils.get_dataset import get_label_list, get_dane_train, \
    get_dane_val
from ner.local_constants import MODEL_DIR
from shared.modelling_utils.custom_modeling_bert import BertOnlyMLMHeadCustom
from shared.modelling_utils.helpers import create_scheduler, get_lr, \
    validate_model
from shared.modelling_utils.modelling import Modelling
from shared.utils.helpers import append_json


class NERModelling(Modelling):
    def __init__(self, args: argparse.Namespace):

        self.args = args

        self.args.output_name = self.output_name

        self.output_dir = os.path.join(MODEL_DIR, self.output_name)
        self.metrics_dir = os.path.join(self.output_dir, 'metrics')

        self.label_list, self.id2label, self.label2id = get_label_list()

        self.class_labels = ClassLabel(
            num_classes=len(self.label_list),
            names=self.label_list)

        if self.args.load_alvenir_pretrained:
            self.model_path = os.path.join(MODEL_DIR,
                                           self.args.model_name,
                                           'best_model')
        else:
            self.model_path = self.args.model_name

        self.tokenizer = self.get_tokenizer()
        self.data_collator = self.get_data_collator()

        self.model = AutoModelForTokenClassification.from_pretrained(
            self.local_alvenir_model_path,
            local_files_only=self.args.load_alvenir_pretrained,
            num_labels=len(self.label_list),
            id2label=self.id2label,
            label2id=self.label2id)
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.local_alvenir_model_path,
            local_files_only=self.args.load_alvenir_pretrained)

        config = BertConfig.from_pretrained(self.local_alvenir_model_path)
        lm_head = BertOnlyMLMHeadCustom(config)
        lm_head = lm_head.to(self.args.device)
        self.model.cls = lm_head

        self.model = AutoModelForTokenClassification.from_pretrained(
            self.args.model_name,
            local_files_only=self.args.load_alvenir_pretrained,
            num_labels=len(self.label_list),
            id2label=self.id2label,
            label2id=self.label2id)
        self.tokenizer = AutoTokenizer.from_pretrained(self.args.model_name)

        # self.label_list = list(self.ner_mapping.values())

        self.data_collator = self.get_data_collator
        self.scheduler = None

        self.scheduler = None
        # ToDo: Experiment with freezing layers - see SPRIN-159
        if not self.args.freeze_layers:
            self.args.freeze_layers_n_steps = 0
            self.args.lr_freezed_warmup_steps = None
            self.args.lr_freezed = None

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
                           attention_mask=batch["attention_mask"].to(
                               self.args.device),
                           labels=batch["labels"].to(self.args.device))

            preds = np.argmax(output.logits.detach().cpu().numpy(), axis=-1)
            labels = batch["labels"].cpu().numpy()

            labels_flat = labels.flatten()

            preds_flat = preds.flatten()

            # We ignore tokens with value "-100" as these are padding tokens set by the tokenizer.
            # See nn.CrossEntropyLoss(): ignore_index for more information
            # ToDo: get f1 score instead of accuracy and remove
            filtered = [[xv, yv] for xv, yv in zip(labels_flat, preds_flat) if
                        xv != -100 or xv != 0]

            eval_acc = accuracy(np.array([x[1] for x in filtered]),
                                np.array([x[0] for x in filtered]))
            eval_loss = output.loss.item()

            if not np.isnan(eval_loss):
                loss_arr.append(eval_loss)
            if not np.isnan(eval_acc):
                accuracy_arr.append(eval_acc)

            sleep(0.001)

        return np.mean(loss_arr), np.mean(accuracy_arr)

    def load_data(self, train: bool = True):

        if train:
            self.data.train = get_dane_train(subset=None)
            print(f'len train: {len(self.train_data)}')
            self.total_steps = int(
                len(self.train_data) / self.args.train_batch_size * self.args.epochs)
            self.args.total_steps = self.total_steps
            print(f'total steps: {self.total_steps}')

        if self.args.compute_delta:
            self.args.delta = 1 / len(self.data.train)
            print(f'delta:{self.args.delta}')

        if self.args.evaluate_during_training:
            self.eval_data = get_dane_val(subset=None)
            print(f'len eval: {len(self.eval_data)}')

    def load_model_and_replace_bert_head(self):
        """
        Load BertForMaskedLM, replace head and freeze all params in embeddings layer
        """
        model = AutoModelForTokenClassification.from_pretrained(
            self.args.model_name)
        config = BertConfig.from_pretrained(self.args.model_name)
        lm_head = BertOnlyMLMHeadCustom(config)
        lm_head = lm_head.to(self.args.device)
        model.cls = lm_head

        return model

    def tokenize_and_wrap_data(self, data: Dataset):
        """
        Tokenize dataset with tokenize_function and wrap with DatasetWrapper
        :param data: Dataset
        :return: DatasetWrapper(Dataset)
        """

        def tokenize_and_align_labels(examples):
            # print(examples['tokens'])
            tokenized_inputs = self.tokenizer(examples["tokens"],
                                              truncation=True,
                                              is_split_into_words=True,
                                              padding='max_length',
                                              max_length=self.args.max_length)

            labels = []
            labels_tokenized = []
            for i, label in enumerate(examples["ner_tags"]):

                label_tokenized = self.tokenizer.tokenize(
                    ' '.join(examples['tokens'][i]))
                label_tokenized.insert(0, "-100")
                label_tokenized.append("-100")

                word_ids = tokenized_inputs.word_ids(
                    batch_index=i)  # Map tokens to their respective word.

                previous_word_idx = None
                label_ids = []
                for word_idx in word_ids:  # Set the special tokens to -100.
                    if word_idx is None:
                        label_ids.append(-100)
                    elif word_idx != previous_word_idx:  # Only label the first token of a given word.
                        label_ids.append(label[word_idx])
                    else:
                        label_ids.append(-100)
                    previous_word_idx = word_idx
                labels.append(label_ids)
                labels_tokenized.append(label_tokenized)

            tokenized_inputs["labels"] = labels
            tokenized_inputs["labels_tokenized"] = labels_tokenized

            return tokenized_inputs

        tokenized_dataset = data.map(tokenize_and_align_labels, batched=True)
        # tokenized_dataset_new = tokenized_dataset.remove_columns(
        #     )
        if self.args.train_data == 'wikiann':
            tokenized_dataset_new = tokenized_dataset.remove_columns(
                ["spans", "langs", "ner_tags", "labels_tokenized", "tokens"])
        else:
            tokenized_dataset_new = tokenized_dataset.remove_columns(
                ['sent_id', 'text', 'tok_ids', 'tokens', 'lemmas', 'pos_tags',
                 'morph_tags',
                 'dep_ids', 'dep_labels', 'ner_tags', 'labels_tokenized'])

        # wrapped = DatasetWrapper(tokenized_dataset_new)

        return tokenized_dataset_new

    def compute_lr_automatically(self):

        if self.args.freeze_layers:
            # self.args.freeze_layers_n_steps = 20000
            self.args.lr_freezed_warmup_steps = int(
                np.ceil(0.1 * self.args.freeze_layers_n_steps))

        self.args.lr_warmup_steps = int(
            np.ceil(0.1 * (self.total_steps - self.args.freeze_layers_n_steps)))
        self.args.lr_start_decay = int(
            np.ceil((self.total_steps - self.args.freeze_layers_n_steps) *
                    0.5 + self.args.freeze_layers_n_steps))

    def save_model_at_step(self, model, epoch, step,
                           save_best_model: bool):
        """
        Save model at step and overwrite best_model if the model
        have improved evaluation performance.
        :param model: Current model
        :param epoch: Current epoch
        :param step: Current step
        """

        if not self.args.save_only_best_model:
            self.save_model(model, output_dir=self.output_dir,
                            data_collator=self.data_collator,
                            tokenizer=self.tokenizer,
                            step=f'/epoch-{epoch}_step-{step}')

        if save_best_model:
            self.save_model(model, output_dir=self.output_dir,
                            data_collator=self.data_collator,
                            tokenizer=self.tokenizer,
                            step='/best_model')

    def get_tokenizer(self):
        # ToDo: Consider do_lower_case=True, otherwise lowercase training data
        return AutoTokenizer.from_pretrained(
            self.model_path,
            local_files_only=self.args.load_alvenir_pretrained)

    def get_data_collator(self):
        """
        load DataCollatorForTokenClassification
        :return: DataCollator
        """
        return DataCollatorForTokenClassification(tokenizer=self.tokenizer)

    def get_model(self):
        return AutoModelForTokenClassification.from_pretrained(
            self.model_path,
            num_labels=len(self.args.labels),
            label2id=self.label2id,
            id2label=self.id2label,
            local_files_only=self.args.load_alvenir_pretrained)

    def modify_learning_rate_and_layers(self, model, optimizer, step: int):
        """
        Modify learning rate and freeze/unfreeze layers based on input args,
        learning rate scheduler and current step
        :return: Modified model
        """

        if self.args.freeze_layers and step == 0:
            model = self.freeze_layers(model)

        if self.args.freeze_layers and step == self.args.freeze_layers_n_steps:
            model = self.unfreeze_layers(model)
            # ToDo: Below operation only works if lr is lower than lr_freezed:
            #  fix this - also for SeqModelling
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
                total_iters=self.total_steps - step
            )
        return model

    @staticmethod
    def freeze_layers(model):
        """
        Freeze all bert layers in model, such that we can train only the head
        :param model: Model of type BertForMaskedLM
        :return: freezed model
        """
        for name, param in model.named_parameters():
            if not name.startswith("cls."):
                param.requires_grad = False
        model.train()
        return model

    @staticmethod
    def unfreeze_layers(model):
        """
        Un-freeze all layers exept for embeddings in model, such that we can train full model
        :param model: Model of type BertForMaskedLM
        :return: un-freezed model
        """
        for name, param in model.named_parameters():
            if not name.startswith("bert.embeddings"):
                param.requires_grad = True
        print("bert layers unfreezed")
        model.train()
        return model

    @staticmethod
    def save_config(output_dir: str, metrics_dir: str,
                    args: argparse.Namespace):
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

    @staticmethod
    def save_model(model, output_dir: str,
                   data_collator,
                   tokenizer,
                   step: str = ""):
        """
        Wrap model in trainer class and save to pytorch object
        :param model: model to save
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
        torch.save(model.state_dict(),
                   os.path.join(output_dir, 'model_weights.json'))


class NERModellingDP(NERModelling):
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

        model, dp_optimizer, dp_train_loader = self.set_up_training()

        all_lrs = []
        step = 0

        if self.eval_data:
            eval_data_wrapped = self.tokenize_and_wrap_data(data=self.eval_data)
            eval_loader = DataLoader(dataset=eval_data_wrapped,
                                     collate_fn=self.data_collator,
                                     batch_size=self.args.eval_batch_size)
            losses = []
            accuracies = []

            for epoch in tqdm(range(self.args.epochs), desc="Epoch",
                              unit="epoch"):
                model, losses, accuracies, step, lrs = self.train_epoch(
                    model=model,
                    train_loader=dp_train_loader,
                    optimizer=dp_optimizer,
                    val_loader=eval_loader,
                    epoch=epoch + 1,
                    step=step,
                    eval_losses=losses,
                    eval_accuracies=accuracies)
                all_lrs.extend(lrs)

        print('dp model is training')

    def train_epoch(self, model: GradSampleModule, train_loader: DPDataLoader,
                    optimizer: DPOptimizer, epoch: int = None,
                    val_loader: DataLoader = None,
                    step: int = 0, eval_losses: List[dict] = None,
                    eval_accuracies: List[dict] = None):
        """
        Train one epoch
        :param eval_accuracies:
        :param eval_losses:
        :param model: Model
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
        with BatchMemoryManager(
            data_loader=train_loader,
            max_physical_batch_size=self.args.train_batch_size,
            optimizer=optimizer
        ) as memory_safe_data_loader:

            for batch in tqdm(memory_safe_data_loader,
                              desc=f'Train epoch {epoch} of {self.args.epochs}',
                              unit="batch"):

                if self.args.freeze_layers and step == 0:
                    model = self.freeze_layers(model)

                if self.args.freeze_layers and step == self.args.freeze_layers_n_steps:
                    model = self.unfreeze_layers(model)
                    # ToDo: Below operation only works if lr is lower than lr_freezed: fix this
                    self.scheduler = create_scheduler(optimizer,
                                                      start_factor=(
                                                                       self.args.learning_rate
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
                append_json(output_dir=self.metrics_dir,
                            filename='learning_rates',
                            data={'epoch': epoch, 'step': step,
                                  'lr': get_lr(optimizer)[0]})
                lrs.append(
                    {'epoch': epoch, 'step': step, 'lr': get_lr(optimizer)[0]})

                optimizer.zero_grad()

                # compute model output
                output = model(
                    input_ids=batch["input_ids"].to(self.args.device),
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

                if val_loader and (
                    step > 0 and (step % self.args.evaluate_steps == 0)):
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
                    append_json(output_dir=self.metrics_dir,
                                filename='eval_losses',
                                data={'epoch': epoch, 'step': step,
                                      'loss': eval_loss})
                    append_json(output_dir=self.metrics_dir,
                                filename='accuracies',
                                data={'epoch': epoch, 'step': step,
                                      'acc': eval_accuracy})
                    eval_losses.append(
                        {'epoch': epoch, 'step': step, 'loss': eval_loss})
                    eval_accuracies.append(
                        {'epoch': epoch, 'step': step, 'acc': eval_accuracy})

                    if self.args.save_steps is not None and (
                        step > 0 and (step % self.args.save_steps == 0)):
                        self.save_model_at_step(model, epoch, step, eval_losses,
                                                eval_accuracies)
                    model.train()
                step += 1
        if self.eval_data:
            return model, eval_losses, eval_accuracies, step, lrs
        return model, step, lrs

    def set_up_training(self):
        """

        :return:
        """

        self.load_data()

        if self.args.auto_lr_scheduling:
            self.compute_lr_automatically()

        if self.args.save_config:
            self.save_config(output_dir=self.output_dir,
                             metrics_dir=self.metrics_dir,
                             args=self.args)

        train_data_wrapped = self.tokenize_and_wrap_data(data=self.train_data)

        if self.args.load_alvenir_pretrained:
            model = AutoModelForTokenClassification.from_pretrained(
                self.local_alvenir_model_path, num_labels=len(self.label_list),
                id2label=self.id2label, label2id=self.label2id)
            # config = BertConfig.from_pretrained(self.local_alvenir_model_path)
            # new_head = BertOnlyMLMHeadCustom(config)
            # new_head.load_state_dict(
            #     torch.load(self.local_alvenir_model_path + '/head_weights.json'))
            # lm_head = new_head.to(self.args.device)
            # model.cls = lm_head

            for param in model.bert.embeddings.parameters():
                param.requires_grad = False

        else:
            if self.args.replace_head:
                print('Replacing bert head')
                model = self.load_model_and_replace_bert_head()

            else:
                model = AutoModelForTokenClassification.from_pretrained(
                    self.args.model_name,
                    num_labels=len(
                        self.label_list),
                    id2label=self.id2label,
                    label2id=self.label2id)
                if self.args.freeze_embeddings:
                    for param in model.bert.embeddings.parameters():
                        param.requires_grad = False

        # dummy_trainer = self.create_dummy_trainer(train_data_wrapped=train_data_wrapped,
        #                                           model=model)

        train_loader = DataLoader(dataset=train_data_wrapped,
                                  batch_size=self.args.lot_size,
                                  collate_fn=self.data_collator)

        dp_model, dp_optimizer, dp_train_loader = self.set_up_privacy(
            train_loader=train_loader,
            model=model)

        if self.args.freeze_layers:
            self.scheduler = create_scheduler(dp_optimizer,
                                              start_factor=self.args.lr_freezed /
                                                           self.args.lr_freezed_warmup_steps,
                                              end_factor=1,
                                              total_iters=self.args.lr_freezed_warmup_steps)
        else:
            self.scheduler = create_scheduler(dp_optimizer,
                                              start_factor=self.args.learning_rate /
                                                           self.args.lr_warmup_steps,
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
        optimizer = torch.optim.AdamW(model.parameters(),
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
    def save_model(model: GradSampleModule, output_dir: str, data_collator,
                   tokenizer,
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
        torch.save(model._module.cls.state_dict(),
                   os.path.join(output_dir, 'head_weights.json'))
        torch.save(model._module.state_dict(),
                   os.path.join(output_dir, 'model_weights.json'))

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
