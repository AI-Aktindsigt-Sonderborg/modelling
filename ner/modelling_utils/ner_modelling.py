import argparse
import datetime
import json
import os
import shutil
import sys
from datetime import datetime
from typing import List

import numpy as np
import torch
from datasets import ClassLabel
from opacus import GradSampleModule
from opacus.data_loader import DPDataLoader
from opacus.optimizers import DPOptimizer
from opacus.utils.batch_memory_manager import BatchMemoryManager
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, DataCollatorForTokenClassification, \
    AutoModelForTokenClassification, Trainer, TrainingArguments, BertConfig

from mlm.local_constants import MODEL_DIR as MLM_MODEL_DIR
from ner.data_utils.get_dataset import get_label_list, get_dane_train, \
    get_dane_val, get_dane_test
from ner.local_constants import MODEL_DIR
from ner.local_constants import PLOTS_DIR
from ner.modelling_utils.helpers import align_labels_with_tokens
from shared.data_utils.custom_dataclasses import EvalScore
from shared.data_utils.helpers import DatasetWrapper
from shared.modelling_utils.custom_modeling_bert import BertOnlyMLMHeadCustom
from shared.modelling_utils.helpers import get_lr, \
    log_train_metrics_dp
from shared.modelling_utils.modelling import Modelling
from shared.utils.visualization import plot_confusion_matrix


class NERModelling(Modelling):
    def __init__(self, args: argparse.Namespace):
        super().__init__(args=args)

        if self.args.custom_model_name:
            self.args.output_name = self.args.custom_model_name

        self.output_dir = os.path.join(MODEL_DIR, self.args.output_name)

        self.metrics_dir = os.path.join(self.output_dir, 'metrics')

        self.args.labels, self.id2label, self.label2id = get_label_list()

        self.class_labels = ClassLabel(
            num_classes=len(self.args.labels),
            names=self.args.labels)

        if self.args.load_alvenir_pretrained:
            self.model_path = os.path.join(MLM_MODEL_DIR,
                                           self.args.model_name,
                                           'best_model')
        else:
            self.model_path = self.args.model_name

        self.tokenizer = self.get_tokenizer()
        self.data_collator = self.get_data_collator()

    def evaluate(self, model, val_loader: DataLoader,
                 conf_plot: bool = False) -> EvalScore:
        """
        Evaluate model at given step
        :param device:
        :param model:
        :param val_loader:
        :return: mean eval loss and mean accuracy
        """
        if not next(model.parameters()).is_cuda:
            model = model.to(self.args.device)

        model.eval()

        y_true, y_pred, loss = [], [], []

        with torch.no_grad():
            # get model predictions and labels
            for batch in tqdm(val_loader, unit="batch", desc="Eval"):
                output = model(
                    input_ids=batch["input_ids"].to(self.args.device),
                    attention_mask=batch["attention_mask"].to(self.args.device),
                    labels=batch["labels"].to(self.args.device))

                batch_loss = output.loss.item()

                preds = np.argmax(output.logits.detach().cpu().numpy(), axis=-1)
                labels = batch["labels"].cpu().numpy()

                labels_flat = labels.flatten()
                preds_flat = preds.flatten()

                # We ignore tokens with value "-100" as these are padding tokens
                # set by the tokenizer.
                # See nn.CrossEntropyLoss(): ignore_index for more information
                filtered = [[xv, yv] for xv, yv in zip(labels_flat, preds_flat)
                            if xv != -100]

                batch_labels = np.array([x[0] for x in filtered]).astype(int)
                batch_preds = np.array([x[1] for x in filtered]).astype(int)

                y_true.extend(batch_labels)
                y_pred.extend(batch_preds)
                loss.append(batch_loss)

        # calculate metrics of interest
        acc = accuracy_score(y_true, y_pred)
        f_1 = f1_score(y_true, y_pred, average='macro')
        f_1_none = f1_score(y_true, y_pred, average=None)
        loss = float(np.mean(loss))

        print(f"\n"
              f"eval loss: {loss} \t"
              f"eval acc: {acc}"
              f"eval f1: {f_1}")

        if conf_plot:
            y_true_plot = list(map(lambda x: self.id2label[int(x)], y_true))
            y_pred_plot = list(map(lambda x: self.id2label[int(x)], y_pred))

            plot_confusion_matrix(
                y_true=y_true_plot,
                y_pred=y_pred_plot,
                labels=self.args.labels,
                model_name=self.args.model_name,
                plots_dir=PLOTS_DIR)

        return EvalScore(accuracy=acc, f_1=f_1, loss=loss,
                         f_1_none=list(f_1_none))

    def load_data(self, train: bool = True, test: bool = False):

        if train:
            self.data.train = get_dane_train(subset=None)
            print(f'len train: {len(self.data.train)}')
            self.args.total_steps = int(
                len(self.data.train) / self.args.train_batch_size
                * self.args.epochs)

            print(f'total steps: {self.args.total_steps}')

            if self.args.differential_privacy and self.args.compute_delta:
                self.args.delta = 1 / len(self.data.train)
                print(f'delta:{self.args.delta}')
            else:
                self.args.delta = None
                self.args.compute_delta = None
                self.args.epsilon = None
                self.args.max_grad_norm = None

        if self.args.evaluate_during_training:
            self.data.eval = get_dane_val(subset=None)
            print(f'len eval: {len(self.data.eval)}')

        if test:
            self.data.test = get_dane_test(subset=None)

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

        def tokenize_and_align_labels(examples):
            tokenized_inputs = self.tokenizer(
                examples["tokens"],
                truncation=True,
                is_split_into_words=True,
                padding='max_length',
                max_length=self.args.max_length
            )
            all_labels = examples["ner_tags"]
            new_labels = []
            for i, labels in enumerate(all_labels):
                word_ids = tokenized_inputs.word_ids(i)
                new_labels.append(align_labels_with_tokens(labels, word_ids))

            tokenized_inputs["labels"] = new_labels
            return tokenized_inputs

        tokenized = data.map(
            tokenize_and_align_labels,
            batched=True,
            remove_columns=data.column_names,
            desc="Running tokenizer on dataset line_by_line",
        )

        tokenized.set_format('torch')
        wrapped = DatasetWrapper(tokenized)

        return wrapped

    def tokenize_and_wrap_data_old(self, data: Dataset):
        """
        Tokenize dataset with tokenize_function and wrap with DatasetWrapper
        :param data: Dataset
        :return: DatasetWrapper(Dataset)
        """

        def tokenize_and_align_labels_old(examples):
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

        tokenized_dataset = data.map(tokenize_and_align_labels_old,
                                     batched=True)
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

    # @staticmethod
    # def save_config(output_dir: str, metrics_dir: str,
    #                 args: argparse.Namespace):
    #     """
    #     Save config file with input arguments
    #     :param output_dir: model directory
    #     :param args: input args
    #     """
    #
    #     if not os.path.exists(output_dir):
    #         os.makedirs(output_dir)
    #
    #     if not os.path.exists(metrics_dir):
    #         os.makedirs(metrics_dir)
    #
    #     with open(os.path.join(metrics_dir, 'training_run_config.json'), 'w',
    #               encoding='utf-8') as outfile:
    #         json.dump(args.__dict__, outfile, indent=2)
    #
    #     if args.print_only_args:
    #         metrics_path = os.path.dirname(os.path.abspath(metrics_dir))
    #         print('Training arguments:\n')
    #         for arg in args.__dict__:
    #             print(f'{arg}: {args.__dict__[arg]}')
    #
    #         print('\nDeleting model path...')
    #         shutil.rmtree(metrics_path)
    #
    #         print('\nExiting.')
    #         sys.exit(0)

    # @staticmethod
    # def save_model(model, output_dir: str,
    #                data_collator,
    #                tokenizer,
    #                step: str = ""):
    #     """
    #     Wrap model in trainer class and save to pytorch object
    #     :param model: model to save
    #     :param output_dir: model directory
    #     :param data_collator:
    #     :param tokenizer:
    #     :param step: if saving during training step should be '/epoch-{epoch}_step-{step}'
    #     """
    #     output_dir = output_dir + step
    #     trainer_test = Trainer(
    #         model=model,
    #         args=TrainingArguments(output_dir=output_dir),
    #         data_collator=data_collator,
    #         tokenizer=tokenizer
    #     )
    #     trainer_test.save_model(output_dir=output_dir)
    #     torch.save(model.state_dict(),
    #                os.path.join(output_dir, 'model_weights.json'))


class NERModellingDP(NERModelling):
    def __init__(self, args: argparse.Namespace):
        super().__init__(args)

        self.privacy_engine = None

        if self.args.custom_model_name:
            self.args.output_name = self.args.custom_model_name
        else:
            self.args.output_name = f'DP-eps-{int(self.args.epsilon)}-'\
                                    + self.args.output_name

        self.output_dir = os.path.join(MODEL_DIR, self.args.output_name)
        self.metrics_dir = os.path.join(self.output_dir, 'metrics')
        if not self.args.freeze_layers:
            self.args.freeze_layers_n_steps = 0

    def train_epoch(self, model: GradSampleModule, train_loader: DPDataLoader,
                    optimizer: DPOptimizer, epoch: int = None,
                    val_loader: DataLoader = None,
                    step: int = 0, eval_scores: List[EvalScore] = None):
        """
        Train one epoch with DP - modification of superclass train_epoch
        :param eval_scores: eval_scores
        :param model: Differentially private model wrapped in GradSampleModule
        :param train_loader: Differentially private data loader of type
        DPDataLoader
        :param optimizer: Differentially private optimizer of type DPOptimizer
        :param epoch: Given epoch: int
        :param val_loader: If evaluate_during_training: DataLoader containing
        validation data
        :param step: Given step
        :return: if self.eval_data: return model, eval_losses, eval_accuracies,
        step, lrs
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
                model, optimizer, eval_scores, train_losses, lrs = \
                    self.train_batch(
                        model=model,
                        optimizer=optimizer,
                        batch=batch,
                        val_loader=val_loader,
                        epoch=epoch,
                        step=step,
                        eval_scores=eval_scores,
                        train_losses=train_losses,
                        learning_rates=lrs
                    )

                log_train_metrics_dp(
                    epoch,
                    step,
                    lr=get_lr(optimizer)[0],
                    loss=float(np.mean(train_losses)),
                    eps=self.privacy_engine.get_epsilon(self.args.delta),
                    delta=self.args.delta,
                    logging_steps=self.args.logging_steps)

                step += 1

        if self.data.eval:
            return model, step, lrs, eval_scores
        return model, step, lrs

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
