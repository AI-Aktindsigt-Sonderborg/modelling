import argparse
import os
from typing import List

import evaluate
import numpy as np
import torch
from datasets import ClassLabel, load_dataset
from opacus import GradSampleModule
from opacus.data_loader import DPDataLoader
from opacus.optimizers import DPOptimizer
from opacus.utils.batch_memory_manager import BatchMemoryManager
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from tqdm import tqdm
from transformers import AutoTokenizer, DataCollatorForTokenClassification, \
    AutoModelForTokenClassification, BertConfig

from ner.data_utils.get_dataset import get_label_list_old, get_dane_train, \
    get_dane_val, get_dane_test
from ner.local_constants import MODEL_DIR, DATA_DIR, PREP_DATA_DIR
from ner.local_constants import PLOTS_DIR
from ner.modelling_utils.helpers import align_labels_with_tokens, get_label_list
from shared.data_utils.custom_dataclasses import EvalScore
from shared.data_utils.helpers import DatasetWrapper
from shared.modelling_utils.custom_modeling_bert import BertOnlyMLMHeadCustom
from shared.modelling_utils.helpers import get_lr, \
    log_train_metrics_dp
from shared.modelling_utils.modelling import Modelling
from shared.utils.helpers import append_json_lines
from shared.utils.visualization import plot_confusion_matrix

seqeval = evaluate.load('seqeval')
f1_seq = evaluate.load('f1')


class NERModelling(Modelling):
    def __init__(self, args: argparse.Namespace):
        super().__init__(args=args)

        if not args.test:
            self.output_dir = os.path.join(MODEL_DIR, self.args.output_name)
        else:
            self.output_dir = os.path.join(MODEL_DIR, self.args.model_name)

        self.metrics_dir = os.path.join(self.output_dir, 'metrics')


        self.args.labels, self.id2label, self.label2id, self.label2weight = get_label_list(self.args.entities)
        self.class_labels = ClassLabel(
            num_classes=len(self.args.labels),
            names=self.args.labels)

        if args.train_data == 'dane':
            self.args.labels, self.id2label, self.label2id, self.label2weight  = get_label_list_old()
            self.class_labels = ClassLabel(
                num_classes=len(self.args.labels),
                names=self.args.labels)


        self.data_dir = PREP_DATA_DIR

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
        # ToDo: check if relevant

        model = model.to(self.args.device)

        model.eval()

        y_true, y_pred, loss, standard_losses = [], [], [], []

        with torch.no_grad():
            # get model predictions and labels
            for batch in tqdm(val_loader, unit="batch", desc="Eval"):
                output = model(
                    input_ids=batch["input_ids"].to(self.args.device),
                    attention_mask=batch["attention_mask"].to(self.args.device),
                    labels=batch["labels"].to(self.args.device))

                # batch_loss = output.loss.item()

                if not self.args.weight_classes:
                    batch_loss = output.loss.item()
                else:
                    logits = output.logits.detach().cpu()
                    labels_flat = batch['labels'].view(-1)
                    logits_flat = logits.view(-1, len(self.args.labels)).float()
                    batch_loss = torch.tensor(torch.mean(
                        self.weighted_loss_function(logits_flat.float(), labels_flat.long()))).item()
                    standard_loss = output.loss.item()
                    loss_function = torch.nn.CrossEntropyLoss()
                    weighted_loss_function = torch.nn.CrossEntropyLoss(weight=self.class_weights)
                    batch_loss = torch.tensor(loss_function(logits_flat.float(), labels_flat.long())).to(self.args.device).item()
                    loss_weighted = torch.tensor(weighted_loss_function(logits_flat.float(), labels_flat.long())).to(self.args.device)


                preds = np.argmax(output.logits.detach().cpu().numpy(), axis=-1)
                labels = batch["labels"].cpu().numpy()

                # We ignore tokens with value "-100" as these are padding tokens
                # set by the tokenizer.
                # See nn.CrossEntropyLoss(): ignore_index for more information

                batch_labels = [[self.id2label[l] for l in label if l != -100]
                                for label in labels]
                batch_preds = [
                    [self.id2label[p] for (p, l) in zip(prediction, label) if
                     l != -100]
                    for prediction, label in zip(preds, labels)
                ]

                y_true.extend(batch_labels)
                y_pred.extend(batch_preds)
                loss.append(batch_loss)
                standard_losses.append(standard_loss)

        # To Søren: this block is only used for comparing different f1 scores
        # Dont worry about it - seqeval not working for sonderborg NER classes
        # all_metrics = seqeval.compute(
        #     predictions=y_pred,
        #     references=y_true,
        #     scheme='IOB2')
        #
        # for k, v in all_metrics.items():
        #     if isinstance(v, dict):
        #         for j, u in v.items():
        #             all_metrics[k][j] = float(u)
        #     else:
        #         all_metrics[k] = float(v)
        # append_json_lines(output_dir=self.metrics_dir, data=all_metrics,
        #             filename='seqeval_metrics')
        # --------------------------------------------------------------------

        y_pred = [item for sublist in y_pred for item in sublist]
        y_true = [item for sublist in y_true for item in sublist]
        # calculate metrics of interest
        acc = accuracy_score(y_true, y_pred)
        f_1 = f1_score(y_true, y_pred, average='macro')
        f_1_none_ = f1_score(y_true, y_pred, average=None, labels=self.args.labels)
        f_1_none = [{self.args.labels[i]: f_1_none_[i]} for i in range(len(self.args.labels))]
        loss = float(np.mean(loss))
        standard_loss = float(np.mean(standard_losses))

        print(f"\n"
              f"eval loss: {loss}\t"
              f"standard eval loss: {standard_loss}\t"
              f"eval acc: {acc}\t"
              f"eval f1: {f_1}\t")

        if conf_plot:
            # y_true_plot = list(map(lambda x: self.id2label[int(x)], y_true))
            # y_pred_plot = list(map(lambda x: self.id2label[int(x)], y_pred))

            plot_confusion_matrix(
                y_true=y_true,
                y_pred=y_pred,
                labels=self.args.labels,
                model_name=self.args.model_name,
                plots_dir=PLOTS_DIR,
                concat_bilu=self.args.concat_bilu)

        return EvalScore(accuracy=acc, f_1=f_1, loss=loss,
                         f_1_none=f_1_none)

    def load_data(self, train: bool = True, test: bool = False):
        """
        Load data using datasets.load_dataset for training and evaluation
        This method is temporarily set to load dane data
        :param train: Whether to train model
        :param test: Whether to load test data
        """

        if train:
            if self.args.train_data == 'dane':
                self.data.train = get_dane_train(subset=self.args.data_subset)
            else:
                self.data.train = load_dataset(
                    'json',
                    data_files=os.path.join(self.data_dir, self.args.train_data),
                    split='train')


            print(f'len train: {len(self.data.train)}')
            self.args.total_steps = int(
                len(self.data.train) / self.args.train_batch_size
                * self.args.epochs)

            print(f'total steps: {self.args.total_steps}')

            if self.args.differential_privacy:
                if self.args.compute_delta:
                    self.args.delta = 1 / len(self.data.train)
            else:
                self.args.delta = None
                self.args.compute_delta = None
                self.args.epsilon = None
                self.args.max_grad_norm = None

        if self.args.evaluate_during_training:
            if self.args.train_data == 'dane':
                self.data.eval = get_dane_val(subset=self.args.data_subset)
            else:
                self.data.eval = load_dataset(
                    'json',
                    data_files=os.path.join(self.data_dir, self.args.eval_data),
                    split='train')

            print(f'len eval: {len(self.data.eval)}')

        if test:
            self.data.test = load_dataset(
                'json',
                data_files=os.path.join(self.data_dir, self.args.test_data),
                split='train')

    def tokenize_and_wrap_data(self, data: Dataset):

        def tokenize_and_align_labels(examples):
            tokenized_inputs = self.tokenizer(
                examples["tokens"],
                truncation=True,
                is_split_into_words=True,
                padding='max_length',
                max_length=self.args.max_length
            )
            # examples["ner_tags"] = [self.label2id(x) for x in examples['tags']]

            all_labels = examples["ner_tags"]
            new_labels = []
            attention_mask = []
            for i, labels in enumerate(all_labels):
                word_ids = tokenized_inputs.word_ids(i)
                new_labels.append(align_labels_with_tokens(labels, word_ids))
                attention_mask.append(list(map(lambda x: 1 if x != -100 else 0, align_labels_with_tokens(labels, word_ids))))
            tokenized_inputs["labels"] = new_labels
            tokenized_inputs["attention_mask"] = attention_mask

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
        Un-freeze all layers exept for embeddings in model, such that we can
        train full model
        :param model: Model of type BertForMaskedLM
        :return: un-freezed model
        """
        for name, param in model.named_parameters():
            if not name.startswith("bert.embeddings"):
                param.requires_grad = True
        print("bert layers unfreezed")
        model.train()
        return model


class NERModellingDP(NERModelling):
    def __init__(self, args: argparse.Namespace):
        super().__init__(args)

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
        Freeze all bert encoder layers in model, such that we can train only
        the head
        :param model: Model of type GradSampleModule
        :return: freezed model
        """
        for name, param in model._module.named_parameters():
            if not (name.startswith('bert.encoder.layer.11') or name.startswith(
                'classifier')):
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
        for name, param in model._module.named_parameters():
            if not name.startswith("bert.embeddings"):
                param.requires_grad = True
        model.train()
        return model
