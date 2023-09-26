# pylint: disable=protected-access
import argparse
import os
from typing import List

import numpy as np
import torch
from opacus import GradSampleModule
from opacus.data_loader import DPDataLoader
from opacus.optimizers import DPOptimizer
from opacus.utils.batch_memory_manager import BatchMemoryManager
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import BertConfig, BertForMaskedLM, TrainingArguments, \
    Trainer, DataCollatorForWholeWordMask, \
    DataCollatorForLanguageModeling

from mlm.local_constants import DATA_DIR, MODEL_DIR
from shared.data_utils.custom_dataclasses import EvalScore
from shared.data_utils.helpers import DatasetWrapper
from shared.modelling_utils.custom_modeling_bert import BertOnlyMLMHeadCustom
from shared.modelling_utils.helpers import get_lr, \
    log_train_metrics_dp
from shared.modelling_utils.modelling import Modelling


class MLMModelling(Modelling):
    """
    Class to train an MLM unsupervised model
    """

    def __init__(self, args: argparse.Namespace):
        super().__init__(args=args)

        if not self.args.custom_model_name:
            self.args.output_name = "mlm-" + self.args.output_name

        self.output_dir = os.path.join(MODEL_DIR, self.args.output_name)
        self.metrics_dir = os.path.join(self.output_dir, 'metrics')
        self.data_dir = DATA_DIR

        if self.args.load_alvenir_pretrained:
            self.model_path = os.path.join(MODEL_DIR,
                                           self.args.model_name,
                                           'best_model')
        else:
            self.model_path = self.args.model_name

        self.tokenizer = self.get_tokenizer()
        self.data_collator = self.get_data_collator()

    def tokenize_and_wrap_data(self, data: Dataset):
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
                line for line in examples['text'] if
                len(line) > 0 and not line.isspace()
            ]

            return self.tokenizer(
                examples['text'],
                padding='max_length',
                truncation=True,
                max_length=self.args.max_length,
                # We use this option because DataCollatorForLanguageModeling
                # (see below)
                # is more efficient when it receives the `special_tokens_mask`.
                return_special_tokens_mask=True,
            )

        tokenized = data.map(
            tokenize_function,
            batched=True,
            remove_columns=data.column_names,
            load_from_cache_file=False,
            desc="Running tokenizer on dataset line_by_line",
        )

        wrapped = DatasetWrapper(tokenized)

        return wrapped

    def get_model(self):
        """
        get model based on model type
        :return: model
        """

        if self.args.load_alvenir_pretrained:
            model = BertForMaskedLM.from_pretrained(
                self.model_path,
                local_files_only=self.args.load_alvenir_pretrained)
            config = BertConfig.from_pretrained(
                self.model_path,
                local_files_only=self.args.load_alvenir_pretrained)
            new_head = BertOnlyMLMHeadCustom(config)
            new_head.load_state_dict(
                torch.load(self.model_path + '/head_weights.json'))

            # lm_head = new_head.to(self.args.device)
            model.cls = new_head
            # We freeze embeddings layer, as this is not implemented in opacus,
            # and it is shown that unfreezing embeddings doesn't provide better
            # output
            for param in model.bert.embeddings.parameters():
                param.requires_grad = False
            return model

        if self.args.replace_head:
            print('Replacing bert head')
            return self.load_model_and_replace_bert_head()

        return BertForMaskedLM.from_pretrained(
            self.model_path,
            local_files_only=self.args.load_alvenir_pretrained)

    def load_model_and_replace_bert_head(self):
        """
        Load BertForMaskedLM, replace head and freeze all embedding-params
        """
        model = BertForMaskedLM.from_pretrained(
            self.model_path,
            local_files_only=self.args.load_alvenir_pretrained)
        config = BertConfig.from_pretrained(self.model_path)
        lm_head = BertOnlyMLMHeadCustom(config)
        # lm_head = lm_head.to(self.args.device)
        model.cls = lm_head

        if self.args.freeze_embeddings:
            for param in model.bert.embeddings.parameters():
                param.requires_grad = False
        return model

    def get_data_collator(self):
        """
        Based on word mask load corresponding data collator
        :return: DataCollator
        """
        collator_class = DataCollatorForWholeWordMask if \
            self.args.whole_word_mask else DataCollatorForLanguageModeling

        return collator_class(tokenizer=self.tokenizer,
                              mlm=True,
                              mlm_probability=self.args.mlm_prob)

    def evaluate(self, model, val_loader: DataLoader) -> EvalScore:
        """
        Evaluate model at given step
        :param device:
        :param model:
        :param val_loader:
        :return: mean eval loss and mean accuracy
        """
        # if not next(model.parameters()).is_cuda:
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

                # batch_labels = [
                #     [l for l in label if l != -100] for label in labels
                # ]
                #
                # batch_preds = [
                #     [p for (p, l) in zip(prediction, label) if l != -100]
                #     for prediction, label in zip(preds, labels)
                # ]

                # labels_flat = labels.flatten()
                # preds_flat = preds.flatten()

                # We ignore tokens with value "-100" as these are padding tokens
                # set by the tokenizer.
                # See nn.CrossEntropyLoss(): ignore_index for more information
                filtered = [[xv, yv] for xv, yv in
                            zip(labels.flatten(), preds.flatten())
                            if xv != -100]

                batch_labels = np.array([x[0] for x in filtered]).astype(int)
                batch_preds = np.array([x[1] for x in filtered]).astype(int)

                y_true.extend(batch_labels)
                y_pred.extend(batch_preds)
                loss.append(batch_loss)

        # calculate metrics of interest
        acc = accuracy_score(y_true, y_pred)
        f_1 = f1_score(y_true, y_pred, average='macro')
        loss = float(np.mean(loss))

        print(f"\n"
              f"eval loss: {loss} \t"
              f"eval acc: {acc}"
              f"eval f1: {f_1}")

        return EvalScore(accuracy=acc, f_1=f_1, loss=loss)

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
        :param model: A model of type BertForMaskedLM
        :return: un-freezed model
        """
        for name, param in model.named_parameters():
            if not name.startswith("bert.embeddings"):
                param.requires_grad = True
        print("bert layers unfreezed")
        model.train()
        return model

    @staticmethod
    def save_model(model, output_dir: str,
                   data_collator,
                   tokenizer,
                   step: str = "",
                   dp: bool = False):
        """
        Wrap model in trainer class and save to pytorch object
        :param model: BertForMaskedLM to save
        :param output_dir: model directory
        :param data_collator:
        :param tokenizer:
        :param step: if saving during training step should be
        '/epoch-{epoch}_step-{step}'
        """
        output_dir = output_dir + step
        if dp:
            model = model._module

        trainer_test = Trainer(
            model=model,
            args=TrainingArguments(output_dir=output_dir),
            data_collator=data_collator,
            tokenizer=tokenizer
        )
        trainer_test.save_model(output_dir=output_dir)
        torch.save(model.cls.state_dict(),
                   os.path.join(output_dir, 'head_weights.json'))
        torch.save(model.state_dict(),
                   os.path.join(output_dir, 'model_weights.json'))


class MLMModellingDP(MLMModelling):
    """
        Class inherited from MLMUnsupervisedModelling to train an unsupervised
        MLM model with differential privacy
    """

    def __init__(self, args: argparse.Namespace):
        super().__init__(args)

        self.output_dir = os.path.join(MODEL_DIR, self.args.output_name)
        self.metrics_dir = os.path.join(self.output_dir, 'metrics')

    def train_epoch(self, model: GradSampleModule, train_loader: DPDataLoader,
                    optimizer: DPOptimizer,
                    val_loader: DataLoader = None, epoch: int = None,
                    step: int = 0, eval_scores: List[EvalScore] = None):
        """
        Train one epoch with DP - modification of superclass train_epoch
        :param model: Differentially private model wrapped in GradSampleModule
        :param train_loader: Differentially private data loader of type
        DPDataLoader
        :param optimizer: Differentially private optimizer of type DPOptimizer
        :param epoch: Given epoch: int
        :param val_loader: DataLoader containing validation data
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
