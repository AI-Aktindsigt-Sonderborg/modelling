# pylint: disable=protected-access
import argparse
import os
from datetime import datetime
from typing import List

import numpy as np
import torch
from opacus import PrivacyEngine, GradSampleModule
from opacus.data_loader import DPDataLoader
from opacus.optimizers import DPOptimizer
from opacus.utils.batch_memory_manager import BatchMemoryManager
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import BertConfig, BertForMaskedLM, TrainingArguments, \
    Trainer, DataCollatorForWholeWordMask, \
    DataCollatorForLanguageModeling

from shared.data_utils.custom_dataclasses import EvalScore
from shared.data_utils.helpers import DatasetWrapper
from shared.modelling_utils.custom_modeling_bert import BertOnlyMLMHeadCustom
from shared.modelling_utils.helpers import create_scheduler, get_lr, \
    validate_model, get_metrics, log_train_metrics_dp, create_data_loader, \
    save_key_metrics
from shared.modelling_utils.modelling import Modelling
from shared.utils.helpers import TimeCode
from shared.utils.visualization import plot_running_results
from sm.local_constants import DATA_DIR, MODEL_DIR


class MLMModelling(Modelling):
    """
    Class to train an MLM unsupervised model
    """

    def __init__(self, args: argparse.Namespace):
        super().__init__(args=args)
        self.output_dir = os.path.join(MODEL_DIR, self.output_name)
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

            lm_head = new_head.to(self.args.device)
            model.cls = lm_head
            # ToDo: For now we are freezing embedding layer until (maybe)
            #  we have implemented grad sampler -
            #  as this is not implemented in opacus
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
        lm_head = lm_head.to(self.args.device)
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
        if self.args.whole_word_mask:
            return DataCollatorForWholeWordMask(
                tokenizer=self.tokenizer,
                mlm=True,
                mlm_probability=self.args.mlm_prob)
        return DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
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
    def save_model(model: BertForMaskedLM, output_dir: str,
                   data_collator,
                   tokenizer,
                   step: str = ""):
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

        self.privacy_engine = None
        self.output_name = f'DP-eps-{int(self.args.epsilon)}-' \
                           f'{self.args.model_name.replace("/", "_")}-' \
                           f'{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}'
        self.args.output_name = self.output_name
        self.output_dir = os.path.join(MODEL_DIR, self.output_name)
        self.metrics_dir = os.path.join(self.output_dir, 'metrics')

    def train_model(self):
        """
        Load data, set up private training and train with differential privacy
        """
        model, dp_optimizer, dp_train_loader = self.set_up_training()

        code_timer = TimeCode()
        all_lrs = []
        step = 0

        if self.eval_data:
            _, eval_loader = create_data_loader(
                data_wrapped=self.tokenize_and_wrap_data(self.eval_data),
                batch_size=self.args.eval_batch_size,
                data_collator=self.data_collator,
                shuffle=False)

            eval_scores = []
            for epoch in tqdm(range(self.args.epochs), desc="Epoch",
                              unit="epoch"):
                model, step, lrs, eval_scores = self.train_epoch(
                    model=model,
                    train_loader=dp_train_loader,
                    optimizer=dp_optimizer,
                    val_loader=eval_loader,
                    epoch=epoch + 1,
                    step=step,
                    eval_scores=eval_scores)

                all_lrs.extend(lrs)

            if step > self.args.freeze_layers_n_steps:
                best_metrics, _ = get_metrics(
                    eval_scores=eval_scores,
                    eval_metrics=self.args.eval_metrics)

                save_key_metrics(output_dir=self.metrics_dir,
                                     args=self.args,
                                     best_metrics=best_metrics,
                                     total_steps=self.args.total_steps)

            if self.args.make_plots:
                plot_running_results(
                    output_dir=self.metrics_dir,
                    epsilon=str(self.args.epsilon),
                    delta=str(self.args.delta),
                    epochs=self.args.epochs,
                    lrs=all_lrs,
                    metrics=eval_scores)

        else:
            for epoch in tqdm(range(self.args.epochs), desc="Epoch",
                              unit="epoch"):
                model, step, lrs = self.train_epoch(
                    model=model,
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
                    step: int = 0, eval_scores: List[EvalScore] = None):
        """
        Train one epoch with DP - modification of superclass train_epoch
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

        if self.eval_data:
            return model, step, lrs, eval_scores
        return model, step, lrs

    def set_up_training(self) -> tuple:
        """
        Load data and set up for training an unsupervised MLM model with
        differential privacy
        :return: model, optimizer and train_loader for DP training
        """
        self.load_data()

        if self.args.auto_lr_scheduling:
            self.compute_lr_automatically()

        if self.args.save_config:
            self.save_config(output_dir=self.output_dir,
                             metrics_dir=self.metrics_dir,
                             args=self.args)

        _, train_loader = create_data_loader(
            data_wrapped=self.tokenize_and_wrap_data(self.train_data),
            batch_size=self.args.lot_size,
            data_collator=self.data_collator)

        model = self.get_model()

        dp_model, dp_optimizer, dp_train_loader = self.set_up_privacy(
            train_loader=train_loader,
            model=model)

        if self.args.freeze_layers:
            self.scheduler = create_scheduler(
                dp_optimizer,
                start_factor=self.args.lr_freezed /
                             self.args.lr_freezed_warmup_steps,
                end_factor=1,
                total_iters=self.args.lr_freezed_warmup_steps)
        else:
            self.scheduler = create_scheduler(
                dp_optimizer,
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
        # opacus version >= 1.0 requires to generate optimizer from
        # model.parameters()
        optimizer = torch.optim.AdamW(model.parameters(),
                                      lr=self.args.learning_rate)

        dp_model, dp_optimizer, dp_train_loader = \
            self.privacy_engine.make_private_with_epsilon(
                module=model,
                optimizer=optimizer,
                data_loader=train_loader,
                epochs=self.args.epochs,
                target_epsilon=self.args.epsilon,
                target_delta=self.args.delta,
                max_grad_norm=self.args.max_grad_norm,
                # ToDo: hooks is the original opacus grad sampler. If we want
                #  to try new features, then see
                #  https://github.com/pytorch/opacus/releases
                grad_sample_mode="hooks"
            )
        self.privacy_engine.get_epsilon(self.args.delta)
        return dp_model, dp_optimizer, dp_train_loader

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
