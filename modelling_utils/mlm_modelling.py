# pylint: disable=protected-access
import argparse
import json
import os
import sys
from datetime import datetime
from typing import List

import numpy as np
from datasets import load_dataset
from opacus import PrivacyEngine, GradSampleModule
from opacus.data_loader import DPDataLoader
from opacus.optimizers import DPOptimizer
from opacus.utils.batch_memory_manager import BatchMemoryManager
from opacus.validators import ModuleValidator
from torch.optim.lr_scheduler import LinearLR
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import BertConfig, BertForMaskedLM, AutoTokenizer, TrainingArguments, \
    Trainer, DataCollatorForWholeWordMask, DataCollatorForLanguageModeling

from local_constants import DATA_DIR, MODEL_DIR
from modelling_utils.custom_modeling_bert import BertOnlyMLMHeadCustom
from utils.helpers import TimeCode
from utils.visualization import plot_running_results


class DatasetWrapper(Dataset):

    def __init__(self, dataset):
        self.dataset = dataset

    def __getitem__(self, item):
        item = int(item)
        return self.dataset[item]

    def __len__(self):
        return len(self.dataset)


class MLMUnsupervisedModelling:
    def __init__(self, args: argparse.Namespace):
        self.total_steps = None
        self.trainer = None
        self.args = args
        self.output_name = f'{self.args.model_name.replace("/", "_")}-' \
                           f'{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}'
        self.output_dir = os.path.join(MODEL_DIR, self.output_name)
        self.train_data = None
        self.eval_data = None
        self.model = None
        self.tokenizer = AutoTokenizer.from_pretrained(self.args.model_name)
        self.data_collator = self.get_data_collator
        self.scheduler = None

    def train_model(self):
        """
        load data, set up private training and train with privacy
        """
        optimizer, train_loader = self.set_up_training()

        code_timer = TimeCode()
        all_lrs = []
        step = 0
        model = self.model
        if self.eval_data:
            eval_data_wrapped = self.tokenize_and_wrap_data(data=self.eval_data)
            eval_loader = DataLoader(dataset=eval_data_wrapped,
                                     collate_fn=self.data_collator,
                                     batch_size=self.args.eval_batch_size)
            losses = []
            accuracies = []

            for epoch in tqdm(range(self.args.epochs), desc="Epoch", unit="epoch"):
                model, eval_loss, eval_accuracy, step, lrs = self.train_epoch(model=model,
                                                                              train_loader=train_loader,
                                                                              optimizer=optimizer,
                                                                              val_loader=eval_loader,
                                                                              epoch=epoch + 1,
                                                                              step=step)
                losses.append({epoch: eval_loss})
                accuracies.append({epoch: eval_accuracy})
                all_lrs.append({epoch: lrs})

            self.save_json(output_dir=self.output_dir, data=all_lrs, filename='learning_rates')
            self.save_json(output_dir=self.output_dir, data=losses, filename='eval_losses')
            self.save_json(output_dir=self.output_dir, data=accuracies, filename='accuracies')

            if self.args.make_plots:
                plot_running_results(
                    output_path=os.path.join(self.output_dir, 'results'),
                    title=f'Epochs: {self.args.epochs}',
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

    def train_epoch(self, model: BertForMaskedLM, train_loader: DataLoader,
                    optimizer, epoch: int = None, val_loader: DataLoader = None,
                    step: int = 0):
        """
        Train one epoch
        :param model:
        :param train_loader:
        :param optimizer:
        :param epoch:
        :param val_loader:
        :param step:
        :return:
        """

        model.train()
        model = model.to(self.args.device)

        train_losses = []
        eval_losses = []
        eval_accuracies = []
        lrs = []

        for batch in train_loader:

            if self.args.freeze_layers and step == 0:
                model = self.freeze_layers(model)

            if self.args.freeze_layers and step == self.args.freeze_layers_n_steps:
                model = self.unfreeze_layers(model)
                # ToDo: Below operation only works if lr is lower than lr_freezed: fix this
                self.scheduler = self.create_scheduler(optimizer,
                                                       start_factor=(self.args.lr
                                                                     / self.args.lr_freezed)
                                                                    / self.args.lr_warmup_steps,
                                                       end_factor=self.args.lr
                                                                  / self.args.lr_freezed,
                                                       total_iters=self.args.lr_warmup_steps)

            if step == self.args.lr_start_decay:
                self.scheduler = self.create_scheduler(optimizer,
                                                       start_factor=1,
                                                       end_factor=self.args.lr / (
                                                           self.total_steps - step),
                                                       total_iters=self.total_steps - step
                                                       )

            lrs.append({step: self.get_lr(optimizer)[0]})

            optimizer.zero_grad()

            # compute models
            output = model(input_ids=batch["input_ids"].to(self.args.device),
                           attention_mask=batch["attention_mask"].to(self.args.device),
                           labels=batch["labels"].to(self.args.device))

            loss = output.loss
            loss.backward()

            train_losses.append(loss.item())

            optimizer.step()
            self.scheduler.step()

            if val_loader and (step > 0 and (step % self.args.evaluate_steps == 0)):
                print(
                    f"\tTrain Epoch: {epoch} \t"
                    f"Step: {step} \t LR: {self.get_lr(optimizer)[0]}\t"
                    f"Loss: {np.mean(train_losses):.6f} "
                )
                eval_loss, eval_accuracy = self.evaluate(model, val_loader)
                print(
                    f"eval loss: {eval_loss} \t"
                    f"eval acc: {eval_accuracy}"
                )
                eval_losses.append({step: eval_loss})
                eval_accuracies.append({step: eval_accuracy})

                if self.args.save_steps is not None and (
                    step > 0 and (step % self.args.save_steps == 0)):
                    self.save_model(model, output_dir=self.output_dir,
                                    data_collator=self.data_collator,
                                    tokenizer=self.tokenizer,
                                    step=f'/epoch-{epoch}_step-{step}')
            step += 1
        if self.eval_data:
            return model, eval_losses, eval_accuracies, step, lrs
        return model, step, lrs

    def evaluate(self, model, val_loader: DataLoader):
        """
        Evaluate model at given step
        :param model:
        :param val_loader:
        :return: mean eval loss and mean accuracy
        """
        model.eval()

        loss_arr = []
        accuracy_arr = []

        for batch in val_loader:
            output = model(input_ids=batch["input_ids"].to('cuda'),
                           attention_mask=batch["attention_mask"].to('cuda'),
                           labels=batch["labels"].to('cuda'))

            preds = np.argmax(output.logits.detach().cpu().numpy(), axis=-1)
            labels = batch["labels"].cpu().numpy()

            labels_flat = labels.flatten()
            preds_flat = preds.flatten()

            filtered = [[xv, yv] for xv, yv in zip(labels_flat, preds_flat) if xv != -100]

            eval_acc = self.accuracy(np.array([x[1] for x in filtered]),
                                     np.array([x[0] for x in filtered]))
            eval_loss = output.loss.item()

            if not np.isnan(eval_loss):
                loss_arr.append(eval_loss)
            if not np.isnan(eval_acc):
                accuracy_arr.append(eval_acc)

        model.train()
        return np.mean(loss_arr), np.mean(accuracy_arr)

    def set_up_training(self):
        if self.args.save_config:
            self.save_config(output_dir=self.output_dir, args=self.args)

        self.load_data()
        train_data_wrapped = self.tokenize_and_wrap_data(data=self.train_data)

        if self.args.replace_head:
            print('Replacing bert head')
            self.load_model_and_replace_bert_head()
        else:
            self.model = BertForMaskedLM.from_pretrained(self.args.model_name)

        self.create_dummy_trainer(train_data_wrapped=train_data_wrapped)

        train_loader = DataLoader(dataset=train_data_wrapped, batch_size=self.args.train_batch_size,
                                  collate_fn=self.data_collator)

        optimizer = self.trainer.create_optimizer()

        self.scheduler = self.create_scheduler(optimizer,
                                               start_factor=self.args.lr / self.args.lr_warmup_steps,
                                               end_factor=1,
                                               total_iters=self.args.lr_warmup_steps)
        return optimizer, train_loader

    def create_dummy_trainer(self, train_data_wrapped: DatasetWrapper):
        """
        Create dummy trainer, such that we get optimizer and can save model object
        :param train_data_wrapped: Train data of type DatasetWrapper
        :return: Add self.trainer to class
        """
        if self.args.freeze_layers:
            learning_rate_init: float = self.args.lr_freezed
        else:
            learning_rate_init: float = self.args.lr

        training_args = TrainingArguments(
            output_dir=self.output_dir,
            learning_rate=learning_rate_init,
            weight_decay=self.args.weight_decay,
            fp16=self.args.use_fp16,
            # do_train=True
            # warmup_steps=self.args.lr_warmup_steps,
            # lr_scheduler_type='polynomial' # bert use linear scheduling
        )

        # Dummy training for optimizer
        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_data_wrapped,
            data_collator=self.data_collator,
            tokenizer=self.tokenizer
        )

    def tokenize_and_wrap_data(self, data: Dataset):

        def tokenize_function(examples):
            # Remove empty lines
            examples['text'] = [
                line for line in examples['text'] if len(line) > 0 and not line.isspace()
            ]

            return self.tokenizer(
                examples['text'],
                padding='max_length',
                truncation=True,
                max_length=self.args.max_length,
                # We use this option because DataCollatorForLanguageModeling (see below) is more efficient when it
                # receives the `special_tokens_mask`.
                return_special_tokens_mask=True,
            )

        tokenized = data.map(
            tokenize_function,
            batched=True,
            num_proc=1,
            remove_columns=['text'],
            load_from_cache_file=False,
            desc="Running tokenizer on dataset line_by_line",
        )

        wrapped = DatasetWrapper(tokenized)

        return wrapped

    def load_data(self, train: bool = True):

        if train:
            self.train_data = load_dataset('json',
                                           data_files=os.path.join(DATA_DIR, self.args.train_data),
                                           split='train')
            self.total_steps = int(
                len(self.train_data) / self.args.train_batch_size * self.args.epochs)

        if self.args.evaluate_during_training:
            self.eval_data = load_dataset('json',
                                          data_files=os.path.join(DATA_DIR, self.args.eval_data),
                                          split='train')

    def load_model_and_replace_bert_head(self):
        model = BertForMaskedLM.from_pretrained(self.args.model_name)

        config = BertConfig.from_pretrained(self.args.model_name)
        lm_head = BertOnlyMLMHeadCustom(config)
        lm_head = lm_head.to(self.args.device)
        model.cls = lm_head

        for param in model.bert.embeddings.parameters():
            param.requires_grad = False

        self.model = model

    @property
    def get_data_collator(self):
        """
        Based on word mask load corresponding data collator
        :return: DataCollator
        """
        if self.args.whole_word_mask:
            return DataCollatorForWholeWordMask(tokenizer=self.tokenizer, mlm=True,
                                                mlm_probability=self.args.mlm_prob)
        return DataCollatorForLanguageModeling(tokenizer=self.tokenizer, mlm=True,
                                               mlm_probability=self.args.mlm_prob)

    @staticmethod
    def freeze_layers(model):
        for name, param in model.named_parameters():
            if name.startswith("bert."):
                param.requires_grad = False
        model.train()
        return model

    @staticmethod
    def unfreeze_layers(model):
        for name, param in model.named_parameters():
            if name.startswith("bert.encoder"):
                param.requires_grad = True
        model.train()
        return model

    @staticmethod
    def save_model(model: BertForMaskedLM, output_dir: str,
                   data_collator,
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
            model=model,
            args=TrainingArguments(output_dir=output_dir),
            data_collator=data_collator,
            tokenizer=tokenizer
        )
        trainer_test.save_model(output_dir=output_dir)
        # trainer_test.save_metrics()

        model.save_pretrained(save_directory=output_dir)

    @staticmethod
    def save_config(output_dir: str, args: argparse.Namespace):
        """
        Save config file with input arguments
        :param output_dir: model directory
        :param args: input args
        """
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        with open(os.path.join(output_dir, 'training_run_config.json'), 'w',
                  encoding='utf-8') as outfile:
            json.dump(args.__dict__, outfile, indent=2)

    @staticmethod
    def get_lr(optimizer: DPOptimizer):
        """
        Get current learning rate from optimizer
        :param optimizer:
        :return: list of learning rates in all groups
        """
        lrs = []
        for param_group in optimizer.param_groups:
            lrs.append(param_group['lr'])
        return lrs

    @staticmethod
    def save_json(output_dir: str, data: List[dict], filename: str):
        """
        Save list of dicts to json dump
        :param output_dir:
        :param data: List[dict]
        :param filename: output file name
        """
        with open(os.path.join(output_dir, filename + '.json'), 'w',
                  encoding='utf-8') as outfile:
            for entry in data:
                json.dump(entry, outfile)
                outfile.write('\n')

    @staticmethod
    def accuracy(preds: np.array, labels: np.array):
        """
        Compute accuracy on predictions and labels
        :param preds: Predicted token
        :param labels: Input labelled token
        :return: Mean accuracy
        """
        return (preds == labels).mean()

    @staticmethod
    def create_scheduler(optimizer, start_factor: float, end_factor: float, total_iters: int):
        return LinearLR(optimizer, start_factor=start_factor,
                        end_factor=end_factor,
                        total_iters=total_iters)


class MLMUnsupervisedModellingDP(MLMUnsupervisedModelling):
    def __init__(self, args: argparse.Namespace):
        super().__init__(args)

        self.privacy_engine = None
        self.output_name = f'DP-eps-{self.args.epsilon}-{self.args.model_name.replace("/", "_")}-' \
                           f'{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}'
        self.output_dir = os.path.join(MODEL_DIR, self.output_name)

    def train_model(self):
        """
        Load data, set up private training and train with privacy
        :return: Add final model to class
        """
        dp_model, dp_optimizer, dp_train_loader = self.set_up_training()

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
                dp_model, eval_loss, eval_accuracy, step, lrs = self.train_epoch(model=dp_model,
                                                                                 train_loader=dp_train_loader,
                                                                                 optimizer=dp_optimizer,
                                                                                 val_loader=eval_loader,
                                                                                 epoch=epoch + 1,
                                                                                 step=step)
                losses.append({epoch: eval_loss})
                accuracies.append({epoch: eval_accuracy})
                all_lrs.append({epoch: lrs})

            self.save_json(output_dir=self.output_dir, data=all_lrs, filename='learning_rates')
            self.save_json(output_dir=self.output_dir, data=losses, filename='eval_losses')
            self.save_json(output_dir=self.output_dir, data=accuracies, filename='accuracies')

            if self.args.make_plots:
                plot_running_results(
                    output_path=os.path.join(self.output_dir,
                                             f'results_eps-{self.args.epsilon}-delta-{self.args.delta}'),
                    title=f'Epochs: {self.args.epochs}, Epsilon: {self.args.epsilon}, '
                          f'Delta: {self.args.delta}',
                    lrs=all_lrs, accs=accuracies, loss=losses)

        else:
            for epoch in tqdm(range(self.args.epochs), desc="Epoch", unit="epoch"):
                dp_model, eval_loss, eval_accuracy, step, lrs = self.train_epoch(model=dp_model,
                                                                                 train_loader=dp_train_loader,
                                                                                 optimizer=dp_optimizer,
                                                                                 step=step)
                all_lrs.append(lrs)
        code_timer.how_long_since_start()
        if self.args.save_model_at_end:
            self.save_model(
                model=dp_model,
                output_dir=self.output_dir,
                data_collator=self.data_collator,
                tokenizer=self.tokenizer
            )
        self.model = dp_model

    def train_epoch(self, model: GradSampleModule, train_loader: DPDataLoader,
                    optimizer: DPOptimizer, epoch: int = None, val_loader: DataLoader = None,
                    step: int = 0):
        """
        Train one epoch
        :param model:
        :param train_loader:
        :param optimizer:
        :param epoch:
        :param val_loader:
        :param step:
        :return:
        """

        model.train()
        model = model.to(self.args.device)

        train_losses = []
        eval_losses = []
        eval_accuracies = []
        lrs = []
        with BatchMemoryManager(
            data_loader=train_loader,
            max_physical_batch_size=self.args.train_batch_size,
            optimizer=optimizer
        ) as memory_safe_data_loader:

            # for i, batch in enumerate(memory_safe_data_loader):
            for batch in memory_safe_data_loader:

                if self.args.freeze_layers and step == 0:
                    model = self.freeze_layers(model)

                if self.args.freeze_layers and step == self.args.freeze_layers_n_steps:
                    model = self.unfreeze_layers(model)
                    # ToDo: Below operation only works if lr is lower than lr_freezed: fix this
                    self.scheduler = self.create_scheduler(optimizer,
                                                           start_factor=(self.args.lr
                                                                         / self.args.lr_freezed)
                                                                        / self.args.lr_warmup_steps,
                                                           end_factor=self.args.lr
                                                                      / self.args.lr_freezed,
                                                           total_iters=self.args.lr_warmup_steps)

                if step == self.args.lr_start_decay:
                    self.scheduler = self.create_scheduler(optimizer,
                                                           start_factor=1,
                                                           end_factor=self.args.lr /
                                                                      (self.total_steps - step),
                                                           total_iters=self.total_steps -
                                                                       self.args.lr_start_decay)

                lrs.append({step: self.get_lr(optimizer)[0]})

                # if step % 100 == 0:
                #     print(f'Step: {step}, lr: {self.get_lr(optimizer)[0]}')
                #     # print(f'Total steps: {self.total_steps}')

                optimizer.zero_grad()

                # compute models
                output = model(input_ids=batch["input_ids"].to(self.args.device),
                               attention_mask=batch["attention_mask"].to(self.args.device),
                               labels=batch["labels"].to(self.args.device))

                loss = output.loss
                loss.backward()

                train_losses.append(loss.item())

                optimizer.step()
                self.scheduler.step()

                eval_loss = None
                eval_accuracy = None

                if val_loader and (step > 0 and (step % self.args.evaluate_steps == 0)):
                    print(
                        f"\tTrain Epoch: {epoch} \t"
                        f"Step: {step} \t LR: {self.get_lr(optimizer)[0]}\t"
                        f"Loss: {np.mean(train_losses):.6f} "
                        f"(ε = {self.privacy_engine.get_epsilon(self.args.delta):.2f}, "
                        f"δ = {self.args.delta})"
                    )
                    eval_loss, eval_accuracy = self.evaluate(model, val_loader)
                    print(
                        f"eval loss: {eval_loss} \t"
                        f"eval acc: {eval_accuracy}"
                    )
                    eval_losses.append({step: eval_loss})
                    eval_accuracies.append({step: eval_accuracy})

                    if self.args.save_steps is not None and (
                        step > 0 and (step % self.args.save_steps == 0)):
                        self.save_model(model, output_dir=self.output_dir,
                                        data_collator=self.data_collator,
                                        tokenizer=self.tokenizer,
                                        step=f'/epoch-{epoch}_step-{step}')
                step += 1
        return model, eval_losses, eval_accuracies, step, lrs

    def set_up_training(self):

        if self.args.save_config:
            self.save_config(output_dir=self.output_dir, args=self.args)

        self.load_data()
        train_data_wrapped = self.tokenize_and_wrap_data(data=self.train_data)

        if self.args.replace_head:
            print('Replacing bert head')
            self.load_model_and_replace_bert_head()
        else:
            self.model = BertForMaskedLM.from_pretrained(self.args.model_name)

        self.create_dummy_trainer(train_data_wrapped=train_data_wrapped)

        train_loader = DataLoader(dataset=train_data_wrapped, batch_size=self.args.lot_size,
                                  collate_fn=self.data_collator)

        dp_model, dp_optimizer, dp_train_loader = self.set_up_privacy(train_loader=train_loader)

        # ToDo: finish head warmup lr
        self.scheduler = self.create_scheduler(dp_optimizer,
                                               start_factor=self.args.lr_freezed / self.args.lr_freezed_warmup_steps,
                                               end_factor=1,
                                               total_iters=self.args.lr_freezed_warmup_steps)

        # self.scheduler = LinearLR(dp_optimizer,
        #                           start_factor=self.args.lr_freezed_warmup / self.args.lr_freezed_warmup_steps,
        #                           end_factor=1,
        #                           total_iters=self.args.lr_freezed_warmup_steps)
        return dp_model, dp_optimizer, dp_train_loader

    def set_up_privacy(self, train_loader: DataLoader):
        self.privacy_engine = PrivacyEngine()

        # for param in self.model.bert.embeddings.parameters():
        #     param.requires_grad = False

        self.model = self.model.train()

        # validate if model works with opacus
        self.validate_model(self.model)

        self.trainer.create_optimizer()

        dp_model, dp_optimizer, dp_train_loader = self.privacy_engine.make_private_with_epsilon(
            module=self.model,
            optimizer=self.trainer.optimizer,
            data_loader=train_loader,
            epochs=self.args.epochs,
            target_epsilon=self.args.epsilon,
            target_delta=self.args.delta,
            max_grad_norm=self.args.max_grad_norm
        )

        self.trainer = None

        return dp_model, dp_optimizer, dp_train_loader

    # def load_model_and_replace_bert_head(self):
    #     model = BertForMaskedLM.from_pretrained(self.args.model_name)
    #
    #     config = BertConfig.from_pretrained(self.args.model_name)
    #     lm_head = BertOnlyMLMHeadCustom(config)
    #     lm_head = lm_head.to(self.args.device)
    #     model.cls = lm_head
    #
    #     for param in model.bert.embeddings.parameters():
    #         param.requires_grad = False
    #
    #     self.model = model

    # @staticmethod
    # def freeze_layers(model):
    #     for name, param in model.named_parameters():
    #         if name.startswith("_module.bert.encoder"):
    #             param.requires_grad = False
    #     model.train()
    #     return model
    #
    # @staticmethod
    # def unfreeze_layers(model):
    #     for name, param in model.named_parameters():
    #         if name.startswith("_module.bert.encoder"):
    #             param.requires_grad = True
    #     model.train()
    #     return model

    @staticmethod
    def validate_model(model: BertForMaskedLM, strict: bool = False):
        """
        Validate whether model is compatible with opacus
        :param model: pytorch model of type BertForMaskedLM
        :param strict: Whether validation should be strict
        """
        errors = ModuleValidator.validate(model, strict=strict)
        if errors:
            print("Model is not compatible for DF with opacus. Please fix errors.")
            print(errors)
            sys.exit(0)
        else:
            print("Model is compatible for DP with opacus.")

    @staticmethod
    def freeze_layers(model):
        for name, param in model.named_parameters():
            if name.startswith("_module.bert.encoder"):
                param.requires_grad = False
        model.train()
        return model

    @staticmethod
    def unfreeze_layers(model):
        for name, param in model.named_parameters():
            if name.startswith("_module.bert.encoder"):
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
            model=model,
            args=TrainingArguments(output_dir=output_dir),
            data_collator=data_collator,
            tokenizer=tokenizer
        )
        trainer_test.save_model(output_dir=output_dir)
        # trainer_test.save_metrics()

        model._module.save_pretrained(save_directory=output_dir)
