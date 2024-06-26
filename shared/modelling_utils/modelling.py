import argparse
import dataclasses
import json
import os
import shutil
import sys
from datetime import datetime
from typing import List

import numpy as np
import torch
from datasets import load_dataset
from opacus import PrivacyEngine
from sklearn.metrics import accuracy_score, f1_score
from sklearn.utils import compute_class_weight
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (
    Trainer,
    TrainingArguments,
    AutoModel,
    AutoTokenizer,
    DataCollator,
)

from mlm.local_constants import MODEL_DIR as MLM_MODEL_DIR
from sc.local_constants import MODEL_DIR as SC_MODEL_DIR
from shared.data_utils.custom_dataclasses import EvalScore, ModellingData
from shared.data_utils.helpers import DatasetWrapper
from shared.modelling_utils.helpers import (
    create_scheduler,
    create_data_loader,
    get_lr,
    get_metrics,
    log_train_metrics,
    save_key_metrics,
    validate_model,
    predefined_hf_models,
)
from shared.utils.helpers import append_json_lines, TimeCode, write_json_lines
from shared.utils.visualization import plot_running_results


class Modelling:
    """
    General class for model training. For input arguments see :class:`.ModellingArgParser`.


    Methods
    -------
    """

    def __init__(self, args: argparse.Namespace):
        self.args = args

        if self.args.custom_model_name:
            self.args.output_name = self.args.custom_model_name
        else:
            self.args.output_name = (
                f'{self.args.model_name.replace("/", "_")}'
                f'-{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}'
            )
            if self.args.differential_privacy:
                self.args.output_name = (
                    f"DP-eps-{int(self.args.epsilon)}-" + self.args.output_name
                )

        # Initialialize all variables relevant for modelling
        self.privacy_engine = None
        self.model = None
        self.scheduler = None
        self.output_dir = None
        self.metrics_dir = None
        self.tokenizer = None
        self.data_collator = None
        self.data = ModellingData

        if not self.args.freeze_layers:
            self.args.freeze_layers_n_steps = 0
            self.args.lr_freezed_warmup_steps = None
            self.args.lr_freezed = None

        if not self.args.differential_privacy:
            self.args.lot_size = self.args.train_batch_size

        if self.args.load_alvenir_pretrained:
            model_dir = (
                args.custom_model_dir if args.custom_model_dir else MLM_MODEL_DIR
            )
            self.model_path = os.path.join(
                model_dir, self.args.model_name, "best_model"
            )
        else:
            self.args.model_name = predefined_hf_models(self.args.model_name)
            self.model_path = self.args.model_name

    def load_data(self, train: bool = True, test: bool = False):
        """
        Load data using datasets.load_dataset for training and evaluation

        :param bool train: Whether to train model
        :param bool test: Whether to load test data
        """
        if train:
            self.data.train = load_dataset(
                "json",
                data_files=os.path.join(self.data_dir, self.args.train_data),
                split="train",
            )
            self.args.total_steps = int(
                len(self.data.train) / self.args.train_batch_size * self.args.epochs
            )

            if self.args.differential_privacy:
                if self.args.compute_delta:
                    self.args.delta = 1 / len(self.data.train)
            else:
                self.args.delta = None
                self.args.compute_delta = None
                self.args.epsilon = None
                self.args.max_grad_norm = None

            self.data.eval = load_dataset(
                "json",
                data_files=os.path.join(self.data_dir, self.args.eval_data),
                split="train",
            )

        if test:
            self.data.test = load_dataset(
                "json",
                data_files=os.path.join(self.data_dir, self.args.test_data),
                split="train",
            )

    def set_up_training(self) -> tuple:
        """
        Load data and set up for training
        :return: model, optimizer and train_loader for training
        """
        self.load_data()

        if self.args.auto_lr_scheduling:
            self.compute_lr_automatically()

        if self.args.weight_classes:
            label_ids = [self.label2id[label] for label in self.args.labels]

            ner_tags = []
            for tag in self.data.train["tags"]:
                ner_tags.extend([self.label2id[label] for label in tag])

            # important that all tags are represented in training set -
            # otherwise we stop training
            assert sorted(list(set(ner_tags))) == sorted(list(set(label_ids)))

            ner_tags = np.array(ner_tags)

            self.class_weights = torch.tensor(
                compute_class_weight(
                    class_weight="balanced",
                    classes=np.unique(label_ids),
                    y=ner_tags,
                )
            ).float()

            self.args.class_weights = self.class_weights.tolist()

        if self.args.save_config:
            self.save_config(
                output_dir=self.output_dir, metrics_dir=self.metrics_dir,
                args=self.args
            )

        train_data_wrapped, train_loader = create_data_loader(
            data_wrapped=self.tokenize_and_wrap_data(self.data.train),
            batch_size=self.args.lot_size,
            data_collator=self.data_collator,
        )

        model = self.get_model()
        dummy_trainer_ = self.create_dummy_trainer(
            train_data_wrapped=train_data_wrapped, model=model
        )
        # optimizer_ = dummy_trainer_.create_optimizer()
        dummy_trainer_.save_model(output_dir=self.output_dir)

        if self.args.freeze_embeddings:
            for param in model.bert.embeddings.parameters():
                param.requires_grad = False

        if self.args.differential_privacy:
            model, optimizer, train_loader = self.set_up_privacy(
                train_loader=train_loader, model=model
            )
        else:
            dummy_trainer = self.create_dummy_trainer(
                train_data_wrapped=train_data_wrapped, model=model
            )
            optimizer = dummy_trainer.create_optimizer()
            dummy_trainer.save_model(output_dir=self.output_dir)

        if self.args.freeze_layers:
            self.scheduler = create_scheduler(
                optimizer,
                start_factor=self.args.lr_freezed / self.args.lr_freezed_warmup_steps,
                end_factor=1,
                total_iters=self.args.lr_freezed_warmup_steps,
            )
        elif self.args.static_lr:
            self.scheduler = create_scheduler(
                optimizer, start_factor=1, end_factor=1, total_iters=1
            )
        else:
            self.scheduler = create_scheduler(
                optimizer,
                start_factor=self.args.learning_rate / self.args.lr_warmup_steps,
                end_factor=1,
                total_iters=self.args.lr_warmup_steps,
            )
        return model, optimizer, train_loader

    def create_dummy_trainer(self, train_data_wrapped: DatasetWrapper, model):
        """
        Create dummy trainer, such that we get optimizer and can save model
        object
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
            fp16=False,
        )

        # Dummy training for optimizer
        return Trainer(
            model=model,
            args=training_args,
            train_dataset=train_data_wrapped,
            data_collator=self.data_collator,
            tokenizer=self.tokenizer,
        )

    def compute_lr_automatically(self):
        """
        Compute the learning rate schedule automatically based on the number
        of training steps.
        This function sets several parameters in the `self.args` object,
        including `lr_freezed_warmup_steps`, `lr_warmup_steps`, and
        `lr_start_decay`. These parameters are used to control the learning
        rate schedule during training.
        If `self.args.freeze_layers` is true, `lr_freezed_warmup_steps` is
        set to 10% of the number of steps used to train the frozen layers.
        `lr_warmup_steps` is set to 10% of the total number of training steps,
        after the frozen layers have been trained.
        `lr_start_decay` is set to be half way through the remaining steps
        after the frozen layers have been trained.
        """

        if self.args.freeze_layers and not self.args.lr_freezed_warmup_steps:
            self.args.lr_freezed_warmup_steps = int(
                np.ceil(0.1 * self.args.freeze_layers_n_steps)
            )

        self.args.lr_warmup_steps = int(
            np.ceil(
                0.1 * (self.args.total_steps - self.args.freeze_layers_n_steps))
        )
        self.args.lr_start_decay = int(
            np.ceil(
                (self.args.total_steps - self.args.freeze_layers_n_steps) * 0.5
                + self.args.freeze_layers_n_steps
            )
        )

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

            self.scheduler = create_scheduler(
                optimizer,
                start_factor=(self.args.learning_rate / self.args.lr_freezed)
                             / self.args.lr_warmup_steps,
                end_factor=self.args.learning_rate / self.args.lr_freezed,
                total_iters=self.args.lr_warmup_steps,
            )

        if (
            step == self.args.lr_start_decay
            and self.args.total_steps > self.args.lr_start_decay
            and not self.args.static_lr
        ):
            self.scheduler = create_scheduler(
                optimizer,
                start_factor=1,
                end_factor=self.args.learning_rate / (
                        self.args.total_steps - step),
                total_iters=self.args.total_steps - step,
            )
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

        with open(
            os.path.join(metrics_dir, "training_run_config.json"), "w",
            encoding="utf-8"
        ) as outfile:
            json.dump(args.__dict__, outfile, indent=2)

        if args.print_only_args:
            metrics_path = os.path.dirname(os.path.abspath(metrics_dir))
            print("Training arguments:\n")
            for arg in args.__dict__:
                print(f"{arg}: {args.__dict__[arg]}")

            print("\nDeleting model path...")
            shutil.rmtree(metrics_path)

            print("\nExiting.")
            sys.exit(0)

    def train_model(self):
        """
        load data, set up training and train model
        """

        model, optimizer, train_loader = self.set_up_training()

        code_timer = TimeCode()
        all_lrs = []
        step = 0

        if self.data.eval:
            _, eval_loader = create_data_loader(
                data_wrapped=self.tokenize_and_wrap_data(self.data.eval),
                batch_size=self.args.eval_batch_size,
                data_collator=self.data_collator,
                shuffle=False,
            )

            eval_scores = []
            for epoch in tqdm(range(self.args.epochs), desc="Epoch",
                              unit="epoch"):
                model, step, lrs, eval_scores = self.train_epoch(
                    model=model,
                    train_loader=train_loader,
                    optimizer=optimizer,
                    val_loader=eval_loader,
                    epoch=epoch + 1,
                    step=step,
                    eval_scores=eval_scores,
                )
                all_lrs.extend(lrs)

            if (
                step > self.args.freeze_layers_n_steps
                and self.args.evaluate_steps <= self.args.total_steps
            ):
                best_metrics, _ = get_metrics(
                    eval_scores=eval_scores, eval_metrics=self.args.eval_metrics
                )

                save_key_metrics(
                    output_dir=self.metrics_dir,
                    args=self.args,
                    best_metrics=best_metrics,
                )

            if self.args.make_plots:
                plot_running_results(
                    output_dir=self.metrics_dir,
                    epochs=self.args.epochs,
                    lrs=all_lrs,
                    metrics=eval_scores,
                )

                all_lrs.append(lrs)
        code_timer.how_long_since_start()
        if self.args.save_model_at_end:
            self.save_model(
                model=model,
                output_dir=self.output_dir,
                data_collator=self.data_collator,
                tokenizer=self.tokenizer,
                dp=self.args.differential_privacy,
            )
        self.model = model

    def train_epoch(
        self,
        model,
        train_loader: DataLoader,
        optimizer,
        val_loader: DataLoader,
        epoch: int = None,
        step: int = 0,
        eval_scores: List[EvalScore] = None,
    ):
        """
        Train one epoch
        :param eval_scores: eval_scores of type List[EvalScore]
        :param model: Model of type BertForMaskedLM
        :param train_loader: Data loader of type DataLoader
        :param optimizer: Default is AdamW optimizer
        :param epoch: Given epoch: int
        :param val_loader: DataLoader containing
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

        for batch in tqdm(
            train_loader,
            desc=f"Train epoch {epoch} of {self.args.epochs}",
            unit="batch",
        ):
            model, optimizer, eval_scores, train_losses, lrs = self.train_batch(
                model=model,
                optimizer=optimizer,
                batch=batch,
                val_loader=val_loader,
                epoch=epoch,
                step=step,
                eval_scores=eval_scores,
                train_losses=train_losses,
                learning_rates=lrs,
            )

            log_train_metrics(
                epoch,
                step,
                lr=get_lr(optimizer)[0],
                loss=float(np.mean(train_losses)),
                logging_steps=self.args.logging_steps,
            )
            step += 1

        return model, step, lrs, eval_scores

    def train_batch(
        self,
        model,
        optimizer,
        batch,
        val_loader: DataLoader,
        epoch: int,
        step: int,
        train_losses: List[float],
        learning_rates: List[dict],
        eval_scores: List[EvalScore] = None,
    ) -> tuple:
        """
        Method to train one batch of data
        :return: tuple of model, optimizer, eval_scores, train_losses,
        learning_rates
        """

        model.train()

        model = self.modify_learning_rate_and_layers(
            model=model, optimizer=optimizer, step=step
        )

        append_json_lines(
            output_dir=self.metrics_dir,
            filename="learning_rates",
            data={"epoch": epoch, "step": step, "lr": get_lr(optimizer)[0]},
        )

        # compute model output
        if not self.args.weight_classes:
            output = model(
                input_ids=batch["input_ids"].to(self.args.device),
                attention_mask=batch["attention_mask"].to(self.args.device),
                labels=batch["labels"].to(self.args.device),
            )
            loss = output.loss
        else:
            output = model(
                input_ids=batch["input_ids"].to(self.args.device),
                attention_mask=batch["attention_mask"].to(self.args.device),
                labels=batch["labels"].to(self.args.device),
                class_weights=self.class_weights.to(self.args.device),
            )
            loss = output.loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        self.scheduler.step()

        if val_loader and (step > 0 and (step % self.args.evaluate_steps == 0)):
            eval_score = self.evaluate(model=model, val_loader=val_loader)
            eval_score.step = step
            eval_score.epoch = epoch

            append_json_lines(
                output_dir=self.metrics_dir,
                filename="eval_scores",
                data=dataclasses.asdict(eval_score),
            )
            eval_scores.append(eval_score)

            if self.args.save_steps is not None and (
                step > self.args.freeze_layers_n_steps
                and step % self.args.save_steps == 0
            ):
                best_metrics_input_dir = (
                    self.output_dir + f"/epoch-{epoch}_step-{step}"
                    if not self.args.save_only_best_model
                    else self.output_dir + "/best_model"
                )

                _, save_best_model = get_metrics(
                    eval_scores=eval_scores,
                    eval_metrics=self.args.eval_metrics,
                    best_model_path=best_metrics_input_dir,
                )

                self.save_model_at_step(
                    model=model,
                    epoch=epoch,
                    step=step,
                    save_best_model=save_best_model,
                    eval_score=eval_score,
                )

        train_losses.append(loss.item())

        append_json_lines(
            output_dir=self.metrics_dir,
            filename="train_loss",
            data={"epoch": epoch, "step": step,
                  "score": float(np.mean(train_losses))},
        )

        learning_rates.append(
            {"epoch": epoch, "step": step, "lr": get_lr(optimizer)[0]}
        )

        return model, optimizer, eval_scores, train_losses, learning_rates

    def save_model_at_step(self, model, epoch, step, save_best_model: bool,
                           eval_score):
        """
        Save model at step and overwrite best_model if the model
        have improved evaluation performance.
        :param save_best_model: whether to update best model
        :param model: Current model
        :param epoch: Current epoch
        :param step: Current step
        """

        if not self.args.save_only_best_model:
            self.save_model(
                model,
                output_dir=self.output_dir,
                data_collator=self.data_collator,
                tokenizer=self.tokenizer,
                step=f"/epoch-{epoch}_step-{step}",
                dp=self.args.differential_privacy,
            )

            write_json_lines(
                out_dir=self.output_dir + f"/epoch-{epoch}_step-{step}",
                filename="eval_scores",
                data=[dataclasses.asdict(eval_score)],
            )

        if save_best_model:
            self.save_model(
                model,
                output_dir=self.output_dir,
                data_collator=self.data_collator,
                tokenizer=self.tokenizer,
                step="/best_model",
                dp=self.args.differential_privacy,
            )

            write_json_lines(
                out_dir=self.output_dir + "/best_model",
                filename="eval_scores",
                data=[dataclasses.asdict(eval_score)],
            )

    def get_data_collator(self):
        """
        Get default data collator - depends on model type
        :return: DataCollator
        """
        return DataCollator(tokenizer=self.tokenizer)

    def get_tokenizer(self):
        """
        Get default tokenizer - depends on model type
        """
        return AutoTokenizer.from_pretrained(
            self.model_path, local_files_only=self.args.load_alvenir_pretrained
        )

    def tokenize_and_wrap_data(self, data):
        pass

    def get_model(self):
        return AutoModel.from_pretrained(self.model_path)

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

        init_learning_rate = (
            self.args.lr_freezed if self.args.freeze_layers else self.args.learning_rate
        )

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=init_learning_rate,
            weight_decay=self.args.weight_decay,
        )

        (
            dp_model,
            dp_optimizer,
            dp_train_loader,
        ) = self.privacy_engine.make_private_with_epsilon(
            module=model,
            optimizer=optimizer,
            data_loader=train_loader,
            epochs=self.args.epochs,
            target_epsilon=self.args.epsilon,
            target_delta=self.args.delta,
            max_grad_norm=self.args.max_grad_norm,
            grad_sample_mode="hooks",
        )
        return dp_model, dp_optimizer, dp_train_loader

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
    def unfreeze_layers(model, freeze_embeddings: bool = False):
        """
        Un-freeze all layers exept for embeddings in model, such that we can
        train full model
        :param model: A model of type BertForMaskedLM
        :return: un-freezed model
        """
        for name, param in model.named_parameters():
            if freeze_embeddings:
                if not name.startswith("bert.embeddings"):
                    param.requires_grad = True
            else:
                param.requires_grad = True

        print("bert layers unfreezed")
        model.train()
        return model

    @staticmethod
    def save_model(
        model,
        output_dir: str,
        data_collator,
        tokenizer,
        step: str = "",
        dp: bool = False,
    ):
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
            # model.config.save_pretrained(output_dir)

        trainer_test = Trainer(
            model=model,
            args=TrainingArguments(output_dir=output_dir),
            data_collator=data_collator,
            tokenizer=tokenizer,
        )

        trainer_test.save_model(output_dir=output_dir)
        torch.save(model.state_dict(),
                   os.path.join(output_dir, "model_weights.json"))

    def evaluate(self, model, val_loader: DataLoader) -> EvalScore:
        """
        :param model:
        :param val_loader: DataLoader containing validation data
        :return: mean loss, accuracy-score, f1-score
        """
        # set up preliminaries
        # if not next(model.parameters()).is_cuda:
        model = model.to(self.args.device)

        model.eval()

        y_true, y_pred, loss = [], [], []

        with torch.no_grad():
            # get model predictions and labels
            for batch in tqdm(val_loader, unit="batch", desc="val"):
                output = model(
                    input_ids=batch["input_ids"].to(self.args.device),
                    attention_mask=batch["attention_mask"].to(self.args.device),
                    labels=batch["labels"].to(self.args.device),
                )

                batch_preds = np.argmax(output.logits.detach().cpu().numpy(),
                                        axis=-1)
                batch_labels = batch["labels"].cpu().numpy()
                batch_loss = output.loss.item()

                y_true.extend(batch_labels)
                y_pred.extend(batch_preds)
                loss.append(batch_loss)

        # calculate metrics of interest
        acc = accuracy_score(y_true, y_pred)
        f_1 = f1_score(y_true, y_pred, average="macro")
        loss = float(np.mean(loss))

        print(
            f"\n" f"eval loss: {loss} \t" f"eval acc: {acc}" f"eval f1: {f_1}")

        return EvalScore(accuracy=acc, f_1=f_1, loss=loss)
