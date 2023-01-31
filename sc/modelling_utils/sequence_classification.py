# pylint: disable=protected-access
import argparse
import os
import pickle
from datetime import datetime
from typing import List

import numpy as np
import torch
from datasets import ClassLabel
from opacus import GradSampleModule
from opacus.data_loader import DPDataLoader
from opacus.optimizers import DPOptimizer
from opacus.utils.batch_memory_manager import BatchMemoryManager
from sklearn.metrics import f1_score, accuracy_score
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, TrainingArguments, Trainer, \
    BertForSequenceClassification, \
    DataCollatorWithPadding

from shared.data_utils.custom_dataclasses import EvalScore, EmbeddingOutput
from shared.data_utils.helpers import DatasetWrapper
from shared.modelling_utils.helpers import get_lr, \
    log_train_metrics_dp
from shared.modelling_utils.modelling import Modelling
from shared.utils.visualization import plot_confusion_matrix
from sc.local_constants import MODEL_DIR, DATA_DIR
from mlm.local_constants import MODEL_DIR as MLM_MODEL_DIR


class SequenceClassification(Modelling):
    """
    Class to train a SequenceClassification model
    """

    def __init__(self, args: argparse.Namespace):
        super().__init__(args=args)
        # self.args = args
        # self.total_steps = None

        self.output_name = f'{self.args.model_name.replace("/", "_")}-' \
                           f'{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}'
        self.args.output_name = self.output_name
        self.output_dir = os.path.join(MODEL_DIR, self.output_name)
        self.metrics_dir = os.path.join(self.output_dir, 'metrics')
        self.data_dir = DATA_DIR

        self.label2id, self.id2label = self.label2id2label()

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
        :param model: Model
        :param val_loader: DataLoader containing validation data
        :param conf_plot: whether to plot confusion_matrix
        :return: mean loss, accuracy-score, f1-score
        """
        # set up preliminaries
        if not next(model.parameters()).is_cuda:
            model = model.to(self.args.device)

        model.eval()

        y_true, y_pred, loss = [], [], []

        with torch.no_grad():
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

        if conf_plot:
            y_true_plot = list(map(lambda x: self.id2label[int(x)], y_true))
            y_pred_plot = list(map(lambda x: self.id2label[int(x)], y_pred))

            plot_confusion_matrix(
                y_true=y_true_plot,
                y_pred=y_pred_plot,
                labels=self.args.labels,
                model_name=self.args.model_name)
        return EvalScore(accuracy=acc, f_1=f_1, loss=loss)

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
        # ToDo: split into words before feeding tokenizer?
        #  use fx word_tokenize(line, language='danish') from nltk

        batch['text'] = [
            line for line in batch['text'] if
            len(line) > 0 and not line.isspace()
        ]

        tokens = self.tokenizer(batch['text'],
                                padding='max_length',
                                max_length=self.args.max_length,
                                truncation=True,
                                # is_split_into_words=True
                                )
        tokens['labels'] = self.class_labels.str2int(batch['label'])
        return tokens

    def tokenize2(self, batch):
        """
        Tokenize each batch in dataset
        @param batch: batch of data
        @return: tokens for each batch
        """
        # ToDo: split into words before feeding tokenizer?
        #  use fx word_tokenize(line, language='danish') from nltk

        batch['text'] = [
            line for line in batch['text'] if
            len(line) > 0 and not line.isspace()
        ]

        tokens = self.tokenizer(batch["text"],
                                max_length=self.args.max_length,
                                padding='max_length',
                                truncation=True,
                                return_overflowing_tokens=True,
                                return_offsets_mapping=True,
                                stride=32)
        token_labels_mapping = []
        for sample_mapping in tokens['overflow_to_sample_mapping']:
            token_labels_mapping.append(batch['label'][sample_mapping])

        tokens['labels'] = self.class_labels.str2int(token_labels_mapping)
        return tokens

    def predict(self, model, sentence: str,
                label: str = None) -> EmbeddingOutput:
        """
        Predict class from input sentence
        :param model: model
        :param sentence: input sentence
        :param label: label if known
        :return: Output including sentence embedding and prediction
        """
        tokenized = self.tokenizer([sentence],
                                   padding='max_length',
                                   max_length=self.args.max_length,
                                   truncation=True,
                                   return_tensors='pt')

        decoded_text = self.tokenizer.decode(
            token_ids=tokenized['input_ids'][0])
        model.to(self.args.device)
        output = model(**tokenized.to(self.args.device),
                       output_hidden_states=True,
                       return_dict=True
                       )
        pred = output.logits.argmax(-1).detach().cpu().numpy()

        pred_actual = self.id2label[pred[0]]
        embedding = torch.mean(
            output.hidden_states[-2], dim=1).detach().cpu().numpy()[0]
        return EmbeddingOutput(sentence=sentence,
                               label=label,
                               prediction=pred_actual,
                               decoded_text=decoded_text,
                               embedding=embedding)

    def create_embeddings_windowed(self, model, save_dict: bool = False):

        tokenized = self.test_data.map(
            self.tokenize2,
            batched=True,
            remove_columns=self.test_data.column_names,
            load_from_cache_file=False,
            desc="Running tokenizer on dataset line_by_line",
        )
        overflows = [x['overflow_to_sample_mapping'] for x in tokenized]
        tokenized.set_format('torch')
        wrapped = DatasetWrapper(tokenized)
        test_loader = DataLoader(dataset=wrapped,
                                 # collate_fn=modelling.data_collator,
                                 batch_size=1,
                                 shuffle=False)

        model.eval()
        model = model.to('cuda')

        # with tqdm(val_loader, unit="batch", desc="Batch") as batches:
        results = []
        tmp_results = []
        with torch.no_grad():
            for i, batch in enumerate(
                tqdm(test_loader, unit="batch", desc="Eval")):
                # for batch in val_loader:
                output = model(
                    input_ids=batch["input_ids"].to(self.args.device),
                    attention_mask=batch["attention_mask"].to(
                        self.args.device),
                    labels=batch["labels"].to(self.args.device),
                    output_hidden_states=True,
                    return_dict=True)
                pred = output.logits.argmax(-1).detach().cpu().numpy()[0]
                pred_actual = self.id2label[pred]

                decoded_text = self.tokenizer.decode(
                    token_ids=batch['input_ids'][0])
                embedding = torch.mean(
                    output.hidden_states[-2], dim=1).detach().cpu().numpy()[0]

                tmp_results.append(
                    [overflows[i], embedding, pred, decoded_text, ])

        grouped = [[x for x in tmp_results if x[0] == y] for y in
                   {x for x in overflows}]

        for i, data_batch in enumerate(grouped):
            if len(data_batch) > 1:
                new_embedding = np.mean([x[1] for x in data_batch], axis=0)
                pred = data_batch[0][2]
                pred_actual = self.id2label[pred]
                decoded_text = ' '.join([x[3] for x in data_batch])
            else:
                new_embedding = data_batch[0][1]

            results.append(
                EmbeddingOutput(sentence=self.data.test[i]['text'],
                                label=self.data.test[i]['label'],
                                prediction=pred_actual,
                                decoded_text=decoded_text,
                                embedding=new_embedding))
            if save_dict:
                with open("data/test_data/test_embeddings", "wb") as fp:
                    pickle.dump(results, fp)

        return results

    def create_embeddings(self, data_loader: DataLoader, model):

        assert data_loader.batch_size == 1, "batch size must be 1 for this demo"

        model.eval()
        model = model.to('cuda')

        # with tqdm(val_loader, unit="batch", desc="Batch") as batches:
        results = []
        with torch.no_grad():
            for i, batch in enumerate(
                tqdm(data_loader, unit="batch", desc="Eval")):
                # for batch in val_loader:
                output = model(
                    input_ids=batch["input_ids"].to(self.args.device),
                    attention_mask=batch["attention_mask"].to(
                        self.args.device),
                    labels=batch["labels"].to(self.args.device),
                    output_hidden_states=True,
                    return_dict=True)
                pred = output.logits.argmax(-1).detach().cpu().numpy()[0]
                pred_actual = self.id2label[pred]

                decoded_text = self.tokenizer.decode(
                    token_ids=batch['input_ids'][0])
                embedding = torch.mean(
                    output.hidden_states[-2], dim=1).detach().cpu().numpy()[0]

                results.append(
                    EmbeddingOutput(sentence=self.data.test[i]['text'],
                                    label=self.data.test[i]['label'],
                                    prediction=pred_actual,
                                    decoded_text=decoded_text,
                                    embedding=embedding))

        return results

    def get_tokenizer(self):
        return AutoTokenizer.from_pretrained(
            self.model_path,
            local_files_only=self.args.load_alvenir_pretrained)
        # ToDo: Consider do_lower_case=True, otherwise lowercase training data

    def get_data_collator(self):
        return DataCollatorWithPadding(tokenizer=self.tokenizer)

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
            if not (name.startswith("bert.pooler") or
                    name.startswith("classifier")):
                param.requires_grad = False
        model.train()
        return model

    @staticmethod
    def unfreeze_layers(model):
        """
        Un-freeze all layers exept for embeddings in model, such that we can
        train full model
        :param model: Model of type BertForSequenceClassification
        :return: un-freezed model
        """
        for name, param in model.named_parameters():
            if not name.startswith("bert.embeddings"):
                param.requires_grad = True
        print("bert layers unfreezed")
        model.train()
        return model

    # @staticmethod
    # def save_model(model, output_dir: str,
    #                data_collator,
    #                tokenizer,
    #                step: str = ""):
    #     """
    #     Wrap model in trainer class and save to pytorch object
    #     :param model: BertForSequenceClassification to save
    #     :param output_dir: model directory
    #     :param data_collator:
    #     :param tokenizer:
    #     :param step: if saving during training step should be
    #     '/epoch-{epoch}_step-{step}'
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


class SequenceClassificationDP(SequenceClassification):
    """
        Class inherited from SequenceClassification to train a
        SequenceClassification model with differential privacy
    """

    def __init__(self, args: argparse.Namespace):
        super().__init__(args)

        # self.privacy_engine = None
        self.output_name = f'DP-eps-{int(self.args.epsilon)}-' \
                           f'{self.args.model_name.replace("/", "_")}-' \
                           f'{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}'
        self.args.output_name = self.output_name
        self.output_dir = os.path.join(MODEL_DIR, self.output_name)
        self.metrics_dir = os.path.join(self.output_dir, 'metrics')

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
        for name, param in model.named_parameters():
            if not name.startswith("_module.cls"):
                param.requires_grad = False
        model.train()
        return model

    @staticmethod
    def unfreeze_layers(model):
        """
        Un-freeze all bert layers in model, such that we can train full model
        :param model: of type GradSampleModule
        :return: un-freezed model
        """
        for name, param in model.named_parameters():
            if not name.startswith("_module.bert.embeddings"):
                param.requires_grad = True
        model.train()
        return model
