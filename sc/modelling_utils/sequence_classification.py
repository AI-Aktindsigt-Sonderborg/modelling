# pylint: disable=protected-access
import argparse
import os
import pickle
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
from transformers import AutoTokenizer, BertForSequenceClassification, \
    DataCollatorWithPadding

from sc.local_constants import MODEL_DIR, DATA_DIR
from shared.data_utils.custom_dataclasses import EvalScore, EmbeddingOutput
from shared.data_utils.helpers import DatasetWrapper
from shared.modelling_utils.helpers import get_lr, \
    log_train_metrics_dp, label2id2label
from shared.modelling_utils.modelling import Modelling
from shared.utils.visualization import plot_confusion_matrix


class SequenceClassification(Modelling):
    """
    Class to train a SequenceClassification model

    Methods
    -------
    """

    def __init__(self, args: argparse.Namespace):
        super().__init__(args=args)

        self.output_dir = os.path.join(MODEL_DIR, self.args.output_name)
        self.metrics_dir = os.path.join(self.output_dir, 'metrics')
        self.data_dir = DATA_DIR

        self.label2id, self.id2label = label2id2label(self.args.labels)

        self.class_labels = ClassLabel(
            num_classes=len(self.args.labels),
            names=self.args.labels)

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

    def tokenize_for_windowed_embeddings(self, batch):
        """
        This function is only used when running a window function on sentences
        that are too long. Only relevant for demo atm.
        @param batch: batch of data
        @return: tokens for each batch
        """

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
        """
        Method only relevant for demo purposes - see also
        tokenize_for_windowed_embeddings
        :param model:
        :param save_dict:
        :return:
        """
        tokenized = self.data.test.map(
            self.tokenize_for_windowed_embeddings,
            batched=True,
            remove_columns=self.data.test.column_names,
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
                with open(os.path.join(DATA_DIR, "test_data/test_embeddings"), "wb") as fp:
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


class SequenceClassificationDP(SequenceClassification):
    """
        Class inherited from SequenceClassification to train a
        SequenceClassification model with differential privacy

        Methods
        -------
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
        :param eval_scores: eval_scores
        :param model: Differentially private model wrapped in GradSampleModule
        :param train_loader: Differentially private data loader of type
            DPDataLoader
        :param optimizer: Differentially private optimizer of type DPOptimizer
        :param epoch: Given epoch: int
        :param val_loader: DataLoader containing validation data
        :param step: Given step
        :return:
            if self.eval_data: return model, eval_losses, eval_accuracies,
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
            if not name.startswith("cls"):
                param.requires_grad = False
        model.train()
        return model

    @staticmethod
    def unfreeze_layers(model, freeze_embeddings: bool = True):
        """
        Un-freeze all bert layers in model, such that we can train full model
        :param model: of type GradSampleModule
        :return: un-freezed model
        """
        for name, param in model._module.named_parameters():
            if freeze_embeddings:
                if not name.startswith("bert.embeddings"):
                    param.requires_grad = True
            else:
                param.requires_grad = True

        model.train()
        return model
