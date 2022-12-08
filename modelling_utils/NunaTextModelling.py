import json
import os
from typing import List

import pandas as pd
import torch
import torch.nn.functional as F
import numpy as np
from datasets import load_dataset, ClassLabel, Dataset
from matplotlib import pyplot as plt
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix, f1_score
from transformers import AutoTokenizer, DataCollatorWithPadding, AutoModelForSequenceClassification, \
    Trainer, EarlyStoppingCallback, BertForSequenceClassification, DataCollatorForLanguageModeling, \
    BertConfig

from local_constants import MODEL_DIR, RESULTS_DIR
from modelling_utils.custom_modeling_bert import BertOnlyMLMHeadCustom
from utils.helpers import compute_metrics
from data_utils.custom_dataclasses import LoadModelType
from data_utils.custom_dataclasses import PredictionOutput
import seaborn as sn

class NunaTextModelling:
    def __init__(self, model_name: str = None, labels: list = None, data_types: List[str] = None,
                 data_path: str = None, load_model_type: LoadModelType = None):
        """

        :param model_name:
        :param labels:
        :param data_types:
        :param data_path:
        :param load_model_type:
        """
        self.model_name = model_name
        self.local_alvenir_model_path = os.path.join(MODEL_DIR, self.model_name,
                                                     'best_model')

        self.tokenizer = AutoTokenizer.from_pretrained(self.local_alvenir_model_path,
                                                       local_files_only=True)

        self.data_path = data_path
        self.labels = labels
        self.data_types = data_types

        self.model_path = 'models/' + model_name

        self.label2id, self.id2label = self.label2id2label()

        self.class_labels = ClassLabel(num_classes=len(self.labels), names=self.labels)

        if load_model_type.value == 1:
            self.data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)
            self.model = BertForSequenceClassification.from_pretrained(
                self.local_alvenir_model_path,
                num_labels=len(self.labels),
                local_files_only=True)

            config = BertConfig.from_pretrained(self.local_alvenir_model_path, local_files_only=True)
            new_head = BertOnlyMLMHeadCustom(config)
            new_head.load_state_dict(
                torch.load(self.local_alvenir_model_path + '/head_weights.json'))

            lm_head = new_head.to('cuda')
            self.model.cls = lm_head

            self.callbacks = EarlyStoppingCallback(early_stopping_patience=10,
                                                   early_stopping_threshold=0.0)
        elif load_model_type.value in [2, 3]:
            self.data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)
            self.model = BertForSequenceClassification.from_pretrained(
                self.local_alvenir_model_path,
                num_labels=len(self.labels),
                label2id=self.label2id,
                id2label=self.id2label,
                local_files_only=True)

    def load_data(self):
        """
        Load data using datasets.load_dataset for training and evaluation
        :param train: Whether to train model
        """

        train_data = load_dataset('json',
                                       data_files=os.path.join(self.data_path,
                                                               'train_classified_mixed.json'),
                                       split='train')

        eval_data = load_dataset('json',
                                      data_files=os.path.join(self.data_path,
                                                              'test_classified_mixed.json'),
                                      split='train')
        return train_data, eval_data

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
                line for line in examples['text'] if len(line) > 0 and not line.isspace()
            ]

            return self.tokenizer(
                examples['text'],
                padding='max_length',
                truncation=True,
                max_length=64,
                # We use this option because DataCollatorForLanguageModeling (see below)
                # is more efficient when it receives the `special_tokens_mask`.
                return_special_tokens_mask=True,
            )


        tokenized = data.map(
            self.tokenize,
            batched=True,
            # num_proc=1,
            # remove_columns=['text', 'label'],
            load_from_cache_file=False,
            desc="Running tokenizer on dataset line_by_line",
        )

        # wrapped = DatasetWrapper(tokenized)
        tokenized.set_format('torch', columns=['input_ids', 'attention_mask', 'labels'])
        return tokenized


    def map_to_hf_dataset(self):

        data_dict = {}
        for data_type in self.data_types:
            data_dict[data_type] = os.path.join(self.data_path, f'{data_type}.json')

        data = load_dataset('json', data_files=data_dict)

        data = data.map(self.tokenize, batched=True)
        # data.set_format('torch', columns=['input_ids', 'attention_mask', 'labels'])
        data.set_format('torch', columns=['input_ids', 'attention_mask', 'labels'])
        return data

    def tokenize(self, batch):
        tokens = self.tokenizer(batch['text'], padding='max_length', max_length=64, truncation=True)
        tokens['labels'] = self.class_labels.str2int(batch['label'])
        return tokens

    def label2id2label(self):
        label2id, id2label = dict(), dict()

        for i, label in enumerate(self.labels):
            label2id[label] = str(i)
            id2label[i] = label
        return label2id, id2label

    def eval_model(self, model_name: str = None, labels: list = None):

        if model_name:
            self.model_name = model_name

            self.model = BertForSequenceClassification.from_pretrained(self.local_alvenir_model_path,
                                                                            num_labels=len(
                                                                                self.labels),
                                                                            label2id=self.label2id,
                                                                            id2label=self.id2label,
                                                                            local_files_only=True)

            config = BertConfig.from_pretrained(self.local_alvenir_model_path, local_files_only=True)
            new_head = BertOnlyMLMHeadCustom(config)
            new_head.load_state_dict(
                torch.load(self.local_alvenir_model_path + '/head_weights.json'))

            # lm_head = new_head.to('cuda')
            self.model.cls = new_head

        if labels:
            self.labels = labels

        train_data, eval_data = self.load_data()
        # train_data_wrapped = nuna_text_modelling.tokenize_and_wrap_data(data=train_data)
        eval_data_wrapped = self.tokenize_and_wrap_data(data=eval_data)

        prediction_path = os.path.join(RESULTS_DIR, self.model_name)
        if not os.path.exists(prediction_path):
            os.mkdir(prediction_path)

        # data = self.map_to_hf_dataset()

        test_trainer = Trainer(self.model, compute_metrics=compute_metrics,
                               data_collator=self.data_collator,
                               )

        pred_logits, label_ids, eval_result = test_trainer.predict(eval_data_wrapped)

        softmax_pred = F.softmax(torch.Tensor(pred_logits), dim=-1).numpy()

        y_true = list(map(lambda x: self.id2label[x], label_ids))
        y_pred = list(map(lambda x: self.id2label[x], np.argmax(pred_logits, axis=1)))

        predictions_list = []
        for i in range(len(y_true)):
            predictions_list.append(
                {'y': y_true[i], 'pred': y_pred[i], 'softmax': str(list(softmax_pred[i]))})

        with open(f'{prediction_path}/true_vs_predictions.json', 'w', encoding='utf-8') as f_out:
            json.dump(predictions_list, f_out)

        return y_true, y_pred, eval_result

    def predict(self, data: str = None):

        inputs = self.tokenizer(data, return_tensors="pt")

        with torch.no_grad():
            logits = self.model(**inputs).logits

        softmax_pred = F.softmax(torch.Tensor(logits), dim=-1).numpy()

        predicted_class_ids = torch.argmax(logits, dim=-1).item()
        predicted_label = self.model.config.id2label[predicted_class_ids]

        label_percentage = {}
        for i in range(len(softmax_pred[0])):
            label_percentage[self.id2label[i]] = softmax_pred[0][i]

        return PredictionOutput(prediction=predicted_label, class_ids=predicted_class_ids,
                                softmax=label_percentage)

    @staticmethod
    def calc_f1_score(y_list, prediction_list, labels, conf_plot: bool = False, normalize: str = None):

        if conf_plot:
            conf_matrix = confusion_matrix(y_list, prediction_list, labels=labels,
                                           normalize=normalize)
            df_cm = pd.DataFrame(conf_matrix, index=labels, columns=labels)

            plt.figure(figsize=(10, 7))
            # plot_labels = ['Besk. og int.', 'Børn/unge', 'Erhverv', 'Klima/tek/miljø',
            #                'Kultur/fritid',
            #                'Social', 'Sundhed/ældre', 'Øko./adm', 'Øko./budget']
            sn.heatmap(df_cm, annot=True, cmap="YlGnBu", fmt='g', xticklabels=labels,
                       yticklabels=labels)
            # plt.savefig('plots/svm_03_confplot.png')
            plt.show()

        return precision_recall_fscore_support(y_list, prediction_list, labels=labels,
                                               average='micro'), \
               f1_score(y_true=y_list, y_pred=prediction_list, labels=labels, average=None)
