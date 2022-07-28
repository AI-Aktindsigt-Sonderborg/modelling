import argparse
import os

from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, DataCollatorForWholeWordMask, \
    DataCollatorForLanguageModeling, TrainingArguments, Trainer

from local_constants import DATA_DIR
from modelling_utils.custom_modeling_bert import BertForMaskedLM
from modelling_utils.mlm_modelling import DatasetWrapper
from utils.input_args import MLMArgParser

class MLMUnsupervisedEvaluation:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.eval_data = None
        self.model = None
        self.tokenizer = AutoTokenizer.from_pretrained(self.args.model_name)
        self.data_collator = self.get_data_collator
        self.model = BertForMaskedLM.from_pretrained(self.args.model_name)

    def load_data(self):
        """
        Load data using datasets.load_dataset for training and evaluation
        :param train: Whether to train model
        """

        self.eval_data = load_dataset('json',
                                      data_files=os.path.join(DATA_DIR, self.args.eval_data),
                                      split='train')

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


if __name__ == '__main__':
    mlm_parser = MLMArgParser()
    args = mlm_parser.parser.parse_args()
    args.eval_data = 'val_10.json'

    mlm_eval = MLMUnsupervisedEvaluation(args=args)
    mlm_eval.load_data()

    eval_data = mlm_eval.tokenize_and_wrap_data(mlm_eval.eval_data)

    training_args = TrainingArguments(
        output_dir='models/eval_model_test',
        weight_decay=0.1,
        do_eval=True
    )

    trainer = Trainer(
        model=mlm_eval.model,
        args=training_args,
        eval_dataset=eval_data,
        data_collator=mlm_eval.data_collator,
        tokenizer=mlm_eval.tokenizer
    )

    trainer.evaluate()

    print()




