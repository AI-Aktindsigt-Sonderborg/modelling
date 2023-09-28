import argparse
import json
import os.path
import random
from typing import List, Optional

import nltk.data
import numpy as np

from mlm.data_utils.data_prep_input_args import DataPrepArgParser
from mlm.local_constants import DATA_DIR, CONF_DATA_DIR
from shared.utils.helpers import write_json_lines

DEFAULT_UNIQUE_SENTENCES_FILE = 'unique_sentences_mlm1.jsonl'


class RawDataPreprocessing:
    """
    Class to preprocess raw data to MLM format from confidential "aktindsigter".

    :param argparse.Namespace args:
        input arguments from :class:`.DataPrepArgParser`.

    :Eksempel:
        ::

            from mlm.data_utils.data_prep_input_args import DataPrepArgParser
            from mlm.data_utils.prep_scrape import RawScrapePreprocessing
            prep_parser = DataPrepArgParser()
            prep_args = prep_parser.parser.parse_args()
            data_preprocessor = RawScrapePreprocessing(args=prep_args)
            data_preprocessor.run()

    :CLI eksempel:
        ::

            python -m mlm.data_utils.prep_data

    Metoder
    -------
    """

    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.sentence_splitter = nltk.data.load(
            'tokenizers/punkt/danish.pickle')
        self.model_id = 'pere/norwegian-gpt2'

    def run(self):
        """
        Generate train and validation data as json line files from raw scrape
        file. Calls class methods:

            - ``extract_danish_and_save_from_raw()``
            - ``create_unique_sentences()``
            - ``split_train_val()``

        """
        # self.extract_danish_and_save_from_raw()
        # self.create_unique_sentences()
        # if self.args.add_ppl:
        #     self.filter_ppl_scores()
        #     self.split_train_val(in_file='approved_sentences_ppl.json')
        self.split_train_val()

    def split_train_val(self,
                        in_file: str = DEFAULT_UNIQUE_SENTENCES_FILE,
                        seed: int = 42):
        """
        Split approved sentences to train and validation set and call
        ``save_datasets()``.

        :param (str, Optional) in_file: jsonlines file
        :param (int, Optional) seed: seed for reproducibility
        """
        sentences = []
        with open(os.path.join(CONF_DATA_DIR, in_file), 'r',
                  encoding='utf-8') as file:
            for line in file:
                data_dict = json.loads(line)
                sentences.append({'text': data_dict['text']})
        random.seed(seed)
        random.shuffle(sentences)
        train_idx = int(len(sentences) * self.args.split)
        train_init = sentences[:train_idx]

        val = sentences[train_idx:]
        val = list(val)
        if self.args.split_train_n_times > 0:
            train_chunks = np.array_split(train_init,
                                          self.args.split_train_n_times)
            for i, train_chunk in enumerate(train_chunks):
                train = list(train_chunk)
                train_outfile = self.args.train_outfile + f'_{i}'
                self.save_datasets(train=train, train_outfile=train_outfile)
            self.save_datasets(val=val)
        else:
            train = list(train_init)
            self.save_datasets(train=train, val=val,
                               train_outfile=self.args.train_outfile)

    def save_datasets(self, train: List[dict] = None, val: List[dict] = None,
                      train_outfile: str = None):
        """
        Take train and validation data as input and saves to json-lines files
        compatible with torch dataset.

        :param List[dict], Optional train: List[str] where each element is a
            valid sentence for training
        :param List[dict], Optional val: List[str] where each element is a valid
             sentence for validation
        :param str, Optional train_outfile: name of train file
        """

        if train and train_outfile:
            write_json_lines(out_dir=DATA_DIR, filename=train_outfile,
                             data=train)
        if val:
            write_json_lines(out_dir=DATA_DIR, filename=self.args.val_outfile,
                             data=val)


if __name__ == '__main__':
    prep_parser = DataPrepArgParser()
    prep_args = prep_parser.parser.parse_args()
    data_preprocessor = RawDataPreprocessing(args=prep_args)
    # category_mapping = read_json(filepath=os.path.join(METADATA_DIR, "ss_annotation_categories.json"))
    # data_preprocessor.run()
    # data_preprocessor.extract_danish_and_save_from_raw()
    # data_preprocessor.create_unique_sentences()
    data_preprocessor.split_train_val()
