import argparse
import itertools
from typing import List

import nltk.data
from sklearn.model_selection import train_test_split

from sc.data_utils.data_prep_input_args import DataPrepArgParser
from sc.local_constants import CONF_DATA_DIR, DATA_DIR
from shared.utils.helpers import read_json_lines, write_json_lines


class DataPreprocessing:
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

            python -m mlm.data_utils.prep_scrape

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
        self.split_train_val()

    @staticmethod
    def train_val_test_to_json_split(class_grouped_data,
                                     train_size: float = None,
                                     test_size: int = None,
                                     train_outfile: str = None,
                                     val_outfile: str = None,
                                     test_outfile: str = None):
        """
        Read grouped data, split to train, val and test and save json
        :param class_grouped_data: grouped data as list og lists of dicts
        :param train_size: float between 0 and 1 specifying the size of the
        train set where
        1 is all data
        :param test_size: int >= 1 specifying the number of sentences in each
        class
        :param train_outfile: if train_outfile specified generate train set
        :param val_outfile: if val_outfile specified generate validation set
        :param test_outfile: if test_outfile specified generate test set
        :return:
        """
        assert train_outfile and test_outfile, \
            '\n At least train_outfile and test_outfile must be specified - ' \
            'see doc: \n' + \
            DataPreprocessing.train_val_test_to_json_split.__doc__

        assert train_size and test_size, \
            'Either train or test size must be specified - see doc: \n' + \
            DataPreprocessing.train_val_test_to_json_split.__doc__

        if train_outfile and test_size and not val_outfile:
            train_test = [
                train_test_split(x, test_size=test_size, random_state=1) for x
                in
                class_grouped_data]
            train = list(
                itertools.chain.from_iterable([x[0] for x in train_test]))
            test = list(
                itertools.chain.from_iterable([x[1] for x in train_test]))

        if train_outfile and val_outfile and test_outfile and train_size and \
            test_size:
            train_val = [
                train_test_split(x, train_size=train_size, random_state=1) for x
                in
                class_grouped_data]
            train = list(
                itertools.chain.from_iterable([x[0] for x in train_val]))
            val_tmp = list(
                itertools.chain.from_iterable([x[1] for x in train_val]))

            val_grouped = DataPreprocessing.group_data_by_class(
                val_tmp)

            val_test = [train_test_split(x, test_size=test_size, random_state=1)
                        for x in
                        val_grouped]

            val = list(itertools.chain.from_iterable([x[0] for x in val_test]))
            test = list(itertools.chain.from_iterable([x[1] for x in val_test]))

        if train_outfile:
            write_json_lines(out_dir=DATA_DIR,
                             data=train,
                             filename=train_outfile)
        if val_outfile:
            write_json_lines(out_dir=DATA_DIR,
                             data=val,
                             filename=val_outfile)
        if test_outfile:
            write_json_lines(out_dir=DATA_DIR,
                             data=test,
                             filename=test_outfile)
        return print('datasets generated')

    @staticmethod
    def group_data_by_class(list_data: List[dict], label_set: List[str] = None):
        """
        Group data by class
        :param data: list of dictionarys of sentences
        :return: list of lists of dicts
        """
        if not label_set:
            label_set = {x['label'] for x in list_data}

        grouped = [[x for x in list_data if x['label'] == y] for y in label_set]
        return grouped


if __name__ == '__main__':
    prep_parser = DataPrepArgParser()
    prep_args = prep_parser.parser.parse_args()
    data_preprocessor = DataPreprocessing(args=prep_args)

    data = read_json_lines(input_dir=CONF_DATA_DIR, filename="unique_sentences_with_label1")
    grouped_data = data_preprocessor.group_data_by_class(list_data=data)
    data_preprocessor.train_val_test_to_json_split(
        class_grouped_data=grouped_data,
        train_outfile='train_classified',
        val_outfile='eval_classified',
        test_outfile='test_classified',
        train_size=0.9,
        test_size=30)
