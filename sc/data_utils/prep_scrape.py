import argparse
import itertools
import os.path
from typing import List

import pandas as pd
from shared.data_utils.helpers import write_json_lines
from sc.local_constants import CLASS_DATA_DIR
from sklearn.model_selection import train_test_split

from mlm.data_utils.data_prep_input_args import DataPrepArgParser
from shared.utils.helpers import read_jsonlines, save_json


class ClassifiedScrapePreprocessing:
    """
    Class to read raw excel file with classified KL categories and split to
    train and validation data
    """

    def __init__(self, args: argparse.Namespace):
        self.args = args

    def read_xls_save_json(self, file_dir: str = CLASS_DATA_DIR,
                           ppl_filters: List[int] = None, drop_na: bool = True):
        """
        Read excel file with labelled data and save to json lines file
        :param file_dir: directory to read file from
        :param ppl_filters: filter data on perplexity scores
        :param drop_na: drop not classified sentences
        """
        # ToDo: find original ppl_filter - between 40 and 3000?
        data = pd.read_excel(
            os.path.join(file_dir, 'raw', self.args.excel_classification_file),
            sheet_name='Klassificering',
            header=1)
        data['index'] = range(len(data))
        data['text_len'] = list(map(lambda x: len(str(x)), data['text']))
        if drop_na:
            data = data.dropna(subset=['klassifikation'])
        if ppl_filters:
            data = data[(data['ppl_score'] > ppl_filters[0]) & (
                    data['ppl_score'] < ppl_filters[1])]

        data_dicts = data.to_dict('records')
        save_json(output_dir=CLASS_DATA_DIR, data=data_dicts,
                  filename=self.args.classified_scrape_file)

    def read_classified_json(self):
        return read_jsonlines(input_dir=CLASS_DATA_DIR,
                              filename=self.args.classified_scrape_file)

    @staticmethod
    def group_data_by_class(list_data: List[dict]):
        """
        Group data by class
        :param data: list of dictionarys of sentences
        :return: list of lists of dicts
        """
        # k = 0
        # with open(os.path.join(DATA_DIR, 'train_1110.json'),
        #           'r', encoding='utf-8') as file:
        #     for i, line in enumerate(file):
        #         if i % 100000 == 0:
        #             print(i)
        #         data_dict = json.loads(line)
        #
        #         if data_dict['text'] in [x['text'] for x in list_data]:
        #             k += 1
        #             # print()

        # print(f'{k} of {len(list_data)} has been trained on')
        label_set = {x['label'] for x in list_data}
        grouped = [[x for x in list_data if x['label'] == y] for y in label_set]
        return grouped

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
        :param test_size: int >= 1 specifying the number of sentences in each class
        :param train_outfile: if train_outfile specified generate train set
        :param val_outfile: if val_outfile specified generate validation set
        :param test_outfile: if test_outfile specified generate test set
        :return:
        """
        assert train_outfile and test_outfile, \
            '\n At least train_outfile and test_outfile must be specified - ' \
            'see doc: \n' + \
            ClassifiedScrapePreprocessing.train_val_test_to_json_split.__doc__

        assert train_size and test_size, \
            'Either train or test size must be specified - see doc: \n' + \
            ClassifiedScrapePreprocessing.train_val_test_to_json_split.__doc__

        if train_outfile and test_size and not val_outfile:
            train_test = [
                train_test_split(x, test_size=test_size, random_state=1) for x
                in
                class_grouped_data]
            train = list(
                itertools.chain.from_iterable([x[0] for x in train_test]))
            test = list(
                itertools.chain.from_iterable([x[1] for x in train_test]))

        if train_outfile and val_outfile and test_outfile and train_size and test_size:
            train_val = [
                train_test_split(x, train_size=train_size, random_state=1) for x
                in
                class_grouped_data]
            train = list(
                itertools.chain.from_iterable([x[0] for x in train_val]))
            val_tmp = list(
                itertools.chain.from_iterable([x[1] for x in train_val]))

            val_grouped = ClassifiedScrapePreprocessing.group_data_by_class(
                val_tmp)

            val_test = [train_test_split(x, test_size=test_size, random_state=1)
                        for x in
                        val_grouped]

            val = list(itertools.chain.from_iterable([x[0] for x in val_test]))
            test = list(itertools.chain.from_iterable([x[1] for x in val_test]))

        if train_outfile:
            save_json(os.path.join(CLASS_DATA_DIR, 'processed'), data=train,
                      filename=train_outfile)
        if val_outfile:
            save_json(os.path.join(CLASS_DATA_DIR, 'processed'), data=val,
                      filename=val_outfile)
        if test_outfile:
            save_json(os.path.join(CLASS_DATA_DIR, 'processed'), data=test,
                      filename=test_outfile)
        return print('datasets generated')

    @staticmethod
    def concat_all_classified_data(data_dir: str = CLASS_DATA_DIR):
        all_data = []
        for filename in os.listdir(data_dir):
            filepath = os.path.join(data_dir, filename)
            if os.path.isfile(filepath):
                tmp_data = read_jsonlines(input_dir=data_dir,
                                          filename=filename.split('.')[0])
                for line in tmp_data:
                    if isinstance(line['text'], str):
                        all_data.append({'label': line['klassifikation'],
                                         'text': line['text']})
                    else:
                        print(f'line {line} is empty')
        write_json_lines(out_dir=os.path.join(CLASS_DATA_DIR, 'processed'),
                         data=all_data, filename='all_sentences')
        return all_data


if __name__ == '__main__':

    prep_parser = DataPrepArgParser()
    prep_args = prep_parser.parser.parse_args()

    prep_args.data_type = 'labelled'
    # prep_args.excel_classification_file = 'aalborg_kommune_done.xlsx'
    # prep_args.classified_scrape_file = 'aalborg_classified_scrape'

    # prep_args.excel_classification_file = 'blandet_mindre_kommuner_done.xlsx'
    # prep_args.classified_scrape_file = 'mixed_classified_scrape'

    class_prep = ClassifiedScrapePreprocessing(prep_args)
    concat_data = class_prep.concat_all_classified_data()
    grouped_data = class_prep.group_data_by_class(list_data=concat_data)

    class_prep.train_val_test_to_json_split(class_grouped_data=grouped_data,
                                            train_outfile='train_classified',
                                            val_outfile='eval_classified',
                                            test_outfile='test_classified',
                                            train_size=0.9,
                                            test_size=30)
