# pylint: skip-file
import json
import os
import re
from typing import List

import nltk
import numpy as np

from local_constants import PROJECT_ROOT, DATA_DIR


class InternalDataPreprocessing:
    """
        Class to preprocess raw data to ML from web scraper
        Attributes
        ----------
        data_dir: str
                main data dir
        train_output: str
                name of the json line file to be used for training
        val_output: str
                name of the json line file to be used for training
        data_sources: List[str]
                sources to pull data from

        Methods
        -------
        from_raw_to_train_val()
                Calls:
                    extract_danish_and_save_from_raw()
                    split_to_sentences()
                    split_train_val()
        concat_and_save_data(out_file_name)
                Extract text from all files in specified sources and concats to one json object
        split_to_sentences(in_file_name, out_file_name, word_count_threshold)
                Split all approved text blocks to sentences with self.sentence_splitter
                - save json
        split_train_val(in_file, save_datasets, split, seed):
                Split approved sentences to train and validation set
        save_datasets(train, val)
                Take train and validation data as input and saves to json-lines files compatible
                with torch dataset

        Example call
        ------------
        data_preprocessor = RawScrapePreprocessing(train_output='train_new_scrape.json',
                                                    val_output='val_new_scrape.json')
        data_preprocessor.from_raw_to_train_val()
        """

    def __init__(self,
                 data_dir: str = os.path.join(PROJECT_ROOT, DATA_DIR),
                 train_output: str = 'train.json',
                 val_output: str = 'validation.json',
                 data_sources: List[str] = ['dr', 'dagw'],
                 split: float = 0.95):

        self.data_dir = data_dir
        self.internal_data_dir = self.data_dir + '/internal_data'
        self.data_sources = data_sources
        self.train_output = train_output
        self.val_output = val_output
        self.sentence_splitter = nltk.data.load('tokenizers/punkt/danish.pickle')
        self.split = split

    def from_raw_to_train_val(self):
        self.concat_and_save_data()
        self.split_to_sentences()
        self.create_train_val()

    def concat_and_save_data(self, out_file_name: str = 'concat_data.json'):
        with open(os.path.join(self.internal_data_dir, out_file_name), 'w',
                  encoding='utf-8') as outfile:
            for root, dir, files in os.walk(self.internal_data_dir):
                if len(files) > 0 and root != self.internal_data_dir:
                    for file in files:

                        with open(os.path.join(root, file), 'r', encoding='utf-8') as file:
                            split = file.name.split('/')
                            text = file.read()
                            try:
                                json.dump({'source': str(split[-3]), 'sub_source': str(split[-2]),
                                           'filename': str(split[-1]), 'text': text}, outfile)
                                outfile.write('\n')
                            except Exception as e:
                                print(e)

    def split_to_sentences(self, in_file_name: str = 'concat_data.json',
                           out_file_name: str = 'all_sentences.json',
                           word_count_threshold: int = 5):
        # approved_sentences = []
        # dissapproved_sentences = []
        # unique_approved = []
        # seen = set()
        with open(os.path.join(self.internal_data_dir, out_file_name), 'w',
                  encoding='utf-8') as outfile:
            with open(os.path.join(self.internal_data_dir, in_file_name), 'rb') as file:
                for line in file:
                    data_dict = json.loads(line)
                    text_splitted = (
                        '\n'.join(self.sentence_splitter.tokenize(data_dict['text'])))
                    sentences = re.split('\n', text_splitted)
                    for i, sentence in enumerate(sentences):
                        final_sentence = sentence.strip().replace('|', '')
                        search = any(c.isalpha() for c in final_sentence)
                        word_count = len(final_sentence.split(' '))
                        if word_count >= word_count_threshold and search:
                            # approved_sentences.append(1)
                            # line_hash = hashlib.md5(final_sentence.encode()).digest()
                            # unique_approved.append(1)
                            json.dump({'filename': data_dict['filename'], 'sentence': i,
                                       'text': final_sentence}, outfile)
                            outfile.write('\n')
                            # if line_hash not in seen:
                            #     unique_approved.append(1)
                            #     json.dump({'filename': data_dict['filename'], 'sentence': i,
                            #                'text': final_sentence}, outfile)
                            #     outfile.write('\n')
                            #
                            #     # seen.add(line_hash)
                            # else:
                            #     dissapproved_sentences.append(1)

        # print(f'Approved sentences: {np.sum(approved_sentences)}')
        # print(f'Dissapproved sentences: {np.sum(dissapproved_sentences)}')
        # print(f'Total unique sentences: {np.sum(unique_approved)}')

    def create_train_val(self, in_file: str = 'all_sentences.json',
                         seed: int = 42):
        np.random.seed(seed)
        with open(os.path.join(self.internal_data_dir, self.train_output), 'w',
                  encoding='utf-8') as train_file:
            with open(os.path.join(self.internal_data_dir, self.val_output), 'w',
                      encoding='utf-8') as val_file:

                with open(os.path.join(self.internal_data_dir, in_file), 'r',
                          encoding='utf-8') as file:

                    for line in file:
                        data_dict = json.loads(line)
                        a = np.random.choice([0, 1], 1, p=[self.split, 1 - self.split])
                        if a == 0:
                            json.dump({'text': data_dict['text']}, train_file)
                            train_file.write('\n')
                        if a == 1:
                            json.dump({'text': data_dict['text']}, val_file)
                            val_file.write('\n')


if __name__ == "__main__":
    data_prep = InternalDataPreprocessing(split=0.95, data_sources=['dagw'],
                                          train_output='train_dagw.json',
                                          val_output='val_dagw.json')
    # data_prep.concat_and_save_data()
    # data_prep.split_to_sentences()
    # data_prep.create_train_val()
    data_prep.from_raw_to_train_val()
