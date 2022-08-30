import hashlib
import json
import os.path
import random
import re
from typing import List

import nltk.data
import numpy as np
from ftfy import fix_encoding
from local_constants import DATA_DIR, FILTERED_SCRAPE_DIR, SCRAPED_DATA_DIR


class RawScrapePreprocessing:
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
    save_data: str

    Methods
    -------
    from_raw_to_train_val()
            Calls:
                extract_danish_and_save_from_raw()
                split_to_sentences()
                split_train_val()
    extract_danish_and_save_from_raw(confidence_threshold)
            Takes raw scrape file(s), extracts danish text and save filtered
    split_to_sentences(out_file_name, word_count_threshold)
            Split all approved text blocks to sentences with self.sentence_splitter.
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
                 train_output: str = 'train.json',
                 val_output: str = 'validation.json',
                 save_data: bool = True):
        self.train_output = train_output
        self.val_output = val_output
        self.save_data = save_data
        self.sentence_splitter = nltk.data.load('tokenizers/punkt/danish.pickle')

    def from_raw_to_train_val(self):
        """
        Generate train and validation data as json line files from raw scrape file
        """
        self.extract_danish_and_save_from_raw()
        self.split_to_sentences(out_file_name='unique_sentences.json')
        self.split_train_val(in_file='unique_sentences.json')

    def extract_danish_and_save_from_raw(self, confidence_threshold: float = 0.6):
        """
        Filters each raw scrape file, keeps line id and page_filtered_text where danish is detected
        and writes to json line file filtered_scrape/<source_name>_filtered
        :param confidence_threshold: Keep all "danish-detected" sentences with a score above this
        threshold
        """
        for filename in os.listdir(SCRAPED_DATA_DIR):
            data = []
            filtered_filename = filename.split('_')[0] + '_filtered'
            false_lang_preds = []
            with open(os.path.join(SCRAPED_DATA_DIR, filename), 'rb') as file:
                for index, line in enumerate(file):
                    data_dict = json.loads(line)
                    if "__label__da" in data_dict['detected_page_lang']:
                        confidence = float(
                            re.findall(r'\d+\.\d+', data_dict['detected_page_lang'])[0])
                        if confidence < confidence_threshold:
                            false_lang_preds.append(1)
                        else:
                            # ToDo: Below does not fix encoding for example 'Ã¥'
                            if ('Ã¥' or 'Ã¸') in data_dict['page_filtered_text']:
                                data_dict['page_filtered_text'] = fix_encoding(
                                    data_dict['page_filtered_text'])
                            data.append({'id': index, 'url': data_dict['redirected_to_url'],
                                         'sha512': data_dict['redirected_to_url_sha512'],
                                         'text': data_dict['page_filtered_text']})
            print(f'Number of false preds {filtered_filename}: {np.sum(false_lang_preds)}')
            with open(os.path.join(FILTERED_SCRAPE_DIR, filtered_filename + '.json'), 'w',
                      encoding='utf-8') as outfile:
                for entry in data:
                    json.dump(entry, outfile)
                    outfile.write('\n')

    def split_to_sentences(self, out_file_name: str = 'unique_sentences.json',
                           word_count_threshold: int = 5):
        """
        Split all approved text blocks to sentences with self.sentence_splitter.
        :param out_file_name: json out file name
        :param word_count_threshold: Keep only sentences with word count above this threshold
        """
        approved_sentences = []
        dissapproved_sentences = []
        unique_approved = []
        seen = set()
        with open(os.path.join(FILTERED_SCRAPE_DIR, out_file_name), 'w',
                  encoding='utf-8') as outfile:
            for filename in os.listdir(FILTERED_SCRAPE_DIR):
                if not filename == out_file_name:
                    with open(os.path.join(FILTERED_SCRAPE_DIR, filename), 'rb') as file:
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
                                    approved_sentences.append(1)
                                    line_hash = hashlib.md5(final_sentence.encode()).digest()
                                    if line_hash not in seen:
                                        seen.add(line_hash)
                                        unique_approved.append(1)
                                        json.dump({'id': data_dict['id'], 'sentence': i,
                                                   'url': data_dict['url'],
                                                   'sha512': data_dict['sha512'],
                                                   'text': final_sentence}, outfile)
                                        outfile.write('\n')
                                else:
                                    dissapproved_sentences.append(1)

        print(f'Approved sentences: {np.sum(approved_sentences)}')
        print(f'Dissapproved sentences: {np.sum(dissapproved_sentences)}')
        print(f'Total unique sentences: {np.sum(unique_approved)}')

    def split_train_val(self,
                        in_file: str = 'unique_sentences.json',
                        split: float = 0.9,
                        seed: int = 42):
        """
        Split approved sentences to train and validation set and save as json
        :param in_file: json line file
        :param split: float between 0 and 1 specifying size of train
        :param seed: seed for reproducibility
        """
        sentences = []
        with open(os.path.join(FILTERED_SCRAPE_DIR, in_file), 'r', encoding='utf-8') as file:
            for line in file:
                data_dict = json.loads(line)
                sentences.append(data_dict['text'])
        random.seed(seed)
        random.shuffle(sentences)
        train_idx = int(len(sentences) * split)
        train = sentences[:train_idx]
        val = sentences[train_idx:]

        self.save_datasets(train=train, val=val)

    def save_datasets(self, train: List[str], val: List[str]):
        """
        Take train and validation data as input and saves to json-lines files compatible with torch
        dataset
        :param train: List[str] where each element is a valid sentence for training
        :param val: List[str] where each element is a valid sentence for validation
        """
        with open(os.path.join(DATA_DIR, self.train_output), 'w', encoding='utf-8') as outfile:
            for entry in train:
                json.dump({'text': entry}, outfile)
                outfile.write('\n')

        with open(os.path.join(DATA_DIR, self.val_output), 'w', encoding='utf-8') as outfile:
            for entry in val:
                json.dump({'text': entry}, outfile)
                outfile.write('\n')


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--train_outfile', type=str,
                        default='train.json', help="Name of final training data file")
    parser.add_argument('--val_outfile', type=str,
                        default='validation.json', help="Name of final validation data file")

    args = parser.parse_args()

    data_preprocessor = RawScrapePreprocessing(train_output=args.train_outfile,
                                               val_output=args.val_outfile)
    data_preprocessor.from_raw_to_train_val()
