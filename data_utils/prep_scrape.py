import hashlib
import json
import os.path
import random
import re
from distutils.util import strtobool
from typing import List

import nltk.data
import numpy as np
from ftfy import fix_encoding
from tqdm import tqdm

from data_utils.helpers import write_json_lines, split_sentences, write_text_lines, score_gpt2, \
    load_model_for_ppl
from local_constants import DATA_DIR, FILTERED_SCRAPE_DIR, SCRAPED_DATA_DIR, PREP_DATA_DIR
from utils.helpers import TimeCode
from utils.helpers import count_num_lines


class RawScrapePreprocessing:
    """
    Class to preprocess raw data to ML from web scraper
    Attributes
    ----------
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

    def __init__(self, args):
        self.args = args
        self.sentence_splitter = nltk.data.load('tokenizers/punkt/danish.pickle')
        self.model_id = 'pere/norwegian-gpt2'

    def from_raw_to_train_val(self):
        """
        Generate train and validation data as json line files from raw scrape file
        """
        self.extract_danish_and_save_from_raw()
        self.create_unique_sentences()
        if self.args.add_ppl:
            self.filter_ppl_scores()
            self.split_train_val(in_file='approved_sentences_ppl.json')
        else:
            self.split_train_val()

    def extract_danish_and_save_from_raw(self, confidence_threshold: float = 0.6):
        """
        Filters each raw scrape file, keeps line id and page_filtered_text where danish is detected
        and writes to json line file filtered_scrape/<source_name>_filtered
        :param confidence_threshold: Keep all "danish-detected" sentences with a score above this
        threshold
        """

        print("Initial extraction of raw text...")
        file_count = len(os.listdir(SCRAPED_DATA_DIR))
        for file_index, filename in enumerate(os.listdir(SCRAPED_DATA_DIR)):
            data = []
            filtered_filename = filename.split('_')[0] + '_filtered'
            false_lang_preds = []
            if '_scrape_output.jsonl' in filename:
                in_file_path = os.path.join(SCRAPED_DATA_DIR, filename)
                total_lines = count_num_lines(file_path=in_file_path)
                with open(in_file_path, 'rb') as file:
                    for index, line in enumerate(tqdm(file,
                                                      total=total_lines,
                                                      desc=f'{filename}: {file_index + 1} of {file_count}',
                                                      unit="line")):
                        data_dict = json.loads(line)
                        if "__label__da" in data_dict['detected_page_lang']:
                            confidence = float(
                                re.findall(r'\d+\.\d+', data_dict['detected_page_lang'])[0])
                            if confidence < confidence_threshold:
                                false_lang_preds.append(1)
                            elif '<div ' in data_dict['page_filtered_text']:
                                false_lang_preds.append(1)
                            elif 'endif' in data_dict['page_filtered_text']:
                                false_lang_preds.append(1)
                            elif '<class' in data_dict['page_filtered_text']:
                                false_lang_preds.append(1)
                            else:
                                # ToDo: Below does not fix encoding for example 'Ã¥'
                                if ('Ã¥' or 'Ã¸') in data_dict['page_filtered_text']:
                                    data_dict['page_filtered_text'] = fix_encoding(
                                        data_dict['page_filtered_text'])
                                data.append({'id': index, 'url': data_dict['redirected_to_url'],
                                             'sha512': data_dict['redirected_to_url_sha512'],
                                             'text': data_dict['page_filtered_text']})
                        else:
                            false_lang_preds.append(1)
                    assert total_lines == index + 1, {'Total lines and index dont match'}
                    print(
                        f'Observations discarded in {filtered_filename}: {np.sum(false_lang_preds)}')
                    print(f'Urls approved in {filtered_filename}: '
                          f'{index + 1 - np.sum(false_lang_preds)} of {index + 1}')
                write_json_lines(out_dir=FILTERED_SCRAPE_DIR, filename=filtered_filename, data=data)
            else:
                print(f'File {filename} is not a scrape file')
        print("Finished extracting text.")

    def create_unique_sentences(self, out_file_name: str = 'unique_sentences.json'):
        """
        Split all approved text blocks to sentences with self.sentence_splitter.
        :param out_file_name: json out file name
        :param word_count_threshold: Keep only sentences with word count above this threshold
        """

        timer = TimeCode()
        if self.args.add_ppl:
            print('Adding perplexity to each sentence')
            model, tokenizer = load_model_for_ppl(model_id=self.model_id)
        approved_sentences = []
        disapproved_sentences = []
        unique_approved = []
        print("Splitting data to sentences...")
        seen = set()
        with open(os.path.join(PREP_DATA_DIR, out_file_name), 'w',
                  encoding='utf-8') as outfile:
            file_count = len(os.listdir(FILTERED_SCRAPE_DIR))
            for file_index, filename in enumerate(os.listdir(FILTERED_SCRAPE_DIR)):
                if '_filtered.json' in filename:
                    # Create file for each municipality for testing
                    with open(os.path.join(DATA_DIR, 'data_testing', f'unique_{filename}'), 'w',
                              encoding='utf-8') as muni_outfile:

                        in_file_path = os.path.join(FILTERED_SCRAPE_DIR, filename)
                        total_lines = count_num_lines(file_path=in_file_path)
                        with open(in_file_path, 'r', encoding='utf-8') as file:
                            for line in tqdm(file,
                                             total=total_lines,
                                             desc=f'{filename}: {file_index + 1} of {file_count}',
                                             unit="line"):
                                data_dict = json.loads(line)

                                # Do initial sentence split - returns list of all sentences
                                sentences, disapproved_sentences = split_sentences(
                                    data_dict=data_dict,
                                    sentence_splitter=self.sentence_splitter,
                                    disapproved_sentences=disapproved_sentences,
                                    filename=filename)

                                for i, sentence in enumerate(sentences):
                                    final_sentence = sentence.strip()
                                    search = any(c.isalpha() for c in final_sentence)
                                    word_count = len(final_sentence.split(' '))
                                    if word_count >= self.args.min_len and search:
                                        approved_sentences.append(1)
                                        line_hash = hashlib.md5(final_sentence.encode()).digest()
                                        if line_hash not in seen:
                                            seen.add(line_hash)
                                            unique_approved.append(1)
                                            dump_data = {'id': data_dict['id'], 'sentence': i,
                                                         'kommune': filename.split('_')[0],
                                                         'url': data_dict['url'],
                                                         'sha512': data_dict['sha512'],
                                                         'text': final_sentence}

                                            if self.args.add_ppl:
                                                ppl_score = score_gpt2(final_sentence, model,
                                                                       tokenizer)
                                                dump_data['ppl_score'] = str(ppl_score)

                                            json.dump(dump_data, outfile)

                                            outfile.write('\n')
                                            muni_outfile.write(
                                                f"{data_dict['id']} - {i} - {final_sentence}\n")
                                    else:
                                        if not len(final_sentence.strip()) == 0:
                                            disapproved_sentences.append(
                                                f'{filename.split("_")[0]} - {data_dict["id"]} - '
                                                f'{i} - {final_sentence}')
        write_text_lines(out_dir=DATA_DIR, filename='data_testing/disapproved_sentences',
                         data=disapproved_sentences)

        timer.how_long_since_start()
        print(f'Approved sentences: {np.sum(approved_sentences)}')
        print(f'Disapproved sentences: {len(disapproved_sentences)}')
        print(f'Total unique sentences: {np.sum(unique_approved)}')
        print()

    def split_train_val(self,
                        in_file: str = 'unique_sentences.json',
                        seed: int = 42):
        """
        Split approved sentences to train and validation set and save as json
        :param in_file: json line file
        :param split: float between 0 and 1 specifying size of train
        :param seed: seed for reproducibility
        """
        sentences = []
        with open(os.path.join(PREP_DATA_DIR, in_file), 'r', encoding='utf-8') as file:
            for line in file:
                data_dict = json.loads(line)
                sentences.append(data_dict['text'])
        random.seed(seed)
        random.shuffle(sentences)
        train_idx = int(len(sentences) * self.args.split)
        train = sentences[:train_idx]
        train = [{'text': x} for x in train]

        val = sentences[train_idx:]
        val = [{'text': x} for x in val]
        self.save_datasets(train=train, val=val)

    def save_datasets(self, train: List[dict], val: List[dict]):
        """
        Take train and validation data as input and saves to json-lines files compatible with torch
        dataset
        :param train: List[str] where each element is a valid sentence for training
        :param val: List[str] where each element is a valid sentence for validation
        """
        write_json_lines(out_dir=DATA_DIR, filename=self.train_output, data=train)
        write_json_lines(out_dir=DATA_DIR, filename=self.val_output, data=val)

    @staticmethod
    def filter_ppl_scores(ppl_threshold: int = 10000, in_file_name: str = 'unique_sentences.json'):
        """
        Filtering ppl scores from unique sentences - creates two files approved/disapproved
        :param ppl_threshold: perplexity threshold - approve all below
        :param in_file_name: unique sentences file
        """

        in_file_path = os.path.join(PREP_DATA_DIR, in_file_name)
        total_lines = count_num_lines(file_path=in_file_path)
        print("Filtering perplexity scores...")
        with open(in_file_path, 'r', encoding='utf-8') as infile, \
            open(os.path.join(PREP_DATA_DIR, 'approved_sentences_ppl.json'),
                 'w', encoding='utf-8') as approved_sentences, \
            open(os.path.join(PREP_DATA_DIR, 'disapproved_sentences_ppl.json'),
                 'w', encoding='utf-8') as disapproved_sentences:

            for index, line in enumerate(tqdm(infile, total=total_lines,
                                              desc=in_file_name, unit="line")):
                data_dict = json.loads(line)
                ppl_score = float(data_dict['ppl_score'])
                if ppl_score < ppl_threshold:
                    json.dump(data_dict, approved_sentences)
                    approved_sentences.write('\n')
                else:
                    json.dump(data_dict, disapproved_sentences)
                    disapproved_sentences.write('\n')
        print("Finished.")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--save_data', type=lambda x: bool(strtobool(x)), default=False,
                        help='whether or not to add ppl_score to unique sentences')
    parser.add_argument('--train_outfile', type=str,
                        default='train', help="Name of final training data file")
    parser.add_argument('--val_outfile', type=str,
                        default='validation', help="Name of final validation data file")
    parser.add_argument('--min_len', type=int, default=8,
                        help='Keep sentences with at least n words')
    parser.add_argument('--split', type=float, default=0.98,
                        help='training set size between 0 and 1')
    parser.add_argument('--add_ppl', type=lambda x: bool(strtobool(x)), default=True,
                        help='whether or not to add ppl_score to unique sentences')
    parser.add_argument('--ppl_threshold', type=int, default=10000,
                        help='ppl_threshold for approving sentences')

    prep_args = parser.parse_args()

    data_preprocessor = RawScrapePreprocessing(args=prep_args)
    # data_preprocessor.split_to_sentences()
    data_preprocessor.from_raw_to_train_val()
    # data_preprocessor.filter_ppl_scores()
