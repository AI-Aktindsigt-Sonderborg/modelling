import argparse
import hashlib
import itertools
import json
import os.path
import random
import re
from typing import List

import nltk.data
import numpy as np
import pandas as pd
from ftfy import fix_encoding
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from data_utils.data_prep_input_args import DataPrepArgParser
from data_utils.helpers import write_json_lines, split_sentences, write_text_lines, score_gpt2, \
    load_model_for_ppl, find_letters_and_word_count
from local_constants import DATA_DIR, FILTERED_SCRAPE_DIR, SCRAPED_DATA_DIR, PREP_DATA_DIR, \
    CLASS_DATA_DIR
from utils.helpers import TimeCode, read_jsonlines, save_json
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

    def __init__(self, args: argparse.Namespace):
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

    def extract_danish_and_save_from_raw(self):
        """
        Filters each raw scrape file, keeps line id and page_filtered_text where danish is detected
        and writes to json line file filtered_scrape/<source_name>_filtered
        :param confidence_threshold: Keep all "danish-detected" sentences with a score above this
        threshold
        """

        print("Initial extraction of raw text...")
        file_count = len(os.listdir(SCRAPED_DATA_DIR))
        for file_index, filename in enumerate(os.listdir(SCRAPED_DATA_DIR)):
            out_data = []
            filtered_filename = filename.split('_')[0] + '_filtered'
            false_lang_preds = []
            if '_scrape_output.jsonl' in filename:
                in_file_path = os.path.join(SCRAPED_DATA_DIR, filename)
                total_lines = count_num_lines(file_path=in_file_path)
                with open(in_file_path, 'rb') as file:
                    index: int
                    for index, line in enumerate(tqdm(file,
                                                      total=total_lines,
                                                      desc=f'{filename}: {file_index + 1} of {file_count}',
                                                      unit="line")):
                        data_dict = json.loads(line)

                        if self.is_correct_danish(
                            data_dict=data_dict,
                            confidence_threshold=self.args.danish_threshold):

                            data_dict = self.fix_utf8_encodings(data_dict=data_dict)
                            out_data.append({'id': index, 'url': data_dict['redirected_to_url'],
                                             'sha512': data_dict['redirected_to_url_sha512'],
                                             'text': data_dict['page_filtered_text']})

                        else:
                            false_lang_preds.append(1)

                    assert total_lines == index + 1, {'Total lines and index dont match'}
                    print(f'Observations discarded in '
                          f'{filtered_filename}: {np.sum(false_lang_preds)}')
                    print(f'Urls approved in {filtered_filename}: '
                          f'{index + 1 - np.sum(false_lang_preds)} of {index + 1}')
                write_json_lines(out_dir=FILTERED_SCRAPE_DIR, filename=filtered_filename,
                                 data=out_data)
            else:
                print(f'File {filename} is not a scrape file')
        print("Finished extracting text.")

    def create_unique_sentences(self, out_file_name: str = 'unique_sentences.json'):
        """
        Split all approved text blocks to sentences with self.sentence_splitter.
        :param out_file_name: json out file name
        """
        # Init TimeCode
        timer = TimeCode()
        if self.args.add_ppl:
            print('Adding perplexity to each sentence')
            model, tokenizer = load_model_for_ppl(model_id=self.model_id)
        else:
            model = None
            tokenizer = None
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
                                    # Strip sentence, search for letters and filter by word count
                                    dump_data = self.create_dump_data(data=data_dict,
                                                                      sentence_counter=i,
                                                                      sentence=sentence, seen=seen,
                                                                      filename=filename,
                                                                      model=model,
                                                                      tokenizer=tokenizer)
                                    if dump_data:
                                        approved_sentences.append(1)
                                        unique_approved.append(1)
                                    else:
                                        if final_sentence.strip():
                                            disapproved_sentences.append(
                                                f'{filename.split("_")[0]} - {data_dict["id"]} - '
                                                f'{i} - {final_sentence}')

                                    final_sentence = sentence.strip()
                                    if find_letters_and_word_count(
                                        text=final_sentence,
                                        word_count_threshold=self.args.min_len):
                                        approved_sentences.append(1)
                                        line_hash = hashlib.md5(final_sentence.encode()).digest()
                                        # add to unique sentences if not already exists
                                        if line_hash not in seen:
                                            seen.add(line_hash)
                                            unique_approved.append(1)
                                            dump_data = {'id': data_dict['id'], 'sentence': i,
                                                         'kommune': filename.split('_')[0],
                                                         'url': data_dict['url'],
                                                         'sha512': data_dict['sha512'],
                                                         'text': final_sentence}

                                            if self.args.add_ppl:
                                                dump_data['ppl_score'] = \
                                                    str(score_gpt2(text=final_sentence,
                                                                   model=model,
                                                                   tokenizer=tokenizer))

                                            json.dump(dump_data, outfile)

                                            outfile.write('\n')

                                            muni_outfile.write(
                                                f"{data_dict['id']} - {i} - {sentence}\n")
                                    else:
                                        if not len(final_sentence.strip()) == 0:
                                            disapproved_sentences.append(
                                                f'{filename.split("_")[0]} - {data_dict["id"]} - '
                                                f'{i} - {final_sentence}')
        write_text_lines(out_dir=DATA_DIR, filename='data_testing/disapproved_sentences',
                         data=disapproved_sentences)
        # Print running time of code
        timer.how_long_since_start()
        print(f'Approved sentences: {np.sum(approved_sentences)}')
        print(f'Disapproved sentences: {len(disapproved_sentences)}')
        print(f'Total unique sentences: {np.sum(unique_approved)}')
        print()

    def create_dump_data(self, data: dict, sentence_counter: int, sentence: str, seen,
                         filename, model, tokenizer):
        final_sentence = sentence.strip()
        if find_letters_and_word_count(
            text=final_sentence,
            word_count_threshold=self.args.min_len):

            line_hash = hashlib.md5(final_sentence.encode()).digest()
            # add to unique sentences if not already exists
            if line_hash in seen:
                return None

            seen.add(line_hash)

            dump_data = {'id': data['id'], 'sentence': sentence_counter,
                         'kommune': filename.split('_')[0],
                         'url': data['url'],
                         'sha512': data['sha512'],
                         'text': final_sentence}

            if self.args.add_ppl:
                dump_data['ppl_score'] = \
                    str(score_gpt2(text=final_sentence,
                                   model=model,
                                   tokenizer=tokenizer))
            return dump_data

        return None

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
        train_init = sentences[:train_idx]

        val = sentences[train_idx:]
        val = [{'text': x} for x in val]
        if self.args.split_train_n_times > 0:
            train_chunks = np.array_split(train_init, self.args.split_train_n_times)
            for i, train_chunk in enumerate(train_chunks):
                train = [{'text': x} for x in list(train_chunk)]
                train_outfile = self.args.train_outfile + f'_{i}'
                self.save_datasets(train=train, train_outfile=train_outfile)
            self.save_datasets(val=val)
        else:
            train = [{'text': x} for x in train_init]
            self.save_datasets(train=train, val=val, train_outfile=self.args.train_outfile)

    def save_datasets(self, train: List[dict] = None, val: List[dict] = None,
                      train_outfile: str = None):
        """
        Take train and validation data as input and saves to json-lines files compatible with torch
        dataset
        :param train_outfile: name of train file
        :param train: List[str] where each element is a valid sentence for training
        :param val: List[str] where each element is a valid sentence for validation
        """
        if train and train_outfile:
            write_json_lines(out_dir=DATA_DIR, filename=train_outfile, data=train)
        if val:
            write_json_lines(out_dir=DATA_DIR, filename=self.args.val_outfile, data=val)

    @staticmethod
    def is_correct_danish(data_dict: dict, confidence_threshold: int):
        """
        Detect correct danish text without raw html code
        :param data: raw text from scraped url
        :param confidence_threshold: confidence in which we filter danish prediction
        :return: boolean whether to keep data
        """
        if "__label__da" not in data_dict['detected_page_lang']:
            return False
        confidence = float(re.findall(r'\d+\.\d+', data_dict['detected_page_lang'])[0])
        discard_conditions = [confidence < confidence_threshold,
                              '<div ' in data_dict['page_filtered_text'],
                              'endif' in data_dict['page_filtered_text'],
                              '<class' in data_dict['page_filtered_text']]

        for condition in discard_conditions:
            if condition:
                return False
        return True

    @staticmethod
    def fix_utf8_encodings(data_dict: dict):
        """
        Use fix_encoding and manually replace wrong danish letters
        :param data: dictionary containing text from scrape url
        :return: dictionary with correct encoded text
        """
        data_dict['page_filtered_text'] = fix_encoding(
            data_dict['page_filtered_text'])
        wrong_encodings = [['Ã¥', 'å'], ['Ã¸', 'ø'], ['Ã¦', 'æ']]
        for wrong_encoding in wrong_encodings:
            if wrong_encoding[0] in data_dict['page_filtered_text']:
                data_dict['page_filtered_text'] = data_dict['page_filtered_text'].replace(
                    wrong_encoding[0],
                    wrong_encoding[1])
        return data_dict

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

            for line in tqdm(infile, total=total_lines,
                             desc=in_file_name, unit="line"):
                data_dict = json.loads(line)
                ppl_score = float(data_dict['ppl_score'])
                if ppl_score < ppl_threshold:
                    json.dump(data_dict, approved_sentences)
                    approved_sentences.write('\n')
                else:
                    json.dump(data_dict, disapproved_sentences)
                    disapproved_sentences.write('\n')
        print("Finished.")


class ClassifiedScrapePreprocessing:
    """
    Class to read raw excel file with classified KL categories and split to train and validation data
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
        data = pd.read_excel(os.path.join(file_dir, 'raw', self.args.excel_classification_file),
                             sheet_name='Klassificering',
                             header=1)
        data['index'] = range(len(data))
        data['text_len'] = list(map(lambda x: len(str(x)), data['text']))
        if drop_na:
            data = data.dropna(subset=['klassifikation'])
        if ppl_filters:
            data = data[(data['ppl_score'] > ppl_filters[0]) & (data['ppl_score'] < ppl_filters[1])]

        data_dicts = data.to_dict('records')
        save_json(output_dir=CLASS_DATA_DIR, data=data_dicts,
                  filename=self.args.classified_scrape_file)

    def read_classified_json(self):
        return read_jsonlines(input_dir=CLASS_DATA_DIR, filename=self.args.classified_scrape_file)

    @staticmethod
    def group_data_by_class(list_data: List[dict]):
        """
        Group data by class
        :param data: list of dictionarys of sentences
        :return: list of lists of dicts
        """

        label_set = set(map(lambda x: x['label'], list_data))

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
        :param train_size: float between 0 and 1 specifying the size of the train set where
        1 is all data
        :param test_size: int >= 1 specifying the number of sentences in each class
        :param train_outfile: if train_outfile specified generate train set
        :param val_outfile: if val_outfile specified generate validation set
        :param test_outfile: if test_outfile specified generate test set
        :return:
        """
        assert train_outfile and test_outfile, \
            '\n At least train_outfile and test_outfile must be specified - see doc: \n' + \
            ClassifiedScrapePreprocessing.train_val_test_to_json_split.__doc__

        assert not train_size and not test_size, \
            'Either train or test size must be specified - see doc: \n' + \
            ClassifiedScrapePreprocessing.train_val_test_to_json_split.__doc__

        if train_outfile and test_size and not val_outfile:
            train_test = [train_test_split(x, test_size=test_size, random_state=1) for x in
                          class_grouped_data]
            train = list(itertools.chain.from_iterable([x[0] for x in train_test]))
            test = list(itertools.chain.from_iterable([x[1] for x in train_test]))

        if train_outfile and val_outfile and test_outfile and train_size and test_size:
            train_val = [train_test_split(x, train_size=train_size, random_state=1) for x in
                         class_grouped_data]
            train = list(itertools.chain.from_iterable([x[0] for x in train_val]))
            val_tmp = list(itertools.chain.from_iterable([x[1] for x in train_val]))

            val_grouped = ClassifiedScrapePreprocessing.group_data_by_class(val_tmp)

            val_test = [train_test_split(x, test_size=test_size, random_state=1) for x in
                        val_grouped]

            val = list(itertools.chain.from_iterable([x[0] for x in val_test]))
            test = list(itertools.chain.from_iterable([x[1] for x in val_test]))

        if train_outfile:
            save_json(os.path.join(CLASS_DATA_DIR, 'processed'), data=train, filename=train_outfile)
        if val_outfile:
            save_json(os.path.join(CLASS_DATA_DIR, 'processed'), data=val, filename=val_outfile)
        if test_outfile:
            save_json(os.path.join(CLASS_DATA_DIR, 'processed'), data=test, filename=test_outfile)
        return print('datasets generated')

    @staticmethod
    def concat_all_classified_data(data_dir: str = CLASS_DATA_DIR):
        all_data = []
        for filename in os.listdir(data_dir):
            filepath = os.path.join(data_dir, filename)
            if os.path.isfile(filepath):
                tmp_data = read_jsonlines(input_dir=data_dir, filename=filename.split('.')[0])
                for line in tmp_data:
                    if isinstance(line['text'], str):
                        all_data.append({'label': line['klassifikation'], 'text': line['text']})
                    else:
                        print(f'line {line} is empty')

        return all_data


if __name__ == '__main__':

    prep_parser = DataPrepArgParser()
    prep_args = prep_parser.parser.parse_args()
    prep_args.add_ppl = False
    # data_preprocessor = RawScrapePreprocessing(args=prep_args)
    # data_preprocessor.from_raw_to_train_val()

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

    print()
