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
from transformers import AutoTokenizer, AutoModelWithLMHead

from data_utils.perplex import score_gpt2
from local_constants import DATA_DIR, FILTERED_SCRAPE_DIR, SCRAPED_DATA_DIR, PREP_DATA_DIR
from utils.helpers import count_num_lines


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
                 save_data: bool = True,
                 split: float = 0.95,
                 add_ppl: bool = True,
                 ppl_threshold: int = 1000):
        self.train_output = train_output
        self.val_output = val_output
        self.save_data = save_data
        self.sentence_splitter = nltk.data.load('tokenizers/punkt/danish.pickle')
        self.split = split
        self.ppl_threshold = ppl_threshold
        self.add_ppl = add_ppl
        self.model_id = 'pere/norwegian-gpt2'

    def from_raw_to_train_val(self):
        """
        Generate train and validation data as json line files from raw scrape file
        """
        self.extract_danish_and_save_from_raw()
        self.split_to_sentences()
        self.filter_ppl_scores()
        self.split_train_val(split=self.split)

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
                print(f'Observations discarded in {filtered_filename}: {np.sum(false_lang_preds)}')
                print(
                    f'Urls approved in {filtered_filename}: {index + 1 - np.sum(false_lang_preds)} of {index + 1}')
                with open(os.path.join(FILTERED_SCRAPE_DIR, filtered_filename + '.json'), 'w',
                          encoding='utf-8') as outfile:
                    for entry in data:
                        json.dump(entry, outfile)
                        outfile.write('\n')
            else:
                print(f'File {filename} is not a scrape file')
        print("Finished extracting text.")

    def split_to_sentences(self, out_file_name: str = 'unique_sentences.json',
                           word_count_threshold: int = 8):
        """
        Split all approved text blocks to sentences with self.sentence_splitter.
        :param out_file_name: json out file name
        :param word_count_threshold: Keep only sentences with word count above this threshold
        """
        if self.add_ppl:
            print('Adding perplexity to each sentence')
            model, tokenizer = self.load_model_for_ppl(model_id=self.model_id)
        approved_sentences = []
        disapproved_sentences = []
        unique_approved = []
        wrong_letter_counter = 0
        wrong_letter_examples = []
        print("Splitting data to sentences...")
        with open(os.path.join(PREP_DATA_DIR, out_file_name), 'w',
                  encoding='utf-8') as outfile:
            file_count = len(os.listdir(FILTERED_SCRAPE_DIR))
            for file_index, filename in enumerate(os.listdir(FILTERED_SCRAPE_DIR)):
                # ToDo: Maybe move seen above loop, such that we dont have same sentences across municipalities
                if '_filtered.json' in filename:
                    seen = set()
                    with open(os.path.join(DATA_DIR, 'data_testing', f'unique_{filename}'), 'w',
                              encoding='utf-8') as muni_outfile:
                        test_sentences = []
                        in_file_path = os.path.join(FILTERED_SCRAPE_DIR, filename)
                        total_lines = count_num_lines(file_path=in_file_path)
                        with open(in_file_path, 'r', encoding='utf-8') as file:
                            for line in tqdm(file,
                                             total=total_lines,
                                             desc=f'{filename}: {file_index + 1} of {file_count}',
                                             unit="line"):
                                data_dict = json.loads(line)
                                raw_text = repr(data_dict['text'])
                                special_chars1 = ['\\r', '\\t', '\\n', '\\xa0', ' | ', '|', '*']
                                prep_text = raw_text
                                for special_char in special_chars1:
                                    prep_text = prep_text.replace(special_char, ' ')
                                special_chars2 = ['”', '"', "'"]
                                for special_char in special_chars2:
                                    prep_text = prep_text.replace(special_char, '')

                                special_words1 = ['NemID', 'MitID', 'minSU', 'LinkedIn', ]
                                for case in special_words1:
                                    if case in prep_text:
                                        prep_text = prep_text.replace(case, case.lower())

                                prep_text = re.sub(' +', ' ', prep_text)

                                find_wrong_break_ids = [
                                    (m.start(0), m.end(0)) for m in
                                    re.finditer("\?[A-Z]|\.[A-Z]", prep_text)]

                                if len(find_wrong_break_ids) > 0:
                                    increment = 1
                                    for id in find_wrong_break_ids:
                                        prep_text = prep_text[:id[0] + increment] + '\n' + \
                                                    prep_text[id[0] + increment:]
                                        increment += 1

                                text_splitted = (
                                    '\n'.join(self.sentence_splitter.tokenize(prep_text)))

                                sentences = re.split('\n', text_splitted)
                                new_sentences = []
                                for j, sentence in enumerate(sentences):

                                    find_wrong_letter_ids = [
                                        (m.start(0), m.end(0)) for m in
                                        re.finditer("[a-z][A-Z]", sentence)]

                                    if len(find_wrong_letter_ids) > 0:
                                        wrong_letter_counter += 1
                                        wrong_letter_examples.append(data_dict['url'])
                                        disapproved_sentences.append(
                                            f'{data_dict["id"]} - {j} - {sentence}')

                                    else:
                                        new_sentences.append(sentence)

                                for i, sentence in enumerate(new_sentences):
                                    final_sentence = sentence.strip()
                                    search = any(c.isalpha() for c in final_sentence)
                                    word_count = len(final_sentence.split(' '))
                                    if word_count >= word_count_threshold and search:
                                        approved_sentences.append(1)
                                        line_hash = hashlib.md5(final_sentence.encode()).digest()
                                        if line_hash not in seen:
                                            seen.add(line_hash)
                                            unique_approved.append(1)
                                            if self.add_ppl:
                                                ppl_score = score_gpt2(final_sentence, model,
                                                                       tokenizer)
                                                json.dump({'id': data_dict['id'], 'sentence': i,
                                                           'kommune': filename.split('_')[0],
                                                           'url': data_dict['url'],
                                                           'sha512': data_dict['sha512'],
                                                           'text': final_sentence,
                                                           'ppl_score': str(ppl_score)}, outfile)
                                            else:
                                                json.dump({'id': data_dict['id'], 'sentence': i,
                                                           'kommune': filename.split('_')[0],
                                                           'url': data_dict['url'],
                                                           'sha512': data_dict['sha512'],
                                                           'text': final_sentence}, outfile)

                                            outfile.write('\n')
                                            muni_outfile.write(
                                                f"{data_dict['id']} - {i} - {final_sentence}\n")
                                            test_sentences.append(
                                                [data_dict['id'], i, data_dict['url'],
                                                 final_sentence])
                                    else:
                                        if not len(final_sentence.strip()) == 0:
                                            disapproved_sentences.append(
                                                f'{data_dict["id"]} - {i} - {final_sentence}')
                    print(f'n unique in {filename}: {len(seen)}')

        with open(os.path.join(DATA_DIR, 'data_testing/disaproved_sentences.txt'), 'w',
                  encoding='utf-8') as dis_outfile:
            for sentence in disapproved_sentences:
                dis_outfile.write(f"{sentence}\n")
        print(f'Approved sentences: {np.sum(approved_sentences)}')
        print(f'Disapproved sentences: {len(disapproved_sentences)}')
        print(f'Total unique sentences: {np.sum(unique_approved)}')
        print()

    def split_train_val(self,
                        in_file: str = 'approved_sentences_ppl.json',
                        split: float = 0.95,
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

    @staticmethod
    def load_model_for_ppl(model_id: str = 'pere/norwegian-gpt2', device: str = 'cuda'):
        tokenizer = AutoTokenizer.from_pretrained(model_id)

        model = AutoModelWithLMHead.from_pretrained(model_id)
        model = model.to(device)
        model.eval()
        return model, tokenizer

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
    parser.add_argument('--train_outfile', type=str,
                        default='train.json', help="Name of final training data file")
    parser.add_argument('--val_outfile', type=str,
                        default='validation.json', help="Name of final validation data file")
    parser.add_argument('--split', type=float, default=0.95,
                        help='training set size between 0 and 1')
    parser.add_argument('--add_ppl', type=lambda x: bool(strtobool(x)), default=True,
                        help='whether or not to add ppl_score to unique sentences')
    parser.add_argument('--ppl_threshold', type=int, default=10000,
                        help='ppl_threshold for approving sentences')

    args = parser.parse_args()

    data_preprocessor = RawScrapePreprocessing(train_output=args.train_outfile,
                                               val_output=args.val_outfile,
                                               split=args.split,
                                               ppl_threshold=args.ppl_threshold,
                                               add_ppl=args.add_ppl)
    # data_preprocessor.split_to_sentences()
    data_preprocessor.from_raw_to_train_val()
    # data_preprocessor.filter_ppl_scores()
