import argparse
import hashlib
import json
import os.path
import random
import re
from typing import List, Tuple, Union, Optional

import nltk.data
import numpy as np
from ftfy import fix_encoding
from tqdm import tqdm

from mlm.data_utils.data_prep_input_args import DataPrepArgParser
from mlm.data_utils.helpers import split_sentences
from mlm.local_constants import DATA_DIR, FILTERED_SCRAPE_DIR, \
    SCRAPED_DATA_DIR, PREP_DATA_DIR
from sc.local_constants import CLASS_DATA_DIR
from shared.data_utils.helpers import score_gpt2, load_model_for_ppl, \
    find_letters_and_word_count
from shared.utils.helpers import TimeCode, read_json_lines, write_json_lines, \
    write_text_lines
from shared.utils.helpers import count_num_lines

DEFAULT_UNIQUE_SENTENCES_FILE = 'unique_sentences.jsonl'


class RawScrapePreprocessing:
    """
    Class to preprocess raw data to MLM format from municipality scraped data.

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
        self.extract_danish_and_save_from_raw()
        self.create_unique_sentences()
        if self.args.add_ppl:
            self.filter_ppl_scores()
            self.split_train_val(in_file='approved_sentences_ppl.json')
        else:
            self.split_train_val()

    def extract_danish_and_save_from_raw(self):
        """
        Filters each raw scrape file, keeps line id and page_filtered_text
        where danish is detected and writes to json line file
        *filtered_scrape/<source_name>_filtered*.
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
                    for index, line in enumerate(tqdm(
                        file,
                        total=total_lines,
                        desc=f'{filename}: {file_index + 1} of {file_count}',
                        unit="line")):

                        data_dict = json.loads(line)

                        if self.is_correct_danish(
                            data_dict=data_dict,
                            confidence_threshold=self.args.danish_threshold):

                            data_dict = self.fix_utf8_encodings(
                                data_dict=data_dict)
                            out_data.append({'id': index, 'url': data_dict[
                                'redirected_to_url'],
                                             'sha512': data_dict[
                                                 'redirected_to_url_sha512'],
                                             'text': data_dict[
                                                 'page_filtered_text']})

                        else:
                            false_lang_preds.append(1)

                    assert total_lines == index + 1, {
                        'Total lines and index dont match'}
                    print(f'Observations discarded in '
                          f'{filtered_filename}: {np.sum(false_lang_preds)}')
                    print(
                        f'Urls approved in {filtered_filename}: '
                        f'{index + 1 - np.sum(false_lang_preds)} of '
                        f'{index + 1}')
                write_json_lines(out_dir=FILTERED_SCRAPE_DIR,
                                 filename=filtered_filename,
                                 data=out_data)
            else:
                print(f'File {filename} is not a scrape file')
        print("Finished extracting text.")

    def create_unique_sentences(
        self,
        out_file_name: str = DEFAULT_UNIQUE_SENTENCES_FILE):
        """
        Split all approved text blocks to sentences with self.sentence_splitter.
        Calls ``create_dump_data`` and helper function ``split_sentences``.

        :param str out_file_name: json out file name
        """

        # Init TimeCode
        timer = TimeCode()
        if self.args.add_ppl:
            print('Adding perplexity to each sentence')
            model, tokenizer = load_model_for_ppl(model_id=self.model_id)
        else:
            model = None
            tokenizer = None

        class_data = read_json_lines(
            input_dir=os.path.join(CLASS_DATA_DIR, 'processed'),
            filename='all_sentences')
        class_texts = [x['text'] for x in class_data]
        if self.args.lower_case:
            class_texts = [x.lower() for x in class_texts]

        class_hash = {hashlib.md5(x.encode()).digest() for x in class_texts}

        approved_sentences = []
        disapproved_sentences = []
        unique_approved = []
        print("Splitting data to sentences...")
        seen = set()
        with open(os.path.join(PREP_DATA_DIR, out_file_name), 'w',
                  encoding='utf-8') as outfile:
            file_count = len(os.listdir(FILTERED_SCRAPE_DIR))
            for file_index, filename in enumerate(
                os.listdir(FILTERED_SCRAPE_DIR)):
                if '_filtered.json' in filename:

                    # Create file for each municipality for testing
                    with open(os.path.join(DATA_DIR, 'data_testing',
                                           f'unique_{filename}'), 'w',
                              encoding='utf-8') as muni_outfile:

                        in_file_path = os.path.join(FILTERED_SCRAPE_DIR,
                                                    filename)
                        total_lines = count_num_lines(file_path=in_file_path)
                        print(f'total_lines: {total_lines}')
                        with open(in_file_path, 'r', encoding='utf-8') as file:
                            for line in tqdm(
                                file,
                                total=total_lines,
                                desc=f'{filename}: {file_index + 1} '
                                     f'of {file_count}', unit="line"):
                                data_dict = json.loads(line)

                                # Do initial sentence split - returns list of
                                # all sentences
                                sentences, disapproved_sentences = \
                                    split_sentences(
                                        data_dict=data_dict,
                                        sentence_splitter=self.sentence_splitter,
                                        disapproved_sentences=disapproved_sentences,
                                        filename=filename)

                                for i, sentence in enumerate(sentences):
                                    # Strip sentence, search for letters and
                                    # filter by word count
                                    dump_data, seen = self.create_dump_data(
                                        data=data_dict,
                                        sentence_counter=i,
                                        sentence=sentence,
                                        seen=seen,
                                        filename=filename,
                                        model=model,
                                        tokenizer=tokenizer,
                                        class_data=class_hash)
                                    if dump_data:
                                        json.dump(dump_data, outfile)
                                        outfile.write('\n')
                                        muni_outfile.write(
                                            f"{data_dict['id']} - {i} - "
                                            f"{sentence}\n")

                                        approved_sentences.append(1)
                                        unique_approved.append(1)
                                    else:
                                        if sentence.strip():
                                            disapproved_sentences.append(
                                                f'{filename.split("_")[0]} - '
                                                f'{data_dict["id"]} - '
                                                f'{i} - {sentence}')

        write_text_lines(out_dir=DATA_DIR,
                         filename='data_testing/disapproved_sentences',
                         data=disapproved_sentences)
        # Print running time of code
        timer.how_long_since_start()
        print(f'Approved sentences: {np.sum(approved_sentences)}')
        print(f'Disapproved sentences: {len(disapproved_sentences)}')
        print(f'Total unique sentences: {np.sum(unique_approved)}')
        print()

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
        with open(os.path.join(PREP_DATA_DIR, in_file), 'r',
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

    def create_dump_data(self, data: dict, sentence_counter: int, sentence: str,
                         seen: set, filename: str, model, tokenizer,
                         class_data: set) -> Tuple[Optional[dict], set]:
        """
        Return dump data if data passes quality check and is not in classified
        data used for Sequence Classification.

        :param dict data: dictionary of input data
        :param int sentence_counter: Sentence number of input url
        :param str sentence: sentence
        :param set seen: whether the sentence has been seen before - we only need
            unique sentences
        :param str filename: Input filename
        :param AutoModelWithLMHead or None model: GPT model needed for perplexity
        :param AutoTokenizer or None tokenizer: Tokenizer needed for perplexity
        :param set class_data: Classified data used for Sequence Classification
        :return: If sentence pass quality check, return Tuple with data for
            modelling and updated set of seen observations, else (None, seen)
        :rtype: Tuple[Optional[dict], set]
        """

        final_sentence = sentence.strip()
        if self.args.lower_case:
            final_sentence = final_sentence.lower()

        if find_letters_and_word_count(
            text=final_sentence,
            word_count_threshold=self.args.min_len):

            line_hash = hashlib.md5(final_sentence.encode()).digest()
            # add to unique sentences if not already exists
            if line_hash in seen:
                return None, seen

            if line_hash in class_data:
                seen.add(line_hash)
                return None, seen

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
            return dump_data, seen

        return None, seen

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

    @staticmethod
    def is_correct_danish(data_dict: dict, confidence_threshold: int):
        """
        Detect correct danish text without raw html code

        :param dict data_dict: raw text from scraped url
        :param int confidence_threshold: confidence in which we filter danish
            prediction
        :return: Whether data is determined as danish
        :rtype: bool
        """

        if "__label__da" not in data_dict['detected_page_lang']:
            return False
        confidence = float(
            re.findall(r'\d+\.\d+', data_dict['detected_page_lang'])[0])
        discard_conditions = [confidence < confidence_threshold,
                              '<div ' in data_dict['page_filtered_text'],
                              'endif' in data_dict['page_filtered_text'],
                              '<class' in data_dict['page_filtered_text']]

        for condition in discard_conditions:
            if condition:
                return False
        return True

    @staticmethod
    def fix_utf8_encodings(data_dict: dict) -> dict:
        """
        Use fix_encoding and manually replace wrong danish letters.

        :param dict data: text and metadata from scraped url
        :return: data updated with correct encoded text
        :rtype: dict
        """
        data_dict['page_filtered_text'] = fix_encoding(
            data_dict['page_filtered_text'])
        wrong_encodings = [['Ã¥', 'å'], ['Ã¸', 'ø'], ['Ã¦', 'æ']]
        for wrong_encoding in wrong_encodings:
            if wrong_encoding[0] in data_dict['page_filtered_text']:
                data_dict['page_filtered_text'] = data_dict[
                    'page_filtered_text'].replace(
                    wrong_encoding[0],
                    wrong_encoding[1])
        return data_dict

    @staticmethod
    def filter_ppl_scores(ppl_threshold: int = 10000,
                          in_file_name: str = DEFAULT_UNIQUE_SENTENCES_FILE):
        """
        Filtering ppl scores from unique sentences - creates two files
        approved/disapproved

        :param int, Optional ppl_threshold: perplexity threshold - approve
            all below
        :param str, Optional in_file_name: unique sentences file
        """

        in_file_path = os.path.join(PREP_DATA_DIR, in_file_name)
        total_lines = count_num_lines(file_path=in_file_path)
        print("Filtering perplexity scores...")
        with open(in_file_path, 'r', encoding='utf-8') as infile, \
            open(os.path.join(PREP_DATA_DIR, 'approved_sentences_ppl.json'),
                 'w', encoding='utf-8') as approved_sentences, \
            open(os.path.join(PREP_DATA_DIR,
                              'disapproved_sentences_ppl.json'),
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


if __name__ == '__main__':
    prep_parser = DataPrepArgParser()
    prep_args = prep_parser.parser.parse_args()
    data_preprocessor = RawScrapePreprocessing(args=prep_args)
    data_preprocessor.run()
    # data_preprocessor.extract_danish_and_save_from_raw()
    # data_preprocessor.create_unique_sentences()
    # data_preprocessor.split_train_val()
