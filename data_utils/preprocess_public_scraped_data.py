import hashlib
import json
import os.path
import random
import re
from typing import List
import nltk.data
import numpy as np
from ftfy import fix_encoding

# class TokenizedSentencesDataset:
#     def __init__(self, sentences, tokenizer, max_length, cache_tokenization=False):
#         self.tokenizer = tokenizer
#         self.sentences = sentences
#         self.max_length = max_length
#         self.cache_tokenization = cache_tokenization
#
#     def __getitem__(self, item):
#         if not self.cache_tokenization:
#             return self.tokenizer(self.sentences[item], add_special_tokens=True,
#                                   truncation=True, max_length=self.max_length,
#                                   return_special_tokens_mask=True)
#
#         if isinstance(self.sentences[item], str):
#             self.sentences[item] = self.tokenizer(self.sentences[item], add_special_tokens=True,
#                                                   truncation=True, max_length=self.max_length,
#                                                   return_special_tokens_mask=True)
#         return self.sentences[item]
#
#     def __len__(self):
#         return len(self.sentences)

class RawScrapePreprocessing:
    """
    Class to preprocess raw data from web scraper
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
    extract_danish_and_save_from_raw(confidence_threshold)
            Takes raw scrape file(s), extracts danish text and save filtered

    """


    def __init__(self, data_dir: str = '../data',
                 train_output: str = 'train.json',
                 val_output: str = 'validation.json',
                 save_data: bool = True):
        self.data_dir = data_dir
        self.filtered_dir = self.data_dir + '/filtered_scrape'
        self.new_scrape_dir = self.data_dir + '/new_scrape'
        self.train_output = train_output
        self.val_output = val_output
        self.save_data = save_data
        self.sentence_splitter = nltk.data.load('tokenizers/punkt/danish.pickle')

    def from_raw_to_train_val(self):
        self.extract_danish_and_save_from_raw()
        self.split_to_sentences(out_file_name='scrape_v2_all_unique_sentences.json')
        self.split_train_val(in_file='scrape_v2_all_unique_sentences.json')

    def extract_danish_and_save_from_raw(self, confidence_threshold: float = 0.6):
        for filename in os.listdir(self.new_scrape_dir):
            data = []
            filtered_filename = filename.split('_')[0] + '_filtered'
            false_lang_preds = []
            with open(os.path.join(self.new_scrape_dir, filename), 'rb') as file:
                for index, line in enumerate(file):
                    data_dict = json.loads(line)
                    if "__label__da" in data_dict['detected_page_lang']:
                        confidence = float(
                            re.findall('\d+\.\d+', data_dict['detected_page_lang'])[0])
                        if confidence > confidence_threshold:
                            if ('Ã¥' or 'Ã¸') in data_dict['page_filtered_text']:
                                data_dict['page_filtered_text'] = fix_encoding(data_dict['page_filtered_text'])
                            data.append({'id': index, 'text': data_dict['page_filtered_text']})
                        else:
                            false_lang_preds.append(1)
            print(f'Number of false preds: {np.sum(false_lang_preds)}')
            with open(os.path.join(self.filtered_dir, filtered_filename + '.json'), 'w', encoding='utf-8') as outfile:
                for entry in data:
                    json.dump(entry, outfile)
                    outfile.write('\n')

    def split_to_sentences(self, out_file_name: str = 'scrape_v2_all_unique_sentences.json'):
        approved_sentences = []
        dissapproved_sentences = []
        unique_approved = []
        seen = set()
        with open(os.path.join('../data', out_file_name), 'w',
                  encoding='utf-8') as outfile:
            for filename in os.listdir(self.filtered_dir):
                with open(os.path.join(self.filtered_dir, filename), 'rb') as file:
                    for line in file:
                        data_dict = json.loads(line)
                        text_splitted = ('\n'.join(self.sentence_splitter.tokenize(data_dict['text'])))
                        sentences = re.split('\n', text_splitted)
                        for i, sentence in enumerate(sentences):
                                final_sentence = sentence.strip().replace('|', '')
                                search = any(c.isalpha() for c in final_sentence)
                                word_count = len(final_sentence.split(' '))
                                if word_count >= 5 and search:
                                    approved_sentences.append(1)
                                    line_hash = hashlib.md5(final_sentence.encode()).digest()
                                    if line_hash not in seen:
                                        seen.add(line_hash)
                                        unique_approved.append(1)
                                        json.dump({'id': data_dict['id'], 'sentence': i, 'text': final_sentence}, outfile)
                                        outfile.write('\n')
                                else:
                                    dissapproved_sentences.append(1)

        print(f'Approved sentences: {np.sum(approved_sentences)}')
        print(f'Dissapproved sentences: {np.sum(dissapproved_sentences)}')
        print(f'Total unique sentences: {np.sum(unique_approved)}')

    def split_train_val(self,
                    in_file: str = 'scrape_v2_all_unique_sentences.json',
                    save_datasets: bool = True,
                    split: float = 0.9,
                    seed: int = 42):
        sentences = []
        with open(os.path.join(self.data_dir, in_file), 'r', encoding='utf-8') as file:
            for line in file:
                data_dict = json.loads(line)
                sentences.append(data_dict['text'])
        random.seed(seed)
        random.shuffle(sentences)
        train_idx = int(len(sentences) * split)
        train = sentences[:train_idx]
        val = sentences[train_idx:]
        if save_datasets:
            self.save_datasets(train=train, val=val)
        return train, val

    def save_datasets(self, train: List[str],
                      val: List[str],
                      train_subset: int = None):
        if train_subset:
            train_file_name = f'train_{train_subset}.json'
        else:
            train_file_name = self.train_output
        with open(os.path.join(self.data_dir, train_file_name), 'w', encoding='utf-8') as outfile:
            for i, entry in enumerate(train):
                json.dump({'text': entry}, outfile)
                outfile.write('\n')
                if train_subset and i == train_subset:
                    break

        with open(os.path.join(self.data_dir, self.val_output), 'w', encoding='utf-8') as outfile:
            for entry in val:
                json.dump({'text': entry}, outfile)
                outfile.write('\n')


if __name__ == '__main__':
    import sys

    sys.path.append("/home/kasper/Code/semantic-modelling/utils")
    from helpers import TimeCode

    code_timer = TimeCode()
    data_preprocessor = RawScrapePreprocessing(train_output='train_new_scrape.json',
                                               val_output='val_new_scrape.json')
    data_preprocessor.from_raw_to_train_val()
    code_timer.how_long_since_start() # about 72 seconds total

    # data_preprocessor.extract_danish_and_save_from_raw()
    # data_preprocessor.split_to_sentences()
    # data_preprocessor.split_train_val()


    print()
