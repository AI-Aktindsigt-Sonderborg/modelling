import hashlib
import json
import os.path
import random
import re
from typing import List

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
    def __init__(self, data_dir: str = '../data/new_scrape',
                 filtered_dir: str = '../data/filtered_scrape',
                 train_output: str = 'train.json',
                 eval_output: str = 'validation.json',
                 languages: List[str] = ['da_DK'],
                 save_data: bool = True):
        self.data_dir = data_dir
        self.filtered_dir = filtered_dir
        self.train_output = train_output
        self.eval_output = eval_output
        self.languages = languages
        self.save_data = save_data

    def extract_text_and_save_from_raw(self):
        for filename in os.listdir(self.data_dir):
            data = []
            filtered_filename = filename.split('_')[0] + '_filtered'
            false_lang_preds = []
            with open(os.path.join(self.data_dir, filename), 'rb') as file:
                for index, line in enumerate(file):
                    data_dict = json.loads(line)
                    if "__label__da" in data_dict['detected_page_lang']:
                        confidence = float(
                            re.findall('\d+\.\d+', data_dict['detected_page_lang'])[0])
                        if confidence > 0.6:
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

    def split_to_sentences(self, input_type: str = 'from_dir'):
        all_sentences = []
        approved_sentences = []
        dissapproved_sentences = []
        unique_approved = []
        seen = set()
        with open(os.path.join('../data', 'concat_new_scrape_unique_sentences5.json'), 'w',
                  encoding='utf-8') as outfile:
            if input_type == 'from_dir':
                for filename in os.listdir(self.filtered_dir):
                    with open(os.path.join(self.filtered_dir, filename), 'rb') as file:
                        for index, line in enumerate(file):
                            data_dict = json.loads(line)
                            if 'forklarer Ann E' in data_dict['text']:
                                print()
                            sentences = re.split(r'(\. [A-Z])|\n', data_dict['text'])
                            for i, sentence in enumerate(sentences):
                                if sentence and not sentence.startswith('.'):
                                    final_sentence = sentence.strip()
                                    if i > 0 and sentences[i - 1]:
                                        tmp_sentence = sentences[i - 1].split(' ')[1] + ' ' + final_sentence
                                        final_sentence = tmp_sentence.strip()
                                    search = any(c.isalpha() for c in final_sentence)
                                    word_count = len(final_sentence.split(' '))
                                    if word_count >= 5 and search:
                                        approved_sentences.append(1)
                                        line_hash = hashlib.md5(final_sentence.encode()).digest()
                                        if line_hash not in seen:
                                            seen.add(line_hash)
                                            unique_approved.append(1)
                                            json.dump({'text': final_sentence}, outfile)
                                            outfile.write('\n')
                                    else:
                                        dissapproved_sentences.append(1)

        print(f'Approved sentences: {np.sum(approved_sentences)}')
        print(f'Dissapproved sentences: {np.sum(dissapproved_sentences)}')
        print(f'Total unique sentences: {np.sum(unique_approved)}')


def split_train_val(sentences: List[str]):
    random.shuffle(sentences)
    train_idx = int(len(sentences) * 0.90)
    train = sentences[:train_idx]
    val = sentences[train_idx:]

    return train, val


def save_datasets(data_dir: str = '../data', train: List[str] = None, val: List[str] = None,
                  train_subset: int = None):
    if train_subset:
        train_file_name = f'train_{train_subset}.json'
    else:
        train_file_name = 'train.json'
    with open(os.path.join(data_dir, train_file_name), 'w', encoding='utf-8') as outfile:
        for i, entry in enumerate(train):
            json.dump({'text': entry}, outfile)
            outfile.write('\n')
            if train_subset and i == train_subset:
                break

    with open(os.path.join(data_dir, 'validation.json'), 'w', encoding='utf-8') as outfile:
        for entry in val:
            json.dump({'text': entry}, outfile)
            outfile.write('\n')


if __name__ == '__main__':
    # preprocess_public_sborg()
    #
    # sentences = split_to_sentences()
    # #
    # train, val = split_train_val(sentences=sentences)
    # # train = train[:4]
    # #
    # save_datasets(train=train, val=val)

    data_preprocessor = RawScrapePreprocessing()
    # data_preprocessor.extract_text_and_save_from_raw()
    data_preprocessor.split_to_sentences()


    print()
