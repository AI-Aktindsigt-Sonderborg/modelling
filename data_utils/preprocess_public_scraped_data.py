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
                            # print(data_dict['detected_page_lang'])
                            # print(data_dict['page_filtered_text'])
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

        if input_type == 'from_dir':
            for filename in os.listdir(self.filtered_dir):
                with open(os.path.join(self.filtered_dir, filename), 'rb') as file:
                    for index, line in enumerate(file):
                        data_dict = json.loads(line)
                        sentences = re.split(r'(\. [A-Z])|\n', data_dict['text'])
                        for i, sentence in enumerate(sentences):
                            if sentence and not sentence.startswith('.'):
                                new_sentence = sentence
                                if i > 0 and sentences[i - 1]:
                                    new_sentence = sentences[i - 1].split(' ')[1] + new_sentence
                                if len(new_sentence) > 10:
                                    all_sentences.append(new_sentence.strip())


        unique_sentences = list(set(all_sentences))

        with open(os.path.join('../data', 'concat_scrape2_unique_sentences.json'), 'w',
                  encoding='utf-8') as outfile:
            for entry in unique_sentences:
                json.dump({'text': entry}, outfile)
                outfile.write('\n')

        return unique_sentences





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
    sentences = data_preprocessor.split_to_sentences()


    print()
