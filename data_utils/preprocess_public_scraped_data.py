import json
import random
import re
# from random import random
from typing import List

class TokenizedSentencesDataset:
    def __init__(self, sentences, tokenizer, max_length, cache_tokenization=False):
        self.tokenizer = tokenizer
        self.sentences = sentences
        self.max_length = max_length
        self.cache_tokenization = cache_tokenization

    def __getitem__(self, item):
        if not self.cache_tokenization:
            return self.tokenizer(self.sentences[item], add_special_tokens=True,
                                  truncation=True, max_length=self.max_length,
                                  return_special_tokens_mask=True)

        if isinstance(self.sentences[item], str):
            self.sentences[item] = self.tokenizer(self.sentences[item], add_special_tokens=True,
                                                  truncation=True, max_length=self.max_length,
                                                  return_special_tokens_mask=True)
        return self.sentences[item]

    def __len__(self):
        return len(self.sentences)


def preprocess_public_sborg(data_path: str = '../data/kommunal_data_test.json',
                            save_data: bool = True, languages: List[str] = ['da_DK']):

    data = []
    with open(data_path, 'rb') as file:
        for index, line in enumerate(file):

            data_dict = json.loads(line)
            data_dict['id'] = index
            title = data_dict['title'].split(' | ')[0]
            data_dict['new_title'] = title

            body = re.split(r'Opdateret \d\d.\d\d.\d\d\d\d', data_dict['text'])
            if len(body) == 2:
                data_dict['body'] = body[1]
            else:
                data_dict['body'] = ''
            data.append(data_dict)

    for lang in languages:
        if lang == 'da_DK':
            # english_locale = [x for x in data if (x['og_locale'] ==
            # 'en_GB' or '//en.' in x['url'] or '/en/' in x['url'])]
            # german_locale = [x for x in data if x['og_locale'] ==
            # 'de_DE' or '//de.' in x['url'] or '/de/' in x['url']]
            # danish_locale = [x for x in data if x['og_locale'] == 'da_DK']
            # empty_locale = [x for x in data if (len(x['og_locale']) == 0 and not
            # ( '//en.' in x['url'] or '/en/' in x['url'] or
            #  '//de.' in x['url'] or '/de/' in x['url']))]

            all_danish = [x for x in data if x['og_locale'] == lang or (len(x['og_locale']) == 0 and not
            ( '//en.' in x['url'] or '/en/' in x['url'] or
             '//de.' in x['url'] or '/de/' in x['url']))]

            subset = [x for x in all_danish if len(x['body']) > 0]
            if save_data:

                # return subset

                with open(f'../data/{lang}_subset.json', "w", encoding='utf-8') as outfile:
                    outfile.write(json.dumps(subset))


def split_to_sentences(data_path: str = "../data/da_DK_subset.json"):

    with open(data_path, "rb") as file:
        data = json.load(file)

    all_sentences = []
    for url in data:
        sentences = re.split('(\. [A-Z])|\n', url['body'])

        for i, sentence in enumerate(sentences):

            if sentence and not sentence.startswith('.'):
                new_sentence = sentence
                if i > 0 and sentences[i - 1]:
                    new_sentence = sentences[i - 1].split(' ')[1] + new_sentence
                    # new_sentences.append(sentence)
                if len(new_sentence) > 10:
                    all_sentences.append(new_sentence.strip())

    return all_sentences

def split_train_dev(sentences: List[str]):
    random.shuffle(sentences)
    train_idx = int(len(sentences)*0.90)
    train = sentences[:train_idx]
    dev = sentences[train_idx:]

    return train, dev

# preprocess_public_sborg()

# sentences = split_to_sentences()
#
# train, dev = split_train_dev(sentences=sentences)
