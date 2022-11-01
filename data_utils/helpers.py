import itertools
import json
import math
import os
from typing import List, Tuple
import re
from torch.utils.data import Dataset
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelWithLMHead
import pandas as pd
from sklearn.model_selection import train_test_split
from local_constants import PREP_DATA_DIR
from utils.helpers import save_json, read_jsonlines


class DatasetWrapper(Dataset):
    """
    Class to wrap dataset such that we can iterate through
    """

    def __init__(self, dataset):
        self.dataset = dataset

    def __getitem__(self, item):
        item = int(item)
        return self.dataset[item]

    def __len__(self):
        return len(self.dataset)

def write_json_lines(out_dir: str, filename: str, data: List[dict]):
    """
    Write json_lines_file based on list of dictionaries
    :param out_dir: directory
    :param filename: filename to write
    :param data: input data
    """
    with open(os.path.join(out_dir, filename + '.json'), 'w',
              encoding='utf-8') as outfile:
        for entry in data:
            json.dump(entry, outfile)
            outfile.write('\n')

def write_text_lines(out_dir: str, filename: str, data: List[str]):
    """
    Write text file based on list of strings
    :param out_dir: directory
    :param filename: filename to write
    :param data: input data
    """
    with open(os.path.join(out_dir, f'{filename}.txt'), 'w',
              encoding='utf-8') as outfile:
        for entry in data:
            outfile.write(f"{entry}\n")

def split_sentences(data_dict: dict, filename: str, sentence_splitter: object,
                   disapproved_sentences: List[str]) -> Tuple[List[str], List[str]]:
    """
    Takes raw html input text and splits to sentences
    :param data_dict: Dictionary containing raw input text from filtered scrape file
    :param sentence_splitter: nltk.tokenize.punkt.PunktSentenceTokenizer
    :param disapproved_sentences: list of disapproved sentences
    :param filename: filename
    :return: list of sentences, list of disapproved sentences
    """

    # Read canonical string representation of the object as we need special regex characters
    prep_text = repr(data_dict['text'])

    # Define list of special chars to replace
    special_chars = [['\\r', '\\t', '\\n', '\\xa0', ' | ', '|', '*'], ['”', '"', "'"],
                     ['NemID', 'MitID', 'minSU', 'LinkedIn']]
    for case in special_chars[0]:
        prep_text = prep_text.replace(case, ' ')

    # Define second list of special chars
    for case in special_chars[1]:
        prep_text = prep_text.replace(case, '')

    for case in special_chars[2]:
        if case in prep_text:
            prep_text = prep_text.replace(case, case.lower())

    # remove consecutive whitespaces
    prep_text = re.sub(' +', ' ', prep_text)

    # Find wrong breaks like "." or "?" followed by capital letters
    find_wrong_break_ids = [
        (m.start(0), m.end(0)) for m in
        re.finditer(r'\?[A-Z]|\.[A-Z]', prep_text)]

    # Add whitespace after each wrong break
    if len(find_wrong_break_ids) > 0:
        increment = 1
        for idx in find_wrong_break_ids:
            prep_text = prep_text[:idx[0] + increment] + '\n' + prep_text[idx[0] + increment:]
            increment += 1

    # split text with sentence_splitter and join by newlines
    text_splitted = (
        '\n'.join(sentence_splitter.tokenize(prep_text)))

    # split text again by newlines
    sentences = re.split('\n', text_splitted)

    new_sentences = []
    for j, sentence in enumerate(sentences):
        # Discard sentences where lower case letters are followed by capital letters
        if len([(m.start(0), m.end(0)) for m in re.finditer("[a-z][A-Z]", sentence)]) > 0:
            disapproved_sentences.append(
                f'{filename.split("_")[0]} - {data_dict["id"]} - {j} - {sentence}')
        else:
            new_sentences.append(sentence)

    return new_sentences, disapproved_sentences


def score_gpt2(text: str, model, tokenizer, device: str = 'cuda'):
    """
    Computes perplexity score on the first 200 words in a sentence
    :param text: input string
    :param model: model
    :param tokenizer: tokenizer
    :param device: should normally be cuda
    :return: returns loss of sentence - the lower the better
    """
    words = text.split(" ")
    new_text = " ".join(words[0:200])
    tensor_input = tokenizer.encode(new_text, return_tensors='pt').to(device)

    if len(tensor_input[0]) > 1000:
        return 50000.0

    if max(tensor_input[0]) > 50257:
        return 51000.0

    with torch.no_grad():
        result = model(tensor_input, labels=tensor_input).loss

    return np.exp(result.cpu().numpy())

def load_model_for_ppl(model_id: str = 'pere/norwegian-gpt2', device: str = 'cuda'):
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    model = AutoModelWithLMHead.from_pretrained(model_id)
    model = model.to(device)
    model.eval()
    return model, tokenizer


def find_letters_and_word_count(text: str, word_count_threshold: int):
    search = any(c.isalpha() for c in text)
    word_count = len(text.split(' '))
    if search and word_count >= word_count_threshold:
        return True
    return False

def read_xls_save_json(file_dir: str = 'data/preprocessed_data',
                       in_file_name: str = 'skrab_01.xlsx',
                       ppl_filters: List[int] = None, drop_na: bool = True,
                       out_file_name: str = 'classified_scrape1'):

    data = pd.read_excel(os.path.join(file_dir, in_file_name), sheet_name='Klassificering', header=1)
    data['index'] = range(len(data))
    data['text_len'] = len(data['text'])
    if drop_na:
        data = data.dropna(subset=['klassifikation'])
    if ppl_filters:
        data = data[(data['ppl_score'] > ppl_filters[0]) & (data['ppl_score'] < ppl_filters[1])]

    data_dicts = data.to_dict('records')
    save_json(output_dir='data/preprocessed_data', data=data_dicts, filename=out_file_name)


def group_data_by_class(data: List[dict]):

    label_set = set(map(lambda x: x['klassifikation'], data))

    new_data = [{'label': x['klassifikation'], 'text': x['text']} for x in data]

    grouped = [[x for x in new_data if x['label'] == y] for y in label_set]
    return grouped

def train_test_to_json_split(data):

    train_test = [train_test_split(x, test_size=10, random_state=1) for x in data]

    train = list(itertools.chain.from_iterable([x[0] for x in train_test]))
    test = list(itertools.chain.from_iterable([x[1] for x in train_test]))

    save_json(PREP_DATA_DIR, data=train, filename='train_classified')
    save_json(PREP_DATA_DIR, data=test, filename='test_classified')


if __name__ == "__main__":

    # read_xls_save_json()
    data = read_jsonlines(input_dir=PREP_DATA_DIR, filename='classified_scrape1')
    data = [x for x in data if not (isinstance(x['text'], float) and math.isnan(x['text'])) and not x['text_len'] > 3000]

    grouped = group_data_by_class(data=data)


    train_test_to_json_split(data=grouped)


    print()

