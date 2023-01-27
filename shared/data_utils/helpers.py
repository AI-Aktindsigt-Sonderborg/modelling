import json
import os
from typing import List

import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoModelWithLMHead


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


def load_model_for_ppl(model_id: str = 'pere/norwegian-gpt2',
                       device: str = 'cuda'):
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
