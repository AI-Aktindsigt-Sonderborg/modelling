import json
import os
from typing import List, Tuple
import re

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelWithLMHead


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


