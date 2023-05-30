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

from ner.data_utils.create_bilou import create_bilou_from_one_document
from ner.data_utils.data_prep_input_args import DataPrepArgParser

from ner.local_constants import DATA_DIR, PREP_DATA_DIR
from sc.local_constants import CLASS_DATA_DIR
from shared.data_utils.helpers import score_gpt2, load_model_for_ppl, \
    find_letters_and_word_count
from shared.utils.helpers import TimeCode, read_json_lines, write_json_lines, \
    write_text_lines
from shared.utils.helpers import count_num_lines

from ner.modelling_utils.helpers import get_label_list


class NERDataPreprocessing:
    """
    Class to prepare data for NER training.
    :param argparse.Namespace args:
        input arguments from :class: `.DataPrepArgParser`.
    """

    def __init__(self, args: argparse.Namespace):
        self.args = args

    def create_bilou(self):
        raw_data = read_json_lines(input_dir=DATA_DIR,
                                   filename=self.args.origin_input_file)

        word_tag_mismatch_errors: int = 0
        wrong_index_errors: int = 0
        wrong_raw_index_errors: int = 0
        correct_indexes: int = 0
        entity_data: List[dict] = []
        total_sentences: int = 0
        deleted_annotations: int = 0
        total_indices_reindexed: int = 0
        total_not_danish_counter: int = 0

        for i, obs in enumerate(raw_data):
            print(f'creating bilou from document number {i + 1}')
            single_obs_data, errors = create_bilou_from_one_document(
                input_data=obs,
                data_number=i)
            word_tag_mismatch_errors += errors[0]
            wrong_index_errors += errors[1]
            correct_indexes += errors[2]
            total_sentences += errors[3]
            deleted_annotations += errors[4]
            wrong_raw_index_errors += errors[5]
            total_indices_reindexed += errors[6]
            total_not_danish_counter += errors[7]
            entity_data.extend(single_obs_data)

        write_json_lines(out_dir=DATA_DIR, data=entity_data,
                         filename=self.args.bilou_input_file)

        print(f'total valid sentences: {len(entity_data)}')
        print(f'word/tag length mismatch errors: {word_tag_mismatch_errors}')
        print(f'wrong index errors: {wrong_index_errors}')
        print(f'wrong raw index errors: {wrong_raw_index_errors}')
        print(f'Total indices reindexed: {total_indices_reindexed}')
        print(f'deleted annotations: {deleted_annotations}')
        print(f'correct indexes: {correct_indexes}')
        print(f'sentences not danish: {total_not_danish_counter}')

        print(f'total sentences: {total_sentences}')

    def filter_entities(self):
        bilou = read_json_lines(input_dir=DATA_DIR,
                                filename=self.args.bilou_input_file)
        labels, id2label, label2id = get_label_list(self.args.entities)

        out_suffix = ''.join([x[0] for x in self.args.entities])

        for i, obs in enumerate(bilou):
            obs['tags'] = [tag if (
                    (tag[2:] in self.args.entities) or (tag == "O")) else "O"
                            for tag in obs['tags']]
            print()

        print()


if __name__ == "__main__":
    prep_parser = DataPrepArgParser()
    prep_args = prep_parser.parser.parse_args()
    prep_args.bilou_input_file = 'bilou_entities_kasper_all'

    data_prep = NERDataPreprocessing(prep_args)
    if prep_args.create_bilou:
        data_prep.create_bilou()

    data_prep.filter_entities()
