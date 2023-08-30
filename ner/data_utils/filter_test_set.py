import argparse
import random
from typing import List
import os
from sklearn.model_selection import train_test_split

from ner.data_utils.create_bilou import create_bilou_from_one_document
from ner.data_utils.data_prep_input_args import DataPrepArgParser
from ner.local_constants import DATA_DIR, PREP_DATA_DIR
from ner.modelling_utils.helpers import get_label_list
from shared.utils.helpers import read_json_lines, write_json_lines, read_json


prep_parser = DataPrepArgParser()
prep_args = prep_parser.parser.parse_args()


# test_data = read_json_lines(input_dir=PREP_DATA_DIR, filename="test_corrected2")

test_data = read_json(filepath=os.path.join(PREP_DATA_DIR, "__test.jsonl"))

print(test_data[2])
