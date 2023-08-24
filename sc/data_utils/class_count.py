from collections import Counter

import pandas as pd

from sc.data_utils.data_prep_input_args import DataPrepArgParser
from sc.local_constants import DATA_DIR, CONF_DATA_DIR, PREP_DATA_DIR
from shared.utils.helpers import read_json_lines

prep_parser = DataPrepArgParser()
prep_args = prep_parser.parser.parse_args()

data_dir = DATA_DIR
if prep_args.custom_data_dir:
    if prep_args.custom_data_dir == "conf":
        data_dir = CONF_DATA_DIR
    if prep_args.custom_data_dir == "prep":
        data_dir = PREP_DATA_DIR
    if prep_args.custom_data_dir == "data":
        data_dir = DATA_DIR

data = read_json_lines(data_dir, prep_args.train_outfile)
ERROR_COUNT = 0
ANNOTATION_ERRORS = 0
ANNOTATION_COUNTER = 0
j = 0

total_labels = [obs['label'] for obs in data]
label_count = Counter(total_labels)

data_types = [x['test'] for x in data]
data_type_count = Counter(data_types)

print()
for items, count in label_count.items():
    print(f"Entity: {items} - sentence count: {count}")

print()
for items, count in data_type_count.items():
    print(f"Entity: {items} - sentence count: {count}")

df = pd.DataFrame(data)
print()
grouped = df.groupby(['label', 'test'])
for (label, test), group_data in grouped:
    print(f"Label: {label}, Test: {test}")
    print(group_data)
    print("--------------------")

print()
print(f"Total sentences: {len(data)}")
print(f"Total labels: {len(total_labels)}")
