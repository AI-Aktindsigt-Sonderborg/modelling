from itertools import groupby

from ner.data_utils.data_prep_input_args import DataPrepArgParser
from ner.local_constants import DATA_DIR, PREP_DATA_DIR
from shared.utils.helpers import read_json_lines, get_sublist_length
from collections import Counter
prep_parser = DataPrepArgParser()
prep_args = prep_parser.parser.parse_args()

data_dir = DATA_DIR
if prep_args.custom_data_dir:
    if prep_args.custom_data_dir == "prep":
        data_dir = PREP_DATA_DIR
    if prep_args.custom_data_dir == "data":
        data_dir = DATA_DIR

data_bilou = read_json_lines(data_dir, prep_args.bilou_input_file)
ERROR_COUNT = 0
ANNOTATION_ERRORS = 0
ANNOTATION_COUNTER = 0
j = 0

BILOU_ERROR_COUNT = 0
total_entities = []
total_labels = []
first_entity = []
for i, obs in enumerate(data_bilou):
    if prep_args.print_entity and prep_args.print_entity in obs["entities"]:
        print("----- Tokens -------")
        print(obs["tokens"])
        print("----- Tags ---------")
        print(obs["tags"])

    if len(obs["tokens"]) != len(obs["tags"]):
        BILOU_ERROR_COUNT = BILOU_ERROR_COUNT + 1
        print(
            f"len(Words) != len(tags): Bilou data med linjenummer {i + 1}" f" fejlede"
        )
        print(obs["sentence"])
        if obs["faulty_annotation"]:
            BILOU_ERROR_COUNT = BILOU_ERROR_COUNT + 1
            print(
                f"faulty_annotation fejl: Bilou data med linjenummer {i + 1}"
                f" fejlede"
            )

    else:
        first_entity.append(obs["entities"][0])
        total_entities.extend(obs["entities"])
        total_labels.extend(obs["tags"])

    for tag in obs["tags"]:
        if tag.count("-") > 1:
            print(obs["tags"])
        if "[" in tag:
            print(obs["tags"])

    for token in obs["tokens"]:
        crap = ["(", ")", "[", "]"]
        for c in crap:
            if c in token and len(token) > 1:
                print("---tokens---")
                print(obs["tokens"])
                print("---tags---")
                print(obs["tags"])
                print("---sentence---")
                print(obs["sentence"])
                print("---sentence_anon---")
                print(obs["sentence_anon"])



    # print(obs["tags"])
print(first_entity)
total_entities.sort()
total_labels.sort()
# first_entity.sort()
first_entity_count = Counter(first_entity)
# print(first_entity_count)

# grouped_first_entity = [group for group in groupby(first_entity)]
# grouped_first_entity = groupby(sorted(first_entity_count.items(), key=lambda x: x[1], reverse=True), key=lambda x: x[1])
grouped_entities = [list(group) for key, group in groupby(total_entities)]
grouped_labels = [list(group) for key, group in groupby(total_labels)]


# sorted_first_entity = sorted(grouped_first_entity, key=get_sublist_length, reverse=True)
sorted_entities = sorted(grouped_entities, key=get_sublist_length, reverse=True)
sorted_labels = sorted(grouped_labels, key=get_sublist_length, reverse=True)

for group in sorted_entities:
    print(f"Entitet: {group[0]} - count: {len(group)}")
print()
for group in sorted_labels:
    print(f"Label: {group[0]} - count: {len(group)}")
print()
for count, items in first_entity_count:
    print(f"Entity: {items} - sentence count: {count}")

# for group in grouped_first_entity:
#     print(f"Sentences with label: {group[0]} - antal: {len(group)}")

print()
print(f"Total sentences: {len(data_bilou)}")
print(f"Total entities: {len(total_entities)}")
print(f"Total labels: {len(total_labels)}")

