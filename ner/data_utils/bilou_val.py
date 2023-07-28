from itertools import groupby

from ner.data_utils.data_prep_input_args import DataPrepArgParser
from ner.local_constants import DATA_DIR, PREP_DATA_DIR
from shared.utils.helpers import read_json_lines, get_sublist_length

prep_parser = DataPrepArgParser()
prep_args = prep_parser.parser.parse_args()

data_bilou = read_json_lines(DATA_DIR, prep_args.bilou_input_file)
ERROR_COUNT = 0
ANNOTATION_ERRORS = 0
ANNOTATION_COUNTER = 0
j = 0

BILOU_ERROR_COUNT = 0
total_entities = []
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
        total_entities.extend(obs["entities"])

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


total_entities.sort()
grouped = [list(group) for key, group in groupby(total_entities)]

sorted_list = sorted(grouped, key=get_sublist_length, reverse=True)

for group in sorted_list:
    print(f"Entitet: {group[0]} - antal: {len(group)}")

print(f"Total sentences: {len(data_bilou)}")
print(f"Total entities: {len(total_entities)}")
