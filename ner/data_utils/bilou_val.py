from itertools import groupby

from ner.data_utils.data_prep_input_args import DataPrepArgParser
from ner.local_constants import PREP_DATA_DIR
from shared.utils.helpers import read_json_lines

prep_parser = DataPrepArgParser()
prep_args = prep_parser.parser.parse_args()

data_bilou = read_json_lines(PREP_DATA_DIR, prep_args.bilou_input_file)
error_count = 0
annotation_errors = 0
annotation_counter = 0
j = 0

bilou_error_count = 0
total_entities = []
for i, obs in enumerate(data_bilou):
    if prep_args.print_entity and prep_args.print_entity in obs['entities']:
        print("----- Tokens -------")
        print(obs['tokens'])
        print("----- Tags ---------")
        print(obs['tags'])

    if len(obs['tokens']) != len(obs['tags']):
        bilou_error_count = bilou_error_count + 1
        print(f'len(Words) != len(tags): Bilou data med linjenummer {i + 1} fejlede')
        print(obs['sentence'])
        if obs['faulty_annotation']:
            bilou_error_count = bilou_error_count + 1
            print(f'faulty_annotation fejl: Bilou data med linjenummer {i + 1} fejlede')
    else:
        total_entities.extend(obs['entities'])

total_entities.sort()
grouped = [list(group) for key, group in groupby(total_entities)]

sorted_list = sorted(grouped, key=lambda x: len(x), reverse=True)

for group in sorted_list:
    print(f"Entitet: {group[0]} - antal: {len(group)}")

print(f"Total sentences: {len(data_bilou)}")
print(f"Total entities: {len(total_entities)}")

