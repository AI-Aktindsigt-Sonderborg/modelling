import sys
from itertools import groupby
from shared.utils.helpers import read_json_lines
from ner.local_constants import DATA_DIR
import traceback
from itertools import groupby

args = sys.argv[1:]


filename_bilou = args[0]


data_bilou = read_json_lines(DATA_DIR, filename_bilou)
error_count = 0
annotation_errors = 0
annotation_counter = 0
j = 0

bilou_error_count = 0
total_entities = []
for i, obs in enumerate(data_bilou):
    # for tag in obs['tags']:
        # if 'FORBRYDELSE' in tag:
            # print("-----------------------")
            # print(obs['words'])
            # print(obs['tags'])
            # print("-----------------------")
            # continue

    if len(obs['words']) != len(obs['tags']):
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

