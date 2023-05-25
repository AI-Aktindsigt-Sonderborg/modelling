import sys

from shared.utils.helpers import read_json_lines
from ner.local_constants import DATA_DIR
import traceback

args = sys.argv[1:]


filename_bilou = args[0]


data_bilou = read_json_lines(DATA_DIR, filename_bilou)
error_count = 0
annotation_errors = 0
annotation_counter = 0
j = 0

bilou_error_count = 0
for i, obs in enumerate(data_bilou):
    print()
    if len(obs['words']) != len(obs['tags']):
        bilou_error_count = bilou_error_count + 1
        print(f'len(Words) != len(tags): Bilou data med linjenummer {i + 1} fejlede')
    if obs['faulty_annotation']:
        bilou_error_count = bilou_error_count + 1
        print(f'faulty_annotation fejl: Bilou data med linjenummer {i + 1} fejlede')

print(f"Total sentences: {len(data_bilou)}")

