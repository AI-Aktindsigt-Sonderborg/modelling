import sys

from ner.local_constants import DATA_DIR
from shared.utils.helpers import read_json_lines

args: list = sys.argv[1:]

bilou = read_json_lines(input_dir=DATA_DIR, filename=args[0])

bilou_filtered = [x for x in bilou if x['document_id'] == args[1]]


for i, x in enumerate(bilou_filtered):
    print(f"------------------{i}-------------------")
    print(x['words'])
    print(x['tags'])
    print(f"-------------------------------------")

