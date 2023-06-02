import sys

from ner.local_constants import DATA_DIR
from shared.utils.helpers import read_json_lines




args: list = sys.argv[1:]

bilou_1 = read_json_lines(DATA_DIR, args[0])
bilou_2 = read_json_lines(DATA_DIR, args[1])

bilou_1_tokens = []
bilou_1_tags = []
for i, obs in enumerate(bilou_1):
    bilou_1_tokens.append(obs['tokens'])
    bilou_1_tags.append(obs['tags'])

bilou_2_tokens = []
bilou_2_tags = []
for i, obs in enumerate(bilou_2):
    bilou_2_tokens.append(obs['tokens'])
    bilou_2_tags.append(obs['tags'])

bilou_2_only = list(set(bilou_2_tokens) - set(bilou_1_tokens))
bilou_1_only = list(set(bilou_1_tokens) - set(bilou_2_tokens))

if args[2] == 2:
    for element in bilou_2_only:
        print(element)

if args[2] == 1:
    for element in bilou_1_only:
        print(element)
