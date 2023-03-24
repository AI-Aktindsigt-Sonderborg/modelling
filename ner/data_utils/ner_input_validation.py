from shared.utils.helpers import read_jsonlines
from ner.local_constants import DATA_DIR

filename = "non_sensitive_bilou_output_2"
filename = 'non_sensitive_output'

data = read_jsonlines(DATA_DIR, filename)

print()
