from shared.utils.helpers import read_jsonlines

input_dir = "ner/data"
filename = "non_sensitive_bilou_output_2"
filename = 'non_sensitive_output'

data = read_jsonlines(input_dir, filename)

print()


