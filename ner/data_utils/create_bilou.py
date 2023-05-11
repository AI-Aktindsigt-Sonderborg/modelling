import sys

from ner.local_constants import DATA_DIR
from shared.utils.helpers import read_json_lines

args = sys.argv[1:]
args.append("non_sensitive_output")
args.append("non_sensitive_bilou_output_2")

data = read_json_lines(input_dir=DATA_DIR, filename=args[0])
bilou = read_json_lines(input_dir=DATA_DIR, filename=args[1])

entity_data = []
for i, obs in enumerate(data):
    page_num
    k = 0
    for page_num in list(obs['text_annotation']):
        for j, annotation in enumerate(obs['text_annotation'][page_num]):
            index = obs['pdf_text'][k][annotation['annotation']['start']:annotation['annotation']['end']]
            content = annotation['annotation']['content']
            if not index == content:
                print()
            else:
                list_content = content.split(" ")


                if len(list_content) == 1:
                    anon_content = annotation['']
                    print("prut")
                elif len(list_content) == 2:
                    print("prut")
                elif len(list_content) > 2:
                    print("prut")
                else:
                    print("prut")



print()