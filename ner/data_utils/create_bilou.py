import sys

from ner.local_constants import DATA_DIR
from shared.utils.helpers import read_json_lines

args = sys.argv[1:]
args.append("origin")
args.append("bilou")

data = read_json_lines(input_dir=DATA_DIR, filename=args[0])
bilou = read_json_lines(input_dir=DATA_DIR, filename=args[1])

entity_data = []
for i, obs in enumerate(data):
    for k, pdf_text in enumerate(obs['pdf_text']):
        index_diff = 0
        pdf_altered = pdf_text
        page_num = f'page_no:{k+1}'
        current_page_annotations = obs['text_annotation'][page_num]

        for j, annotation in enumerate(current_page_annotations):
            index = pdf_altered[annotation['annotation']['start'] + index_diff:annotation['annotation']['end'] + index_diff]
            content = annotation['annotation']['content']
            if not index == content:
                print()
            else:
                pdf_altered = pdf_altered[:annotation['annotation']['start']] + \
                              annotation['annotation']['annotation'] + pdf_altered[annotation['annotation']['end']:]

                index_diff = len(pdf_altered) - len(pdf_text)

                list_content = content.split()
                if len(list_content) == 1:
                    # anon_content = annotation['']
                    print("prut")
                elif len(list_content) == 2:
                    print("prut")
                elif len(list_content) > 2:
                    print("prut")
                else:
                    print("prut")



print()