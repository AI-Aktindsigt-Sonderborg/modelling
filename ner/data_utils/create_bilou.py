import re
import sys
import traceback

import nltk

from ner.data_utils.helpers import split_sentences_bilou
from ner.local_constants import DATA_DIR
from shared.utils.helpers import read_json_lines

args = sys.argv[1:]
args.append("origin")
args.append("bilou")

data = read_json_lines(input_dir=DATA_DIR, filename=args[0])
bilou = read_json_lines(input_dir=DATA_DIR, filename=args[1])

sentence_splitter = nltk.data.load('tokenizers/punkt/danish.pickle')



entity_data = []
for i, obs in enumerate(data):
    for k, pdf_text in enumerate(obs['pdf_text']):
        index_diff = 0
        pdf_altered = pdf_text
        page_num = f'page_no:{k+1}'
        if len(obs['text_annotation']) >= k+1:
            current_page_annotations = obs['text_annotation'][page_num]

            for j, annotation in enumerate(current_page_annotations):
                index = pdf_altered[annotation['annotation']['start'] + index_diff:annotation['annotation']['end'] + index_diff]
                content = annotation['annotation']['content']
                entity = annotation['annotation']['annotation']
                if not index == content:
                    print()
                else:

                    list_content = content.split()
                    if len(list_content) == 1:
                        annotation_to_insert = 'U-' + entity

                    elif len(list_content) == 2:
                        annotation_to_insert = 'B-' + entity + ' L-' + entity
                    elif len(list_content) > 2:
                        annotation_to_insert = 'B-' + entity
                        for inside_element in list_content[1:-1]:
                            annotation_to_insert = annotation_to_insert + ' I-' + entity
                        annotation_to_insert = annotation_to_insert + ' L-' + entity
                    else:
                        annotation_to_insert = "prut"
                        print("prut")

                    pdf_altered = pdf_altered[:annotation['annotation']['start'] + index_diff] + \
                                  annotation_to_insert + pdf_altered[annotation['annotation']['end']+ index_diff:]

                    new_sentences, discarded = split_sentences_bilou(data=pdf_text, sentence_splitter=sentence_splitter)
                    new_sentences_anon, _ = split_sentences_bilou(data=pdf_altered, sentence_splitter=sentence_splitter)
                    try:
                        assert len(new_sentences_anon) == len(new_sentences)
                        for s, (sentence, sentence_anon) in enumerate(
                            zip(new_sentences, new_sentences_anon)):

                            words = re.split(r'( |,|\.)', sentence)
                            tags = re.split(r'( |,|\.)', sentence_anon)

                            to_remove = [" ", ""]
                            words_no_whitespace = list(filter(lambda word: word not in to_remove, words))
                            tags_no_whitespace = list(filter(lambda tag: tag not in to_remove, tags))
                            tags_final = [tag if tag.startswith(("B-", "U-", "I-", "L-")) else "O" for tag in tags_no_whitespace]
                            # words = sentence.split().split(',').split('.')
                            # tags = sentence_anon.split()
                            assert len(tags_final) == len(words_no_whitespace)
                            print()
                    except Exception as e:
                        print(f"data med linjenummer {i + 1} med id {obs['id']} fejlede paa annotation nummer {j}.")
                        print(traceback.format_exc())
                    # print()

                    index_diff = len(pdf_altered) - len(pdf_text)



        entity_data.append(pdf_altered)



print()