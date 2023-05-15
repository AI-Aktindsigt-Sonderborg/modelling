import re
import sys
import traceback
from typing import List

import nltk

from ner.data_utils.helpers import split_sentences_bilou
from ner.local_constants import DATA_DIR
from shared.utils.helpers import read_json_lines, write_json_lines

args: list = sys.argv[1:]
# args.append("origin")
# args.append("bilou")

raw_data = read_json_lines(input_dir=DATA_DIR, filename=args[0])
bilou = read_json_lines(input_dir=DATA_DIR, filename=args[1])

sentence_splitter = nltk.data.load('tokenizers/punkt/danish.pickle')

word_tag_mismatch_errors: int = 0
wrong_index_errors: int = 0
entity_data: List[dict] = []


def create_bilou_from_one_document(input_data: dict, data_number: int,
                                   print_stats: bool = False,
                                   print_each_sentence: int = 0):
    word_tag_mismatch_error: int = 0
    wrong_index: int = 0
    output_data = []
    for k, pdf_text in enumerate(input_data['pdf_text']):
        index_diff = 0
        pdf_altered = pdf_text
        page_num = f'page_no:{k + 1}'
        if len(input_data['text_annotation']) >= k + 1:
            try:
                current_page_annotations = input_data['text_annotation'][page_num]
            except Exception as ex:
                print(f"wtf error - check line {data_number + 1}")
                #                 print(page_num)
                #                 print(traceback.format_exc())
                current_page_annotations = None

            if current_page_annotations:
                for j, annotation in enumerate(current_page_annotations):
                    content = annotation['annotation']['content']
                    content = content.strip()
                    if content.endswith('.') or content.endswith('. '):
                        if print_stats:
                            print(f'weird content: {content}')
                        if not index == content:
                            print(f'index error with removed dot line {data_number+1}')

                        content = content[:-1]
                        if print_stats:
                            print(f'weird content2: {content}')
                        index = pdf_altered[
                                annotation['annotation']['start'] + index_diff:
                                annotation['annotation'][
                                    'end'] + index_diff - 1]
                    else:
                        print(f'index error {data_number+1}')
                        index = pdf_altered[
                                annotation['annotation']['start'] + index_diff:
                                annotation['annotation']['end'] + index_diff]
                    entity = annotation['annotation']['annotation']
                    if not index == content:
                        wrong_index += 1
                    else:

                        list_content = re.split(r'( |,|\. |\.\n)', content)
                        to_remove = [" ", ""]
                        list_content = list(filter(lambda tag:
                                                   tag.strip() not
                                                   in to_remove, list_content))

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

                        pdf_altered = pdf_altered[:annotation['annotation'][
                                                       'start'] + index_diff] + \
                                      annotation_to_insert + pdf_altered[
                                                             annotation[
                                                                 'annotation'][
                                                                 'end'] + index_diff:]

                        new_sentences, discarded = split_sentences_bilou(
                            data=pdf_text, sentence_splitter=sentence_splitter)
                        new_sentences_anon, _ = split_sentences_bilou(
                            data=pdf_altered,
                            sentence_splitter=sentence_splitter)
                        if print_stats:
                            print(
                                f'len(new_sentences_anon): {len(new_sentences_anon)}, len(new_sentences): {len(new_sentences)}')
                            print(
                                f'len(discarded_anon): {len(_)}, len(discarded): {len(discarded)}')

                        try:
                            assert len(new_sentences_anon) == len(new_sentences)
                            for s, (sentence, sentence_anon) in enumerate(
                                zip(new_sentences, new_sentences_anon)):
                                words = re.split(r'( |,|\. |\.\n)', sentence)
                                tags = re.split(r'( |,|\. |\.\n)',
                                                sentence_anon)
                                if print_stats:
                                    print(
                                        f'len(words): {len(words)}, len(tags): {len(tags)}')

                                to_remove = [" ", ""]
                                words_final = list(filter(lambda word:
                                                          word.strip(
                                                          ).rstrip(
                                                              '\\n').strip()
                                                          not
                                                          in to_remove, words))
                                words_final[-1] = words_final[-1].strip()
                                tags_no_whitespace = list(filter(lambda tag:
                                                                 tag not in
                                                                 to_remove,
                                                                 tags))
                                tags_final = [tag if
                                              tag.startswith(("B-",
                                                              "U-", "I-",
                                                              "L-")) else "O"
                                              for tag in tags_no_whitespace]
                                if print_stats:
                                    print(
                                        f'len(words_final): {len(words_final)}, len(tags_final): {len(tags_final)}')

                                if print_each_sentence == 1:
                                    print("---------------------")
                                    print(words_final)
                                    print()
                                    print(tags_final)
                                    print("---------------------")

                                output_data.append(
                                    {'words': words_final, 'tags':
                                        tags_final})

                                index_diff = len(pdf_altered) - len(pdf_text)
                                assert len(tags_final) == len(words_final)

                        except Exception as e:
                            word_tag_mismatch_error += 1
                            print(
                                f"data med linjenummer {data_number + 1} med id {input_data['id']} fejlede paa annotation nummer {j}.")
                            print(traceback.format_exc())
    return output_data, [word_tag_mismatch_error, wrong_index]


if len(args) == 4:
    args[2] = int(args[2])
    single_obs_data, errors = create_bilou_from_one_document(input_data=raw_data[args[2] - 1],
                                                     data_number=args[2] - 1,
                                                     print_stats=True,
                                                     print_each_sentence=args[3])
else:
    for i, obs in enumerate(raw_data):
        single_obs_data, errors = create_bilou_from_one_document(input_data=obs,
                                                         data_number=i)
        word_tag_mismatch_errors += errors[0]
        wrong_index_errors += errors[1]
        entity_data.extend(single_obs_data)

print(f'mismatch errors: {word_tag_mismatch_errors}')
print(f'wrong index errors: {wrong_index_errors}')

write_json_lines(out_dir=DATA_DIR, data=entity_data,
                 filename='bilou_entities_kasper')
