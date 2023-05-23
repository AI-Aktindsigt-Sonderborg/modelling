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
wrong_raw_index_errors: int = 0
correct_indexes: int = 0
entity_data: List[dict] = []
total_sentences: int = 0
deleted_annotations: int = 0


def create_bilou_from_one_document(input_data: dict, data_number: int,print_stats: bool = False,print_each_sentence: int = 0):
    word_tag_mismatch_error: int = 0
    wrong_raw_index: int = 0
    wrong_index: int = 0
    correct_index: int = 0
    total_sentence: int = 0
    deleted_annotation: int = 0
    output_data = []

    for k, pdf_text in enumerate(input_data['pdf_text']):
        sentences_anon = []



        index_diff = 0
        pdf_altered = pdf_text
        page_num = f'page_no:{k + 1}'
        current_page_annotations = input_data['text_annotation'][page_num]

        # for s, annotation in enumerate(current_page_annotations):
        #
        #
        #     print()
        
        new_sentences, new_sentences2 = split_sentences_bilou(data=pdf_text, sentence_splitter=sentence_splitter)
        
        if len(input_data['text_annotation']) >= 1:
            try:
                current_page_annotations = input_data['text_annotation'][page_num]
            except Exception as ex:
                print(f"wtf error - check line {data_number + 1}")
                #                 print(page_num)
                print(traceback.format_exc())
                current_page_annotations = None

            if current_page_annotations:

                TRUE_LENGTH = len(new_sentences2)
                TRUE_LENGTH_OLD = len(new_sentences)                
                
                for i, sentence in enumerate(new_sentences2):
                    sentence_anon = sentence
                    if i == 0:
                        page_index_diff = 0
                    else:
                        page_index_diff += len(new_sentences2[i - 1]) + 2
                    current_sentence_annotations = [x for x in current_page_annotations if (x['annotation']['start'] < (len(sentence) + page_index_diff) and x['annotation']['start'] >= (page_index_diff))]
                    sorted_sentence_annotations = sorted(current_sentence_annotations, key=lambda x: x['annotation']['start'], reverse=True)
                    filtered_sentence_annotations = [x for x in sorted_sentence_annotations if x['annotation']['annotation'] not in ['EMAIL']]

                    for j, annotation in enumerate(filtered_sentence_annotations):
                        manual_match = False

                        if j == 0:
                            sentence_index_diff = 0
                        if annotation['annotation']['state'] == 'deleted':
                            deleted_annotation += 1
                            continue


                        start_index_init = annotation['annotation']['start']
                        end_index_init = annotation['annotation']['end']
                        true_orig_init = pdf_text[start_index_init:end_index_init]
                        
                        annotated_content = annotation['annotation']['content'].replace(" |", "")
                        annotated_content_last = annotated_content[-1]
                        
                        

                        while annotated_content_last.isspace() | (annotated_content_last == " "):
                            end_index_init = end_index_init - 1
                            annotated_content = annotated_content[:-1]
                            annotated_content_last = annotated_content[-1]
                            
                        # content_index_diff = 0
                        # content = annotated_content.strip()
                        # if content.endswith('.'):
                        #     if print_stats:
                        #         print(f'weird content: {content}')
                        #
                        #     content = content[:-1]
                        #     if print_stats:
                        #         print(f'weird content2: {content}')
                        #     content_index_diff = len(raw_content) - len(content)
                        # else:
                        #     print(f'index error {data_number+1}')

                        true_content = sentence[start_index_init - page_index_diff:end_index_init - page_index_diff]

                        true_original = pdf_text[annotation['annotation']['start']:end_index_init]
                        
                        if print_stats:
                            print(f'----------annotation_stats: {data_number + 1} - page_number: {k+1} - {i} - anno_nr: {j} -------------')
                            print(f'true original index content: |{true_orig_init}|')
                            print(f'true modified original index content: |{true_original}|')
                            print(f'true index content: |{true_content}|')
                            print(f'annotated content:  |{annotated_content}|')
                            print(f"true == annotated: {true_original == annotated_content}")
                            print(len(pdf_text) > annotation['annotation']['end'])
                            if len(pdf_text) > annotation['annotation']['end']:
                                print(annotation['annotation']['end'])
                                print("|" + pdf_text[annotation['annotation']['end']] + "|")
                                print("|" + pdf_text[annotation['annotation']['end']-15: annotation['annotation']['end'] + 15] + "|")
                                print(sentence[annotation['annotation']['start'] - page_index_diff + 1:annotation['annotation']['end'] - page_index_diff + 1])
                        
                        if true_original == annotated_content and len(pdf_text) > annotation['annotation']['end']:
                            if print_stats:
                                if pdf_text[annotation['annotation']['end']].isalpha():
                                    print('fuck')
                                    if pdf_text[annotation['annotation']['end'] - 1].isalpha():
                                        print("testing4")
                                        continue
                                    if pdf_text[annotation['annotation']['end'] - 1] == ' ':
                                        print("stupid1")
                                        print(data_number + 1)
                                        print(i)
                                        print(page_num)
                                        print(j)
                                        print("|" + true_original + "|")
                                        print(pdf_text[annotation['annotation']['end']])
                                        print(sentence)
                                        print("---------------")

                                try:
                                    if pdf_text[end_index_init+1] == '`':
                                        print("stupid2")
                                        print("|" + true_original + "|")
                                        print(sentence)
                                        print("---------------")
                                except Exception as ex:
                                    print("end index out of range")
                                    print(traceback.format_exc())
                                
                        entity = annotation['annotation']['annotation']

                        content_index_diff = 0

                        if true_original != annotated_content:
                            wrong_raw_index += 1
                        else:
                            try:                            
                                match1 = re.search(re.escape(annotated_content +'[^0-9a-zA-Z]'), sentence_anon)
                                match2 = re.search(annotated_content +'[^0-9a-zA-Z]', sentence_anon)
                                
                                if match1:
                                    print()
                                    print(f"manual match1: {match1}")
                                    print()
                                    start_index = match1.start()
                                    end_index = match1.end() - 1
                                    manual_match = True
                                    #manual_computed_diff = (annotation['annotation']['start'] - page_index_diff) - start_index
                                    #if (manual_computed_diff < 10) and (manual_computed_diff > -10):
                                     #   manual_match = True
                                elif match2:
                                    print()
                                    print(f"manual match2: {match1}")
                                    print()                                    
                                    start_index = match2.start()
                                    end_index = match2.end() - 1
                                    manual_match = True                                    
                                else: 
                                    if true_content == annotated_content:
                                        manual_match = False                        
                            except Exception as ex:
                                print(f"weird search error at {data_number + 1}")
                                print(traceback.format_exc())
                                manual_match = False

                        if not manual_match:
                            if true_content != annotated_content:
                                true_content_skewed = sentence[start_index_init - page_index_diff + 1:end_index_init - page_index_diff + 1]
                                true_content_skewed2 = sentence[start_index_init - page_index_diff - 1:end_index_init - page_index_diff - 1]
                                if true_content_skewed == annotated_content:
                                    true_content = true_content_skewed
                                    content_index_diff = 1
                                elif true_content_skewed2 == annotated_content:
                                    true_content = true_content_skewed2
                                    content_index_diff = -1
                            else:
                                print(f'index error at {data_number + 1}')
                                wrong_index += 1

                        if (annotated_content == true_content) or manual_match:
                            correct_index += 1
                            list_content = re.split(r'( |,|\. |\.\n)',
                                                    annotated_content)
                            to_remove = [" ", ""]
                            list_content = list(filter(lambda tag: tag.strip() not in to_remove, list_content))
                            insert_annotation: bool = True

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
                                insert_annotation = False
                                print("prut")
                                continue    

                            if insert_annotation:
                                if not manual_match:
                                    start_index = start_index_init - page_index_diff + content_index_diff
                                    end_index = end_index_init - page_index_diff + content_index_diff
                                try:
                                    if sentence_anon[end_index] == '\.':
                                        annotation_to_insert = annotation_to_insert + ' '
                                    if sentence_anon[end_index].isalpha() and \
                                        sentence_anon[end_index].islower():
                                        annotation_to_insert = annotation_to_insert + ' '
                                except Exception as ex:
                                    print("weird out of range error")
                                    print(traceback.format_exc())

                                sentence_anon = sentence_anon[:start_index] + annotation_to_insert + sentence_anon[end_index:]
                                # print("---------------------")
                                # print(sentence_anon)
                                # print('-')
                                # sentence_index_diff = len(sentence_anon) - len(sentence)

#                    if print_each_sentence:
#                        print('--- Sentence ------')
 #                       print(sentence)
  #                      print('--')
   #                     print(sentence_anon)
    #                    print('-------------------')                    
                    sentences_anon.append(sentence_anon + '\n')
                    
        try:
            ANON_LENGTH = len(sentences_anon)                        

            # assert len(new_sentences_anon2) == len(new_sentences2)
            for s, (sentence, sentence_anon) in enumerate(
                zip(new_sentences2, sentences_anon)):
                
                sentence = sentence + '\n'
                sentence_anon = sentence_anon + '\n'

                words = re.split(r'( |,|\. |\.\n)', sentence)
                tags = re.split(r'( |,|\. |\.\n)', sentence_anon)
                if print_stats and (len(words) != len(tags)):
                    print(
                        f'len(words): {len(words)}, len(tags): {len(tags)}')

                to_remove = [" ", "", "\n"]
                words_final = list(filter(lambda word: word.strip().rstrip('\\n').strip() not in to_remove, words))
                words_final[-1] = words_final[-1].strip()

                tags_no_whitespace = list(filter(lambda tag: tag.strip().rstrip('\\n').strip() not in to_remove,tags))
                
                tags_final = [tag if (tag.startswith(("B-", "U-", "I-", "L-")) and tag[2:].isupper()) else "O" for tag in tags_no_whitespace]
                tags_final[-1] = tags_final[-1].strip()

                if print_stats and (
                    len(words_final) != len(tags_final)):
                    print(f'len(words_final): {len(words_final)}, len(tags_final): {len(tags_final)}')
                    print(f'len(words_final): {len(words_final)}, len(tags_no_whitespace): {len(tags_no_whitespace)}')
                    print("--------------Annotations---------------")
                    for annot in sorted_sentence_annotations:
                        print(f"start: {annot['annotation']['start']}, end: {annot['annotation']['end']}, content: {annot['annotation']['content']}, annotation: {annot['annotation']['annotation']}")
                    print('----------------------------------------')
                        
                    if print_each_sentence == 1:
                        print("---------------------")
                        print(sentence)
                        print('-')
                        print(sentence_anon)
                        print('--')
                        print(words_final)
                        print('-')
                        print(tags_no_whitespace)
                        print('-')
                        print(tags_final)
                        print("---------------------")
                if len(tags_final) != len(words_final):
                    word_tag_mismatch_error += 1
                    print(f"word/tag mismatch linje {data_number + 1} med document_id {input_data['document_id']}.")
                
                assert len(tags_final) == len(words_final)
                total_sentence += 1
                output_data.append(
                    {'words': words_final, 'tags': tags_final})
                

        except Exception as e:
            print(f"data med linjenummer {data_number + 1} med id {input_data['id']} fejlede.")
            print(traceback.format_exc())

    return output_data, [word_tag_mismatch_error, wrong_index, correct_index,
                         total_sentence, deleted_annotation, wrong_raw_index]


if len(args) == 4:
    args[2] = int(args[2])
    args[3] = int(args[3])
    single_obs_data, errors = create_bilou_from_one_document(
        input_data=raw_data[args[2] - 1],
        data_number=args[2] - 1,
        print_stats=True,
        print_each_sentence=args[3])

    word_tag_mismatch_errors += errors[0]
    wrong_index_errors += errors[1]
    correct_indexes += errors[2]
    total_sentences += errors[3]
    deleted_annotations += errors[4]
    wrong_raw_index_errors += errors[5]

else:
    for i, obs in enumerate(raw_data):
        single_obs_data, errors = create_bilou_from_one_document(input_data=obs,
                                                                 data_number=i)
        word_tag_mismatch_errors += errors[0]
        wrong_index_errors += errors[1]
        correct_indexes += errors[2]
        total_sentences += errors[3]
        deleted_annotations += errors[4]
        wrong_raw_index_errors += errors[5]

        entity_data.extend(single_obs_data)

    write_json_lines(out_dir=DATA_DIR, data=entity_data,
                     filename='bilou_entities_kasper')
    print(f'total valid sentences: {len(entity_data)}')

print(f'word/tag length mismatch errors: {word_tag_mismatch_errors}')
print(f'wrong index errors: {wrong_index_errors}')
print(f'wrong raw index errors: {wrong_raw_index_errors}')
print(f'deleted annotations: {deleted_annotations}')
print(f'correct indexes: {correct_indexes}')

print(f'total sentences: {total_sentences}')
