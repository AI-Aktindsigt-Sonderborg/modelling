# pylint: disable=line-too-long
import os
import re
import sys
import traceback
from typing import List

import nltk

from ner.data_utils.custom_dataclasses import DataPrepConstants
from ner.data_utils.helpers import filter_language, \
    delete_duplicate_annotations, reindexing_first_or_last, fix_skewed_indices, \
    handle_wrong_annotation_end_letter
from ner.local_constants import DATA_DIR
from shared.utils.helpers import read_json, read_json_lines, write_json_lines

sentence_splitter = nltk.data.load("tokenizers/punkt/danish.pickle")


def fix_faulty_indices(current_page_annotations, pdf_text, document_num, ):
    indices_reindexed = 0
    annotation_errors = 0

    if os.path.isfile(os.path.join(DATA_DIR, "blacklist_helbred.json")):
        BLACKLIST_HELBRED = read_json(
            filepath=os.path.join(DATA_DIR, "blacklist_helbred.json")
        )

        BLACKLIST_FORB = read_json(
            filepath=os.path.join(DATA_DIR, "blacklist_forbrydelse.json")
        )

    filtered_annotation_list = []
    for annotation in current_page_annotations:
        annotation_error = 0

        annotation, filtered_annotation_list, annotation_error = delete_duplicate_annotations(
            data=annotation,
            filtered_list=filtered_annotation_list)

    for annotation_num, annotation in enumerate(current_page_annotations):

        if os.path.isfile(os.path.join(DATA_DIR, "blacklist_helbred.json")):
            if annotation["annotation"]["content"] in BLACKLIST_HELBRED:
                annotation_error = 1
                del current_page_annotations[annotation_num]
                print(
                    f"deleted helbred annot: {annotation['annotation']['content']}")
                continue

            if annotation["annotation"]["content"] in BLACKLIST_FORB:
                annotation_error = 1
                del current_page_annotations[annotation_num]
                print(
                    f"deleted forbrydelse annot: {annotation['annotation']['content']}")
                continue

        annotated_class = annotation["annotation"]["annotation"]
        start_index_init = annotation["annotation"]["start"]
        end_index_init = annotation["annotation"]["end"]
        true_original_init = pdf_text[start_index_init:end_index_init]
        annotated_content_init = annotation["annotation"]["content"]

        annotation, annotation_error = reindexing_first_or_last(data=annotation)

        start_index = annotation["annotation"]["start"]
        end_index = annotation["annotation"]["end"]

        annotated_content = annotation["annotation"]["content"]
        # true_original = pdf_text[start_index:end_index]

        annotation, true_original, indices_reindexed, annotation_error = fix_skewed_indices(
            text=pdf_text,
            data=annotation,
            reindex_counter=indices_reindexed)

        if current_page_annotations[annotation_num]['annotation']['start'] != annotation["annotation"]["start"]:
            print("This should not happen!!")
            # sys.exit(1)

        # Handle special cases
        if true_original.lower() == annotated_content.lower():
            annotation, annotation_error = handle_wrong_annotation_end_letter(
                data=annotation, text=pdf_text,
                patterns=DataPrepConstants.annotation_replacements,
                end_index=end_index_init
            )

        annotation_errors += annotation_error

    return current_page_annotations, indices_reindexed, annotation_errors


def create_bilou_from_one_document(
    input_data: dict,
    data_number: int,
    print_stats: bool = False,
    print_each_sentence: int = 0,
):
    print("reach this?")
    word_tag_mismatch_error: int = 0
    wrong_raw_index: int = 0
    wrong_index: int = 0
    correct_index: int = 0
    total_sentence: int = 0
    deleted_annotation: int = 0
    not_danish_counter: int = 0
    output_data = []
    document_annotation_errors: int = 0

    for k, pdf_text in enumerate(input_data["pdf_text"]):
        sentence_data = []
        sentences_anon = []
        modified_sentences = []
        page_num = f"page_no:{k + 1}"
        keys_list = list(input_data["text_annotation"])
        if page_num not in keys_list:
            current_page_annotations = None
            sorted_page_annotations = None
            current_page_annotations_original = None
        else:
            current_page_annotations = input_data["text_annotation"][page_num]
            current_page_annotations_original = current_page_annotations

            (
                current_page_annotations,
                indices_reindexed,
                annotation_errors,
            ) = fix_faulty_indices(current_page_annotations, pdf_text,
                                   data_number)
            document_annotation_errors += annotation_errors

            sorted_page_annotations = sorted(
                current_page_annotations, key=lambda x: x["annotation"]["start"]
            )

        splitted_sentences2 = pdf_text.split("\n\n")

        for i, sentence in enumerate(splitted_sentences2):
            is_danish = True
            if not is_danish:
                print(f"----sentence is not danish-----: {sentence}")
                not_danish_counter += 1
            entities = []
            sentence_anon = sentence

            if current_page_annotations:
                if i == 0:
                    page_index_diff = 0
                else:
                    page_index_diff += len(splitted_sentences2[i - 1]) + 2
                current_sentence_annotations = [
                    x
                    for x in current_page_annotations
                    if (
                        x["annotation"]["start"] < (
                        len(sentence) + page_index_diff)
                        and x["annotation"]["start"] >= (page_index_diff)
                    )
                ]
                wrong_sentence_annotations = [
                    x
                    for x in current_page_annotations
                    if (
                        x["annotation"]["start"] > (
                        len(sentence) + page_index_diff)
                        or x["annotation"]["start"] <= (page_index_diff)
                    )
                ]
                # if len(wrong_sentence_annotations) > 0:
                # for element in wrong_sentence_annotations:
                # if element["annotation"]["annotation"] == "HELBRED":
                # print("---Wrong sentence annotation----")
                # print(
                #   f"document_num: {data_number + 1} - page_num: {page_num}"
                # )
                # print(element)

                sorted_sentence_annotations = sorted(
                    current_sentence_annotations,
                    key=lambda x: x["annotation"]["start"],
                    reverse=True,
                )
                filtered_sentence_annotations = [
                    x
                    for x in sorted_sentence_annotations
                    if x["annotation"]["annotation"]
                       not in DataPrepConstants.none_ner_entities
                ]

                for j, annotation in enumerate(filtered_sentence_annotations):
                    manual_match = False

                    if j == 0:
                        sentence_index_diff = 0
                    if annotation["annotation"]["state"] == "deleted":
                        deleted_annotation += 1
                        continue

                    start_index_init = annotation["annotation"]["start"]
                    end_index_init = annotation["annotation"]["end"]
                    true_orig_init = pdf_text[start_index_init:end_index_init]

                    annotated_content = annotation["annotation"][
                        "content"].replace(
                        " |", ""
                    )

                    annotated_content_last = annotated_content[-1]
                    annotated_content_first = annotated_content[0]

                    while annotated_content_last.isspace() | (
                        annotated_content_last == " "
                    ):
                        end_index_init = end_index_init - 1
                        annotated_content = annotated_content[:-1]
                        annotated_content_last = annotated_content[-1]

                    first_is_space = False
                    if annotated_content_first.isspace() | (
                        annotated_content_first == " "
                    ):
                        first_is_space = True

                    true_content = sentence[
                                   start_index_init
                                   - page_index_diff: end_index_init
                                                      - page_index_diff
                                   ]
                    try:
                        if (
                            (annotated_content.lower() == true_content.lower())
                            and sentence[
                            end_index_init - page_index_diff].isalpha()
                            and sentence[
                            end_index_init - page_index_diff - 1].isdigit()
                        ):
                            sentence = (
                                sentence[: start_index_init - page_index_diff]
                                + annotated_content
                                + " "
                                + sentence[end_index_init - page_index_diff:]
                            )
                    except IndexError:
                        print("weird index error")
                        print(traceback.format_exc())

                    true_original = pdf_text[
                                    annotation["annotation"][
                                        "start"]: end_index_init
                                    ]

                    if (
                        annotation["annotation"]["content"] == "Ankestyrelsen"
                        or annotation["annotation"][
                        "content"] == "Ankestyrelsens"
                    ):
                        print(
                            f"----------annotation_stats: {data_number + 1} - page_number: {k + 1} - {i} - anno_nr: {j} -------------"
                        )
                        print(
                            f"true original index content: |{true_orig_init}|")
                        print(
                            f"true modified original index content: |{true_original}|"
                        )
                        print(f"true index content: |{true_content}|")
                        print(f"annotated content:  |{annotated_content}|")
                        print(
                            f"true == annotated: {true_original == annotated_content}"
                        )
                        print(len(pdf_text) > annotation["annotation"]["end"])
                        if len(pdf_text) > annotation["annotation"]["end"]:
                            print(annotation["annotation"]["end"])
                            print("|" + pdf_text[
                                annotation["annotation"]["end"]] + "|")
                            print(
                                "|"
                                + pdf_text[
                                  annotation["annotation"]["end"]
                                  - 15: annotation["annotation"]["end"]
                                        + 15
                                  ]
                                + "|"
                            )
                            print(
                                sentence[
                                annotation["annotation"]["start"]
                                - page_index_diff
                                + 1: annotation["annotation"]["end"]
                                     - page_index_diff
                                     + 1
                                ]
                            )

                    entity = annotation["annotation"]["annotation"]
                    content_index_diff = 0

                    index_match = True
                    if true_original.lower() != annotated_content.lower():
                        index_match = False
                        wrong_raw_index += 1
                        skewness_list = [-5, -4, -3, -2, -1, 1, 2, 3, 4, 5]
                        if true_content.lower() != annotated_content.lower():
                            for skewness in skewness_list:
                                true_content_skewed = sentence[
                                                      start_index_init
                                                      - page_index_diff
                                                      + skewness: end_index_init
                                                                  - page_index_diff
                                                                  + skewness
                                                      ]
                                if (
                                    true_content_skewed.lower()
                                    == annotated_content.lower()
                                ):
                                    true_content = true_content_skewed
                                    content_index_diff = skewness
                                    index_match = True
                                    break
                                else:
                                    continue

                            if not index_match:
                                print(f"index error at {data_number + 1}")
                                wrong_index += 1
                                index_match = False

                    if (
                        annotated_content.lower() == true_content.lower()
                    ) or manual_match:
                        correct_index += 1
                        list_content = re.split(r"( |,|\. |\.\n)",
                                                annotated_content)
                        to_remove = [" ", ""]
                        list_content = list(
                            filter(
                                lambda tag: tag.strip() not in to_remove,
                                list_content
                            )
                        )
                        insert_annotation: bool = True

                        if len(list_content) == 1:
                            annotation_to_insert = "U-" + entity
                            if first_is_space:
                                annotation_to_insert = " " + annotation_to_insert
                        elif len(list_content) == 2:
                            annotation_to_insert = "B-" + entity + " L-" + entity
                            if first_is_space:
                                annotation_to_insert = " " + annotation_to_insert
                        elif len(list_content) > 2:
                            annotation_to_insert = "B-" + entity
                            if first_is_space:
                                annotation_to_insert = " " + annotation_to_insert
                            for inside_element in list_content[1:-1]:
                                annotation_to_insert = (
                                    annotation_to_insert + " I-" + entity
                                )
                            annotation_to_insert = annotation_to_insert + " L-" + entity
                        else:
                            insert_annotation = False
                            continue

                        if insert_annotation:
                            entities.append(entity)
                            if not manual_match:
                                start_index = (
                                    start_index_init
                                    - page_index_diff
                                    + content_index_diff
                                )
                                end_index = (
                                    end_index_init
                                    - page_index_diff
                                    + content_index_diff
                                )
                            try:
                                if sentence_anon[end_index] == r"\.":
                                    annotation_to_insert = annotation_to_insert + " "
                                if (
                                    sentence_anon[end_index].isalpha()
                                    and sentence_anon[end_index].islower()
                                ):
                                    annotation_to_insert = annotation_to_insert + " "
                            except IndexError:
                                print("error: end of sentence reached")
                                print(traceback.format_exc())

                            sentence_anon = (
                                sentence_anon[:start_index]
                                + annotation_to_insert
                                + sentence_anon[end_index:]
                            )
            if is_danish:
                sentences_anon.append(sentence_anon + "\n")
                modified_sentences.append(sentence + "\n")
                sentence_data.append(
                    {
                        "sentence": sentence + "\n",
                        "sentence_anon": sentence_anon + "\n",
                        "doc_id": input_data["document_id"],
                        "page_no": page_num,
                        "sentence_no": i,
                        "entities": entities,
                    }
                )

        try:
            sentence_index_diff = 0
            for s, (sentence, sentence_anon) in enumerate(
                zip(modified_sentences, sentences_anon)
            ):
                patterns = {
                    r"\s+": " ",
                    r"\|": " | ",
                    r"\)": " ) ",
                    r"\(": " ( ",
                    r"\[": " [ ",
                    r"\]": " ] ",
                    r"\;": " ; ",
                    r"\:": " : ",
                    r"\<": " < ",
                    r"\>": " > ",
                    "\?": " ? ",
                }

                for pattern, replacement in patterns.items():
                    sentence = re.sub(pattern, replacement, sentence + "\n")
                    sentence_anon = re.sub(pattern, replacement,
                                           sentence_anon + "\n")

                special_chars = [
                    ")",
                    "(",
                    "]",
                    "[",
                    ".",
                    "-",
                    "=",
                    ",",
                    ";",
                    ":",
                    "?",
                    "/",
                    "_",
                ]
                split_pattern = r"( |,|\. |\.\n|:|\(|\[|\]|=|;)"
                # split_pattern = r"(\w+|[.,:;()?\[\]_])"
                words = re.split(split_pattern, sentence)
                tags = re.split(split_pattern, sentence_anon)
                if print_stats and (len(words) != len(tags)):
                    print(f"len(words): {len(words)}, len(tags): {len(tags)}")

                to_remove = [" ", "", "\n", "[", "]", "(", ")", ":", ";"]
                words_final = list(
                    filter(
                        lambda word: word.strip().rstrip("\\n").strip()
                                     not in to_remove,
                        words,
                    )
                )
                words_final[-1] = words_final[-1].strip()

                tags_no_whitespace = list(
                    filter(
                        lambda tag: tag.strip().rstrip(
                            "\\n").strip() not in to_remove,
                        tags,
                    )
                )
                tags_no_whitespace[-1] = tags_no_whitespace[-1].strip()

                if words_final[-1] == "." and not (
                    tags_no_whitespace[-1] == "."):
                    tags_no_whitespace.append(".")

                tags_final = [
                    tag
                    if (
                        tag.startswith(("B-", "U-", "I-", "L-"))
                        and tag[2:].isupper()
                        and tag[2:].isalpha()
                        and tag not in ("U-ADRESSE", "U-KOMMUNE")
                    )
                    else "O"
                    for tag in tags_no_whitespace
                ]

                if print_stats and (len(words_final) != len(tags_final)):
                    # if '0-17' in sentence:
                    print(f"docnumber: {data_number + 1}")
                    print(
                        f"len(words_final): {len(words_final)}, len(tags_final): {len(tags_final)}"
                    )
                    print(
                        f"len(words_final): {len(words_final)}, len(tags_no_whitespace): {len(tags_no_whitespace)}"
                    )
                    print(
                        f"--------------Annotations - start: {sentence_index_diff} - end: {sentence_index_diff + len(sentence)}---------------"
                    )
                    for annot in sorted_page_annotations:
                        print(
                            f"start: {annot['annotation']['start']}, end: {annot['annotation']['end']}, content: {annot['annotation']['content']}, annotation: {annot['annotation']['annotation']}"
                        )

                    if print_each_sentence == 1:
                        print("----sentence-----")
                        print(f"|{sentence}|")
                        print("----sentence_anon-----")
                        print(f"|{sentence_anon}|")
                        print("----words_final-----")
                        print(words_final)
                        print("----tags_no_whitespace-----")
                        print(tags_no_whitespace)
                        print("----tags_final-----")
                        print(tags_final)
                        print("-------LANGUAGES--------------")
                        is_danish, language_codes = filter_language(
                            string=sentence, approved_languages=["da", "no"]
                        )
                        print(language_codes)
                        print("---------------------")
                        for count in range(len(words_final)):
                            print(
                                f"{words_final[count]} - {tags_no_whitespace[count]}")

                if len(tags_final) != len(words_final):
                    print("fucktards")
                    word_tag_mismatch_error += 1
                    print(
                        f"word/tag mismatch linje {data_number + 1} med document_id {input_data['document_id']}."
                    )
                    print(
                        f"--------------Full dicts - start: {sentence_index_diff} - end: {sentence_index_diff + len(sentence)}---------------"
                    )
                    for annot in sorted_page_annotations:
                        print(annot)
                    print(
                        "--------------Sorted annotations - start------------------")
                    for annot in sorted_page_annotations:
                        print(
                            f"start: {annot['annotation']['start']}, end: {annot['annotation']['end']}, content: {annot['annotation']['content']}, annotation: {annot['annotation']['annotation']}"
                        )
                    print(
                        f"----------Original Annotations - start: {sentence_index_diff} - end: {sentence_index_diff + len(sentence)}---------------"
                    )
                    for annot in current_page_annotations_original:
                        print(
                            f"start: {annot['annotation']['start']}, end: {annot['annotation']['end']}, content: {annot['annotation']['content']}, annotation: {annot['annotation']['annotation']}"
                        )

                    print("----sentence-----")
                    print(f"|{sentence}|")
                    print("----sentence_anon-----")
                    print(f"|{sentence_anon}|")
                    print("----words_final-----")
                    print(words_final)
                    print("----tags_no_whitespace-----")
                    print(tags_no_whitespace)
                    print("----tags_final-----")
                    print(tags_final)
                    print("-------LANGUAGES--------------")
                    is_danish, language_codes = filter_language(
                        string=sentence, approved_languages=["da", "no"]
                    )
                    print(language_codes)
                    print("---------------------")
                    for count in range(len(words_final)):
                        print(
                            f"{words_final[count]} - {tags_no_whitespace[count]}")

                sentence_index_diff += len(sentence)

                assert len(tags_final) == len(words_final)
                total_sentence += 1
                if len(tags_final) >= 5:
                    output_data.append(
                        {
                            "tokens": words_final,
                            "tags": tags_final,
                            "sentence": sentence_data[s]["sentence"],
                            "sentence_anon": sentence_data[s]["sentence_anon"],
                            "doc_id": input_data["document_id"],
                            "page_no": sentence_data[s]["page_no"],
                            "sentence_no": sentence_data[s]["sentence_no"],
                            "origin_line_no": data_number + 1,
                            "entities": sentence_data[s]["entities"],
                        }
                    )

        except Exception:
            print(
                f"data med linjenummer {data_number + 1} med id {input_data['document_id']} fejlede."
            )
            print(traceback.format_exc())
            # assert len(tags_final) == len(words_final)
            # raise Exception("tralala")

    return output_data, [
        word_tag_mismatch_error,
        wrong_index,
        correct_index,
        total_sentence,
        deleted_annotation,
        wrong_raw_index,
        indices_reindexed,
        not_danish_counter,
        document_annotation_errors,
    ]


if __name__ == "__main__":
    args: list = sys.argv[1:]

    raw_data = read_json_lines(input_dir=DATA_DIR, filename=args[0])
    bilou = read_json_lines(input_dir=DATA_DIR, filename=args[1])

    sentence_splitter = nltk.data.load("tokenizers/punkt/danish.pickle")

    word_tag_mismatch_errors: int = 0
    wrong_index_errors: int = 0
    wrong_raw_index_errors: int = 0
    correct_indexes: int = 0
    entity_data: List[dict] = []
    total_sentences: int = 0
    deleted_annotations: int = 0
    total_indices_reindexed: int = 0
    total_not_danish_counter: int = 0
    total_annotation_errors: int = 0

    if len(args) == 4:
        args[2] = int(args[2])
        args[3] = int(args[3])
        single_obs_data, errors = create_bilou_from_one_document(
            input_data=raw_data[args[2] - 1],
            data_number=args[2] - 1,
            print_stats=True,
            print_each_sentence=args[3],
        )

        word_tag_mismatch_errors += errors[0]
        wrong_index_errors += errors[1]
        correct_indexes += errors[2]
        total_sentences += errors[3]
        deleted_annotations += errors[4]
        wrong_raw_index_errors += errors[5]
        total_indices_reindexed += errors[6]
        total_not_danish_counter += errors[7]
        total_annotation_errors += errors[8]

    else:
        for i, obs in enumerate(raw_data):
            print(f"creating bilou from document number {i + 1}")
            single_obs_data, errors = create_bilou_from_one_document(
                input_data=obs, data_number=i
            )
            word_tag_mismatch_errors += errors[0]
            wrong_index_errors += errors[1]
            correct_indexes += errors[2]
            total_sentences += errors[3]
            deleted_annotations += errors[4]
            wrong_raw_index_errors += errors[5]
            total_indices_reindexed += errors[6]
            total_not_danish_counter += errors[7]
            total_annotation_errors += errors[8]

            entity_data.extend(single_obs_data)

        write_json_lines(out_dir=DATA_DIR, data=entity_data,
                         filename="bilou_0808")

    print(f"Total valid sentences: {len(entity_data)}")
    print(f"word/tag length mismatch errors: {word_tag_mismatch_errors}")
    print(f"wrong index errors: {wrong_index_errors}")
    print(f"wrong raw index errors: {wrong_raw_index_errors}")
    print(f"Total indices reindexed: {total_indices_reindexed}")
    print(f"deleted annotations: {deleted_annotations}")
    print(f"correct indexes: {correct_indexes}")
    print(f"sentences not danish: {total_not_danish_counter}")
    print(f"total annotation errors: {total_annotation_errors}")
    print(f"total sentences: {total_sentences}")
