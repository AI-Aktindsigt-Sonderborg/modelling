# pylint: disable=line-too-long
import re
import sys
import traceback
from typing import List
import os
import nltk

from ner.data_utils.helpers import split_sentences_bilou, filter_language
from ner.local_constants import DATA_DIR
from shared.utils.helpers import read_json, read_json_lines, write_json_lines
from langdetect import detect_langs

sentence_splitter = nltk.data.load("tokenizers/punkt/danish.pickle")


def fix_faulty_indices(current_page_annotations, pdf_text, document_num):
    indices_reindexed = 0

    BLACKLIST_HELBRED = read_json(
        filepath=os.path.join(DATA_DIR, "blacklist_helbred.json")
    )

    BLACKLIST_FORB = read_json(
        filepath=os.path.join(DATA_DIR, "blacklist_forbrydelse.json")
    )

    for annotation_num, annotation in enumerate(current_page_annotations):
        # ToDo: Fix this guid thing as they may not be unique
        if (
            annotation["annotation"]["category"]["guid"]
            == "1f51090c-9454-49c0-ade8-d4adb24fcf0a"
        ) and (annotation["annotation"]["content"] == "Børn og Familie"):
            del current_page_annotations[annotation_num]
            continue
        if annotation["annotation"]["content"] in BLACKLIST_HELBRED:
            del current_page_annotations[annotation_num]
            print(f"deleted helbred annot: {annotation['annotation']['content']}")
            continue

        if annotation["annotation"]["content"] in BLACKLIST_FORB:
            del current_page_annotations[annotation_num]
            print(f"deleted forbrydelse annot: {annotation['annotation']['content']}")
            continue

        annotated_content_init = annotation["annotation"]["content"]
        annotated_class = annotation["annotation"]["annotation"]

        start_index_init = annotation["annotation"]["start"]
        end_index_init = annotation["annotation"]["end"]
        true_original_init = pdf_text[start_index_init:end_index_init]
        annotated_content_init = annotation["annotation"]["content"]
        annotated_content = annotated_content_init
        annotated_content_last = annotated_content[-1]

        start_index = start_index_init
        end_index = end_index_init
        while annotated_content_last.isspace() | (annotated_content_last == " "):
            end_index = end_index - 1
            annotation["annotation"]["end"] -= 1
            annotated_content = annotated_content[:-1]
            annotation["annotation"]["content"] = annotated_content
            annotated_content_last = annotated_content[-1]

        special_chars = [")", "(", "]", "[", ".", "-", "=", ",", ";", ":"]
        while annotated_content_last in special_chars:
            end_index = end_index - 1
            annotation["annotation"]["end"] -= 1
            annotated_content = annotated_content[:-1]
            annotation["annotation"]["content"] = annotated_content
            try:
                annotated_content_last = annotated_content[-1]
            except IndexError:
                print(f"removing annotation {annotation_num} from {document_num + 1}")
                print(current_page_annotations[annotation_num])
                del current_page_annotations[annotation_num]
                print(f"removed annotation {annotation_num} from {document_num + 1}")
                break

        true_original = pdf_text[start_index:end_index]

        if true_original.lower() != annotated_content.lower():
            skewness_list = list(range(-10, 10))
            true_content_skewed = true_original
            for skewness in skewness_list:
                try:
                    true_content_skewed = pdf_text[
                        start_index + skewness : end_index + skewness
                    ]
                except IndexError:
                    print("skewness search error")
                    print(traceback.format_exc())
                if true_content_skewed.lower() == annotated_content.lower():
                    current_page_annotations[annotation_num]["annotation"]["start"] = (
                        start_index + skewness
                    )
                    current_page_annotations[annotation_num]["annotation"]["end"] = (
                        end_index + skewness
                    )
                    true_original = pdf_text[
                        start_index + skewness : end_index + skewness
                    ]
                    indices_reindexed += 1
                    print("sentence reindexed")
                    print(current_page_annotations[annotation_num])
                    break
                print(current_page_annotations[annotation_num])
                continue

        if true_original.lower() != annotated_content.lower():
            try:
                match1 = re.search(
                    re.escape(annotated_content.lower() + "[^0-9a-zA-Z]"),
                    pdf_text.lower(),
                )
                match2 = re.search(
                    annotated_content.lower() + "[^0-9a-zA-Z]", pdf_text.lower()
                )
                if (
                    match1
                    and (abs(match1.start() - annotation["annotation"]["start"]) > 10)
                ) or (
                    match2
                    and (abs(match2.start() - annotation["annotation"]["start"]) > 10)
                ):
                    match1 = None
                    match2 = None

                if match1:
                    manual_start_index = match1.start()
                    manual_end_index = match1.end() - 1
                    current_page_annotations[annotation_num]["annotation"][
                        "start"
                    ] = manual_start_index
                    current_page_annotations[annotation_num]["annotation"][
                        "end"
                    ] = manual_end_index
                    indices_reindexed += 1
                elif match2:
                    manual_start_index = match2.start()
                    manual_end_index = match2.end() - 1
                    current_page_annotations[annotation_num]["annotation"][
                        "start"
                    ] = manual_start_index
                    current_page_annotations[annotation_num]["annotation"][
                        "end"
                    ] = manual_end_index
                    indices_reindexed += 1
                else:
                    print(f"found no manual match - doc_num: {document_num}")
                    print(f"true original content: |{true_original_init}|")
                    print(f"true modified content: |{true_original}|")
                    print(f"annotated content:  |{annotated_content}|")
                    print(f"|{pdf_text[start_index_init - 15:end_index_init + 15]}|")
                    print(f"entity: |{annotation['annotation']['annotation']}|")
            except Exception:
                print("manual search error")
                print(traceback.format_exc())

        # Handle special cases
        if true_original.lower() == annotated_content.lower():
            # where annotation does not include the last s in word
            if len(pdf_text) > end_index + 2:
                if pdf_text[end_index_init : end_index_init + 2] == "s ":
                    current_page_annotations[annotation_num]["annotation"][
                        "content"
                    ] = (
                        current_page_annotations[annotation_num]["annotation"][
                            "content"
                        ]
                        + "s"
                    )
                    current_page_annotations[annotation_num]["annotation"]["end"] = (
                        current_page_annotations[annotation_num]["annotation"]["end"]
                        + 1
                    )
                if pdf_text[end_index_init : end_index_init + 3] == "'s ":
                    current_page_annotations[annotation_num]["annotation"][
                        "content"
                    ] = (
                        current_page_annotations[annotation_num]["annotation"][
                            "content"
                        ]
                        + "s"
                    )
                    current_page_annotations[annotation_num]["annotation"]["end"] = (
                        current_page_annotations[annotation_num]["annotation"]["end"]
                        + 1
                    )
                if pdf_text[end_index_init : end_index_init + 2] == "k ":
                    current_page_annotations[annotation_num]["annotation"][
                        "content"
                    ] = (
                        current_page_annotations[annotation_num]["annotation"][
                            "content"
                        ]
                        + "k"
                    )
                    current_page_annotations[annotation_num]["annotation"]["end"] = (
                        current_page_annotations[annotation_num]["annotation"]["end"]
                        + 1
                    )
                if pdf_text[end_index_init : end_index_init + 1] == "e":
                    current_page_annotations[annotation_num]["annotation"][
                        "content"
                    ] = (
                        current_page_annotations[annotation_num]["annotation"][
                            "content"
                        ]
                        + "e"
                    )
                    current_page_annotations[annotation_num]["annotation"]["end"] = (
                        current_page_annotations[annotation_num]["annotation"]["end"]
                        + 1
                    )
                if pdf_text[end_index_init : end_index_init + 1] == "r":
                    current_page_annotations[annotation_num]["annotation"][
                        "content"
                    ] = (
                        current_page_annotations[annotation_num]["annotation"][
                            "content"
                        ]
                        + "r"
                    )
                    current_page_annotations[annotation_num]["annotation"]["end"] = (
                        current_page_annotations[annotation_num]["annotation"]["end"]
                        + 1
                    )
                if pdf_text[end_index_init : end_index_init + 1] == "n":
                    current_page_annotations[annotation_num]["annotation"][
                        "content"
                    ] = (
                        current_page_annotations[annotation_num]["annotation"][
                            "content"
                        ]
                        + "n"
                    )
                    current_page_annotations[annotation_num]["annotation"]["end"] = (
                        current_page_annotations[annotation_num]["annotation"]["end"]
                        + 1
                    )
                if pdf_text[end_index : end_index + 1] == "s":
                    current_page_annotations[annotation_num]["annotation"][
                        "content"
                    ] = (
                        current_page_annotations[annotation_num]["annotation"][
                            "content"
                        ]
                        + "s"
                    )
                    current_page_annotations[annotation_num]["annotation"]["end"] = (
                        current_page_annotations[annotation_num]["annotation"]["end"]
                        + 1
                    )

    return current_page_annotations, indices_reindexed


def create_bilou_from_one_document(
    input_data: dict,
    data_number: int,
    print_stats: bool = False,
    print_each_sentence: int = 0,
):
    word_tag_mismatch_error: int = 0
    wrong_raw_index: int = 0
    wrong_index: int = 0
    correct_index: int = 0
    total_sentence: int = 0
    deleted_annotation: int = 0
    not_danish_counter: int = 0
    output_data = []

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

            current_page_annotations, indices_reindexed = fix_faulty_indices(
                current_page_annotations, pdf_text, data_number
            )

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
                        x["annotation"]["start"] < (len(sentence) + page_index_diff)
                        and x["annotation"]["start"] >= (page_index_diff)
                    )
                ]
                wrong_sentence_annotations = [
                    x
                    for x in current_page_annotations
                    if (
                        x["annotation"]["start"] > (len(sentence) + page_index_diff)
                        or x["annotation"]["start"] <= (page_index_diff)
                    )
                ]
                if len(wrong_sentence_annotations) > 0:
                    for element in wrong_sentence_annotations:
                        if element["annotation"]["annotation"] == "HELBRED":
                            print("---Wrong sentence annotation----")
                            print(
                                f"document_num: {data_number + 1} - page_num: {page_num}"
                            )
                            print(element)

                sorted_sentence_annotations = sorted(
                    current_sentence_annotations,
                    key=lambda x: x["annotation"]["start"],
                    reverse=True,
                )
                filtered_sentence_annotations = [
                    x
                    for x in sorted_sentence_annotations
                    if x["annotation"]["annotation"]
                    not in ["EMAIL", "PRIS", "DATO", "URL", "CVR"]
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

                    annotated_content = annotation["annotation"]["content"].replace(
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
                        - page_index_diff : end_index_init
                        - page_index_diff
                    ]
                    try:
                        if (
                            (annotated_content.lower() == true_content.lower())
                            and sentence[end_index_init - page_index_diff].isalpha()
                            and sentence[end_index_init - page_index_diff - 1].isdigit()
                        ):
                            sentence = (
                                sentence[: start_index_init - page_index_diff]
                                + annotated_content
                                + " "
                                + sentence[end_index_init - page_index_diff :]
                            )
                    except IndexError:
                        print("weird index error")
                        print(traceback.format_exc())

                    true_original = pdf_text[
                        annotation["annotation"]["start"] : end_index_init
                    ]

                    if (
                        annotation["annotation"]["content"] == "Ankestyrelsen"
                        or annotation["annotation"]["content"] == "Ankestyrelsens"
                    ):
                        print(
                            f"----------annotation_stats: {data_number + 1} - page_number: {k + 1} - {i} - anno_nr: {j} -------------"
                        )
                        print(f"true original index content: |{true_orig_init}|")
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
                            print("|" + pdf_text[annotation["annotation"]["end"]] + "|")
                            print(
                                "|"
                                + pdf_text[
                                    annotation["annotation"]["end"]
                                    - 15 : annotation["annotation"]["end"]
                                    + 15
                                ]
                                + "|"
                            )
                            print(
                                sentence[
                                    annotation["annotation"]["start"]
                                    - page_index_diff
                                    + 1 : annotation["annotation"]["end"]
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
                                    + skewness : end_index_init
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
                        list_content = re.split(r"( |,|\. |\.\n)", annotated_content)
                        to_remove = [" ", ""]
                        list_content = list(
                            filter(
                                lambda tag: tag.strip() not in to_remove, list_content
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
                sentence = re.sub(r"\s+", " ", sentence + "\n")
                sentence_anon = re.sub(r"\s+", " ", sentence_anon + "\n")

                words = re.split(r"( |,|\. |\.\n|:)", sentence)
                tags = re.split(r"( |,|\. |\.\n|:)", sentence_anon)
                if print_stats and (len(words) != len(tags)):
                    print(f"len(words): {len(words)}, len(tags): {len(tags)}")

                to_remove = [" ", "", "\n", "[", "]", "(", ")", "*", ":", ";"]
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
                        lambda tag: tag.strip().rstrip("\\n").strip() not in to_remove,
                        tags,
                    )
                )
                tags_no_whitespace[-1] = tags_no_whitespace[-1].strip()

                if words_final[-1] == "." and not (tags_no_whitespace[-1] == "."):
                    tags_no_whitespace.append(".")

                tags_final = [
                    tag
                    if (
                        tag.startswith(("B-", "U-", "I-", "L-"))
                        and tag[2:].isupper()
                        and tag[2:].isalpha()
                        and tag != "U-ADRESSE"
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
                            print(f"{words_final[count]} - {tags_no_whitespace[count]}")

                if len(tags_final) != len(words_final):
                    word_tag_mismatch_error += 1
                    print(
                        f"word/tag mismatch linje {data_number + 1} med document_id {input_data['document_id']}."
                    )
                    print(
                        f"--------------Full dicts - start: {sentence_index_diff} - end: {sentence_index_diff + len(sentence)}---------------"
                    )
                    for annot in sorted_page_annotations:
                        print(annot)
                    print("--------------Sorted annotations - start------------------")
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
                        print(f"{words_final[count]} - {tags_no_whitespace[count]}")

                    # sys.exit(0)

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

    return output_data, [
        word_tag_mismatch_error,
        wrong_index,
        correct_index,
        total_sentence,
        deleted_annotation,
        wrong_raw_index,
        indices_reindexed,
        not_danish_counter,
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
            entity_data.extend(single_obs_data)

        write_json_lines(out_dir=DATA_DIR, data=entity_data, filename="bilou_1507")

    print(f"total valid sentences: {len(entity_data)}")
    print(f"word/tag length mismatch errors: {word_tag_mismatch_errors}")
    print(f"wrong index errors: {wrong_index_errors}")
    print(f"wrong raw index errors: {wrong_raw_index_errors}")
    print(f"Total indices reindexed: {total_indices_reindexed}")
    print(f"deleted annotations: {deleted_annotations}")
    print(f"correct indexes: {correct_indexes}")
    print(f"sentences not danish: {total_not_danish_counter}")
    print(f"total sentences: {total_sentences}")
