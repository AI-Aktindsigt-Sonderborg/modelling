# pylint: disable=line-too-long
import os
import re
import sys
import traceback
from typing import List
from collections import Counter
from itertools import groupby

import nltk

from ner.data_utils.custom_dataclasses import DataPrepConstants
from ner.data_utils.helpers import (
    filter_language,
    delete_duplicate_annotations,
    reindexing_first_or_last,
    fix_skewed_indices,
    handle_wrong_annotation_end_letter,
)
from mlm.local_constants import DATA_DIR, CONF_DATA_DIR, METADATA_DIR
from shared.utils.helpers import read_json, read_json_lines, write_json_lines

sentence_splitter = nltk.data.load("tokenizers/punkt/danish.pickle")


def create_sentences_from_one_document(
    input_data: dict, data_number: int, file_number: int, label_mapping: List[dict]
):
    total_sentence: int = 0
    not_danish_counter: int = 0
    output_data = []

    for k, pdf_text in enumerate(input_data["pdf_text"]):
        page_num = k + 1
        splitted_sentences = pdf_text.split("\n\n")

        for i, sentence in enumerate(splitted_sentences):
            if len(sentence) == 0 or sentence.isspace():
                continue

            if len(sentence.split(" ")) >= 5:
                total_sentence += 1
                try:
                    output_data.append(
                        {
                            "text": sentence,
                            "file_no": file_number,
                            "doc_id": input_data["document_id"],
                            "page_no": page_num,
                            "sentence_no": i,
                            "origin_line_no": data_number + 1,
                            "label": label_mapping[input_data['text_category_guid']]
                        }
                    )
                except Exception:
                    print(input_data)


    return output_data, [
        total_sentence,
        not_danish_counter,
    ]


if __name__ == "__main__":
    total_sentences: int = 0
    total_not_danish_counter: int = 0
    category_mapping = read_json(filepath=os.path.join(METADATA_DIR, "ss_annotation_categories.json"))

    label_mapping = {}

    for x in category_mapping:
        label_mapping[x["guid"]] = x["tag_name"]

    file_list = sorted(
        [
            f.split(".")[0]
            for f in os.listdir(CONF_DATA_DIR)
            if os.path.isfile(os.path.join(CONF_DATA_DIR, f)) and f.startswith("raw")
        ]
    )
    print(file_list)

    out_data: List[dict] = []
    for j, file in enumerate(file_list):
        print(file)
        print(os.path.join(CONF_DATA_DIR, file))
        raw_data = read_json_lines(input_dir=CONF_DATA_DIR, filename=file)

        for i, obs in enumerate(raw_data):
            if obs["pdf_text"]:
                single_obs_data, errors = create_sentences_from_one_document(
                    input_data=obs, data_number=i, file_number=j, label_mapping=label_mapping
                )
                total_sentences += errors[0]
                total_not_danish_counter += errors[1]

                out_data.extend(single_obs_data)


    write_json_lines(out_dir=CONF_DATA_DIR, data=out_data, filename="all_sentences")

    out_data.sort(key=lambda x: x['text'])
    out_data2 = [{"text": x["text"], "label": x["label"]} for x in out_data]
    out_data2.sort(key=lambda x: x['text'])

    grouped_data = {key: list(group) for key, group in groupby(out_data2, key=lambda x: x['text'])}

    unique1 = []
    for text, group in grouped_data.items():
        # print(group)
        # print(f"sentence: {text}")
        # if len(group) > 1:
        #     # print(f"sentence: {text}")
        #     for label in group:
        #         # print(f"  {label}")
        #         # print(f"  {label['label']}")
        if len(group) == 1:
            unique1.append({"text": text, "label": group[0]['label']})


    unique = list(set([x["text"] for x in out_data]))


    write_json_lines(out_dir=CONF_DATA_DIR, data=unique, filename="unique_sentences")
    write_json_lines(out_dir=CONF_DATA_DIR, data=unique1, filename="unique_sentences_with_label")

    label_count = Counter([x['label'] for x in unique1])

    print(f"label count: {label_count}")

    print(f"sentences not danish: {total_not_danish_counter}")
    print(f"total sentences: {total_sentences}")
    print(f"total unique sentences: {len(unique)}")
    print(f"len unique sentences with label: {len(unique1)}")

