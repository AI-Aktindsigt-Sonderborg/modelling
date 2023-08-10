# pylint: disable=line-too-long
import os
import re
import sys
import traceback
from typing import List

import nltk

from ner.data_utils.custom_dataclasses import DataPrepConstants
from ner.data_utils.helpers import (
    filter_language,
    delete_duplicate_annotations,
    reindexing_first_or_last,
    fix_skewed_indices,
    handle_wrong_annotation_end_letter,
)
from mlm.local_constants import DATA_DIR, CONF_DATA_DIR
from shared.utils.helpers import read_json, read_json_lines, write_json_lines

sentence_splitter = nltk.data.load("tokenizers/punkt/danish.pickle")

def create_sentences_from_one_document(
    input_data: dict,
    data_number: int,
    file_number: int
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

                #             for pattern, replacement in DataPrepConstants.tag_replacements.items():
                #                 sentence = re.sub(pattern, replacement, sentence)
                # is_danish, language_codes = filter_language(
                #                         string=sentence, approved_languages=["da", "no"]
                #                     )
                #                 if not is_danish:
                #                     not_danish_counter += 1
                #                     print("---")
                #                     print(language_codes)
                #                     print("-")
                #                     print(sentence)
                #                     print("---\n")
                output_data.append(
                    {
                        "text": sentence,
                        "file_no": file_number,
                        "doc_id": input_data["document_id"],
                        "page_no": page_num,
                        "sentence_no": i,
                        "origin_line_no": data_number + 1,
                    }
                )

    return output_data, [
        total_sentence,
        not_danish_counter,
    ]


if __name__ == "__main__":
    total_sentences: int = 0
    total_not_danish_counter: int = 0

    file_list = sorted([f.split(".")[0] for f in os.listdir(CONF_DATA_DIR) if
             os.path.isfile(os.path.join(CONF_DATA_DIR, f)) and f.startswith("raw")])


    out_data: List[dict] = []
    for j, file in enumerate(file_list):

        raw_data = read_json_lines(input_dir=CONF_DATA_DIR, filename=file)


        for i, obs in enumerate(raw_data):
            single_obs_data, errors = create_sentences_from_one_document(
                input_data=obs, data_number=i, file_number=j
            )
            total_sentences += errors[0]
            total_not_danish_counter += errors[1]

            out_data.extend(single_obs_data)

    write_json_lines(out_dir=CONF_DATA_DIR, data=out_data, filename="all_sentences")

    unique = list(set([x['text'] for x in out_data]))

    write_json_lines(out_dir=CONF_DATA_DIR, data=unique, filename="unique_sentences")

    print(f"sentences not danish: {total_not_danish_counter}")
    print(f"total sentences: {total_sentences}")
    print(f"total sentences: {len(unique)}")
