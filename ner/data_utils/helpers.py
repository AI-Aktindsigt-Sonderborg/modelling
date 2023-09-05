import re
import traceback
from typing import Tuple, List

from langdetect import detect_langs
from nltk.tokenize import sent_tokenize

from ner.data_utils.custom_dataclasses import DataPrepConstants


def split_sentences_bilou(
    data: str,
    sentence_splitter: object) -> Tuple[List[str], List[str]]:
    """
    Takes raw html input text and splits to sentences

    :param dict data_dict: Dictionary containing raw input text from filtered
        scrape file
    :param str filename: filename
    :param PunktSentenceTokenizer sentence_splitter: Tokenizer to split
        sentences
    :param List[str] disapproved_sentences: list of disapproved sentences
    :return: List of sentences, list of disapproved sentences
    :rtype: Tuple[List[str], List[str]]
    """
    # Read canonical string representation of the object as we need special
    # regex characters
    prep_text = repr(data)

    # Define list of special chars to replace
    special_chars = {
        'specials': ['\\r', '\\t', '\\n', '\\xa0', ' | ', '|', '*'],
        'citations': ['”', '"', "'"],
        'spellings': ['NemID', 'MitID', 'minSU', 'LinkedIn']}

    for case in special_chars['specials']:
        prep_text = prep_text.replace(case, ' ')

    for case in special_chars['citations']:
        prep_text = prep_text.replace(case, '')

    for case in special_chars['spellings']:
        if case in prep_text:
            prep_text = prep_text.replace(case, case.lower())

    # remove consecutive whitespaces
    prep_text = re.sub(' +', ' ', prep_text)

    # Find wrong breaks like "." or "?" followed by capital letters
    find_wrong_break_ids = [
        (m.start(0), m.end(0)) for m in
        re.finditer(r'\?[A-Z]|\.[A-Z]', prep_text)]

    # Add whitespace after each wrong break
    if len(find_wrong_break_ids) > 0:
        increment = 1
        for idx in find_wrong_break_ids:
            prep_text = prep_text[:idx[0] + increment] + '\n' \
                        + prep_text[idx[0] + increment:]
            increment += 1

    # split text with sentence_splitter and join by newlines
    text_splitted = (
        '\n'.join(sentence_splitter.tokenize(prep_text)))

    # ToDo: Consider spacy to tokenize: https://github.com/alvenirai/rnd-ataja/blob/858e7e4ee5f35f2809ae396376b36398fe1c42c0/ataja/nlp/ataja_inference/modeling_ataja.py#L336
    text_splitted2 = sent_tokenize(prep_text, language='danish')

    # split text again by newlines
    sentences = re.split('\n', text_splitted)

    new_sentences = []
    new_sentences2 = []

    for j, sentence in enumerate(text_splitted2):
        new_sentences2.append(sentence + "\n")

    for j, sentence in enumerate(sentences):
        new_sentences.append(sentence + "\n")

    return new_sentences, text_splitted2


def filter_language(string: str, approved_languages: List[str]):
    try:
        language_codes = detect_langs(string)

        languages = [lang.lang for lang in language_codes]
        result = any(item for item in languages if item in approved_languages)
        if result:
            probs = [lang.prob for lang in language_codes if
                     lang.lang in approved_languages]
            return True, language_codes

        return False, language_codes
    except:
        print('No language detected')
        return False, None


def map_bilou_to_bio(data):
    bilou_to_bio_mapping = {"U-": "B-", "L-": "I-"}
    bio_mapped = [bilou_to_bio_mapping[tag[:2]] + tag[2:] if tag[:2] in ["U-",
                                                                         "L-"] else tag
                  for tag in data]
    return bio_mapped


def delete_duplicate_annotations(data, filtered_list):
    """
    Remove duplicate annotations, i.e. annotatione with same content, annotation and indices
    Parameters
    ----------
    data:
        Dictionary containing a single annotation
    filtered_list:
        List of existing approved annotations

    Returns:
        updated data, filtered_list and whether annotation was deleted
    -------
    """
    error: int = 0
    if data["annotation"]["state"] != "deleted":
        filtered_annotation = {
            "content": data["annotation"]["content"],
            "annotation": data["annotation"]["annotation"],
            "start": data["annotation"]["start"],
            "end": data["annotation"]["end"],
        }
        if filtered_annotation in filtered_list:
            error = 1
            data["annotation"]["state"] = "deleted"
            print("deleted duplicate annotation")
        else:
            filtered_list.append(filtered_annotation)

    return data, filtered_list, error


def reindexing_first_or_last(data):
    """
    Method to remove wrong first or last characters from annotations
    Parameters
    ----------
    data:
        Dictionary containing a single annotation

    Returns:
        Updated data and whether error was detected
    -------
    """

    error = 0
    annotated_content = data["annotation"]["content"]
    annotated_content_last = annotated_content[-1]
    annotated_content_first = annotated_content[0]
    start_index = data["annotation"]["start"]
    end_index = data["annotation"]["end"]

    while annotated_content_last.isspace() | (annotated_content_last == " "):
        error = 1
        end_index = end_index - 1
        data["annotation"]["end"] -= 1
        annotated_content = annotated_content[:-1]
        data["annotation"]["content"] = annotated_content
        annotated_content_last = annotated_content[-1]

    while annotated_content_last in DataPrepConstants.special_characters:
        error = 1
        end_index = end_index - 1
        data["annotation"]["end"] -= 1
        annotated_content = annotated_content[:-1]
        data["annotation"]["content"] = annotated_content
        try:
            annotated_content_last = annotated_content[-1]
        except IndexError:
            data["annotation"]["state"] = "deleted"
            break

    while annotated_content_first.isspace() | (annotated_content_first == " "):
        error = 1
        start_index = start_index + 1
        data["annotation"]["start"] += 1
        annotated_content = annotated_content[1:]
        data["annotation"]["content"] = annotated_content
        annotated_content_first = annotated_content[0]

    while annotated_content_first in DataPrepConstants.special_characters:
        error = 1
        start_index = start_index + 1
        data["annotation"]["start"] += 1
        annotated_content = annotated_content[1:]
        data["annotation"]["content"] = annotated_content
        try:
            annotated_content_first = annotated_content[0]
        except IndexError:
            try:
                data["annotation"]["state"] = "deleted"
                # print(f"removing annotation {annotation_num} from {document_num + 1}")
                # del current_page_annotations[annotation_num]
                # print(f"removed annotation {annotation_num} from {document_num + 1}")
            except Exception:
                print("data might already be removed")
            break

    return data, error


def handle_wrong_annotation_end_letter(data: dict, text: str, patterns: dict,
                                       end_index):
    error = 0
    for pattern, replacement in patterns.items():
        if len(text) > end_index + len(pattern):
            if text[end_index: end_index + len(pattern)] == pattern:
                error = 1
                data["annotation"]["content"] = (
                    data["annotation"]["content"] + replacement
                )
                data["annotation"]["end"] = (
                        data["annotation"]["end"] + len(replacement)
                        )
                return data, error

    return data, error



def fix_skewed_indices(text, data, reindex_counter,
                       skew_bottom: int = -10, skew_top: int = 10):
    """
    Method to reindex wrong indices
    Parameters
    ----------
    text:
        Original input text
    data:
        Dictionary containing a single annotation
    reindex_counter:
        reindex counter
    Returns
    -------

    """
    error = 0
    start_index = data["annotation"]["start"]
    end_index = data["annotation"]["end"]

    annotated_content = data["annotation"]["content"]
    true_content = text[start_index:end_index]

    if true_content.lower() != annotated_content.lower():

        error = 1
        skewness_list = list(range(skew_bottom, skew_top))
        true_content_skewed = true_content
        for skewness in skewness_list:
            try:
                true_content_skewed = text[
                                      start_index + skewness: end_index + skewness]
            except IndexError:
                print("skewness search error")
                print(traceback.format_exc())
            if true_content_skewed.lower() == annotated_content.lower():
                data["annotation"]["start"] = (start_index + skewness)
                data["annotation"]["end"] = (end_index + skewness)
                true_content = text[
                               start_index + skewness: end_index + skewness]
                reindex_counter += 1
                break
            continue

    return data, true_content, reindex_counter, error
