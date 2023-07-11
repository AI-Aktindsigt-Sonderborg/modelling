import re
from typing import Tuple, List

from langdetect import detect_langs
from nltk.tokenize import sent_tokenize


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
