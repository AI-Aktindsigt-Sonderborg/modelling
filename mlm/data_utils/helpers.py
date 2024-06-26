import re
from typing import List, Tuple


def split_sentences(
    data_dict: dict,
    filename: str,
    sentence_splitter: object,
    disapproved_sentences: List[str]) -> Tuple[List[str], List[str]]:
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
    prep_text = repr(data_dict['text'])

    # Define list of special chars to replace
    special_chars = {'specials': ['\\r', '\\t', '\\n', '\\xa0', ' | ', '|', '*'],
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
            prep_text = prep_text[:idx[0] + increment] + '\n'\
                        + prep_text[idx[0] + increment:]
            increment += 1

    # split text with sentence_splitter and join by newlines
    text_splitted = (
        '\n'.join(sentence_splitter.tokenize(prep_text)))

    # split text again by newlines
    sentences = re.split('\n', text_splitted)

    new_sentences = []
    for j, sentence in enumerate(sentences):
        # Discard sentences where lower case letters are followed by capital
        # letters
        if len([(m.start(0), m.end(0)) for m in re.finditer("[a-z][A-Z]",
                                                            sentence)]) > 0:
            disapproved_sentences.append(
                f'{filename.split("_")[0]} - {data_dict["id"]}'
                f' - {j} -{sentence}')
        else:
            new_sentences.append(sentence)

    return new_sentences, disapproved_sentences
