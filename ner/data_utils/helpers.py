import re
from typing import Tuple, List

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
    disapproved_sentences = []
    # Read canonical string representation of the object as we need special
    # regex characters
    prep_text = repr(data)

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

    text_splitted2 = '\n'.join(sent_tokenize(prep_text, language='danish'))

    # split text again by newlines
    sentences = re.split('\n', text_splitted)

    new_sentences = []
    new_sentences2 = []

    for j, sentence in enumerate(text_splitted2):
        new_sentences2.append(sentence + "\n")

    for j, sentence in enumerate(sentences):
        # Discard sentences where lower case letters are followed by capital
        # letters
#        if len([(m.start(0), m.end(0)) for m in re.finditer("[a-z][A-Z]",
                                           #             sentence)]) > 0:
 #           disapproved_sentences.append(
  #              f'{j} - {sentence}')
   #     else:
        new_sentences.append(sentence + "\n")


    return new_sentences, new_sentences2
