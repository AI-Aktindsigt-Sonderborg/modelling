from dataclasses import dataclass


@dataclass(frozen=True)
class DataPrepConstants:
    dane_to_akt_label_mapping = {"O": "O", "B-ORG": "B-ORGANISATION",
                                 "I-ORG": "I-ORGANISATION",
                                 "B-LOC": "B-LOKATION", "I-LOC": "I-LOKATION",
                                 "B-PER": "B-PERSON", "I-PER": "I-PERSON",
                                 "B-MISC": "O", "I-MISC": "O"}

    special_characters = [
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
        " ",
    ]

    annotation_replacements = {
        "s ": "s",
        "'s ": "s",
        "k ": "k",
        "e": "e",
        "r": "r",
        "n": "n",
        "s": "s",
        "i": "i"
        }

    standard_ner_entities = ["PERSON", "LOKATION", "ADRESSE", "HELBRED",
                         "ORGANISATION", "KOMMUNE", "TELEFONNUMMER"]

    none_ner_entities = ["EMAIL", "PRIS", "DATO", "URL", "CVR"]
