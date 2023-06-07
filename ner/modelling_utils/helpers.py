from typing import List
from collections import OrderedDict

def align_labels_with_tokens(labels, word_ids):
    new_labels = []
    current_word = None
    for word_id in word_ids:
        if word_id != current_word:
            # Start of a new word!
            current_word = word_id
            label = -100 if word_id is None else labels[word_id]
            new_labels.append(label)
        elif word_id is None:
            # Special token
            new_labels.append(-100)
        else:
            # Same word as previous token
            label = labels[word_id]
            # If the label is B-XXX we change it to I-XXX
            if label % 2 == 1:
                label += 1
            new_labels.append(label)

    return new_labels

def get_label_list(ner_entities: List[str]):
    # ner_entities = ["PERSON", "LOKATION", "ADRESSE", "HELBRED", "ORGANISATION", "KOMMUNE", "TELEFONNUMMER"]
    id2label = {}
    label_list = ["O"]
    for entity in ner_entities:
        begin = "B-" + entity
        inside = "I-" + entity
        last = "L-" + entity
        unique = "U-" + entity
        label_list.extend([begin, inside, last, unique])

    for i, label in enumerate(label_list):
        id2label[i] = label
        
    label2id = OrderedDict()
    for i, label in enumerate(label_list):
        label2id[label] = i

    return label_list, id2label, label2id
