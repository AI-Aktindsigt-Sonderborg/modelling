import os
from root_constants import NER_ROOT

MODEL_DIR = os.path.join(NER_ROOT, 'models')
DATA_DIR = os.path.join(NER_ROOT, 'data')
PLOTS_DIR = os.path.join(NER_ROOT, 'plots')
METADATA_DIR = os.path.join(NER_ROOT, 'metadata')
PREP_DATA_DIR = os.path.join(DATA_DIR, 'preprocessed_data')

DANE_TO_AKT_LABEL_MAPPING = {"O": "O",
                             "B-ORG": "B-ORGANISATION",
                             "I-ORG": "I-ORGANISATION",
                             "B-LOC": "B-LOKATION",
                             "I-LOC": "I-LOKATION",
                             "B-PER": "B-PERSON",
                             "I-PER": "I-PERSON", "B-MISC": "O",
                             "I-MISC": "O"}
