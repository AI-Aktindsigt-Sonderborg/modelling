import os
from root_constants import ABS_ROOT

PROJECT_ROOT = os.path.join(ABS_ROOT, 'ner')
MODEL_DIR = os.path.join(PROJECT_ROOT, 'models')
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
PLOTS_DIR = os.path.join(PROJECT_ROOT, 'plots')
PREP_DATA_DIR = os.path.join(DATA_DIR, 'preprocessed_data')
METADATA_DIR = os.path.join(PROJECT_ROOT, 'metadata')
