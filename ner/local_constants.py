import os

PROJECT_ROOT = os.path.dirname(os.path.abspath(
    '../modelling/ner/local_constants.py'))
MODEL_DIR = os.path.join(PROJECT_ROOT, 'models')
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
PLOTS_DIR = os.path.join(PROJECT_ROOT, 'plots')
PREP_DATA_DIR = os.path.join(DATA_DIR, 'preprocessed_data')
METADATA_DIR = os.path.join(PROJECT_ROOT, 'metadata')
