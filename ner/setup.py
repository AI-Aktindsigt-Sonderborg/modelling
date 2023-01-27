"""
Script to create relevant directories necessary for data and training
"""
import os

from ner.local_constants import DATA_DIR, MODEL_DIR, PREP_DATA_DIR,\
    PROJECT_ROOT, PLOTS_DIR

import nltk
nltk.download('punkt')
nltk.download('stopwords')

if not os.path.exists(DATA_DIR):
    # Create a new directory because it does not exist
    os.makedirs(DATA_DIR)
    print("Create data dir!")

if not os.path.exists(MODEL_DIR):
    # Create a new directory because it does not exist
    os.makedirs(MODEL_DIR)
    print("Create models dir!")

if not os.path.exists(os.path.join(PROJECT_ROOT, 'plots')):
    # Create a new directory because it does not exist
    os.makedirs(os.path.join(PROJECT_ROOT, 'plots'))
    print("Create plots dir!")

if not os.path.exists(PREP_DATA_DIR):
    # Create a new directory because it does not exist
    os.makedirs(PREP_DATA_DIR)
    print("Create preprocessed data dir!")
