"""
Script to create relevant directories necessary for data and training
"""
import os

import nltk

from ner.local_constants import DATA_DIR, MODEL_DIR, PREP_DATA_DIR, \
    PLOTS_DIR

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

if not os.path.exists(PLOTS_DIR):
    # Create a new directory because it does not exist
    os.makedirs(PLOTS_DIR)
    print("Create plots dir!")

if not os.path.exists(PREP_DATA_DIR):
    # Create a new directory because it does not exist
    os.makedirs(PREP_DATA_DIR)
    print("Create preprocessed data dir!")
