"""
Script to create relevant directories necessary for data and training
"""
import os

import nltk

from local_constants import DATA_DIR, CONFIG_DIR, MODEL_DIR, FILTERED_SCRAPE_DIR

nltk.download('punkt')

if not os.path.exists(CONFIG_DIR):
    # Create a new directory because it does not exist
    os.makedirs(CONFIG_DIR)
    print("Create config dir!")

if not os.path.exists(DATA_DIR):
    # Create a new directory because it does not exist
    os.makedirs(DATA_DIR)
    print("Create data dir!")

if not os.path.exists(MODEL_DIR):
    # Create a new directory because it does not exist
    os.makedirs(MODEL_DIR)
    print("Create models dir!")

if not os.path.exists(FILTERED_SCRAPE_DIR):
    # Create a new directory because it does not exist
    os.makedirs(FILTERED_SCRAPE_DIR)
    print("Create filtered data dir!")

if not os.path.exists('data/new_scrape'):
    # Create a new directory because it does not exist
    os.makedirs('data/new_scrape')
    print("Create new scrape data dir!")

if not os.path.exists('data/internal_data'):
    # Create a new directory because it does not exist
    os.makedirs('data/internal_data')
    print("Create internal data dir!")
