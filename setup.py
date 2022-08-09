"""
Script to create relevant directories necessary for data and training
"""
import os

import nltk

from local_constants import DATA_DIR, MODEL_DIR, FILTERED_SCRAPE_DIR

nltk.download('punkt')

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

if not os.path.exists('data/scraped_data'):
    # Create a new directory because it does not exist
    os.makedirs('data/scraped_data')
    print("Create scrape data dir!")

if not os.path.exists('data/internal_data'):
    # Create a new directory because it does not exist
    os.makedirs('data/internal_data')
    print("Create internal data dir!")
