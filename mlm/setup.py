"""
Script to create relevant directories necessary for data and training
"""
import os

import nltk

from mlm.local_constants import DATA_DIR, MODEL_DIR, FILTERED_SCRAPE_DIR,\
    SCRAPED_DATA_DIR, PREP_DATA_DIR, PLOTS_DIR, RESULTS_DIR, PROJECT_ROOT

# Download nltk punkt and stopwords relevant for data preprocessing
nltk.download('punkt')
nltk.download('stopwords')

if not os.path.exists(DATA_DIR):
    # Create a new directory because it does not exist
    os.makedirs(DATA_DIR)
    print("Create data dir!")

if not os.path.exists(os.path.join(PROJECT_ROOT, 'metadata')):
    # Create a new directory because it does not exist
    os.makedirs(os.path.join(PROJECT_ROOT, 'metadata'))
    print("Create metadata dir!")

if not os.path.exists(MODEL_DIR):
    # Create a new directory because it does not exist
    os.makedirs(MODEL_DIR)
    print("Create models dir!")

if not os.path.exists(FILTERED_SCRAPE_DIR):
    # Create a new directory because it does not exist
    os.makedirs(FILTERED_SCRAPE_DIR)
    print("Create filtered data dir!")

if not os.path.exists(SCRAPED_DATA_DIR):
    # Create a new directory because it does not exist
    os.makedirs(SCRAPED_DATA_DIR)
    print("Create scrape data dir!")

if not os.path.exists(PLOTS_DIR):
    # Create a new directory because it does not exist
    os.makedirs(PLOTS_DIR)
    print("Create plots dir!")

if not os.path.exists(PREP_DATA_DIR):
    # Create a new directory because it does not exist
    os.makedirs(PREP_DATA_DIR)
    print("Create preprocessed data dir!")

if not os.path.exists(os.path.join(PROJECT_ROOT, 'data/internal_data')):
    # Create a new directory because it does not exist
    os.makedirs(os.path.join(PROJECT_ROOT, 'data/internal_data'))
    print("Create internal data dir!")

if not os.path.exists(RESULTS_DIR):
    # Create a new directory because it does not exist
    os.makedirs(RESULTS_DIR)
    print("Create results dir!")
