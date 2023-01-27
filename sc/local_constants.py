import os

PROJECT_ROOT = os.path.dirname(
    os.path.abspath('../modelling/sc/local_constants.py'))
MODEL_DIR = os.path.join(PROJECT_ROOT, 'models')
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
PLOTS_DIR = os.path.join(PROJECT_ROOT, 'plots')
SCRAPED_DATA_DIR = os.path.join(DATA_DIR, 'scraped_data')
FILTERED_SCRAPE_DIR = os.path.join(DATA_DIR, 'filtered_scrape')
PREP_DATA_DIR = os.path.join(DATA_DIR, 'preprocessed_data')
CLASS_DATA_DIR = os.path.join(DATA_DIR, 'classified_data')
RESULTS_DIR = os.path.join(PROJECT_ROOT, 'results')
