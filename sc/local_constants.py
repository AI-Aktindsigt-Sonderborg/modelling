import os

from root_constants import SC_ROOT

MODEL_DIR = os.path.join(SC_ROOT, 'models')
DATA_DIR = os.path.join(SC_ROOT, 'data')
PLOTS_DIR = os.path.join(SC_ROOT, 'plots')
RESULTS_DIR = os.path.join(SC_ROOT, 'results')
METADATA_DIR = os.path.join(SC_ROOT, 'metadata')
PREP_DATA_DIR = os.path.join(DATA_DIR, 'preprocessed_data')

# specific to SC
# SCRAPED_DATA_DIR = os.path.join(DATA_DIR, 'scraped_data')
# FILTERED_SCRAPE_DIR = os.path.join(DATA_DIR, 'filtered_scrape')
CLASS_DATA_DIR = os.path.join(DATA_DIR, 'classified_data')
