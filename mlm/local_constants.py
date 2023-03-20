import os
from root_constants import MLM_ROOT

MODEL_DIR = os.path.join(MLM_ROOT, 'models')
DATA_DIR = os.path.join(MLM_ROOT, 'data')
PLOTS_DIR = os.path.join(MLM_ROOT, 'plots')
RESULTS_DIR = os.path.join(MLM_ROOT, 'results')
METADATA_DIR = os.path.join(MLM_ROOT, 'metadata')

# specific to mlm
SCRAPED_DATA_DIR = os.path.join(DATA_DIR, 'scraped_data')
FILTERED_SCRAPE_DIR = os.path.join(DATA_DIR, 'filtered_scrape')
PREP_DATA_DIR = os.path.join(DATA_DIR, 'preprocessed_data')
