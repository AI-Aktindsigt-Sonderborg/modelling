"""
Script to create relevant directories necessary for data and training
"""
import os

from mlm.local_constants import DATA_DIR, FILTERED_SCRAPE_DIR, \
    SCRAPED_DATA_DIR, PREP_DATA_DIR_OPEN, CONF_DATA_DIR

dirs_to_create = [FILTERED_SCRAPE_DIR, SCRAPED_DATA_DIR, PREP_DATA_DIR_OPEN,
                  CONF_DATA_DIR, os.path.join(DATA_DIR, 'internal_data')]

for dir_to_create in dirs_to_create:
    os.makedirs(dir_to_create, exist_ok=True)
