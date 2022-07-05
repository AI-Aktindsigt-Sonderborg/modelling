import os
from local_constants import DATA_DIR, CONFIG_DIR, MODEL_DIR, FILTERED_SCRAPE_DIR

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
