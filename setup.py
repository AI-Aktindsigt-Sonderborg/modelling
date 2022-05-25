import os
from local_constants import DATA_DIR, CONFIG_DIR, OUTPUT_DIR

if not os.path.exists(CONFIG_DIR):
    # Create a new directory because it does not exist
    os.makedirs(CONFIG_DIR)
    print("Create config dir!")

if not os.path.exists(DATA_DIR):
    # Create a new directory because it does not exist
    os.makedirs(DATA_DIR)
    print("Create data dir!")

if not os.path.exists(OUTPUT_DIR):
    # Create a new directory because it does not exist
    os.makedirs(OUTPUT_DIR)
    print("Create output dir!")
