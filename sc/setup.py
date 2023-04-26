"""
Script to create relevant directories necessary for data and training
"""
import os

from sc.local_constants import CLASS_DATA_DIR

DIRS_TO_CREATE = [CLASS_DATA_DIR,
                  os.path.join(CLASS_DATA_DIR, 'raw'),
                  os.path.join(CLASS_DATA_DIR, 'processed')]

for dir_to_create in DIRS_TO_CREATE:
    os.makedirs(dir_to_create, exist_ok=True)
