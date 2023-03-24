"""
Script to create relevant directories necessary for data and training
"""
import os

import nltk

from root_constants import SC_ROOT, MLM_ROOT, NER_ROOT

# Download nltk punkt and stopwords relevant for data preprocessing
nltk.download('punkt')
nltk.download('stopwords')

PROJECT_DIRS = [MLM_ROOT, NER_ROOT, SC_ROOT]
GENERIC_DIRS = ['data', 'metadata', 'models', 'plots', 'preprocessed_data',
                'results']

for project_dir in PROJECT_DIRS:
    for generic_dir in GENERIC_DIRS:
        os.makedirs(os.path.join(project_dir, generic_dir), exist_ok=True)
