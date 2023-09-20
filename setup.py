"""
Script to create relevant directories necessary for data and training
"""
import os

import nltk

from root_constants import SC_ROOT, MLM_ROOT, NER_ROOT

# Download nltk punkt and stopwords relevant for data preprocessing
nltk.download("punkt")
nltk.download("stopwords")
nltk.download("averaged_perceptron_tagger")
nltk.download("tagsets")
nltk.download("universal_tagset")

GENERIC_DIRS = ["logs", "tests"]
PROJECT_DIRS = [MLM_ROOT, NER_ROOT, SC_ROOT]
PROJECT_GENERIC_DIRS = [
    "data",
    "metadata",
    "models",
    "plots",
    "data/preprocessed_data",
    "results",
]

for generic_dir in GENERIC_DIRS:
    os.makedirs(generic_dir, exist_ok=True)

for project_dir in PROJECT_DIRS:
    for project_generic_dir in PROJECT_GENERIC_DIRS:
        os.makedirs(os.path.join(project_dir, project_generic_dir), exist_ok=True)


