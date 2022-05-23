import sys
import traceback

import argparse
import numpy as np
import torch
from opacus import PrivacyEngine
from opacus.utils.batch_memory_manager import BatchMemoryManager
# from helpers import validate_model, evaluate, accuracy, TimeCode
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, DataCollatorForTokenClassification, AutoModelForTokenClassification, Trainer, \
    TrainingArguments, HfArgumentParser


parser = argparse.ArgumentParser()
parser.add_argument("--epsilon", type=float, default=1, help="privacy parameter epsilon")
parser.add_argument("--lot_size", type=int, default=16, help="Lot size specifies the sample size of which noise is injected into")
parser.add_argument("--batch_size", type=int, default=8,
                    help="Batch size specifies the sample size of which the gradients are computed. Depends on memory available")
parser.add_argument("--delta", type=float, default=0.00002,
                    help="privacy parameter delta")
parser.add_argument("--max_grad_norm", type=float, default=1.2,
                    help="maximum norm to clip gradient")
parser.add_argument("--epochs", type=int, default=20,
                    help="Number of epochs to train model")
parser.add_argument("--lr", type=float, default=0.00002,
                    help="Learning rate")
parser.add_argument("--weight_decay", type=float, default=0.01,
                    help="Weight decay")
# ToDo: evaluate steps must be smaller than number of steps in each epoch
parser.add_argument("--evaluate_steps", type=int, default=1000,
                    help="evaluate model accuracy after number of steps")
parser.add_argument("--model_name", type=str, default=1000,
                    help="foundation model from huggingface")


args = parser.parse_args()