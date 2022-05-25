import datetime
import sys
import traceback
from datetime import datetime
import os
import argparse
import numpy as np
import torch
from opacus import PrivacyEngine
from opacus.utils.batch_memory_manager import BatchMemoryManager
# from helpers import validate_model, evaluate, accuracy, TimeCode
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, DataCollatorForTokenClassification, \
    AutoModelForTokenClassification, Trainer, \
    TrainingArguments, HfArgumentParser, AutoModelForMaskedLM, DataCollatorForWholeWordMask, \
    DataCollatorForLanguageModeling

from data_utils.preprocess_public_scraped_data import split_to_sentences, split_train_dev, \
    TokenizedSentencesDataset
os.environ["WANDB_DISABLED"] = "true"



parser = argparse.ArgumentParser()
parser.add_argument("--epsilon", type=float, default=1, help="privacy parameter epsilon")
parser.add_argument("--lot_size", type=int, default=16,
                    help="Lot size specifies the sample size of which noise is injected into")
parser.add_argument("--batch_size", type=int, default=16,
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
parser.add_argument("--evaluate_steps", type=int, default=200,
                    help="evaluate model accuracy after number of steps")
parser.add_argument("--logging_steps", type=int, default=100,
                    help="Log model accuracy after number of steps")
parser.add_argument("--model_name", type=str, default='jonfd/electra-small-nordic',
                    help="foundation model from huggingface")
parser.add_argument("--save_steps", type=int, default=1000,
                    help="save checkpoint after number of steps")
parser.add_argument("--max_length", type=int, default=100,
                    help="Max length for a text input")

parser.add_argument("--use_fp16", type=bool, default=False,
                    help="Set to True, if your GPU supports FP16 operations")
parser.add_argument("--whole_word_mask", type=bool, default=True,
                    help="If set to true, whole words are masked")
parser.add_argument("--mlm_prob", type=float, default=0.15,
                    help="Probability that a word is replaced by a [MASK] token")


args = parser.parse_args()

# get data
sentences = split_to_sentences(data_path='data/da_DK_subset.json')

train_sentences, dev_sentences = split_train_dev(sentences=sentences)


# Load the model
model = AutoModelForMaskedLM.from_pretrained(args.model_name)
tokenizer = AutoTokenizer.from_pretrained(args.model_name)

output_dir = "output/{}-{}".format(args.model_name.replace("/", "_"),  datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
print("Save checkpoints to:", output_dir)

train_dataset = TokenizedSentencesDataset(train_sentences, tokenizer, args.max_length)
dev_dataset = TokenizedSentencesDataset(dev_sentences, tokenizer, args.max_length, cache_tokenization=True) if len(dev_sentences) > 0 else None

if args.whole_word_mask:
    data_collator = DataCollatorForWholeWordMask(tokenizer=tokenizer, mlm=True, mlm_probability=args.mlm_prob)
else:
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=args.mlm_prob)

training_args = TrainingArguments(
    output_dir=output_dir,
    overwrite_output_dir=True,
    num_train_epochs=args.epochs,
    evaluation_strategy='steps',
    per_device_train_batch_size=args.batch_size,
    eval_steps=args.evaluate_steps,
    save_steps=args.save_steps,
    logging_steps=args.logging_steps,
    save_total_limit=1,
    prediction_loss_only=True,
    fp16=args.use_fp16,
    push_to_hub=False
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_dataset,
    eval_dataset=dev_dataset
)

print("Save tokenizer to:", output_dir)
tokenizer.save_pretrained(output_dir)

trainer.train()

print("Save model to:", output_dir)
model.save_pretrained(output_dir)

print("Training done")




