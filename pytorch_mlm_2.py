import os

import numpy as np
import torch.optim as optim
from datasets import load_dataset
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import AutoTokenizer, DataCollatorForLanguageModeling, \
    BertForMaskedLM

from local_constants import DATA_DIR


class DatasetWrapper(Dataset):

    def __init__(self, dataset):
        self.dataset = dataset

    def __getitem__(self, item):
        item = int(item)
        return self.dataset[item]

    def __len__(self):
        return len(self.dataset)


def tokenize_and_wrap_data(data: Dataset, tokenizer):

    def tokenize_function(examples):
        # Remove empty lines
        examples['text'] = [
            line for line in examples['text'] if len(line) > 0 and not line.isspace()
        ]

        return tokenizer(
            examples['text'],
            padding='max_length',
            truncation=True,
            max_length=32,
            # We use this option because DataCollatorForLanguageModeling (see below) is more efficient when it
            # receives the `special_tokens_mask`.
            return_special_tokens_mask=True,
        )

    tokenized = data.map(
        tokenize_function,
        batched=True,
        num_proc=1,
        remove_columns=['text'],
        load_from_cache_file=False,
        desc="Running tokenizer on dataset line_by_line",
    )

    wrapped = DatasetWrapper(tokenized)

    return wrapped


MODEL_NAME = 'NbAiLab/nb-bert-base'

# get tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True,
                                                   mlm_probability=0.15)

# Get training data from huggingface
train_data = load_dataset('json', data_files=os.path.join(DATA_DIR, 'train.json'), split='train')

training_dataset = tokenize_and_wrap_data(train_data, tokenizer)
training_dataset_wrapped = DatasetWrapper(training_dataset)
train_loader = DataLoader(dataset=training_dataset_wrapped, batch_size=2, collate_fn=data_collator)

# get model
model = BertForMaskedLM.from_pretrained(MODEL_NAME)


criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=0.002)
losses = []
for epoch in range(20):

    running_loss = 0.0
    for i, batch in enumerate(train_loader, 0):

        # inputs, labels = data

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(input_ids=batch["input_ids"],
                       attention_mask=batch["attention_mask"],
                       labels=batch["labels"])

        # print(outputs.loss.item())
        # loss = criterion(outputs, batch['labels'])
        outputs.loss.backward()
        optimizer.step()

        # print statistics
        running_loss += outputs.loss.item()
        if i % 200 == 199:    # print every 2000 mini-batches
            losses.append(outputs.loss.item())
            print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 200:.3f}')
            running_loss = 0.0

print(losses)

