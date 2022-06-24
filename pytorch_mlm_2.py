import math
import os

import numpy as np
import torch
import torch.optim as optim
from datasets import load_dataset
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import AutoTokenizer, DataCollatorForLanguageModeling, \
    BertForMaskedLM

from local_constants import DATA_DIR
from utils.input_args import MLMArgParser

mlm_parser = MLMArgParser()
args = mlm_parser.parser.parse_args()


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
            max_length=args.max_length,
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
train_loader = DataLoader(dataset=training_dataset_wrapped, batch_size=4, collate_fn=data_collator)

# get model
model = BertForMaskedLM.from_pretrained(MODEL_NAME)

loss_fct = nn.CrossEntropyLoss()

optimizer = optim.AdamW(model.parameters(), lr=args.lr)
model_losses = []
CE_losses = []
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


for epoch in range(args.epochs):
    model.train()
    model = model.to(device)
    loss_len = 0
    running_loss_1 = 0.0
    running_loss_2 = 0.0
    for i, batch in enumerate(train_loader):

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(input_ids=batch["input_ids"].to(device),
                       attention_mask=batch["attention_mask"].to(device),
                       labels=batch["labels"].to(device))

        # np.save('data/logits_manual', outputs.logits.cpu().detach().numpy())
        # np.save('data/labels_manual', batch['labels'].cpu().detach().numpy())

       # -100 index = padding token
       #  loss = loss_fct(outputs.logits.cpu().view(-1, 119547), batch['labels'].cpu().view(-1))
        loss = loss_fct(outputs.logits.view(-1, 119547), batch["labels"].to(device).view(-1))
        CE_losses.append(loss.item())
        # print(outputs.loss.item())
        # loss = criterion(outputs, batch['labels'])
        model_loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]
        model_losses.append(model_loss)
        loss.backward()
        optimizer.step()


        # print statistics
        if not math.isnan(loss):
            loss_len += 1
            running_loss_1 += outputs.loss.item()
            running_loss_2 += loss.item()
        else:
            print("loss is nan")
        if i % 600 == 599:    # print every 600 mini-batches
            print(f'CE loss: {loss}')
            print(f'model loss: {outputs.loss.item()}')
            print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss_1 / loss_len :.3f}')
            print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss_2 / loss_len :.3f}')
            running_loss_1 = 0.0
            running_loss_2 = 0.0
            loss_len = 0

print(CE_losses)
print(model_losses)
