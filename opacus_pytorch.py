import os
import sys
from datasets import load_dataset
from torch.utils.data import Dataset
import numpy as np
from opacus import PrivacyEngine
from opacus.utils.batch_memory_manager import BatchMemoryManager
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, DataCollatorForTokenClassification, \
    AutoModelForTokenClassification, Trainer, TrainingArguments, DataCollatorForLanguageModeling, \
    BertForMaskedLM

from opacus.validators import ModuleValidator

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

def validate_model(model):
    errors = ModuleValidator.validate(model, strict=False)
    if errors:
        print("Model sucks for DP :( Please fix errors....")
        print(errors)
        sys.exit(0)
    else:
        print("Model is totally compatible yaaaay")

def accuracy(preds, labels):
    # a = preds == labels
    return (preds == labels).mean()

def evaluate(model, val_dataloader):
    model.eval()

    loss_arr = []
    accuracy_arr = []

    for batch in val_dataloader:
        # compute output
        output = model(input_ids=batch["input_ids"].to('cuda'),
                       attention_mask=batch["attention_mask"].to('cuda'),
                       labels=batch["labels"].to('cuda'))

        loss, logits = output[:2]

        preds = np.argmax(logits.detach().cpu().numpy(), axis=-1)
        labels = batch["labels"].cpu().numpy()

        labels_flat = labels.flatten()
        preds_flat = preds.flatten()
        filtered = [[xv, yv] for xv, yv in zip(labels_flat, preds_flat) if xv != -100]

        labels_filtered = np.array([x[0] for x in filtered])
        preds_filtered = np.array([x[1] for x in filtered])
        loss_arr.append(loss.item())
        accuracy_arr.append(accuracy(preds_filtered, labels_filtered))

    model.train()
    return np.mean(loss_arr), np.mean(accuracy_arr)


MODEL_NAME = 'NbAiLab/nb-bert-base'

# get tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True,
                                                   mlm_probability=0.15)

# Get training data from huggingface
train_data = load_dataset('json', data_files=os.path.join(DATA_DIR, 'train.json'), split='train')

training_dataset = tokenize_and_wrap_data(train_data, tokenizer)
training_dataset_wrapped = DatasetWrapper(training_dataset)
train_loader = DataLoader(dataset=training_dataset_wrapped, batch_size=64, collate_fn=data_collator)

val_data = load_dataset('json', data_files=os.path.join(DATA_DIR, 'validation.json'), split='train')
val_dataset = tokenize_and_wrap_data(val_data, tokenizer)
val_dataset_wrapped = DatasetWrapper(val_dataset)
val_loader = DataLoader(dataset=val_dataset_wrapped, batch_size=4, collate_fn=data_collator)


# get model
model = BertForMaskedLM.from_pretrained(MODEL_NAME)

# set model to train mode
model = model.train()

# validate if model works with opacus
validate_model(model)

training_args = TrainingArguments(
    output_dir="./results",
    learning_rate=0.002,
    weight_decay=0.01
)

# Dummy training for optimizer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=training_dataset,
    eval_dataset=["train"],
    data_collator=data_collator,
    tokenizer=tokenizer
)

privacy_engine = PrivacyEngine()

for p in model.bert.embeddings.parameters():
    p.requires_grad = False

model, optimizer, dp_train_loader = privacy_engine.make_private_with_epsilon(
    module=model,
    optimizer=trainer.create_optimizer(),
    data_loader=train_loader,
    epochs=20,
    target_epsilon=3,
    target_delta=0.00002,
    max_grad_norm=1.2,
)

trainer = None

# print(f"Using sigma={optimizer.noise_multiplier} and C={args.max_grad_norm}")

def train(model, train_loader, optimizer, epoch, device, val_loader):
    model.train()
    model = model.to(device)

    losses = []

    with BatchMemoryManager(
        data_loader=train_loader,
        max_physical_batch_size=4,
        optimizer=optimizer
    ) as memory_safe_data_loader:

        for i, batch in enumerate(memory_safe_data_loader):

            optimizer.zero_grad()

            # compute output
            output = model(input_ids=batch["input_ids"].to(device),
                           attention_mask=batch["attention_mask"].to(device),
                           labels=batch["labels"].to(device))

            loss = output[0]
            loss.backward()
            losses.append(loss.item())
            optimizer.step()

            if i > 0 and (i + 1) % 600 == 0:

                epsilon = privacy_engine.get_epsilon(0.00002)
                print(
                    f"\tTrain Epoch: {epoch} \t"
                    f"Loss: {np.mean(losses):.6f} "
                    f"(ε = {epsilon:.2f}, δ = {0.00002})"
                )
                eval_loss, eval_accuracy = evaluate(model, val_loader)
                print(
                    f"eval loss: {eval_loss} \t"
                    f"eval acc: {eval_accuracy}"
                )

    return model, eval_loss, eval_accuracy

for epoch in tqdm(range(20), desc="Epoch", unit="epoch"):
    model, eval_loss, eval_accuracy = train(model, train_loader=dp_train_loader,
                                            optimizer=optimizer,
                                            epoch=epoch + 1,
                                            device="cuda",
                                            val_loader=val_loader)


