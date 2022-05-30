import json
import os
import sys
import traceback
from datetime import datetime

import torch
from opacus import PrivacyEngine
from opacus.utils.batch_memory_manager import BatchMemoryManager
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, Trainer, \
    TrainingArguments, AutoModelForMaskedLM, DataCollatorForWholeWordMask, \
    DataCollatorForLanguageModeling

from data_utils.preprocess_public_scraped_data import split_to_sentences, split_train_val, \
    TokenizedSentencesDataset, DatasetWrapper
from local_constants import OUTPUT_DIR, DATA_DIR, CONFIG_DIR
from utils.helpers import validate_model, TimeCode
from utils.input_args import create_parser

os.environ["WANDB_DISABLED"] = "true"

if not torch.cuda.is_available():
    print("Cuda not found. Exiting..")
    sys.exit(0)

parser = create_parser()
args = parser.parse_args()

output_name = f'{args.model_name.replace("/", "_")}-{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}'
output_dir = os.path.join(OUTPUT_DIR, output_name)

with open(os.path.join(CONFIG_DIR, output_name), 'w', encoding='utf-8') as f:
    json.dump(args.__dict__, f, indent=2)

print("Save checkpoints to:", output_dir)

# get data
sentences = split_to_sentences(data_path=os.path.join(DATA_DIR, args.data_file))

train_sentences, val_sentences = split_train_val(sentences=sentences)

# Load foundation model, tokenizer and collator
model = AutoModelForMaskedLM.from_pretrained(args.model_name)
tokenizer = AutoTokenizer.from_pretrained(args.model_name)
# data_collator = DataCollatorForTokenClassification(tokenizer)
if args.whole_word_mask:
    data_collator = DataCollatorForWholeWordMask(tokenizer=tokenizer, mlm=True,
                                                 mlm_probability=args.mlm_prob)
else:
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True,
                                                    mlm_probability=args.mlm_prob)

# create train dataset
train_dataset = TokenizedSentencesDataset(train_sentences, tokenizer, args.max_length)
train_dataset_wrapped = DatasetWrapper(train_dataset)
train_data_loader = DataLoader(dataset=train_dataset_wrapped, batch_size=args.lot_size,
                               collate_fn=data_collator)

# create val dataset
val_dataset = TokenizedSentencesDataset(val_sentences, tokenizer, args.max_length,
                                        cache_tokenization=True) if len(val_sentences) > 0 else None
val_dataset_wrapped = DatasetWrapper(val_dataset)
val_data_loader = DataLoader(dataset=val_dataset_wrapped, batch_size=args.lot_size,
                             collate_fn=data_collator)

# set model to train mode
model = model.train()

# validate if model works with opacus
validate_model(model)

training_args = TrainingArguments(
    output_dir=output_dir,
    learning_rate=args.lr,
    weight_decay=args.weight_decay
)

# Dummy training for optimizer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset_wrapped,
    eval_dataset=["train"],
    data_collator=data_collator,
    tokenizer=tokenizer
)

privacy_engine = PrivacyEngine()

for p in model.electra.embeddings.parameters():
    p.requires_grad = False

model, optimizer, train_loader = privacy_engine.make_private_with_epsilon(
    module=model,
    optimizer=trainer.create_optimizer(),
    data_loader=train_data_loader,
    epochs=args.epochs,
    target_epsilon=args.epsilon,
    target_delta=args.delta,
    max_grad_norm=args.max_grad_norm,
)

trainer = None

print(f"Using sigma={optimizer.noise_multiplier} and C={args.max_grad_norm}")


def train(model, train_loader, optimizer, epoch, device):
    model.train()
    model = model.to(device)

    losses = []

    with BatchMemoryManager(
        data_loader=train_loader,
        max_physical_batch_size=args.batch_size,
        optimizer=optimizer
    ) as memory_safe_data_loader:

        try:

            for i, batch in enumerate(memory_safe_data_loader):

                try:
                    optimizer.zero_grad()
                    print(batch)
                    # compute output
                    output = model(input_ids=batch["input_ids"].to(device),
                                   attention_mask=batch["attention_mask"].to(device),
                                   labels=batch["labels"].to(device))

                    loss = output[0]
                    loss.backward()
                    losses.append(loss.item())
                    optimizer.step()

                    # if i > 0 and (i + 1) % args.evaluate_steps == 0:
                    #     epsilon = privacy_engine.get_epsilon(args.delta)
                    #     print(
                    #         f"\tTrain Epoch: {epoch} \t"
                    #         f"Loss: {np.mean(losses):.6f} "
                    #         f"(ε = {epsilon:.2f}, δ = {args.delta})"
                    #     )
                    #     eval_loss, eval_accuracy = evaluate(model, val_data_loader)
                    #     print(
                    #         f"eval loss: {eval_loss} \t"
                    #         f"eval acc: {eval_accuracy}"
                    #     )

                    # print(f"i = {i} finished")
                except Exception as e:
                    trace = traceback
                    traceback.print_exc()
                    print(e)

        except Exception as e:
            trace = traceback.print_exc()
            print(e)
    return model  # , eval_loss, eval_accuracy


losses = []
accuracies = []
code_timer = TimeCode()

for epoch in tqdm(range(args.epochs), desc="Epoch", unit="epoch"):
    model = train(model, train_loader=train_loader, optimizer=optimizer, epoch=epoch + 1,
                  device="cuda")
    # losses.append({epoch: eval_loss})
    # accuracies.append({epoch: eval_accuracy})

print()
