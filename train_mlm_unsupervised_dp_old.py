import json
import os
import sys
import traceback
from datetime import datetime
from typing import Dict

from opacus.grad_sample import register_grad_sampler
from modelling_utils.grad_sampler_mlm_head import replace_bert_head

import numpy as np
import torch
from datasets import load_dataset
from opacus import PrivacyEngine, GradSampleModule
from opacus.data_loader import DPDataLoader
from opacus.optimizers import DPOptimizer
from opacus.utils.batch_memory_manager import BatchMemoryManager
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, Trainer, \
    TrainingArguments, AutoModelForMaskedLM, DataCollatorForWholeWordMask, \
    DataCollatorForLanguageModeling, BertForMaskedLM, AutoModelForPreTraining, BertConfig
# from transformers.models.bert.modeling_bert import

from data_utils.preprocess_public_scraped_data import DatasetWrapper, TokenizedSentencesDataset, \
    split_to_sentences, split_train_val
from local_constants import OUTPUT_DIR, CONFIG_DIR
# from modelling_utils.custom_modeling_bert import BertLMPredictionHead, BertOnlyMLMHead
from utils.helpers import validate_model, TimeCode
from utils.input_args import create_parser
from utils.helpers import fix_and_validate

os.environ["WANDB_DISABLED"] = "true"

if not torch.cuda.is_available():
    print("Cuda not found. Exiting..")
    sys.exit(0)

parser = create_parser()
args = parser.parse_args()
args.model_name = 'Geotrend/distilbert-base-da-cased'


output_name = f'{args.model_name.replace("/", "_")}-{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}'
output_dir = os.path.join(OUTPUT_DIR, output_name)

with open(os.path.join(CONFIG_DIR, output_name), 'w', encoding='utf-8') as f:
    json.dump(args.__dict__, f, indent=2)

train_data = load_dataset('json', data_files='data/train.json', split='train')

val_data = load_dataset('json', data_files='data/validation.json')

column_names = train_data.column_names

# Load foundation model, tokenizer and collator

# model = BertForMaskedLM.from_pretrained(args.model_name)
model = replace_bert_head(model_tag=args.model_name)

tokenizer = AutoTokenizer.from_pretrained(args.model_name)


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


train_dataset = train_data.map(
    tokenize_function,
    batched=True,
    num_proc=1,
    remove_columns=['text'],
    load_from_cache_file=False,
    desc="Running tokenizer on dataset line_by_line",
)

val_dataset = val_data.map(
    tokenize_function,
    batched=True,
    num_proc=1,
    # remove_columns=['text'],
    # load_from_cache_file=not data_args.overwrite_cache,
    desc="Running tokenizer on dataset line_by_line",
)

print("Save checkpoints to:", output_dir)

# data_collator = DataCollatorForTokenClassification(tokenizer)
if args.whole_word_mask:
    data_collator = DataCollatorForWholeWordMask(tokenizer=tokenizer, mlm=True,
                                                 mlm_probability=args.mlm_prob)
else:
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True,
                                                    mlm_probability=args.mlm_prob)

# consider using TokenizedSentencesDataset class
# sentences = split_to_sentences(data_path='data/da_DK_subset.json')
# train, val = split_train_val(sentences=sentences)

# train_dataset = TokenizedSentencesDataset(train, tokenizer, args.max_length)
train_dataset_wrapped = DatasetWrapper(train_dataset)
train_data_loader = DataLoader(dataset=train_dataset_wrapped, batch_size=args.lot_size,
                               collate_fn=data_collator, shuffle=True)

# for index, x in enumerate(train_data_loader.dataset):
#     train_data_loader.dataset[index] = x.data
# create val dataset
# val_dataset = TokenizedSentencesDataset(val_sentences, tokenizer, args.max_length,
#                                         cache_tokenization=True) if len(val_sentences) > 0 else None
val_dataset_wrapped = DatasetWrapper(val_dataset)
val_data_loader = DataLoader(dataset=val_dataset_wrapped, batch_size=args.lot_size,
                             collate_fn=data_collator)

# set model to train mode
model = model.train()

# validate if model works with opacus
validate_model(model)
# fix_and_validate(model)

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


dp_model, dp_optimizer, dp_train_loader = privacy_engine.make_private_with_epsilon(
    module=model,
    optimizer=trainer.create_optimizer(),
    data_loader=train_data_loader,
    epochs=args.epochs,
    target_epsilon=args.epsilon,
    target_delta=args.delta,
    max_grad_norm=args.max_grad_norm
)

trainer = None

print(f"Using sigma={dp_optimizer.noise_multiplier} and C={args.max_grad_norm}")


def train(model: GradSampleModule, train_loader: DPDataLoader, val_loader: DataLoader,
          optimizer: DPOptimizer, epoch: int, device: str = 'cuda:0'):
    model.train()
    model = model.to(device)
    train_losses = []
    with BatchMemoryManager(
        data_loader=train_loader,
        max_physical_batch_size=args.batch_size,
        optimizer=optimizer
    ) as memory_safe_data_loader:
        batch_dims = []
        for i, batch in enumerate(memory_safe_data_loader):

            try:
                optimizer.zero_grad()

                # compute output
                output = model(input_ids=batch["input_ids"].to(device),
                               attention_mask=batch["attention_mask"].to(device),
                               labels=batch["labels"].to(device))
                batch_dims.append(output.logits.size()[0])
                # if i == 10:
                #     print(batch_dims)

                loss = output.loss
                # loss.backward(create_graph=False, retain_graph=False, inputs=batch['input_ids'])

                loss.backward()

                train_losses.append(loss.item())
                optimizer.step()

                # if i > 0 and (i + 1) % args.evaluate_steps == 0:
                #     epsilon = privacy_engine.get_epsilon(args.delta)
                #     print(
                #         f"\tTrain Epoch: {epoch} \t"
                #         f"Loss: {np.mean(losses):.6f} "
                #         f"(ε = {epsilon:.2f}, δ = {args.delta})"
                #     )
                #     model.eval()
                #     val_losses = []
                #     val_logits_arr = []
                #     for batch in val_loader:
                #         val_output = model(input_ids=batch["input_ids"].to('cuda'),
                #                        attention_mask=batch["attention_mask"].to('cuda'),
                #                        labels=batch["labels"].to('cuda'))
                #
                #         val_loss, val_logits = val_output[:2]
                #         print(f'eval loss: {val_loss} \t')
                #         val_losses.append(val_loss.item())
                #     model.train()
                #     val_logits_arr.append(val_logits)
                #     np.mean(val_losses)

            except Exception as e:
                traceback.print_exc()
                print(e)

        # except Exception as e:
        #     traceback.print_exc()
        #     print(e)
    return model  # , eval_loss, eval_accuracy


losses = []
accuracies = []
code_timer = TimeCode()

for epoch in tqdm(range(args.epochs), desc="Epoch", unit="epoch"):
    dp_model = train(model=dp_model, train_loader=dp_train_loader, val_loader=val_data_loader, optimizer=dp_optimizer,
                     epoch=epoch + 1)
    # losses.append({epoch: eval_loss})
    # accuracies.append({epoch: eval_accuracy})

print()
