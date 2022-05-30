import json
import os
from datetime import datetime

import torch
from transformers import AutoTokenizer, Trainer, \
    TrainingArguments, AutoModelForMaskedLM, DataCollatorForWholeWordMask, \
    DataCollatorForLanguageModeling

from data_utils.preprocess_public_scraped_data import split_to_sentences, split_train_val, \
    TokenizedSentencesDataset
from local_constants import OUTPUT_DIR, DATA_DIR, CONFIG_DIR
from utils.input_args import create_parser
os.environ["WANDB_DISABLED"] = "true"

print(torch.cuda.is_available())


parser = create_parser()
args = parser.parse_args()

# get data
sentences = split_to_sentences(data_path=os.path.join(DATA_DIR, args.data_file))

train_sentences, dev_sentences = split_train_val(sentences=sentences)

# Load the model
model = AutoModelForMaskedLM.from_pretrained(args.model_name)
tokenizer = AutoTokenizer.from_pretrained(args.model_name)

output_name = f'{args.model_name.replace("/", "_")}-{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}'
output_dir = os.path.join(OUTPUT_DIR, output_name)

with open(os.path.join(CONFIG_DIR, output_name), 'w', encoding='utf-8') as f:
    json.dump(args.__dict__, f, indent=2)

print("Save checkpoints to:", output_dir)

train_dataset = TokenizedSentencesDataset(train_sentences, tokenizer, args.max_length)
dev_dataset = TokenizedSentencesDataset(dev_sentences, tokenizer, args.max_length,
                                        cache_tokenization=True) if len(dev_sentences) > 0 else None

if args.whole_word_mask:
    data_collator = DataCollatorForWholeWordMask(tokenizer=tokenizer, mlm=True,
                                                 mlm_probability=args.mlm_prob)
else:
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True,
                                                    mlm_probability=args.mlm_prob)

training_args = TrainingArguments(
    output_dir=os.path.join(OUTPUT_DIR, output_name),
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
