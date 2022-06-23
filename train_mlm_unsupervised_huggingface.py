import os
from datetime import datetime

import numpy as np

from modelling_utils.mlm_modelling import MLMUnsupervisedModelling
from transformers import AutoTokenizer, Trainer, \
    TrainingArguments, AutoModelForMaskedLM, DataCollatorForWholeWordMask, \
    DataCollatorForLanguageModeling

from local_constants import MODEL_DIR
from utils.input_args import MLMArgParser
import os
from datetime import datetime

import numpy as np
from transformers import AutoTokenizer, Trainer, \
    TrainingArguments, AutoModelForMaskedLM, DataCollatorForWholeWordMask, \
    DataCollatorForLanguageModeling

from local_constants import MODEL_DIR
from modelling_utils.mlm_modelling import MLMUnsupervisedModelling
from utils.input_args import MLMArgParser

os.environ["WANDB_DISABLED"] = "true"
from datasets import load_metric

mlm_parser = MLMArgParser()
args = mlm_parser.parser.parse_args()
# args.model_name = 'Geotrend/distilbert-base-da-cased'
# args.train_data = 'train_200.json'
# args.logging_steps = 20
# args.evaluate_steps = 20
# args.save_steps = 20
# args.train_batch_size = 2
# args.eval_batch_size = 2


mlm_modelling = MLMUnsupervisedModelling(args=args)

# get data
mlm_modelling.load_data()

# Load the model
model = AutoModelForMaskedLM.from_pretrained(args.model_name)
tokenizer = AutoTokenizer.from_pretrained(args.model_name)

output_name = f'{args.model_name.replace("/", "_")}-{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}'
output_dir = os.path.join(MODEL_DIR, output_name)
mlm_modelling.save_config(output_dir=output_dir, args=args)

print("Save checkpoints to:", output_dir)

train_data_wrapped = mlm_modelling.tokenize_and_wrap_data(data=mlm_modelling.train_data)
eval_data_wrapped = mlm_modelling.tokenize_and_wrap_data(data=mlm_modelling.eval_data)

if args.whole_word_mask:
    data_collator = DataCollatorForWholeWordMask(tokenizer=tokenizer, mlm=True,
                                                 mlm_probability=args.mlm_prob)
else:
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True,
                                                    mlm_probability=args.mlm_prob)

metric = load_metric('accuracy')


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    # print(logits)
    # print(labels)

    labels = labels.flatten()
    predictions = np.argmax(logits, axis=-1)
    predictions = predictions.flatten()


    filtered = [[xv, yv] for xv, yv in zip(labels, predictions) if xv != -100]

    labels_filtered = np.array([x[0] for x in filtered])
    preds_filtered = np.array([x[1] for x in filtered])

    print(labels_filtered[:20])
    print(preds_filtered[:20])


    return metric.compute(predictions=preds_filtered, references=labels_filtered)


training_args = TrainingArguments(
    output_dir=os.path.join(MODEL_DIR, output_name),
    overwrite_output_dir=True,
    num_train_epochs=args.epochs,
    evaluation_strategy='steps',
    do_eval=True,
    per_device_train_batch_size=args.train_batch_size,
    per_device_eval_batch_size=args.eval_batch_size,
    eval_steps=args.evaluate_steps,
    save_steps=args.save_steps,
    logging_steps=args.logging_steps,
    save_total_limit=1,
    warmup_steps=args.lr_warmup_steps,
    # prediction_loss_only=True,
    fp16=args.use_fp16,
    push_to_hub=False,
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_data_wrapped,
    eval_dataset=eval_data_wrapped,
    compute_metrics=compute_metrics,
)

print("Save tokenizer to:", output_dir)
tokenizer.save_pretrained(output_dir)

trainer.train()

print("Save model to:", output_dir)
# model.save_pretrained(output_dir)
trainer.save_model()
print("Training done")
