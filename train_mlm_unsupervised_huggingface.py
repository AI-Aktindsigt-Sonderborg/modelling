import os
from datetime import datetime
from sklearn.metrics import accuracy_score

import numpy as np
from torch import nn
from torch.utils.data import DataLoader

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

# args.local_testing = True
if args.local_testing:
    args.model_name = 'Geotrend/distilbert-base-da-cased'
    args.train_data = 'train_4.json'
    args.logging_steps = 20
    args.evaluate_steps = 5
    args.save_steps = 200
    args.train_batch_size = 4
    args.eval_batch_size = 2
    args.max_length = 8


mlm_modelling = MLMUnsupervisedModelling(args=args)

# get data
mlm_modelling.load_data()

# Load the model
model = AutoModelForMaskedLM.from_pretrained(args.model_name)
tokenizer = AutoTokenizer.from_pretrained(args.model_name)

output_name = f'HF-{args.model_name.replace("/", "_")}-{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}'
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

class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):

        labels = inputs.get("labels")
        # forward pass
        outputs = model(**inputs)
        logits = outputs.get("logits")

        loss_fct = nn.CrossEntropyLoss()  # -100 index = padding token
        # ToDo: Fix below hardcoded token length
        loss = loss_fct(logits.view(-1, 119547), labels.view(-1))


        return (loss, outputs) if return_outputs else loss

    def get_train_dataloader(self) -> DataLoader:
        """
        Returns the training [`~torch.utils.data.DataLoader`].

        Will use no sampler if `train_dataset` does not implement `__len__`, a random sampler (adapted to distributed
        training if necessary) otherwise.

        Subclass and override this method if you want to inject some custom behavior.
        """
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        return DataLoader(
            self.train_dataset,
            batch_size=self.args.per_device_train_batch_size,
            collate_fn=self.data_collator
            )

def compute_metrics(eval_pred):
    logits, labels = eval_pred

    labels = labels.flatten()
    predictions = np.argmax(logits, axis=-1)
    predictions = predictions.flatten()


    filtered = [[xv, yv] for xv, yv in zip(labels, predictions) if xv != -100]

    labels_filtered = np.array([x[0] for x in filtered])
    preds_filtered = np.array([x[1] for x in filtered])

    return metric.compute(predictions=preds_filtered, references=labels_filtered)


training_args = TrainingArguments(
    output_dir=os.path.join(MODEL_DIR, output_name),
    overwrite_output_dir=True,
    num_train_epochs=args.epochs,
    evaluation_strategy='steps',
    lr_scheduler_type='linear',
    learning_rate=args.lr,
    gradient_accumulation_steps=args.grad_accum_steps,
    per_device_train_batch_size=args.train_batch_size,
    per_device_eval_batch_size=args.eval_batch_size,
    eval_steps=args.evaluate_steps,
    save_steps=args.save_steps,
    logging_steps=args.logging_steps,
    # save_total_limit=1,
    warmup_steps=args.lr_warmup_steps,
    # prediction_loss_only=True,
    fp16=args.use_fp16,
    push_to_hub=False,
)

trainer = CustomTrainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_data_wrapped,
    eval_dataset=eval_data_wrapped,
    compute_metrics=compute_metrics,
)

# print("Save tokenizer to:", output_dir)
# tokenizer.save_pretrained(output_dir)

trainer.train()

print("Save model to:", output_dir)
# model.save_pretrained(output_dir)
trainer.save_model()
print("Training done")
