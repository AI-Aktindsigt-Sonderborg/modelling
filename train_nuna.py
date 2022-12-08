import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from transformers import TrainingArguments, Trainer
# from data_utils.preprocess_nuna_text import NunaTextPreprocessing
from modelling_utils.NunaTextModelling import NunaTextModelling, compute_metrics
from data_utils.custom_dataclasses import LoadModelType
from local_constants import DATA_DIR
MODEL_NAME = 'last_model'
OUTPUT_DIR = "models/" + MODEL_NAME + '_supervised'
# nuna_text_processing = NunaTextPreprocessing(model_name=MODEL_NAME)
# nuna_text_processing.preproces_data_for_train()
os.environ["WANDB_DISABLED"] = "true"

# LABELS = [x[0] for x in nuna_text_processing.class_weights]
# WEIGHTS = [float(x[1]) for x in nuna_text_processing.class_weights]


# class CustomTrainer(Trainer):
#     def compute_loss(self, model, inputs, return_outputs=False):
#         labels = inputs.get("labels")
#         # forward pass
#         outputs = model(**inputs)
#         logits = outputs.get("logits")
#         # compute custom loss (suppose one has 3 labels with different weights)
#         loss_fct = nn.CrossEntropyLoss(weight=torch.tensor(WEIGHTS).cuda())
#         loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
#         return (loss, outputs) if return_outputs else loss
label_dict = {'Beskæftigelse og integration': 0, 'Børn og unge': 1, 'Erhverv og turisme': 2,
            'Klima, teknik og miljø': 3, 'Kultur og fritid': 4, 'Socialområdet': 5,
            'Sundhed og ældre': 6, 'Økonomi og administration': 7}

LABELS = list(label_dict)

nuna_text_modelling = NunaTextModelling(labels=LABELS,
                                        data_types=['train_10', 'val_10'],
                                        # data_types=['train_classified_mixed', 'test_classified_mixed'],
                                        data_path=DATA_DIR,
                                        model_name=MODEL_NAME, load_model_type=LoadModelType.TRAIN)

train_data, eval_data = nuna_text_modelling.load_data()
train_data_wrapped = nuna_text_modelling.tokenize_and_wrap_data(data=train_data)
eval_data_wrapped = nuna_text_modelling.tokenize_and_wrap_data(data=eval_data)



training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    # overwrite_output_dir=True,
    num_train_epochs=50,
    # weight_decay=0.01,
    evaluation_strategy='steps',
    learning_rate=5e-5,
    # initial_learning_rate=0.0002,
    per_device_train_batch_size=32,
    # gradient_accumulation_steps=4,  # 2 * 4 = 8
    per_device_eval_batch_size=32,
    save_steps=100,
    warmup_steps=50,
    do_eval=True,
    do_predict=True,
    metric_for_best_model="accuracy",
    save_strategy="steps",
    logging_steps=10,
    eval_steps=10,
    load_best_model_at_end=True,
)

trainer = Trainer(
    model=nuna_text_modelling.model,
    args=training_args,
    train_dataset=train_data_wrapped,
    eval_dataset=eval_data_wrapped,
    tokenizer=nuna_text_modelling.tokenizer,
    data_collator=nuna_text_modelling.data_collator,
    compute_metrics=compute_metrics,
    # callbacks=[nuna_text_modelling.callbacks],
)

trainer.train()

trainer.save_model(OUTPUT_DIR)
trainer.save_state()

model_eval = trainer.evaluate()

# y_true, y_pred, eval_result = nuna_text_modelling.eval_model(data_types=['test'], model_name=MODEL_NAME)

print()
