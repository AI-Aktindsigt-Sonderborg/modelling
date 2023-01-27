import os

import evaluate
import numpy as np
from transformers import TrainingArguments, Trainer

from data_utils.get_dataset import get_label_list
from modelling_utils.helpers import align_labels_with_tokens
from modelling_utils.input_args import NERArgParser
from modelling_utils.ner_modelling import NERModelling

os.environ["WANDB_DISABLED"] = "true"

metric = evaluate.load("seqeval")

ner_parser = NERArgParser()
args = ner_parser.parser.parse_args()
# args.load_alvenir_pretrained = False

OUTPUT_DIR = "models/" + args.model_name.replace('/',
                                                 '_') + '_ner_finetuned' + args.out_suffix

label_list, id2label, label2id = get_label_list()

ner_modelling = NERModelling(args=args)

ner_modelling.load_data()
ner_feature = ner_modelling.train_data.features["ner_tags"]
label_names = ner_feature.feature.names


def compute_metrics(eval_preds):
    logits, labels = eval_preds
    predictions = np.argmax(logits, axis=-1)

    # Remove ignored index (special tokens) and convert to labels
    true_labels = [[label_names[l] for l in label if l != -100] for label in
                   labels]
    true_predictions = [
        [label_names[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    all_metrics = metric.compute(predictions=true_predictions,
                                 references=true_labels)
    return {
        "precision": all_metrics["overall_precision"],
        "recall": all_metrics["overall_recall"],
        "f1": all_metrics["overall_f1"],
        "accuracy": all_metrics["overall_accuracy"],
    }


# test stuff
# inputs = ner_modelling.tokenizer(ner_modelling.train_data[0]["tokens"], is_split_into_words=True)
# labels = ner_modelling.train_data[0]['ner_tags']
# word_ids = inputs.word_ids()
# tokenized_test = align_labels_with_tokens(labels, word_ids)

def tokenize_and_align_labels(examples):
    tokenized_inputs = ner_modelling.tokenizer(
        examples["tokens"], truncation=True, is_split_into_words=True
    )
    all_labels = examples["ner_tags"]
    new_labels = []
    for i, labels in enumerate(all_labels):
        word_ids = tokenized_inputs.word_ids(i)
        new_labels.append(align_labels_with_tokens(labels, word_ids))

    tokenized_inputs["labels"] = new_labels
    return tokenized_inputs


tokenized_train = ner_modelling.train_data.map(
    tokenize_and_align_labels,
    batched=True,
    remove_columns=ner_modelling.train_data.column_names,
)

tokenized_eval = ner_modelling.eval_data.map(
    tokenize_and_align_labels,
    batched=True,
    remove_columns=ner_modelling.eval_data.column_names,
)

# train_data_wrapped = ner_modelling.tokenize_and_wrap_data(ner_modelling.train_data)
# eval_data_wrapped = ner_modelling.tokenize_and_wrap_data(ner_modelling.eval_data)

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    # overwrite_output_dir=True,
    num_train_epochs=args.epochs,
    evaluation_strategy='steps',
    learning_rate=5e-5,
    # weight_decay=0.01,
    # initial_learning_rate=0.0002,
    # gradient_accumulation_steps=4,  # 2 * 4 = 8
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    save_steps=50,
    warmup_steps=1000,
    do_eval=True,
    do_predict=True,
    metric_for_best_model="eval_f1",
    save_strategy="steps",
    logging_steps=50,
    eval_steps=50,
    load_best_model_at_end=True,
    push_to_hub=False
)

trainer = Trainer(
    model=ner_modelling.model,
    args=training_args,
    train_dataset=tokenized_train.shuffle(seed=1),
    eval_dataset=tokenized_eval,
    tokenizer=ner_modelling.tokenizer,
    data_collator=ner_modelling.data_collator,
    compute_metrics=compute_metrics,
    # callbacks=[nuna_text_modelling.callbacks],
)

trainer.train()

trainer.save_model(OUTPUT_DIR)
trainer.save_state()

model_eval = trainer.evaluate()
