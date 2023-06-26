import evaluate
import numpy as np
from datasets import load_metric
from optuna.pruners import SuccessiveHalvingPruner, ThresholdPruner, PercentilePruner
from optuna.samplers import TPESampler, RandomSampler, NSGAIISampler
from transformers import TrainingArguments, Trainer

from ner.data_utils.get_dataset import get_label_list_old
from ner.modelling_utils.helpers import get_label_list
from ner.modelling_utils.helpers import align_labels_with_tokens
from ner.modelling_utils.input_args import NERArgParser
from ner.modelling_utils.ner_modelling import NERModelling
from sklearn.gaussian_process import GaussianProcessRegressor
import sys
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score


metric = evaluate.load("seqeval")

ner_parser = NERArgParser()
args = ner_parser.parser.parse_args()
args.test = False
args.differential_privacy = False
# args.train_data = "dane"
# args.load_alvenir_pretrained = False
# args.model_name = "base"
args.entities = ["PERSON", "LOKATION", "ADRESSE", "HELBRED", "ORGANISATION", "KOMMUNE", "TELEFONNUMMER"]
label_list, id2label, label2id, label2weight = get_label_list(args.entities)

ner_modelling = NERModelling(args=args)

ner_modelling.load_data()

def tokenize_and_wrap_data(data: Dataset):

    def tokenize_and_align_labels(examples):
        tokenized_inputs = ner_modelling.tokenizer(
            examples["tokens"],
            truncation=True,
            is_split_into_words=True,
            padding='max_length',
            max_length=args.max_length
        )

        all_labels = examples["ner_tags"]
        new_labels = []
        for i, labels in enumerate(all_labels):
            word_ids = tokenized_inputs.word_ids(i)
            new_labels.append(align_labels_with_tokens(labels, word_ids))

        tokenized_inputs["labels"] = new_labels
        return tokenized_inputs

    tokenized = data.map(
        tokenize_and_align_labels,
        batched=True,
        remove_columns=data.column_names,
        desc="Running tokenizer on dataset line_by_line",
    )

    tokenized.set_format('torch')
    return tokenized


from sklearn.metrics import precision_recall_fscore_support, accuracy_score

tokenized_train = tokenize_and_wrap_data(data=ner_modelling.data.train)
tokenized_eval = tokenize_and_wrap_data(data=ner_modelling.data.eval)

def compute_metrics(eval_preds):
    logits, labels = eval_preds
    predictions = np.argmax(logits, axis=-1)

    # Remove ignored index (special tokens) and convert to labels
    true_labels = [[id2label[l] for l in label if l != -100] for label in
                   labels]
    true_labels = [item for sublist in true_labels for item in sublist]
    true_predictions = [
        [id2label[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_predictions = [item for sublist in true_predictions for item in sublist]
    f_1 = f1_score(true_labels, true_predictions, average='macro')
    #all_metrics = metric.compute(predictions=true_predictions,references=true_labels)
    return {'f1': f_1}


ner_modelling.save_config(output_dir=ner_modelling.output_dir,
                          metrics_dir=ner_modelling.metrics_dir,
                          args=args)
a
training_args = TrainingArguments(
    output_dir=ner_modelling.output_dir,
    overwrite_output_dir=True,
    evaluation_strategy='steps',
    # num_train_epochs=5,
    # weight_decay=0.01,
    # initial_learning_rate=0.0002,
    # gradient_accumulation_steps=4,  # 2 * 4 = 8
    per_device_train_batch_size=args.train_batch_size,
    per_device_eval_batch_size=args.eval_batch_size,
    save_steps=10000,
    metric_for_best_model="eval_f1",
    save_strategy="steps",
    logging_steps=250,
    eval_steps=250,
    load_best_model_at_end=True,
    push_to_hub=False,
    report_to='none'
)

# model = ner_modelling.get_model()

trainer = Trainer(
    model=None,
    args=training_args,
    train_dataset=tokenized_train.shuffle(seed=1),
    eval_dataset=tokenized_eval,
    tokenizer=ner_modelling.tokenizer,
    data_collator=ner_modelling.data_collator,
    compute_metrics=compute_metrics,
    model_init=ner_modelling.get_model,
    # callbacks=[nuna_text_modelling.callbacks],
)


def optuna_hp_space(trial):
    return {
        "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True),
        "per_device_train_batch_size": trial.suggest_categorical(
            "per_device_train_batch_size", [8, 16, 32, 64]),
        "optimizer": trial.suggest_categorical("optimizer", ["MomentumSGD", "Adam", "AdamW"]),
        # "num_layers": trial.suggest_int("num_layers", 1, 15),
        "dropout_rate": trial.suggest_float("dropout_rate", 0.0, 1.0),
        "warmup_steps": trial.suggest_int("warmup_steps", 0, 10000),
        "num_epochs": trial.suggest_int("num_epochs", 1, 10),
    }


best_trial = trainer.hyperparameter_search(
    direction="maximize",
    backend="optuna",
    sampler=TPESampler(), # TPESampler(),
    pruner=PercentilePruner(percentile=25.0), # SuccessiveHalvingPruner(),
    hp_space=optuna_hp_space,
    n_trials=40,
    # compute_objective=compute_metrics,
)

print(best_trial)

# trainer.train()
#
# trainer.save_model(ner_modelling.output_dir)
# trainer.save_state()
#
# model_eval = trainer.evaluate()
