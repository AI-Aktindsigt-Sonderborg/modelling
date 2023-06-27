# pylint: skip-file
import os
import sys

import optuna
from torch import optim
from tqdm import tqdm
from transformers import BertConfig


from ner.modelling_utils.input_args import NERArgParser
from ner.modelling_utils.ner_modelling import NERModelling, NERModellingDP
from shared.modelling_utils.custom_modeling_bert import BertForTokenClassification
from shared.modelling_utils.helpers import create_data_loader


def train_model_hp_search(param, model, trial, modelling: NERModelling):
    """
    Training loop for optuna hyper parameter optimization
    """

    modelling.load_data()

    all_lrs = []
    step = 0
    train_data_wrapped, train_loader = create_data_loader(
        data_wrapped=modelling.tokenize_and_wrap_data(modelling.data.train),
        batch_size=modelling.args.lot_size,
        data_collator=modelling.data_collator)

    _, eval_loader = create_data_loader(
        data_wrapped=modelling.tokenize_and_wrap_data(modelling.data.eval),
        batch_size=modelling.args.eval_batch_size,
        data_collator=modelling.data_collator,
        shuffle=False)

    eval_scores = []
    optimizer = getattr(optim, param['optimizer'])(model.parameters(), lr= param['learning_rate'])

    for epoch in tqdm(range(modelling.args.epochs), desc="Epoch",
                      unit="epoch"):
        model, step, lrs, eval_scores = modelling.train_epoch(
            model=model,
            train_loader=train_loader,
            optimizer=optimizer,
            val_loader=eval_loader,
            epoch=epoch + 1,
            step=step,
            eval_scores=eval_scores)
        all_lrs.extend(lrs)

        # Add prune mechanism
        trial.report(eval_scores[-1].f_1, epoch)

        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    return eval_scores[-1].f_1


ner_parser = NERArgParser()
args = ner_parser.parser.parse_args()
args.cmd_line_args = sys.argv
print(sys.argv)

args.test = False
args.local_testing = True
if args.local_testing:
    args.train_data = 'dane'
    args.load_alvenir_pretrained = False
    args.replace_head = False
    args.freeze_layers = False
    args.freeze_embeddings = True
    args.differential_privacy = False
    # args.lr_freezed_warmup_steps = 50
    args.freeze_layers_n_steps = 50
    args.lr_freezed = 0.002

    args.learning_rate = 0.001
    args.model_name = 'base'
    # args.model_name = 'NbAiLab/nb-bert-base'
    args.train_batch_size = 32
    args.lot_size = 64
    args.eval_batch_size = 32

    args.epochs = 5
    args.evaluate_steps = 250
    args.save_steps = 250
    args.custom_model_name = 'optim'
    args.weight_classes = False


ner_modelling = NERModelling(args=args)

ner_modelling.load_data()



def objective(trial):
    params = {
        'learning_rate': trial.suggest_loguniform('learning_rate', 1e-5, 1e-1),
        "num_layers": trial.suggest_int("num_layers", 1, 15),
        "dropout_rate": trial.suggest_float("dropout_rate", 0.0, 1.0),
        "warmup_steps": trial.suggest_int("warmup_steps", 0, 10000),
        "num_epochs": trial.suggest_int("num_epochs", 1, 5),
        "optimizer": trial.suggest_categorical("optimizer", ["MomentumSGD", "Adam", "AdamW"]),
        "per_device_train_batch_size": trial.suggest_categorical(
            "per_device_train_batch_size", [8, 16, 32, 64]),
    }

    model_config = BertConfig(params)

    model = BertForTokenClassification.from_pretrained(
            pretrained_model_name_or_path=ner_modelling.model_path,
            num_labels=len(ner_modelling.args.labels),
            label2id=ner_modelling.label2id,
            id2label=ner_modelling.id2label,
            local_files_only=ner_modelling.args.load_alvenir_pretrained)

    model.config = model_config

    f_1 = train_model_hp_search(params, model, trial, ner_modelling)

    return f_1

os.makedirs("ner/models/" + args.custom_model_name, exist_ok=True)
os.makedirs("ner/models/" + args.custom_model_name + "/metrics", exist_ok=True)

study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(), pruner=optuna.pruners.MedianPruner())
study.optimize(objective, n_trials=10)

best_trial = study.best_trial

for key, value in best_trial.params.items():
    print("{}: {}".format(key, value))

