# pylint: skip-file
import dataclasses

import numpy as np
from sklearn.utils import compute_class_weight

import optuna
import torch
import wandb

# from opacus import PrivacyEngine
# from opacus.utils.batch_memory_manager import BatchMemoryManager
from tqdm import tqdm

from ner.modelling_utils.input_args import NERArgParser
from ner.modelling_utils.ner_modelling import NERModelling
from shared.modelling_utils.helpers import create_data_loader
from shared.utils.helpers import write_json_lines, append_json_lines

ner_parser = NERArgParser()

args, leftovers = ner_parser.parser.parse_known_args()
args.test = False
args.train_data = "bio_train1.jsonl"
args.eval_data = "bio_val1.jsonl"
args.data_format = "bio"
args.evaluate_steps = 300
args.logging_steps = 300
# args.save_steps = 300
args.train_batch_size = 64
args.eval_batch_size = 64
args.epochs = 5
args.n_trials = 10
args.load_alvenir_pretrained = True
args.model_name = "SAS"
args.differential_privacy = False
# args.model_name = "base"


args.entities = [
    "PERSON",
    "LOKATION",
    "ADRESSE",
    "HELBRED",
    "ORGANISATION",
    "KOMMUNE",
    "TELEFONNUMMER",
]

model_name_to_print = (
    args.custom_model_name if args.custom_model_name else args.model_name
)

ner_modelling = NERModelling(args=args)


def train_model(trial, learning_rate, max_length, weight_decay):
    model = ner_modelling.get_model()

    for param in model.bert.embeddings.parameters():
        param.requires_grad = False

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    ner_modelling.load_data()

    if args.weight_classes:
        label_ids = [ner_modelling.label2id[label] for label in args.labels]

        ner_tags = []
        for tag in ner_modelling.data.train["tags"]:
            ner_tags.extend([ner_modelling.label2id[label] for label in tag])

        # important that all tags are represented in training set -
        # otherwise we stop training
        assert sorted(list(set(ner_tags))) == sorted(list(set(label_ids)))

        ner_tags = np.array(ner_tags)

        ner_modelling.class_weights = torch.tensor(
            compute_class_weight(
                class_weight="balanced",
                classes=np.unique(label_ids),
                y=ner_tags,
            )
        ).float()

        ner_modelling.args.class_weights = ner_modelling.class_weights.tolist()


    ner_modelling.save_config(
        output_dir=ner_modelling.output_dir,
        metrics_dir=ner_modelling.metrics_dir,
        args=ner_modelling.args,
    )

    train_wrapped, train_loader = create_data_loader(
        data_wrapped=ner_modelling.tokenize_and_wrap_data(ner_modelling.data.train),
        batch_size=args.train_batch_size,
        data_collator=ner_modelling.data_collator,
    )
    ner_modelling.args.max_length = max_length
    _, eval_loader = create_data_loader(
        data_wrapped=ner_modelling.tokenize_and_wrap_data(ner_modelling.data.eval),
        batch_size=ner_modelling.args.eval_batch_size,
        data_collator=ner_modelling.data_collator,
        shuffle=False,
    )

    model = model.train()

    step = 0
    eval_scores = []
    for epoch in tqdm(range(ner_modelling.args.epochs), desc="Epoch", unit="epoch"):
        model = model.to(ner_modelling.args.device)

        for batch in tqdm(
            train_loader,
            desc=f"Epoch {epoch} of {ner_modelling.args.epochs}",
            unit="batch",
        ):
            model.train()
            if not args.weight_classes:
                output = model(
                    input_ids=batch["input_ids"].to(ner_modelling.args.device),
                    attention_mask=batch["attention_mask"].to(ner_modelling.args.device),
                    labels=batch["labels"].to(ner_modelling.args.device),
                )
            else:
                output = model(
                    input_ids=batch["input_ids"].to(ner_modelling.args.device),
                    attention_mask=batch["attention_mask"].to(ner_modelling.args.device),
                    labels=batch["labels"].to(ner_modelling.args.device),
                    class_weights=ner_modelling.class_weights.to(ner_modelling.args.device),
                )

            loss = output.loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if step > 0 and (step % ner_modelling.args.evaluate_steps == 0):
                eval_score = ner_modelling.evaluate(model=model, val_loader=eval_loader)
                eval_score.step = step
                eval_score.epoch = epoch
                eval_scores.append(eval_score)
                wandb.log({"eval f1": eval_score.f_1})
                wandb.log({"eval loss": eval_score.loss})
                wandb.log({"accuracy": eval_score.accuracy})
                wandb.log({"step": eval_score.step})
                wandb.log({"learning rate": learning_rate})

                if step >= int(
                    11 * args.evaluate_steps
                ):  # and eval_score.accuracy < 0.15:
                    tenth_last = eval_scores[-10]
                    last_nine = eval_scores[-9:]

                    max_acc = max(last_nine, key=lambda x: x.accuracy)

                    if (
                        step >= int(5 * args.evaluate_steps)
                        and eval_score.accuracy < 0.10
                    ):
                        print("Pruning trial")
                        raise optuna.exceptions.TrialPruned()

                    if max_acc.accuracy < tenth_last.accuracy:
                        print("Pruning trial")
                        raise optuna.exceptions.TrialPruned()
            step += 1

    max_f1 = max(reversed(eval_scores), key=lambda x: x.f_1)
    append_json_lines(
        output_dir=ner_modelling.metrics_dir,
        data=dataclasses.asdict(max_f1),
        filename="best_f1",
    )

    return max_f1.f_1


# e99e480bf10627b2fa2ed6f2a9fe58472e3cb992
def objective(trial):
    max_length = trial.suggest_categorical("max_length", [64, 128, 256])
    learning_rate = trial.suggest_float("learning_rate", 0.00004, 0.0006, log=True)
    weight_decay = trial.suggest_float("weight_decay", 0.0001, 0.999, log=True)

    wandb.login(key="3c41fac754b2accc46e0705fa9ae5534f979884a")

    wandb.init(
        reinit=True,
        name=f"lap-{args.load_alvenir_pretrained}-{round(learning_rate, 5)}-"
        f"{round(weight_decay, 5)}-{max_length}",
    )
    wandb.run.tags = ['NER', 'HP-tuning']
    f_1 = train_model(
        trial=trial,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        max_length=max_length,
    )

    trial.report(f_1, step=10)

    # Perform pruning check
    if trial.should_prune():
        raise optuna.exceptions.TrialPruned()

    return f_1


if __name__ == "__main__":
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=args.n_trials)

    print("Best trial:")
    best_trial = study.best_trial
    print("  Value: {:.6f}".format(best_trial.value))
    print("  Params: ")
    for key, value in best_trial.params.items():
        print("{}: {}".format(key, value))
