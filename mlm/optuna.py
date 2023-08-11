# pylint: skip-file
import dataclasses

import torch
import wandb
from tqdm import tqdm

import optuna
from mlm.modelling_utils.input_args import MLMArgParser
from mlm.modelling_utils.mlm_modelling import MLMModelling
from shared.modelling_utils.helpers import create_data_loader
from shared.utils.helpers import append_json_lines

mlm_parser = MLMArgParser()

args, leftovers = mlm_parser.parser.parse_known_args()
args.test = False
args.replace_head = False
args.train_data = "train.jsonl"
args.eval_data = "test.jsonl"
args.evaluate_steps = 25
args.logging_steps = 25
args.train_batch_size = 32
args.eval_batch_size = 32
args.epochs = 5
args.n_trials = 10
args.load_alvenir_pretrained = False
args.model_name = "base"
args.differential_privacy = False

model_name_to_print = (
    args.custom_model_name if args.custom_model_name else args.model_name
)

mlm_modelling = MLMModelling(args=args)


def train_model(trial, learning_rate, max_length):
    model = mlm_modelling.get_model()

    for param in model.bert.embeddings.parameters():
        param.requires_grad = False

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=mlm_modelling.args.weight_decay,
    )

    mlm_modelling.load_data()

    mlm_modelling.save_config(
        output_dir=mlm_modelling.output_dir,
        metrics_dir=mlm_modelling.metrics_dir,
        args=mlm_modelling.args,
    )

    train_wrapped, train_loader = create_data_loader(
        data_wrapped=mlm_modelling.tokenize_and_wrap_data(mlm_modelling.data.train),
        batch_size=args.train_batch_size,
        data_collator=mlm_modelling.data_collator,
    )

    mlm_modelling.args.max_length = max_length
    _, eval_loader = create_data_loader(
        data_wrapped=mlm_modelling.tokenize_and_wrap_data(mlm_modelling.data.eval),
        batch_size=mlm_modelling.args.eval_batch_size,
        data_collator=mlm_modelling.data_collator,
        shuffle=False,
    )

    model = model.train()
    step = 0
    eval_scores = []
    for epoch in tqdm(range(mlm_modelling.args.epochs), desc="Epoch", unit="epoch"):
        model = model.to(mlm_modelling.args.device)

        for batch in tqdm(
            train_loader,
            desc=f"Epoch {epoch} of {mlm_modelling.args.epochs}",
            unit="batch",
        ):
            model.train()

            output = model(
                input_ids=batch["input_ids"].to(mlm_modelling.args.device),
                attention_mask=batch["attention_mask"].to(mlm_modelling.args.device),
                labels=batch["labels"].to(mlm_modelling.args.device),
            )
            loss = output.loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if step > 0 and (step % mlm_modelling.args.evaluate_steps == 0):
                eval_score = mlm_modelling.evaluate(model=model, val_loader=eval_loader)
                eval_score.step = step
                eval_score.epoch = epoch
                eval_scores.append(eval_score)

                wandb.log({"eval f1": eval_score.f_1})
                wandb.log({"eval loss": eval_score.loss})
                wandb.log({"accuracy": eval_score.accuracy})
                wandb.log({"step": eval_score.step})
                wandb.log({"learning rate": learning_rate})
                if step >= 200:  # and eval_score.accuracy < 0.15:
                    print(f"\n"
                          f"step: {eval_score.step}\t"
                          f"eval loss: {eval_score.loss}\t"
                          f"eval acc: {eval_score.accuracy}\t"
                          f"eval f1: {eval_score.f_1}")
                    last_10 = eval_scores[-2:]
                    max_acc = max(last_10, key=lambda x: x.accuracy)
                    if not max_acc.accuracy >= last_10[0].accuracy:
                        print("Pruning trial")
                        raise optuna.exceptions.TrialPruned()

            step += 1

    max_f1 = max(reversed(eval_scores), key=lambda x: x.f_1)
    append_json_lines(
        output_dir=mlm_modelling.metrics_dir,
        data=dataclasses.asdict(max_f1),
        filename="best_f1",
    )

    return max_f1.f_1


# e99e480bf10627b2fa2ed6f2a9fe58472e3cb992
def objective(trial):
    # epsilon = trial.suggest_float("epsilon", 1.0, 10.0)
    # lot_size = trial.suggest_categorical("lot_size", [64, 128, 256, 512])
    max_length = trial.suggest_categorical("max_length", [64, 128, 256])
    # max_length = trial.suggest_categorical("max_length", [2, 4, 8])
    #    delta = trial.suggest_float("delta", 1e-6, 1e-2)
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
    # wandb.login(key="388da466a818b5fcfcc2e6c5365e971daa713566")
    wandb.login(key="3c41fac754b2accc46e0705fa9ae5534f979884a")

    wandb.init(
        reinit=True,
        name=f"lap-{args.load_alvenir_pretrained}-{round(learning_rate, 5)}-{max_length}",
    )


    f_1 = train_model(
        trial=trial,
        learning_rate=learning_rate,
        # epsilon=epsilon,
        # delta=delta,
        # lot_size=lot_size,
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
