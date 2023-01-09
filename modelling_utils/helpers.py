import dataclasses
import json
import os
import sys
from typing import List, Tuple

from opacus.validators import ModuleValidator
from torch.optim.lr_scheduler import LinearLR

from data_utils.custom_dataclasses import EvalScore


def validate_model(model, strict_validation: bool = False):
    errors = ModuleValidator.validate(model, strict=strict_validation)
    if errors:
        print("Model is not compatible for DF with opacus. Please fix errors.")
        print(errors)
        sys.exit(1)
    else:
        print("Model is compatible for DP with opacus.")


def create_scheduler(optimizer, start_factor: float, end_factor: float, total_iters: int):
    """
    Create scheduler for learning rate warmup and decay
    :param optimizer: optimizer - for DP training of type DPOPtimizer
    :param start_factor: learning rate factor to start with
    :param end_factor: learning rate factor to end at
    :param total_iters: Number of steps to iterate
    :return: learning rate scheduler of type LinearLR
    """
    return LinearLR(optimizer, start_factor=start_factor,
                    end_factor=end_factor,
                    total_iters=total_iters)


def get_lr(optimizer):
    """
    Get current learning rate from optimizer
    :param optimizer:
    :return: list of learning rates in all groups
    """
    lrs = []
    for param_group in optimizer.param_groups:
        lrs.append(param_group['lr'])
    return lrs


def get_metrics(eval_scores: List[EvalScore],
                eval_metrics: List[str]) -> Tuple[dict, bool]:
    """
    Compute min loss and max accuracy based on all values from evaluation
    :param f1s: List[dict] of all f1 scores computed by evaluate()
    :param losses: List[dict] of all losses computed by evaluate()
    :param accuracies: List[dict] of all accuracies computed by evaluate()
    :param freeze_layers_n_steps: only get the best performance after model is un-freezed
    :return: The best metric for loss, accuracy and f1
    """
    save_best_model = True

    # For max and min we reverse the list, such that we get the last element if equal
    min_loss = min(reversed(eval_scores), key=lambda x: x.loss)
    max_acc = max(reversed(eval_scores), key=lambda x: x.accuracy)
    max_f1 = max(reversed(eval_scores), key=lambda x: x.f_1)
    current_step = max(eval_scores, key=lambda x: x.step)

    best_steps = {'loss': min_loss.step, 'accuracy': max_acc.step, 'f_1': max_f1.step}
    for metric in eval_metrics:
        if best_steps[metric] != current_step.step:
            save_best_model = False
            break

    best_metric_dict = {'loss': {'epoch': min_loss.epoch,
                                 'step': min_loss.step,
                                 'score': min_loss.loss},
                        'accuracy': {'epoch': max_acc.epoch,
                                     'step': max_acc.step,
                                     'score': max_acc.accuracy},
                        'f_1': {'epoch': max_f1.epoch,
                                'step': max_f1.step,
                                'score': max_f1.f_1}
                        }

    return best_metric_dict, save_best_model


def log_train_metrics(epoch: int, step: int, lr: float, loss: float,
                      logging_steps: int):
    if step % logging_steps == 0:
        print(f"\tTrain Epoch: {epoch} \t"
              f"Step: {step} \t LR: {lr}\t"
              f"Loss: {loss:.6f} ")


def log_train_metrics_dp(epoch: int, step: int, lr: float,
                         loss: float, eps: float, delta: float,
                         logging_steps: int):
    if step % logging_steps == 0:
        print(
            f"\tTrain Epoch: {epoch} \t"
            f"Step: {step} \t LR: {lr}\t"
            f"Loss: {loss:.6f} "
            f"(ε = {eps:.2f}, "
            f"δ = {delta})")


# def get_best_metric(data: List[dict], threshold: int, max_better: bool = True):
#     """
#     Return best metric at step higher than threshold
#     @param data: metric data
#     @param threshold: Get only best metric above threshold
#     @param max_better: True if higher is better
#     @return: The best metric score
#     """
#     if max_better:
#         return max([x for x in data if x['step'] > threshold], key=lambda x: x['score'])
#
#     return min([x for x in data if x['step'] > threshold], key=lambda x: x['score'])
#

def save_key_metrics_mlm(output_dir: str, args,
                         best_metrics: dict,
                         total_steps: int, filename: str = 'key_metrics'):
    """
    Save important args and performance for benchmarking
    :param output_dir: output directory
    :param args: args parsed from ArgParser
    :param best_acc: The best accuracy dict from list of dicts
    :param best_loss: The best loss dict from list of dicts
    :param filename: output filename
    :return:
    """
    metrics = {'output_name': args.output_name,
               'best_metrics': best_metrics,
               'lr': args.learning_rate,
               'epochs': args.epochs,
               'train_batch_size': args.train_batch_size,
               'eval_batch_size': args.eval_batch_size,
               'max_length': args.max_length,
               'lr_warmup_steps': args.lr_warmup_steps,
               'replace_head': args.replace_head,
               'freeze_layers_n_steps': args.freeze_layers_n_steps,
               'lr_freezed': args.lr_freezed,
               'lr_freezed_warmup_steps': args.lr_freezed_warmup_steps,
               'dp': args.differential_privacy,
               'epsilon': args.epsilon,
               'delta': args.delta,
               'lot_size': args.lot_size,
               'whole_word_mask': args.whole_word_mask,
               'train_data': args.train_data,
               'total_steps': total_steps}

    with open(os.path.join(output_dir, filename + '.json'), 'w',
              encoding='utf-8') as outfile:
        json.dump(metrics, outfile)


def save_key_metrics_sc(output_dir: str, args,
                        best_metrics: dict,
                        total_steps: int,
                        filename: str = 'key_metrics'):
    """
    Save important args and performance for benchmarking
    :param output_dir: output directory
    :param args: args parsed from ArgParser
    :param best_acc: The best accuracy dict from list of dicts
    :param best_loss: The best loss dict from list of dicts
    :param filename: output filename
    :return:
    """
    metrics = {'output_name': args.output_name,
               'best_metrics': best_metrics,
               'lr': args.learning_rate,
               'epochs': args.epochs,
               'train_batch_size': args.train_batch_size,
               'eval_batch_size': args.eval_batch_size,
               'max_length': args.max_length,
               'lr_warmup_steps': args.lr_warmup_steps,
               'freeze_layers_n_steps': args.freeze_layers_n_steps,
               'lr_freezed': args.lr_freezed,
               'lr_freezed_warmup_steps': args.lr_freezed_warmup_steps,
               'dp': args.differential_privacy,
               'epsilon': args.epsilon,
               'delta': args.delta,
               'lot_size': args.lot_size,
               'train_data': args.train_data,
               'total_steps': total_steps}

    with open(os.path.join(output_dir, filename + '.json'), 'w',
              encoding='utf-8') as outfile:
        json.dump(metrics, outfile)
