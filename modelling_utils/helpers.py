import json
import os
import sys
from typing import List

from opacus.validators import ModuleValidator
from torch.optim.lr_scheduler import LinearLR


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


def get_metrics(freeze_layers_n_steps,
                losses: List[dict] = None,
                accuracies: List[dict] = None,
                f1s: List[dict] = None):
    """
    Compute min loss and max accuracy based on all values from evaluation
    :param f1s: List[dict] of all f1 scores computed by evaluate()
    :param losses: List[dict] of all losses computed by evaluate()
    :param accuracies: List[dict] of all accuracies computed by evaluate()
    :param freeze_layers_n_steps: only get the best performance after model is un-freezed
    :return: Best metric for loss, accuracy and f1
    """
    min_loss = None
    max_acc = None
    max_f1 = None

    # if not [x for x in losses if x['step'] > freeze_layers_n_steps]:
    #     return

    if losses:
        min_loss = get_best_metric(data=losses, threshold=freeze_layers_n_steps, max_better=False)
        # min_loss = min([x for x in losses if x['step'] > freeze_layers_n_steps], key=lambda x: x['score'])
    if accuracies:
        max_acc = get_best_metric(data=accuracies, threshold=freeze_layers_n_steps)
        # max_acc = max([x for x in accuracies if x['step'] > freeze_layers_n_steps], key=lambda x: x['score'])
    if f1s:
        max_f1 = get_best_metric(data=f1s, threshold=freeze_layers_n_steps)
        # max_f1 = max([x for x in f1s if x['step'] > freeze_layers_n_steps], key=lambda x: x['score'])

    return {'loss': min_loss, 'acc': max_acc, 'f1': max_f1}


def get_best_metric(data: List[dict], threshold: int, max_better: bool = True):
    """
    Return best metric at step higher than threshold
    @param data: metric data
    @param threshold: Get only best metric above threshold
    @param max_better: True if higher is better
    @return: The best metric score
    """
    try:
        if max_better:
            return max([x for x in data if x['step'] > threshold], key=lambda x: x['score'])

        return min([x for x in data if x['step'] > threshold], key=lambda x: x['score'])
    except:
        return None


def save_key_metrics_mlm(output_dir: str, args,
                         metrics: dict,
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
    metrics = {'key_metrics': [{'output_name': args.output_name,
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
                                'total_steps': total_steps,
                                'best_metrics': metrics}]}

    with open(os.path.join(output_dir, filename + '.json'), 'w',
              encoding='utf-8') as outfile:
        json.dump(metrics, outfile)


def save_key_metrics_sc(
    output_dir: str,
    args,
    metrics: dict,
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
               'total_steps': total_steps,
               'best_metrics': metrics}

    with open(os.path.join(output_dir, filename + '.json'), 'w',
              encoding='utf-8') as outfile:
        json.dump(metrics, outfile)
