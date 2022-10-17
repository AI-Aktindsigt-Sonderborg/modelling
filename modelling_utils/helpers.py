import sys
from typing import List

from opacus.validators import ModuleValidator
from torch.optim.lr_scheduler import LinearLR

def validate_model(model):
    errors = ModuleValidator.validate(model, strict=False)
    if errors:
        print("Model is not compatible for DF with opacus. Please fix errors.")
        print(errors)
        sys.exit(0)
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

def get_max_acc_min_loss(losses: List[dict], accuracies: List[dict], freeze_layers_n_steps):
    """
    Compute min loss and max accuracy based on all values from evaluation
    :param losses: List[dict] of all losses evaluate()
    :param accuracies: List[dict] of all accuracies computed by evaluate()
    :param freeze_layers_n_steps: only get best performance after model is un-freezed
    :return: Best metric for loss and accuracy
    """
    min_loss = min([x for x in losses if x['step'] > freeze_layers_n_steps],
                   key=lambda x: x['loss'])
    max_acc = max([x for x in accuracies if x['step'] > freeze_layers_n_steps],
                  key=lambda x: x['acc'])
    return min_loss, max_acc