from dataclasses import dataclass
from enum import Enum

import torch


@dataclass
class PredictionOutput:
    prediction: str = None
    logits: torch.Tensor = None
    softmax: dict = None
    class_ids: int = None

@dataclass
class EvalScore:
    epoch: int = None
    step: int = None
    loss: float = None
    accuracy: float = None
    f_1: dict = None


class DataType(Enum):
    TRAIN = 'train.json'
    VAL = 'validation.json'
    TEST = 'test.json'


class LoadModelType(Enum):
    TRAIN = 1
    EVAL = 2
    INFERENCE = 3
