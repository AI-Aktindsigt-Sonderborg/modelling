from dataclasses import dataclass
from enum import Enum

import numpy as np
import torch


@dataclass
class EmbeddingOutput:
    sentence: str = None
    label: str = None
    prediction: str = None
    decoded_text: str = None
    embedding: np.array = None

@dataclass
class CosineSimilarity:
    input_sentence: str = None
    reference_sentence: str = None
    input_label: str = None
    reference_label: str = None
    cosine_sim: float = None



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
