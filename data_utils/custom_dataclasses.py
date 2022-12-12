from dataclasses import dataclass
from enum import Enum
import numpy as np
import torch


@dataclass
class AsrInput:
    audio: np.ndarray
    reference: str

@dataclass
class PredictionOutput:
    prediction: str = None
    logits: torch.Tensor = None
    softmax: {} = None
    class_ids: int = None


class DataType(Enum):
    TRAIN = 'train.json'
    VAL = 'validation.json'
    TEST = 'test.json'
    FULL = 'one_row_per_video_clip.json'


class LoadModelType(Enum):
    TRAIN = 1
    EVAL = 2
    INFERENCE = 3
