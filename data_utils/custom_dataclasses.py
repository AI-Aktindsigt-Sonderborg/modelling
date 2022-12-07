import numpy as np
from dataclasses import dataclass
from enum import Enum

import torch


@dataclass
class AsrInput:
    audio: np.ndarray
    reference: str

@dataclass
class NunaEmotions:
    anger: float
    disgust: float
    fear: float
    happiness: float
    sadness: float
    surprise: float

@dataclass
class NunaData:
    video_id: str
    emotions: NunaEmotions
    label: str = None
    audio: np.ndarray = None
    transcript: str = None

@dataclass
class PredictionOutput:
    prediction: str = None
    # logits: torch.Tensor = None
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

@dataclass
class MultiModalOutput:
    id: str
    label: str
    pred_audio: str
    pred_text: str
    pred_final: str
    conf_audio: float
    conf_text: float
    text: str