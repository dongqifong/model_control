from pathlib import Path
import json

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from ..abs_model_controller import ControllerBase
from .make_data_loader import get_loader
from . import trainer
from . import predictor
from . import saver
from ...models.retrain_clf.model import ModelMonitor
from ...evaluation.metrics import MCC


class ModelController(ControllerBase):
    def __init__(self, train_x: np.ndarray, train_y: np.ndarray, valid_x: np.ndarray = None, valid_y: np.ndarray = None) -> None:
        self.train_x = train_x
        self.train_y = train_y
        self.valid_x = valid_x
        self.valid_y = valid_y
        self.model = None
        self.train_loader = None
        self.valid_loader = None
        self.train_loss = []
        self.valid_loss = []
        self.model_name = "dafault_model_name"
        self.model_config = {}
        self.compile_config = {}

    def read_config(self, config_name: str):
        # read config for model and compiling
        # return: None
        return None

    def build(self, **kwargs):
        # build model with respective to model config
        # return: None
        return None

    def load_weight(self, model_path: str):
        # load exsisted model weights
        # return: None
        return None

    def compile(self, **kwargs):
        # load training config
        # return: None
        return None

    def train(self, epochs=1, verbose=1, period_show=1):
        # training process
        # return: None
        return None

    def predict(self, x: np.ndarray, batch_size=1):
        # predict output
        # return: y_pred(np.ndarray)
        pass

    def save(self):
        # save model
        # return: None
        return None

    def evaluate(self, y_true: np.ndarray, y_pred: np.ndarray):
        # calculate metrics
        # return metrics's score
        pass
