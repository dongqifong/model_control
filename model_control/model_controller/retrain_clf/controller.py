from pathlib import Path

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

    def build(self, model_config=None):
        self.model = ModelMonitor(model_config)
        return None

    def load_weight(self, model_path: str):
        model_path = Path(model_path)
        print("model_path:", str(model_path))
        self.model.load_state_dict(torch.load(model_path))
        return None

    def compile(self, config={}):
        lr = 1e-4 if "lr" not in config.keys() else config["lr"]
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.loss_func = nn.CrossEntropyLoss()

        loader_config = {}
        loader_config["batch_size"] = 1 if "batch_size" not in config.keys(
        ) else config["batch_size"]
        loader_config["shuffle"] = True if "shuffle" not in config.keys(
        ) else config["shuffle"]
        self.train_loader = get_loader(self.train_x, self.train_y)
        if self.valid_x is not None and self.valid_y is not None:
            self.valid_loader = get_loader(
                self.valid_x, self.valid_y)

        if "model_name" in config.keys():
            self.model_name = config["model_name"]
        return None

    def train(self, epochs=1, verbose=1, period_show=1):
        train_loss, valid_loss = trainer.train(epochs=epochs, model=self.model, optimizer=self.optimizer, loss_func=self.loss_func,
                                               train_loader=self.train_loader, valid_loader=self.valid_loader, verbose=verbose, period_show=period_show)
        self.train_loss = self.train_loss + train_loss
        if valid_loss is not None:
            self.valid_loss = self.valid_loss + valid_loss
        return None

    def predict(self, x: np.ndarray, batch_size=1):
        y_pred = predictor.predict(self.model, x=x, batch_size=batch_size)
        return y_pred

    def save(self):
        saver.save(self.model, self.model_name)
        return None

    def evaluate(self, y_true: np.ndarray, y_pred: np.ndarray):
        score = MCC.evaluate(y_true, y_pred)
        return score
