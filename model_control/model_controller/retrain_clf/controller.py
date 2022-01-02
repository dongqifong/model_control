import abc

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from .make_data_loader import get_loader
from . import trainer
from . import predictor
from . import saver
from .test_model import ModelMonitor
from ...evaluation.metrics import MCC


class ControllerBase(abc.ABC):
    @abc.abstractmethod
    def build(self):
        pass

    @abc.abstractmethod
    def compile(self):
        pass

    @abc.abstractmethod
    def train(self, epoch):
        pass

    @abc.abstractmethod
    def predict(self, x: np.ndarray):
        pass

    @abc.abstractmethod
    def save(self, model, model_name: str):
        pass

    @abc.abstractmethod
    def evaluate(self, y_true: np.ndarray, y_pred: np.ndarray):
        pass


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
        trainer.train(epochs=epochs, model=self.model, optimizer=self.optimizer, loss_func=self.loss_func,
                      train_loader=self.train_loader, valid_loader=self.valid_loader, train_loss=self.train_loss, valid_loss=self.valid_loss, verbose=verbose, period_show=period_show)
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


if __name__ == "__main__":

    n_sample = 30
    input_size = 51200

    train_x = np.random.random((n_sample, input_size))
    train_y = np.random.choice(2, n_sample)

    valid_x = np.random.random((n_sample, input_size))
    valid_y = np.random.choice(2, n_sample)

    model_controller = ModelController(train_x, train_y, valid_x, valid_y)
    model_controller.build()
    model_controller.compile({"model_name": "test_model"})
    model_controller.train(5)
    y_pred = model_controller.predict(train_x, batch_size=5)
    # model_controller.save()
    score = model_controller.evaluate(train_y, y_pred)

    print(model_controller.model_name)
    print(model_controller.model)
    print(model_controller.train_loader)
    print(model_controller.valid_loader)
    print(score)
