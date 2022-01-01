import abc
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from make_data_loader import MakeLoader
import trainer
import predictor
import saver
from test_model import ModelMonitor


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
        self.train_loader = MakeLoader(self.train_x, self.train_y)
        if self.valid_x is not None and self.valid_y is not None:
            self.valid_loader = MakeLoader(self.valid_x, self.valid_y)

        if "model_name" in config.keys():
            self.model_name = config["model_name"]
        return None

    def train(self, epochs, verbose=1, period_show=1):
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
        return None
