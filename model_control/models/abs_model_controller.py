import abc
import numpy as np


class ControllerBase(abc.ABC):
    @abc.abstractmethod
    def read_config(self, config_name: str):
        pass

    @abc.abstractmethod
    def build(self):
        pass

    @abc.abstractmethod
    def load_weight(self):
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
