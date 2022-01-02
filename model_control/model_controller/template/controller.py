import numpy as np
from ..abs_model_controller import ControllerBase


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
        # return: None
        pass

    def load_weight(self, model_path: str):
        # return: None
        pass

    def compile(self, config={}):
        # return: None
        pass

    def train(self, epochs=1, verbose=1, period_show=1):
        # return: None
        pass

    def predict(self, x: np.ndarray, batch_size=1):
        # return: predicted value(np.ndarray)
        pass

    def save(self):
        # return: None
        pass

    def evaluate(self, y_true: np.ndarray, y_pred: np.ndarray):
        # return: metrics score
        pass
