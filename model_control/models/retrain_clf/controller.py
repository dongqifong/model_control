from pathlib import Path
import json

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from model_control.models.abs_model_controller import ControllerBase
from model_control.models.retrain_clf.make_data_loader import get_loader
from model_control.models.retrain_clf import trainer
from model_control.models.retrain_clf import predictor
from model_control.models.retrain_clf import saver
from model_control.models.retrain_clf.model import ModelMonitor
from model_control.evaluation.metrics import MCC


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
        config_root_dir = Path.cwd() / "model_control/configs/retrain_clf"
        config_path = config_root_dir / config_name
        with open(config_path) as f:
            config_all = json.load(f)
        self.model_config = config_all["model_config"]
        self.compile_config = config_all["compile_config"]
        return None

    def build(self, **kwargs):
        self.model_config.update(kwargs)
        self.model = ModelMonitor(self.model_config)
        return None

    def load_weight(self, model_path: str):
        model_path = Path(model_path)
        print("model_path:", str(model_path))
        self.model.load_state_dict(torch.load(model_path))
        return None

    def compile(self, **kwargs):
        self.compile_config.update(kwargs)
        if "lr" not in self.compile_config.keys():
            lr = 1e-4
        else:
            lr = self.compile_config["lr"]
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.loss_func = nn.CrossEntropyLoss()

        self.train_loader = get_loader(
            self.train_x, self.train_y, self.compile_config)

        if self.valid_x is not None and self.valid_y is not None:
            self.valid_loader = get_loader(
                self.valid_x, self.valid_y)

        self.model_name = self.compile_config["model_name"]

        return None

    def train(self, epochs=1, verbose=1, period_show=1):
        train_loss, valid_loss = trainer.train(epochs=epochs, model=self.model, optimizer=self.optimizer, loss_func=self.loss_func,
                                               train_loader=self.train_loader, valid_loader=self.valid_loader, verbose=verbose, period_show=period_show)
        self.train_loss = self.train_loss + train_loss
        if len(valid_loss) > 0:
            self.valid_loss = self.valid_loss + valid_loss
        return None

    def predict(self, x: np.ndarray, batch_size=1):
        y_pred = predictor.predict(self.model, x=x, batch_size=batch_size)
        return y_pred

    def save(self, verbose=1):
        saver.save(self.model, self.model_name)
        return None

    def evaluate(self, y_true: np.ndarray, y_pred: np.ndarray):
        score = MCC.evaluate(y_true, y_pred)
        return score
