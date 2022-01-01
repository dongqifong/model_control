import abc
import numpy as np
from sklearn.metrics import (
    matthews_corrcoef, confusion_matrix, recall_score, precision_score, f1_score)


class MetricsBase(abc.ABC):
    @abc.abstractmethod
    def evaluate(y_true: np.ndarray, y_pred: np.ndarray, **kwargs):
        pass


class ConfusionMatrix(MetricsBase):
    @staticmethod
    def evaluate(y_true: np.ndarray, y_pred: np.ndarray, **kwargs):
        return confusion_matrix(y_true, y_pred, **kwargs)


class Recall(MetricsBase):
    @staticmethod
    def evaluate(y_true: np.ndarray, y_pred: np.ndarray, **kwargs):
        return recall_score(y_true, y_pred, **kwargs)


class Precision(MetricsBase):
    @staticmethod
    def evaluate(y_true: np.ndarray, y_pred: np.ndarray, **kwargs):
        return precision_score(y_true, y_pred, **kwargs)


class F1Score(MetricsBase):
    @staticmethod
    def evaluate(y_true: np.ndarray, y_pred: np.ndarray, **kwargs):
        return f1_score(y_true, y_pred, **kwargs)


class MCC(MetricsBase):
    @staticmethod
    def evaluate(y_true: np.ndarray, y_pred: np.ndarray, **kwargs):
        return matthews_corrcoef(y_true, y_pred, **kwargs)


if __name__ == "__main__":
    y_true = [+1, +1, +1, -1]
    y_pred = [+1, -1, +1, +1]
    print(ConfusionMatrix.evaluate(y_true, y_pred))
    print(Recall.evaluate(y_true, y_pred))
    print(Precision.evaluate(y_true, y_pred))
    print(F1Score.evaluate(y_true, y_pred))
    print(MCC.evaluate(y_true, y_pred))
