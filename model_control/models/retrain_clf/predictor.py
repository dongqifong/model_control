import numpy as np
import torch

from model_control.models.retrain_clf.make_data_loader import get_loader


def predict(model, x: np.ndarray, batch_size=1):
    dummy_y = np.random.random((len(x)))
    test_loader = get_loader(
        x, dummy_y, config={"batch_size": batch_size})
    y_pred = []
    with torch.no_grad():
        for batch_idx, (x, y) in enumerate(test_loader):
            out = model(x)
            # out = torch.randn((10, 2))
            y_out = out.argmax(dim=-1)
            y_pred.append(y_out.numpy())
    return np.concatenate(y_pred, axis=0)
