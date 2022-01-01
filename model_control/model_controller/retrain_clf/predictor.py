import numpy as np
import torch
from make_data_loader import MakeLoader
from test_model import ModelMonitor


def predict(model, x: np.ndarray, batch_size=1):
    dummy_y = np.random.random((len(x)))
    test_loader = MakeLoader(
        x, dummy_y, config={"batch_size": batch_size}).get_loader()
    y_pred = []
    with torch.no_grad():
        for batch_idx, (x, y) in enumerate(test_loader):
            out = model(x)
            # out = torch.randn((10, 2))
            y_out = out.argmax(dim=-1)
            y_pred.append(y_out.numpy())
    return np.concatenate(y_pred, axis=0)


if __name__ == "__main__":
    n_sample = 30
    input_size = 51200
    test_x = np.random.random((n_sample, input_size))

    model = ModelMonitor()
    y_pred = predict(model, x=test_x, batch_size=5)
    print(y_pred)
