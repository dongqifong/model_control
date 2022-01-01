from numpy import ndarray
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


class ModelDataset(Dataset):
    def __init__(self, x: np.ndarray, y: np.ndarray) -> None:
        super().__init__()
        self.x = x
        self.y = y

    def __getitem__(self, index):
        x = torch.tensor(self.x[index:index+1, :]).float()
        if self.y is not None:
            y = torch.tensor(self.y[index]).long()
        return x, y

    def __len__(self):
        return len(self.x)


class MakeLoader:
    def __init__(self, x, y, config={}) -> None:
        shuffle = config["shuffle"] if "shuffle" in config.keys() else True
        batch_size = config["batch_size"] if "batch_size" in config.keys(
        ) else 1
        self.dataset = ModelDataset(x, y)
        self.dataloader = DataLoader(
            self.dataset, batch_size=batch_size, shuffle=shuffle)
        return None

    def get_loader(self):
        return self.dataloader


if __name__ == "__main__":
    from test_model import ModelMonitor
    model = ModelMonitor()

    n_sample = 30
    input_size = 51200
    xx = np.random.random((n_sample, input_size))
    yy = np.random.random(n_sample,)

    dataloader = MakeLoader(xx, yy, config={"batch_size": 15}).get_loader()

    with torch.no_grad():
        for epoch in range(10):
            for x, y in dataloader:
                print("x_shape=", x.shape)
                print("y_shape=", y.shape)
                out = model(x)
                break
            break
    print(out.shape)
    print(y.shape)
