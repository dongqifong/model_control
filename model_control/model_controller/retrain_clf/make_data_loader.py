import numpy as np
import torch
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


def get_loader(x, y, config={}):
    shuffle = config["shuffle"] if "shuffle" in config.keys() else True
    batch_size = config["batch_size"] if "batch_size" in config.keys() else 1
    dataset = ModelDataset(x, y)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader
