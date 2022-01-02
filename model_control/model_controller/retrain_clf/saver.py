import torch


def save(model, model_name: str):
    torch.save(model.state_dict(), model_name + "_state_dict.pth")
