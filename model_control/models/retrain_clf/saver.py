import torch


def save(model, model_name: str, verbose=1):
    saved_model_name = model_name + "_state_dict.pth"
    torch.save(model.state_dict(), saved_model_name)
    if verbose == 1:
        print(f"Saved model name:{saved_model_name }")
    return None
