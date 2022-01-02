import torch
import torch.nn as nn
import torch.optim as optim


def train_one_epoch(model, optimizer, loss_func, train_loader):
    running_loss = 0
    for batch_idx, (x, y) in enumerate(train_loader):
        out = model(x)
        optimizer.zero_grad()
        loss = loss_func(out, y)
        loss.backward()
        optimizer.step()
        running_loss = running_loss + loss.item()
    return round(running_loss / (batch_idx+1), 5)


def valid_one_epoch(model, loss_func, valid_loader):
    running_loss = 0
    with torch.no_grad():
        for batch_idx, (x, y) in enumerate(valid_loader):
            out = model(x)
            loss = loss_func(out, y)
            running_loss = running_loss + loss.item()
    return round(running_loss / (batch_idx+1), 5)


def show_progress(epoch, epochs, period_show, train_loss=[], valid_loss=[], verbose=1):
    if verbose == 1 and (((epoch+1) % period_show == 0) or epoch == 0):
        print("\r" +
              f"Progress: [{epoch+1}/{epochs}], training loss:[{train_loss[-1]}], valid loss:[{None if len(valid_loss)==0 else valid_loss[-1]}]", end="")
    return None


def train(epochs, model, optimizer, loss_func, train_loader, valid_loader=None, train_loss=[], valid_loss=[], verbose=1, period_show=1):
    for epoch in range(epochs):
        if valid_loader is not None:
            running_loss_valid = valid_one_epoch(
                model, loss_func, valid_loader)
            valid_loss.append(running_loss_valid)
        running_loss_train = train_one_epoch(
            model, optimizer, loss_func, train_loader)
        train_loss.append(running_loss_train)
        show_progress(epoch, epochs, period_show, train_loss,
                      valid_loss, verbose=verbose)
    print()
    print("Training finished !")
    return None


if __name__ == "__main__":
    import numpy as np
    from make_data_loader import get_loader
    from test_model import ModelMonitor
    model = ModelMonitor()

    n_sample = 30
    input_size = 51200

    train_x = np.random.random((n_sample, input_size))
    train_y = np.random.random(n_sample,)
    train_loader = get_loader(train_x, train_y, config={
        "batch_size": 15})

    valid_x = np.random.random((n_sample, input_size))
    valid_y = np.random.random(n_sample,)
    valid_loader = get_loader(
        valid_x, valid_y, config={"batch_size": 15})

    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    loss_func = nn.CrossEntropyLoss()

    train(epochs=10, model=model, optimizer=optimizer, loss_func=loss_func,
          train_loader=train_loader, valid_loader=None, train_loss=[], valid_loss=[], verbose=1, period_show=1)
