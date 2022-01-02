import torch


def train_one_epoch(model, optimizer, loss_func, train_loader):
    model.train()
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
    model.eval()
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


def train(epochs, model, optimizer, loss_func, train_loader, valid_loader=None, verbose=1, period_show=1):
    train_loss = []
    valid_loss = []
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
    return train_loss, valid_loss
