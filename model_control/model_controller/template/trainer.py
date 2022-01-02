
def show_progress(epoch, epochs, period_show, train_loss=[], valid_loss=[], verbose=1):
    # Used to show loss during training process
    if verbose == 1 and (((epoch+1) % period_show == 0) or epoch == 0):
        print("\r" +
              f"Progress: [{epoch+1}/{epochs}], training loss:[{train_loss[-1]}], valid loss:[{None if len(valid_loss)==0 else valid_loss[-1]}]", end="")
    return None


def train(epochs, model, optimizer, loss_func, train_loader, valid_loader=None, verbose=1, period_show=1):
    # return training loss and validation loss
    train_loss = []
    valid_loss = []
    for epoch in range(epochs):
        show_progress(epoch, epochs, period_show, train_loss,
                      valid_loss, verbose=verbose)
    print()
    print("Training finished !")
    return train_loss, valid_loss
