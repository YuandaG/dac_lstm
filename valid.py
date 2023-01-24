import torch

@torch.inference_mode()
def valid_one_epoch(model, loss_fn, valid_dl, device):
    model.eval()


    y_train = valid_dl.to(device)
    y_pred = model(y_train)
    loss = loss_fn(y_pred, y_train)

    return loss