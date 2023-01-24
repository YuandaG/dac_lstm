import numpy as np
import torch
from torch.optim import lr_scheduler


def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def fetch_scheduler(optimizer, schedule='CosineAnnealingLR'):
    if schedule == 'CosineAnnealingLR':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=500, eta_min=1e-6)
    elif schedule == 'CosineAnnealingWarmRestarts':
        scheduler = lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=100, eta_min=1e-6)
    elif schedule == None:
        return None
    return scheduler


def R2(y_test, y_true):
    return 1 - ((y_test - y_true) ** 2).sum() / ((y_true - y_true.mean()) ** 2).sum()


def MAPE(x, y):
    return np.mean(abs((y - x) / (x + 1)))



