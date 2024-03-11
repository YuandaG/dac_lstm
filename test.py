import torch
from tqdm import tqdm
import numpy as np

from itertools import chain
# target = list(chain.from_iterable(target.data.tolist()))


@torch.inference_mode()
def test_full(model, args, test, device):
    pred = np.empty((0, args.output_size))
    y = np.empty((0, args.output_size))
    print('*******************loading models*******************')
    model.load_state_dict(torch.load(args.load_ckpt)['models'])
    model.eval()
    print('*******************Predicting*******************')

    for (seq, target) in tqdm(test):
        target = target.detach().cpu().numpy()
        y = np.append(y, target, axis=0)
        seq = seq.to(device)
        with torch.no_grad():
            y_pred = model(seq)
            y_pred = y_pred.detach().cpu().numpy()
            pred = np.append(pred, y_pred, axis=0)
    return y, pred
