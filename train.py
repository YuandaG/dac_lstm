import torch
from tqdm import tqdm
import numpy as np
import copy
import os


def train_one_epoch(model, optimizer, loss_fn, train_dl, device, scheduler):
    model.train()
    train_loss = []
    for (seq, label) in train_dl:
        seq = seq.to(device)
        label = label.to(device)

        '''model output and losses'''
        y_pred = model(seq)
        loss = loss_fn(y_pred, label)
        train_loss.append(loss.item())

        '''1.clear the gradient 2.bp 3.parameters optimisation'''
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    scheduler.step()

    '''Validation'''
    # val_loss = get_val_loss(args, model, Val)
    return train_loss


def train_full(model, optimizer, loss_fn, train_dl, device, args, scheduler):
    train_loss_app = []
    min_loss = 10
    epoch = args.epoch
    best_model_path = os.path.join(args.checkpoint, 'ckpts_best/')
    if not os.path.exists(best_model_path):
        os.mkdir(best_model_path)

    ckpts_path = os.path.join(args.checkpoint, 'ckpts/')
    if not os.path.exists(ckpts_path):
        os.mkdir(ckpts_path)

    print(f'*******************Start training*******************')
    for i in tqdm(range(epoch)):
        train_loss = train_one_epoch(model, optimizer, loss_fn, train_dl, device, scheduler)
        loss_value = np.mean(train_loss)

        if loss_value < min_loss or i == 0:
            min_loss = loss_value
            best_model = copy.deepcopy(model)
            state_best = {'models': best_model.state_dict()}
            best_epoch = i

        # print the progress and save the model for every 10 epochs
        if (i+1) % 10 == 0:
            print(f'epoch {i+1} | train_loss {loss_value:.8f}')
            # print('epoch {:03d} train_loss {:.8f}'.format(i, train_loss))
            current_model = copy.deepcopy(model)
            state = {'models': current_model.state_dict()}
            torch.save(state, ckpts_path + 'epoch_%i'%(i+1))

        # if save the improved model
        if args.save_improved_model:
            print(f'\nSaving the best model: epoch {i+1} | train loss {loss_value}')
            torch.save(state_best, best_model_path + 'epoch_%i_trainloss%.4f' % ((i+1), loss_value))

        # collect losses
        train_loss_app.append(loss_value)

    # if save the best model
    if not args.save_improved_model:
        print(f'\nThe best model saved: epoch {best_epoch+1} | train loss {min_loss}')
        torch.save(state_best, best_model_path + 'epoch_%i_trainloss%.4f' % ((best_epoch+1), min_loss))
    print(f'*******************Done training*******************')

    return train_loss_app, (best_model_path + 'epoch_%i_trainloss%.4f' % ((best_epoch+1), min_loss))
