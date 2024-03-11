import torch
from tqdm import tqdm
import numpy as np
import copy
import os


def train_one_epoch_1(model, optimizer, loss_fn, train_dl, device, scheduler):
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

def train_one_epoch(model, optimizer, loss_fn, train_dl, val_dl, device, scheduler):
    model.train()
    train_loss = []
    for (seq, label) in train_dl:
        seq = seq.to(device)
        label = label.to(device)

        # Model output and losses
        y_pred = model(seq)
        train_loss_value = loss_fn(y_pred, label)
        train_loss.append(train_loss_value.item())

        # 1.Clear the gradient 2.Backpropagation 3.Parameters optimization
        optimizer.zero_grad()
        train_loss_value.backward()
        optimizer.step()
    
    scheduler.step()

    # Validation
    model.eval() 
    val_loss = []
    with torch.no_grad():  # Do not calculate gradients to speed up computation
        for (seq, label) in val_dl:
            seq = seq.to(device)
            label = label.to(device)

            # Model output for validation data
            y_pred = model(seq)
            val_loss_value = loss_fn(y_pred, label)
            val_loss.append(val_loss_value.item())

    return train_loss, val_loss, train_loss_value.item(), val_loss_value.item()

def train_full(model, optimizer, loss_fn, train_dl, val_dl, device, args, scheduler):
    train_loss_app = []
    val_loss_app = []
    epoch = args.epoch
    min_loss = np.inf
    early_stop = False
    epochs_no_improve = 0

    best_model_path = os.path.join(args.checkpoint, 'ckpts_best/')

    if not os.path.exists(best_model_path):
        os.mkdir(best_model_path)

    ckpts_path = os.path.join(args.checkpoint, 'ckpts/')
    if not os.path.exists(ckpts_path):
        os.mkdir(ckpts_path)

    print(f'*******************Start training*******************')
    for i in tqdm(range(epoch)):
        train_loss, val_loss, train_loss_value, val_loss_value = train_one_epoch(model, optimizer, loss_fn, train_dl, val_dl, device, scheduler)
        # loss_value = np.mean(train_loss)
        loss_value = val_loss_value

        if val_loss_value < min_loss:
            min_loss = val_loss_value
            best_model = copy.deepcopy(model)
            state_best = {'models': best_model.state_dict()}
            epochs_no_improve = 0  # Reset counter
            best_epoch = i
        else:
            epochs_no_improve += 1

        # collect losses
        train_loss_app.append(train_loss_value)
        val_loss_app.append(val_loss_value)

        # print the progress and save the model for every 10 epochs
        if (i+1) % 10 == 0:
            print(f'epoch {i+1} | val_loss {val_loss_value:.8f}')
            # print('epoch {:03d} train_loss {:.8f}'.format(i, train_loss))
            current_model = copy.deepcopy(model)
            state = {'models': current_model.state_dict()}
            torch.save(state, ckpts_path + 'epoch_%i'%(i+1))

        # if save the improved model
        if args.save_improved_model:
            print(f'\nSaving the best model: epoch {i+1} | val loss {val_loss_value}')
            torch.save(state_best, best_model_path + 'epoch_%i_valloss%.4f' % ((i+1), val_loss_value))

        # Check for early stopping
        if epochs_no_improve == args.early_stop:
            print(f'\nEarly stopping triggered after {i+1} epochs!')
            early_stop = True
            break

    # if save the best model
    if not args.save_improved_model:
        print(f'\nThe best model saved: epoch {best_epoch+1} | val loss {min_loss}')
        torch.save(state_best, best_model_path + 'epoch_%i_valloss%.4f' % ((best_epoch+1), min_loss))
    print(f'*******************Done training*******************')

    if not early_stop:
        print(f'Finished training for {epoch} epochs without early stopping.')

    return train_loss_app, val_loss_app, (best_model_path + 'epoch_%i_valloss%.4f' % ((best_epoch+1), min_loss))
