import time
import os
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.utils.data as data
import torch.nn.functional as F

def normalize_to_unit_interval(x, normalize_by = None):
    if normalize_by == None:
        max_val = np.max(x)
        return x / max_val, max_val
    else:
        return x / normalize_by, normalize_by

def load_CIFAR_batch(filename):
    """ load single batch of cifar """
    with open(filename, 'rb') as f:
        datadict = pickle.load(f, encoding='latin1')
        X = datadict['data']
        Y = datadict['labels']
        X = X.reshape(10000, 3, 32, 32).astype('float32')
        Y = np.array(Y)
    return X, Y


def load_dataset(dataset, num_training_samples=-1):
    if dataset == 'cifar-10':
        datapath = 'data/cifar-10-batches-py'
        xs = []
        ys = []
        for b in range(1,6):
            f = os.path.join(datapath, 'data_batch_%d' % (b, ))
            X, Y = load_CIFAR_batch(f)
            xs.append(X)
            ys.append(Y)
        x_all = np.concatenate(xs)
        y_all = np.concatenate(ys)
        x_train = x_all[:-5000]
        x_val = x_all[-5000:]
        y_train = y_all[:-5000]
        y_val = y_all[-5000:]
        del X, Y
        x_test, y_test = load_CIFAR_batch(os.path.join(datapath, 'test_batch'))

        x_train, normalize_by = normalize_to_unit_interval(x_train)
        x_val, _ = normalize_to_unit_interval(x_val, normalize_by)
        x_test, _ = normalize_to_unit_interval(x_test, normalize_by)

    else:
        raise ValueError('Import for the dataset you have provided is not yet implemented.')

    if num_training_samples > 0:
        x_train = x_train[:num_training_samples]
        y_train = y_train[:num_training_samples]

    return x_train, y_train, x_val, y_val, x_test, y_test


def compute_loss_and_accuracy(model, loss_fn, x, y, batch_size=64, device='cuda'):
    num_samples = x.shape[0]
    data = torch.utils.data.TensorDataset(torch.from_numpy(x), torch.from_numpy(y))
    loader = torch.utils.data.DataLoader(dataset=data, batch_size=batch_size, shuffle=False)

    correct_samples = 0
    loss = 0.
    num_batches = 0
    model.train(False)
    with torch.no_grad():
        for sample_x, sample_y in loader:
            num_batches += 1
            sample_x = sample_x.to(device)
            sample_y = sample_y.to(device)
            sample_out = model(sample_x)
            loss += loss_fn(sample_out, sample_y).item()
            _, y_pred = sample_out.max(dim=1)
            correct_samples  += sample_y.numel() - torch.nonzero(y_pred - sample_y).numel()
    acc = float(correct_samples) / float(num_samples)
    loss = float(loss) / float(num_batches)
    model.train(True)
    return loss, acc

def train(model, loss_fn, optimizer, data, num_epochs, batch_size, scheduler=None, device='cuda'):
    x_train, y_train, x_val, y_val, x_test, y_test = data
    train_data = torch.utils.data.TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train))
    train_loader = torch.utils.data.DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)

    start_time = time.time()
    training_metrics = {}
    training_metrics['minibatch_avg_loss'] = []
    training_metrics['full_batch_loss'] = []
    training_metrics['full_batch_acc'] = []
    training_metrics['val_loss'] = []
    training_metrics['val_acc'] = []
    training_metrics['epoch_time'] = []

    #initial loss and accuracies
    epoch_full_batch_loss, epoch_full_batch_training_acc = compute_loss_and_accuracy(model, loss_fn, x_train, y_train, batch_size=batch_size, device=device)
    val_loss, val_acc = compute_loss_and_accuracy(model, loss_fn, x_val, y_val, batch_size=batch_size, device=device)
    training_metrics['minibatch_avg_loss'].append(epoch_full_batch_loss)
    training_metrics['full_batch_loss'].append(epoch_full_batch_loss)
    training_metrics['full_batch_acc'].append(epoch_full_batch_training_acc)
    training_metrics['val_loss'].append(val_loss)
    training_metrics['val_acc'].append(val_acc)
    training_metrics['epoch_time'].append(0)


    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        epoch_summed_loss = 0
        epoch_batch_counter = 0
        for x_batch, y_batch in train_loader:
            x = x_batch.to(device)
            y = y_batch.to(device)

            x_out= model(x)
            loss = loss_fn(x_out, y)
            optimizer.zero_grad()
            loss.backward()
            if scheduler is not None:
                scheduler.step(epoch=epoch)
            optimizer.step()
            epoch_summed_loss += loss.item()
            epoch_batch_counter += 1

        epoch_end_time = time.time()
        epoch_time = epoch_end_time - epoch_start_time
        training_metrics['epoch_time'].append(epoch_time)
        epoch_full_batch_loss, epoch_full_batch_training_acc = compute_loss_and_accuracy(model, loss_fn, x_train, y_train, batch_size=batch_size, device=device)
        val_loss, val_acc = compute_loss_and_accuracy(model, loss_fn, x_val, y_val, device=device)
        epoch_avg_loss = epoch_summed_loss / epoch_batch_counter
        training_metrics['minibatch_avg_loss'].append(epoch_avg_loss)
        training_metrics['full_batch_loss'].append(epoch_full_batch_loss)
        training_metrics['full_batch_acc'].append(epoch_full_batch_training_acc)
        training_metrics['val_loss'].append(val_loss)
        training_metrics['val_acc'].append(val_acc)
        print(str(epoch) + ': mini batch avg loss: ' + str(epoch_avg_loss) + ', full batch loss: ' + str(epoch_full_batch_loss) + ', epoch time: ' + str(epoch_time) + 's')
    print('Trained in {0} seconds.'.format(int(time.time() - start_time)))
    test_loss, test_acc = compute_loss_and_accuracy(model, loss_fn, x_test, y_test, batch_size=256, device=device)
    training_metrics['test_loss_acc'] = (test_loss, test_acc)
    print('Avg. test loss: {0}, avg. test accuracy: {1}'.format(test_loss, test_acc))
    return training_metrics
