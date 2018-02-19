import pickle, gzip
import sys
import os
import argparse
import time
import datetime
from functools import reduce
from operator import mul
import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.utils.data as data
import torch.nn.functional as F
from torch.autograd import Variable
from util import load_dataset, train, compute_loss_and_accuracy
from models import ProxPropMLP, ProxPropConvNet

"""
Define all default parameters here.
"""
model_default = 'ConvNet'
optimizer_default = 'adam'
batch_size_default = 500
learning_rate_default = 1e-3
weight_decay_default = 0.
num_epochs_default = 50
tau_prox_default = 1.
momentum_default = 0.95
nesterov_default = True
optimization_mode_default = 'prox_cg1'
use_cuda_default = True
dataset_default = 'cifar-10'
num_training_samples_default = -1

outfile_default = ''
cuda_device_default = 0

# make these parameters parsable with above defined default values
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default=dataset_default)
parser.add_argument('--num_training_samples', type=int, default=num_training_samples_default)
parser.add_argument('--optimization_mode', type=str, default=optimization_mode_default)
parser.add_argument('--no-nesterov', dest='nesterov', action='store_false', default=nesterov_default)
parser.add_argument('--momentum', type=float, default=momentum_default)
parser.add_argument('--tau_prox', type=float, default=tau_prox_default)
parser.add_argument('--num_epochs', type=int, default=num_epochs_default)
parser.add_argument('--weight_decay', type=float, default=weight_decay_default)
parser.add_argument('--learning_rate', type=float, default=learning_rate_default)
parser.add_argument('--batch_size', type=int, default=batch_size_default)
parser.add_argument('--optimizer', type=str, default=optimizer_default)
parser.add_argument('--model', type=str, default=model_default)
parser.add_argument('--outfile', type=str, default=outfile_default)
parser.add_argument('--cuda_device', type=int, default=cuda_device_default)
parser.add_argument('--no-cuda', dest='use_cuda', action='store_false', default=use_cuda_default)
args = parser.parse_args()
params = vars(args)

##########################################################################################################################
x_train, y_train, x_val, y_val, x_test, y_test = load_dataset(params['dataset'], num_training_samples=params['num_training_samples'])

params['train_size'] = x_train.shape
params['val_size'] = x_val.shape
params['test_size'] = x_test.shape

print('x_train dims: ' + str(x_train.shape))
print('x_val dims: ' + str(x_val.shape))
print('x_test dims: ' + str(x_test.shape))
print('Maximum value of training set: ' + str(np.max(x_train)))
print('Minimum value of training set: ' + str(np.min(x_train)))

if params['use_cuda']:
    torch.cuda.set_device(params['cuda_device'])

print('Training parameters:')
for k,v in params.items():
    print(k, v)

input_size = x_train.shape[1:]
if params['model'] == 'MLP':
    model = ProxPropMLP(input_size, hidden_sizes=[4000, 1000, 4000], num_classes=10, tau_prox=params['tau_prox'], optimization_mode=params['optimization_mode'])
elif params['model'] == 'ConvNet':
    model = ProxPropConvNet(input_size, 10, params['tau_prox'], optimization_mode=params['optimization_mode'])
else:
    raise ValueError('The model {} you have provided is not valid.'.format(params['model']))
    
if params['use_cuda']:
    model = model.cuda()
print('model: \n' + str(model))

loss_fn = torch.nn.CrossEntropyLoss()
if params['optimizer'] == 'sgd':
    optimizer = torch.optim.SGD(model.parameters(), lr=params['learning_rate'], momentum=params['momentum'], weight_decay=params['weight_decay'], nesterov=params['nesterov'])
elif params['optimizer'] == 'adam':
    optimizer = torch.optim.Adam(model.parameters(), lr=params['learning_rate'], weight_decay=params['weight_decay'])
else:
    raise ValueError('The optimizer {} you have provided is not valid.'.format(params['optimizer']))

data = x_train, y_train, x_val, y_val, x_test, y_test
training_metrics = train(model, loss_fn, optimizer, data, params['num_epochs'], params['batch_size'], cuda=params['use_cuda'])

if params['outfile'] != '':
    pickle_data = {}
    pickle_data['timestamp'] =datetime.datetime.fromtimestamp(int(time.time())).strftime('%Y-%m-%d %H:%M:%S') 
    pickle_data['params'] = params
    pickle_data['training_metrics'] = training_metrics
    pickle.dump(pickle_data, open( os.path.join(params['outfile'] + '.p'), "wb" ) )

