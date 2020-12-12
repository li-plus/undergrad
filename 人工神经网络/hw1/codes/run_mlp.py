from network import Network
from utils import LOG_INFO
from layers import Relu, Sigmoid, Linear
from loss import EuclideanLoss, SoftmaxCrossEntropyLoss
from solve_net import train_net, test_net
from load_data import load_mnist_2d
import argparse
import numpy as np
import random
import os
import pickle
import time
from tensorboardX import SummaryWriter


train_data, test_data, train_label, test_label = load_mnist_2d('data')

# Your model defintion here
# You should explore different model architecture

parser = argparse.ArgumentParser()
parser.add_argument('--layers', type=int, choices=[0, 1, 2], default=2)
parser.add_argument('--loss', type=str, choices=['mse', 'sce'], default='sce')
parser.add_argument('--activation', type=str, choices=['relu', 'sigmoid'], default='relu')
parser.add_argument('--std', type=float, default=0.01)
parser.add_argument('--batch-size', type=int, default=100)
parser.add_argument('--momentum', type=float, default=0.9)
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--weight-decay', type=float, default=0.0005)
parser.add_argument('--max-epoch', type=int, default=100)
args = parser.parse_args()

output_dir = "{}_{}_{}_{}_{}_{}".format(args.layers, args.activation, args.loss, args.batch_size, args.lr, args.momentum)

model = Network()

activation = {'relu': Relu, 'sigmoid': Sigmoid}[args.activation]

if args.layers == 0:
    model.add(Linear('fc', 784, 10, args.std))
elif args.layers == 1:
    model.add(Linear('fc1', 784, 256, args.std))
    model.add(activation('act'))
    model.add(Linear('fc2', 256, 10, args.std))
else:
    model.add(Linear('fc1', 784, 256, args.std))
    model.add(activation('act'))
    model.add(Linear('fc2', 256, 128, args.std))
    model.add(activation('act'))
    model.add(Linear('fc3', 128, 10, args.std))

if args.loss == 'mse':
    model.add(Sigmoid('sigmoid'))
    loss = EuclideanLoss('loss')
else:
    loss = SoftmaxCrossEntropyLoss('loss')

# Training configuration
# You should adjust these hyperparameters
# NOTE: one iteration means model forward-backwards one batch of samples.
#       one epoch means model has gone through all the training samples.
#       'disp_freq' denotes number of iterations in one epoch to display information.

config = {
    'learning_rate': args.lr,
    'weight_decay': args.weight_decay,
    'momentum': args.momentum,
    'batch_size': args.batch_size,
    'max_epoch': args.max_epoch,
    'disp_freq': 50,
    'test_epoch': 1
}

loss_list = []
acc_list = []

start = time.time()

for epoch in range(config['max_epoch']):
    LOG_INFO('Training @ %d epoch...' % (epoch))
    loss_epoch, acc_epoch = train_net(model, loss, config, train_data, train_label, config['batch_size'], config['disp_freq'])

    loss_list += loss_epoch
    acc_list += acc_epoch

    if epoch % config['test_epoch'] == 0:
        LOG_INFO('Testing @ %d epoch...' % (epoch))
        test_loss, test_acc = test_net(model, loss, test_data, test_label, config['batch_size'])

last_loss, last_acc = test_net(model, loss, test_data, test_label, config['batch_size'])

elapsed = time.time() - start

print("time elapsed {:.0f}, loss {:.4f}, acc {:.4f}".format(elapsed, last_loss, last_acc))

# save results
num_iter = len(loss_list)
with SummaryWriter('result_{}'.format(output_dir)) as tb_writer:
    for step in range(num_iter):
        tb_writer.add_scalar('train/loss', loss_list[step], global_step=step)
        tb_writer.add_scalar('train/acc', acc_list[step], global_step=step)

result_dir = "results"
os.makedirs(result_dir, exist_ok=True)
with open(os.path.join(result_dir, "loss_{}.pk".format(output_dir)), 'wb') as f:
    pickle.dump(loss_list, f)

with open(os.path.join(result_dir, "acc_{}.pk".format(output_dir)), 'wb') as f:
    pickle.dump(acc_list, f)
