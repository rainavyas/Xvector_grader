import torch
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
import numpy as np
import sys
import os
import argparse
from models import Xvector
import time
import pickle

def get_default_device():
    if torch.cuda.is_available():
        print("Got CUDA!")
        return torch.device('cuda')
    else:
        print("No CUDA found")
        return torch.device('cpu')

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def train(train_loader, model, criterion, optimizer, epoch, print_freq=1):
    """
        Run one train epoch
    """
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (x, m, yb) in enumerate(train_loader):

        # measure data loading time
        data_time.update(time.time() - end)

        x = x.to(device)
        m = m.to(device)
        yb = yb.to(device)

        # compute output
        output = model(x, m)
        loss = criterion(output, yb)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        output = output.float()
        loss = loss.float()
        # Record loss
        losses.update(loss.item(), x.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
                      epoch, i, len(train_loader), batch_time=batch_time,
                      data_time=data_time, loss=losses))


def validate(val_loader, model, criterion):
    """
    Run evaluation
    """
    batch_time = AverageMeter()
    losses = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    with torch.no_grad():
        for i, (x, m, yb) in enumerate(val_loader):

            x = x.to(device)
            m = m.to(device)
            yb = yb.to(device)

            # compute output
            output = model(x, m)
            loss = criterion(output, yb)

            output = output.float()
            loss = loss.float()

            # record loss
            losses.update(loss.item(), x.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()


    print('Test\t  Loss@1: {loss.avg:.3f}\n'
          .format(loss=losses))

# Get command line arguments
commandLineParser = argparse.ArgumentParser()
commandLineParser.add_argument('PKL', type=str, help='Specify prepped pkl file')
commandLineParser.add_argument('OUT', type=str, help='Specify output th file')
commandLineParser.add_argument('--seed', type=int, default=1, help='Specify seed')

args = commandLineParser.parse_args()
pkl_file = args.PKL
out_file = args.OUT
seed = args.seed
torch.manual_seed(seed)

# Get the device
device = get_default_device()

# Save the command run
if not os.path.isdir('CMDs'):
    os.mkdir('CMDs')
with open('CMDs/training.cmd', 'a') as f:
    f.write(' '.join(sys.argv)+'\n')

# Load the batched data
pkl = pickle.load(open(pkl_file, "rb"))
print("Loaded pkl")

X = pkl[0]
M = pkl[1]
y = pkl[2]

# Transform to tensors
X = torch.FloatTensor(X)
M = torch.FloatTensor(M)
y = torch.FloatTensor(y)

# Separate into training and validation set
validation_size = 100
X_train = X[validation_size:]
X_val = X[:validation_size]
M_train = M[validation_size:]
M_val = M[:validation_size]
y_train = y[validation_size:]
y_val = y[:validation_size]

# Store as a dataset
train_ds = TensorDataset(X_train, M_train, y)
val_ds = TensorDataset(X_val, M_val, y_val)

# Use dataloader to handle batches easily
batch_size = 50
train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=True)


# Training algorithm

lr = 8*1e-2
epochs = 100
sch = 0.985

mfcc_features = 13

model = Xvector(mfcc_features)
model.to(device)
criterion = torch.nn.MSELoss(reduction='mean')
optimizer = torch.optim.SGD(deep_model.parameters(), lr=lr, momentum = 0.9, nesterov=True)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = 1, gamma = sch)

for epoch in range(epochs):
    # train for one epoch
    print('current lr {:.5e}'.format(optimizer.param_groups[0]['lr']))
    train(train_dl, model, criterion, optimizer, epoch)
    scheduler.step()

    # Evaluate on validation set
    validate(val_dl, model, criterion)

# Save the model
state = model.state_dict()
torch.save(state, out_file)
