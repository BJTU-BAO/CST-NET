#-*- coding:utf-8 –*-
#!/usr/bin/env python3
import os
import gc
# import resource
import torch.utils.data as Data
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import math
import argparse

from preprocessing import APSDataSet
from model.model import GAT_TCN
from loss import MESLoss


# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--lr', type=float, default=0.0005, help='learning rate.')
parser.add_argument('--cuda', action='store_true', help='use CUDA training.')
parser.add_argument('--seed', type=int, default=72, help='Random seed.')

args = parser.parse_args()
args.cuda = args.cuda and torch.cuda.is_available()
# np.random.seed(args.seed)
# torch.manual_seed(args.seed)
# if args.cuda:
#     torch.cuda.manual_seed(args.seed)


rootpath = os.path.abspath('.')

MINIBATCHSIZE = 256
batch_size = 1
EPOCH = 1000    # train the training data n times


def weight_schedule(epoch, max_val=30, mult=-5, max_epochs=30):
    if epoch == 0:
        return 0.
    w = max_val * np.exp(mult * (1. - float(epoch) / max_epochs) ** 2)
    w = float(w)
    if epoch > max_epochs:
        return max_val
    return w


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1 ** (epoch // 2))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
        # print(param_group)


def read_cascadeLine_data(data_type, filename):
    with open(rootpath+'/data/aps_data_'+filename+'/cascade_' + data_type + '.txt', 'r') as f:
        cascade_all = {}
        cascade_idx = 0
        for line in f:
            origin = line
            cascade_all[cascade_idx] = origin
            cascade_idx += 1
    return cascade_all


def test_net(model, test_type, loss_function, observation_time, prediction_time, n_time_interval):
    cascades_train = read_cascadeLine_data(test_type, str(observation_time) + 'y_' + str(prediction_time) + 'y')
    test_dataset = APSDataSet(cascades_train, observation_time, n_time_interval)
    test_loader = Data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    loss = 0
    acc_cnt = 0
    acc_cnt15 = 0
    acc_cnt2 = 0
    mae = 0
    acc_mean = 0
    losses = []
    with torch.no_grad():
        model.eval()
        for step, (adj, features, label) in enumerate(test_loader):
            if args.cuda:
                adj = adj.cuda()
                features = features.cuda()
                label = label.cuda()
            features = features.squeeze()
            adj = adj.squeeze() 
            output = model(features, adj)
            loss_ = output - label
            mae += torch.abs(loss_).sum().item()
            acc_mean_ = torch.abs(loss_) / (label + 1)
            acc_mean += acc_mean_.sum().item()
            acc_ = loss_

            for accu in acc_mean_:
                if accu < 0.1:
                    acc_cnt += 1
                if accu < 0.15:
                    acc_cnt15 += 1
                if accu < 0.2:
                    acc_cnt2 += 1
            loss_ = loss_ * loss_
            for l in loss_.cpu().numpy():
                losses.append(l)
            loss += float(loss_.sum().item())
        cnt = len(cascades_train)
        loss = loss / cnt
        acc = acc_cnt / cnt
        acc15 = acc_cnt15 / cnt
        acc2 = acc_cnt2 / cnt
        acc_mean = acc_mean / cnt
        mae = mae / cnt
        median = np.median(losses)
    return loss, (mae, median), (acc, acc15, acc2, acc_mean)


def main(observation_time, prediction_time, n_time_interval):
    cascades_train = read_cascadeLine_data('train', str(observation_time) + 'y_' + str(prediction_time) + 'y')
    print('len', len(cascades_train))
    
    train_dataset = APSDataSet(cascades_train, observation_time, n_time_interval)
    train_loader = Data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    community_dict = {}
    cascades_train = []
    print('load data finished.')
    
    model = GAT_TCN(nfeat=(n_time_interval) * 12, 
                        nhid=256, 
                        nclass=128, 
                        dropout=0.6, 
                        nheads=10, 
                        alpha=0.2, 
                        n_time_interval=n_time_interval)

    # print(model)
    if args.cuda:
        model.cuda()
    print('cuda is valiable:', args.cuda)

    # optimizer = torch.optim.Adam(
    #     [
    #         {"params": model.gat.parameters(), 'lr': 5e-5},
    #         {"params": model.tcn.parameters(), 'lr': 5e-5},
    #     ],
    #     lr=5e-4,
    #     betas=(0.9, 0.999)
    # )

    optimizer = torch.optim.Adam(model.parameters(), lr=5e-5, betas=(0.9, 0.999), weight_decay=5e-6)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=5)
    # loss_function = nn.MSELoss()
    loss_function = MESLoss.mse_loss(1, 0.2)

    mminloss = 100
    maxtry = 100
    trycnt = 0
    best_val = 100
    best_test = 100

    for epoch in range(EPOCH):
        minibatch = 0
        labels = 0
        outputs1 = 0
        outputs2 = 0
        
        w = weight_schedule(epoch)
        loss_function = MESLoss.mse_loss(1, w)
        print('weight', w, type(w))

        for step, (adj, features, label) in enumerate(train_loader):
            noise = torch.zeros(features.shape)
            noise.data.normal_(0, std=0.005)
            # print(noise)

            if args.cuda:
                noise = noise.cuda()
                adj = adj.cuda()
                features = features.cuda()
                label = label.cuda()

            features = features + noise

            features = features.squeeze()
            adj = adj.squeeze()
            # model.train()
            output1 = model(features, adj)
            # model.eval()
            output2 = model(features, adj)
            if minibatch == 0:
                outputs1 = output1
                outputs2 = output2
                labels = label
            if minibatch < MINIBATCHSIZE and minibatch > 0:
                outputs1 = torch.cat((outputs1, output1))
                outputs2 = torch.cat((outputs2, output2))
                labels = torch.cat((labels, label))
            minibatch += 1
            # print('shape', outputs.shape, labels.shape)
            if minibatch == MINIBATCHSIZE:
                minibatch = 0
                loss = loss_function(outputs1, outputs2, labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                # model.zero_grad()
                print('loss', loss, label)
        
        train_loss = test_net(model, 'train', loss_function, observation_time, prediction_time, n_time_interval)
        test_loss = test_net(model, 'test', loss_function, observation_time, prediction_time, n_time_interval)
        val_loss = test_net(model, 'val', loss_function, observation_time, prediction_time, n_time_interval)

        # scheduler.step(val_loss)

        if mminloss > val_loss:
            torch.save(model, 'model_'+str(observation_time)+'y_20y_cpt.pkl')
            mminloss = val_loss
            best_test = test_loss
            best_val = val_loss
            trycnt = 0
        # log = 'Epoch: ' + str(epoch) + '  |train loss: ' + str(train_loss) + '  |val loss: ' + str(val_loss) + ' |test loss: ' + str(test_loss) + '  |val acc mean: ' + str(val_acc_mean) + '  |val acc: ' + str(val_acc)+ '  |test acc mean: ' + str(test_acc_mean) + '  |test acc: ' + str(test_acc) + ' |best val loss: ' + str(best_val) + ' | best test loss: ' + str(best_test)
        log = 'Epoch: ' + str(epoch) + '  |train loss: ' + str(train_loss) + '  |val loss: ' + str(val_loss) + ' |test loss: ' + str(test_loss) + ' |best val loss: ' + str(best_val) + ' | best test loss: ' + str(best_test)
        print(log)
        with open(rootpath+'/process'+str(observation_time)+'y_20y_cpt.txt', 'a+') as f:
            f.write(log+'\r\n')
        trycnt += 1
        if trycnt > maxtry:
            break

if __name__ == "__main__":
    main(3, 20, 3)
