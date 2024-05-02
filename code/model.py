import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from model import GAT as GAT
from model import TCN as TCN


class GAT_TCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, alpha, nheads, n_time_interval):
        super(GAT_TCN, self).__init__()
        self.n_time_interval = n_time_interval
        self.gat = GAT.GAT(nfeat=nfeat,
                nhid=nhid,
                nclass=nclass,
                dropout=dropout,
                nheads=nheads,
                alpha=alpha)
        self.dropout = nn.Dropout(dropout)
        self.tcn = TCN.TemporalConvNet(num_inputs=2*n_time_interval, num_channels=[300, 300, 300, 300],
                                       kernel_size=2, dropout=dropout, momentum=0.1)
        self.output = nn.Sequential(
            nn.Conv1d(300, 100, 65),
            nn.Dropout(dropout),
            nn.Tanh(),
            nn.Conv1d(100, 1, 64),
        )

    def forward(self, x, adj):
        x = self.gat(x, adj)
        x = x[adj.shape[0]-2*self.n_time_interval:]     # 这里为什么去掉x前面的一部分元素
        x = x.view(1, x.shape[0], -1)
        x = self.dropout(x)
        x = self.tcn(x)
        # print(x.shape)
        # x = x.view(1, -1)
        x = self.output(x)
        # print(x.shape)
        x = x.view(1, -1)
        # x = F.relu(x)
        return x


class GAT_TCN_v2(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, alpha, nheads, n_time_interval):
        super(GAT_TCN_v2, self).__init__()
        self.n_time_interval = n_time_interval
        self.gat = GAT.GAT(nfeat=nfeat,
                nhid=nhid,
                nclass=nclass,
                dropout=dropout,
                nheads=nheads,
                alpha=alpha)
        self.w = nn.Parameter(nn.init.xavier_uniform(torch.Tensor(nclass, 2*n_time_interval), gain=np.sqrt(2.0)),
                              requires_grad=True)
        self.dropout = nn.Dropout(dropout)
        self.tcn = TCN.TemporalConvNet(num_inputs=2*n_time_interval, num_channels=[300, 300, 300, 300],
                                       kernel_size=2, dropout=dropout, momentum=0.1)
        self.output = nn.Sequential(
            nn.Conv1d(300, 100, 65),
            nn.Dropout(dropout),
            nn.Tanh(),
            nn.Conv1d(100, 1, 64),
        )

    def forward(self, x, adj):
        n_interval = adj[:adj.shape[0] - 2*self.n_time_interval, adj.shape[0] - 2*self.n_time_interval:]
        x = self.gat(x, adj)
        x = x[:adj.shape[0] - 2*self.n_time_interval]

        e = x @ self.w
        zero_vec = -9e15 * torch.ones_like(n_interval)
        attention = torch.where(n_interval > 0, e, zero_vec)
        attention = F.softmax(attention, dim=0)  # softmax for every list
        # attention = F.dropout(attention, self.dropout, training=self.training)
        x = torch.matmul(torch.t(attention), x)

        x = x.view(1, x.shape[0], -1)
        x = self.dropout(x)
        x = self.tcn(x)
        # print(x.shape)
        # x = x.view(1, -1)
        x = self.output(x)
        # print(x.shape)
        x = x.view(1, -1)
        # x = F.relu(x)
        return x


class GAT_TCN_v3(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, alpha, nheads, n_time_interval):
        super(GAT_TCN_v3, self).__init__()
        self.n_time_interval = n_time_interval
        self.gat = GAT.GAT(nfeat=nfeat,
                nhid=nhid,
                nclass=nclass,
                dropout=dropout,
                nheads=nheads,
                alpha=alpha)
        self.w = nn.Parameter(nn.init.xavier_uniform(torch.Tensor(nclass, 2*n_time_interval), gain=np.sqrt(2.0)),
                              requires_grad=True)
        self.dropout = nn.Dropout(dropout)
        self.tcn = TCN.TemporalConvNet(num_inputs=2*n_time_interval, num_channels=[300, 300, 300, 300],
                                       kernel_size=2, dropout=dropout, momentum=0.1)
        self.output = nn.Sequential(
            nn.Conv1d(300, 100, 65),
            nn.Dropout(dropout),
            nn.Tanh(),
            nn.Conv1d(100, 1, 64),
        )

    def forward(self, x, adj):
        n_interval = adj[:adj.shape[0] - 2*self.n_time_interval, adj.shape[0] - 2*self.n_time_interval:]
        x = self.gat(x, adj)
        x = x[:adj.shape[0] - 2*self.n_time_interval]
        x = self.dropout(x)
        x = torch.matmul(torch.t(n_interval), x)

        x = x.view(1, x.shape[0], -1)
        x = self.dropout(x)
        x = self.tcn(x)
        # print(x.shape)
        # x = x.view(1, -1)
        x = self.output(x)
        # print(x.shape)
        x = x.view(1, -1)
        # x = F.relu(x)
        return x


class TCNN(nn.Module):
    def __init__(self, dropout, n_time_interval):
        super(TCNN, self).__init__()
        self.n_time_interval = n_time_interval
        # self.W = nn.Parameter(nn.init.xavier_uniform(torch.Tensor(128, 128), gain=np.sqrt(2.0)), requires_grad=True)
        self.dropout = nn.Dropout(dropout)
        self.con1d = nn.Sequential(
            nn.Conv1d(1, 16, 1),
            nn.Dropout(dropout),
            nn.Tanh(),
            nn.Conv1d(16, 128, 1),
            nn.Dropout(dropout),
            nn.Tanh(),
            nn.Conv1d(128, 128, 1)
        )
        self.tcn = TCN.TemporalConvNet(num_inputs=2*n_time_interval, num_channels=[300, 300, 300, 300],
                                       kernel_size=2, dropout=dropout, momentum=0.1)
        self.output = nn.Sequential(
            nn.Conv1d(300, 100, 65),
            nn.Dropout(dropout),
            nn.Tanh(),
            nn.Conv1d(100, 1, 64),
        )

    def forward(self, x, adj):
        # x = self.gat(x, adj)
        x = adj[:adj.shape[0]-2*self.n_time_interval, adj.shape[0]-2*self.n_time_interval:]
        x = torch.t(x)                  # 2 * self.n_time_interval x N
        x = self.con1d(x.unsqueeze(1))  # 2 * self.n_time_interval x 128 x N
        x, _ = torch.max(x, dim=2)
        x = x.view(1, x.shape[0], -1)
        x = self.dropout(x)
        x = self.tcn(x)
        # print(x.shape)
        # x = x.view(1, -1)
        x = self.output(x)
        # print(x.shape)
        x = x.view(1, -1)
        # x = F.relu(x)
        return x
