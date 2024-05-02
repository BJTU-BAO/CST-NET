import sys,os
sys.path.append("../")
from TimeEncoder import TimeEncode
from preprocessing.APSDataSet import APSDataSet

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
import numpy as np
import torch.utils.data as Data

rootpath = os.path.abspath('.')


class DiGCN(nn.Module):
    def __init__(self, in_feats, hid_feats, out_feats):
        super(DiGCN, self).__init__()
        self.conv1 = GCNConv(in_feats, hid_feats)
        self.conv2 = GCNConv(hid_feats, out_feats)
        self.time_encode = TimeEncode(in_feats)

    def forward(self, x, edge_index, t_index):
        time_encoder = self.time_encode(t_index)
        x = x + time_encoder
        h = self.conv1(x, edge_index)
        h_out = self.conv2(h, edge_index)
        return h_out


class GCN_Net(nn.Module):
    def __init__(self, in_feats, hid_feats, out_feats):
        super(GCN_Net, self).__init__()
        self.InDegreeGCN = DiGCN(in_feats, hid_feats, out_feats)
        self.OutDegreeGCN = DiGCN(in_feats, hid_feats, out_feats)
        self.fc = nn.Linear((out_feats + hid_feats)*2, 2)

    def forward(self, x, in_edge_index, out_edge_index, t_index):
        InDegree_x = self.InDegreeGCN(x, in_edge_index, t_index)
        OutDegree_x = self.OutDegreeGCN(x, out_edge_index, t_index)
        x = torch.cat((InDegree_x, OutDegree_x), 1)
        x = self.fc(x)
        x = F.log_softmax(x, dim=1)
        return x



def read_cascadeLine_data(data_type, filename):
    with open(rootpath+'/data/aps_data_'+filename+'/cascade_' + data_type + '.txt', 'r') as f:
        cascade_all = {}
        cascade_idx = 0
        for line in f:
            origin = line
            cascade_all[cascade_idx] = origin
            cascade_idx += 1
    return cascade_all


observation_time = 3
prediction_time = 20
n_time_interval = 3
batch_size = 1
cascades_train = read_cascadeLine_data('train', str(observation_time) + 'y_' + str(prediction_time) + 'y')
print('len', len(cascades_train))

train_dataset = APSDataSet(cascades_train, observation_time, n_time_interval)
train_loader = Data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)