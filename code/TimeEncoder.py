import logging

import numpy as np
import torch

import torch.nn as nn
import torch.nn.functional as F

class TimeEncode(torch.nn.Module):
    def __init__(self, expand_dim, factor=5):
        super(TimeEncode, self).__init__()
        # init_len = np.array([1e8**(i/(time_dim-1)) for i in range(time_dim)])
        # 时间的维度应该和node的嵌入维度一致
        time_dim = expand_dim
        self.factor = factor
        # np.linspace(0, 9, time_dim) 将0-9均匀地分割为dim份  **表示平方
        a = np.linspace(0, 9, time_dim)
        b = torch.from_numpy(1 / 10 ** a).float()  # 应该等于1 / (10 ** a)
        # basis_freq是可被训练的参数
        self.basis_freq = torch.nn.Parameter((torch.from_numpy(1 / 10 ** np.linspace(0, 9, time_dim))).float())
        # from_numpy 新建一个张量，数组和张量共享内存，当在张量或数组中修改其内容时，数值会有对应的反应
        self.phase = torch.nn.Parameter(torch.zeros(time_dim).float())

        #self.dense = torch.nn.Linear(time_dim, expand_dim, bias=False)
        #torch.nn.init.xavier_normal_(self.dense.weight)

    def forward(self, ts):
        # ts: [N, L]
        batch_size = ts.size(0)
        seq_len = ts.size(1)

        # view相当于reshape,元素个数不变，形状整形，-1表示该维度的shape不确定，根据其他维度动态变化
        ts = ts.view(batch_size, seq_len, 1) # [N, L, 1]
        # 三维矩阵乘法第一个维度相同即可，剩余两个维度会自动的广播，如(2,1,5)和(2,5,1)相乘会得到(2,5,5)
        # (batch_size,seq_len,1) * (1,1,dim) = (batch_size,seq_len,dim)
        # print(self.basis_freq)
        # print(ts)
        map_ts = ts * self.basis_freq.view(1, 1, -1) # [N, L, time_dim]
        print(map_ts)
        map_ts += self.phase.view(1, 1, -1)
        print(map_ts)

        harmonic = torch.cos(map_ts)
        return harmonic  # self.dense(harmonic)



class PosEncode(torch.nn.Module):
    def __init__(self, expand_dim, seq_len):
        super().__init__()

        self.pos_embeddings = nn.Embedding(num_embeddings=seq_len, embedding_dim=expand_dim)

    def forward(self, ts):
        # ts: [N, L]
        order = ts.argsort()
        ts_emb = self.pos_embeddings(order)
        return ts_emb

n_dims = 8
SequenceData = torch.tensor([[1,2,3,4,5,6],[7,8,9,10,11,12],[6,5,4,3,2,1]])
pos_encode = PosEncode(n_dims, 6)
TimeVector = []
pos_vector = pos_encode(SequenceData)
# 返回向量的维度 batch, 序列长度， 每个序列中每个数字所对应的32维向量
print(pos_vector.size())
print(pos_vector)
print(pos_vector[0][0][0])
TimeVector.append(pos_vector)

class EmptyEncode(torch.nn.Module):
    def __init__(self, expand_dim):
        super().__init__()
        self.expand_dim = expand_dim

    def forward(self, ts):
        out = torch.zeros_like(ts).float()
        # 在最后一维增加维度，其值用1表示
        out = torch.unsqueeze(out, dim=-1)
        # 将out变形为expand()参数的形状 且原来的tensor和之后的不共享内存
        out = out.expand(out.shape[0], out.shape[1], self.expand_dim)
        # 返回一个形状为（batch_size,seq_len,dim）全零矩阵
        return out


        outrow = list(edgeindex[1])
        outcol = list(edgeindex[0])
        if self.outdroprate > 0:
            length = len(outrow)
            poslist = random.sample(range(length), int(length * (1 - self.outdroprate)))
            poslist = sorted(poslist)
            row = list(np.array(burow)[poslist])
            col = list(np.array(bucol)[poslist])
            # bu为反向构建有向图，将edgeindex行和列颠倒之后再进行随机的采样
            outnew_edgeindex = [row, col]
        else:
            outnew_edgeindex = [burow,bucol]
        return Data(x=torch.tensor(data['x'],dtype=torch.float32),
                    edge_index=torch.LongTensor(new_edgeindex),BU_edge_index=torch.LongTensor(outnew_edgeindex),
                    y=torch.LongTensor([int(data['y'])]), root=torch.LongTensor(data['root']),
                    rootindex=torch.LongTensor([int(data['rootindex'])]))

'''
n_time_interval = 6
n_dims = 8
SequenceData = torch.tensor([[1,2,3,4,5,6],[7,8,9,10,11,12],[6,5,4,3,2,1]])
TimeVector = []
time_encode = TimeEncode(n_dims)
time_vector = time_encode(SequenceData)
# 返回向量的维度 batch, 序列长度， 每个序列中每个数字所对应的32维向量
print(time_vector.size())
print(time_vector)
print(time_vector[0][0][0])
TimeVector.append(time_vector)
'''