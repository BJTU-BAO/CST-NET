
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# class GraphAttentionLayer(nn.Module):
#     """
#     Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
#     """

#     def __init__(self, in_features, out_features, dropout, alpha, concat=True):
#         super(GraphAttentionLayer, self).__init__()
#         self.dropout = dropout
#         self.in_features = in_features
#         self.out_features = out_features
#         self.alpha = alpha
#         self.concat = concat

#         self.W = nn.Parameter(nn.init.xavier_uniform(torch.Tensor(in_features, out_features), gain=np.sqrt(2.0)), requires_grad=True)
#         #self.a = nn.Parameter(nn.init.xavier_uniform(torch.Tensor(2*out_features, 1).type(torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor), gain=np.sqrt(2.0)), requires_grad=True)
#         self.a1 = nn.Parameter(nn.init.xavier_uniform(torch.Tensor(out_features, 1), gain=np.sqrt(2.0)), requires_grad=True)
#         self.a2 = nn.Parameter(nn.init.xavier_uniform(torch.Tensor(out_features, 1), gain=np.sqrt(2.0)), requires_grad=True)
        

#         self.leakyrelu = nn.LeakyReLU(self.alpha)

#     def forward(self, input, adj):
#         h = torch.mm(input, self.W)
#         N = h.size()[0]

#         # a_input = torch.cat([h.repeat(1, N).view(N * N, -1), h.repeat(N, 1)], dim=1).view(N, -1, 2 * self.out_features)
#         # e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(2))
#         f_1 = h @ self.a1
#         f_2 = h @ self.a2
#         e = self.leakyrelu(f_1 + f_2.transpose(0,1))

#         zero_vec = -9e15*torch.ones_like(e)
#         attention = torch.where(adj > 0, e, zero_vec)
#         attention = F.softmax(attention, dim=1)
#         attention = F.dropout(attention, self.dropout, training=self.training)
#         h_prime = torch.matmul(attention, h)

#         if self.concat:
#             # return F.elu(h_prime)
#             return F.tanh(h_prime)
#         else:
#             return h_prime

#     def __repr__(self):
#         return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


class GraphAttentionLayer(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    """

    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(nn.init.xavier_uniform(torch.Tensor(in_features, out_features), gain=np.sqrt(2.0)), requires_grad=True)
        self.a1 = nn.Parameter(nn.init.xavier_uniform(torch.Tensor(out_features, 1), gain=np.sqrt(2.0)), requires_grad=True)
        self.a2 = nn.Parameter(nn.init.xavier_uniform(torch.Tensor(out_features, 1), gain=np.sqrt(2.0)), requires_grad=True)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, input, adj):
        h = torch.mm(input, self.W)
        N = h.size()[0]

        # a_input = torch.cat([h.repeat(1, N).view(N * N, -1), h.repeat(N, 1)], dim=1).view(N, -1, 2 * self.out_features)
        # e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(2))
        f_1 = h @ self.a1
        f_2 = h @ self.a2
        e = self.leakyrelu(f_1 + f_2.transpose(0,1))

        zero_vec = -9e15*torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)     # softmax for every row
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, h)

        if self.concat:
            return F.elu(h_prime)
            # return F.tanh(h_prime)
        else:
            return h_prime

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


class SpecialSpmmFunction(torch.autograd.Function):
    """Special function for only sparse region backpropataion layer."""

    @staticmethod
    def forward(ctx, indices, values, shape, b):
        assert indices.requires_grad == False
        a = torch.sparse_coo_tensor(indices, values, shape)
        ctx.save_for_backward(a, b)
        ctx.N = shape[0]
        return torch.matmul(a, b)

    @staticmethod
    def backward(ctx, grad_output):
        a, b = ctx.saved_tensors
        grad_values = grad_b = None
        if ctx.needs_input_grad[1]:
            grad_a_dense = grad_output.matmul(b.t())
            edge_idx = a._indices()[0, :] * ctx.N + a._indices()[1, :]
            grad_values = grad_a_dense.view(-1)[edge_idx]
        if ctx.needs_input_grad[3]:
            grad_b = a.t().matmul(grad_output)
        return None, grad_values, None, grad_b


class SpecialSpmm(nn.Module):
    def forward(self, indices, values, shape, b):
        return SpecialSpmmFunction.apply(indices, values, shape, b)


class SpGraphAttentionLayer(nn.Module):
    """
    Sparse version GAT layer, similar to https://arxiv.org/abs/1710.10903
    """

    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(SpGraphAttentionLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        # self.W = nn.Parameter(nn.init.xavier_uniform(torch.Tensor(in_features, out_features), gain=np.sqrt(2.0)),
        #                       requires_grad=True)
        # self.a = nn.Parameter(nn.init.xavier_uniform(torch.Tensor(1, 2*out_features), gain=np.sqrt(2.0)),
        #                        requires_grad=True)

        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_normal_(self.W.data, gain=1.414)

        self.a = nn.Parameter(torch.zeros(size=(1, 2 * out_features)))
        nn.init.xavier_normal_(self.a.data, gain=1.414)

        self.dropout = nn.Dropout(dropout)
        self.leakyrelu = nn.LeakyReLU(self.alpha)
        self.special_spmm = SpecialSpmm()

    def forward(self, input, adj):
        N = input.size()[0]  # the number of node
        edge = adj.nonzero().t()  # get the position of elements with weight bigger than 0 in adj

        h = torch.mm(input, self.W)  # transform feature of every node
        # h: N x out
        assert not torch.isnan(h).any()

        # Self-attention on the nodes - Shared attention mechanism
        edge_h = torch.cat((h[edge[0, :], :], h[edge[1, :], :]), dim=1).t()
        # edge: 2*D x E

        edge_e = torch.exp(-self.leakyrelu(self.a.mm(edge_h).squeeze()))
        assert not torch.isnan(edge_e).any()
        # edge_e: E

        e_rowsum = self.special_spmm(edge, edge_e, torch.Size([N, N]), torch.ones(size=(N, 1)).cuda())
        e_rowsum += 1e-5
        # e_rowsum: N x 1

        edge_e = self.dropout(edge_e)
        # edge_e: E

        h_prime = self.special_spmm(edge, edge_e, torch.Size([N, N]), h)
        assert not torch.isnan(h_prime).any()
        # h_prime: N x out

        h_prime = h_prime.div(e_rowsum)
        # h_prime: N x out
        assert not torch.isnan(h_prime).any()

        if self.concat:
            # if this layer is not last layer,
            return F.elu(h_prime)
        else:
            # if this layer is last layer,
            return h_prime

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


class GAT(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, alpha, nheads):
        """Dense version of GAT."""
        super(GAT, self).__init__()
        self.dropout = dropout
        self.nheads = nheads
        self.attentions = [GraphAttentionLayer(nfeat, nhid, dropout=dropout, alpha=alpha, concat=True) for _ in range(nheads)]
        self.attentions2 = [GraphAttentionLayer(nhid * nheads, nhid, dropout=dropout, alpha=alpha, concat=True) for _ in range(nheads)]
        # self.attentions3 = [GraphAttentionLayer(nhid * nheads, nhid, dropout=dropout, alpha=alpha, concat=True) for _ in range(nheads)]
        
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)
        for i, attention in enumerate(self.attentions2):
            self.add_module('attention_2{}'.format(i), attention)

        # self.out_att = GraphAttentionLayer(nhid * nheads, nclass, dropout=dropout, alpha=alpha, concat=True)
        self.out_att = [GraphAttentionLayer(nhid * nheads, nclass, dropout=dropout, alpha=alpha, concat=True) for _ in range(nheads)]
        for i, att in enumerate(self.out_att):
            self.add_module('att_{}'.format(i), att)
    
    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj) for att in self.attentions2], dim=1)
        x = F.dropout(x, self.dropout, training=self.training)
        tmp = [att(x, adj) for att in self.out_att]
        output = 0
        for t in tmp:
            output += t
        output = output / self.nheads
        # output = F.tanh(output)
        output = F.elu(output)
        return output


class SpGAT(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, alpha, nheads):
        """Dense version of GAT."""
        super(SpGAT, self).__init__()
        self.dropout = dropout
        self.nheads = nheads
        self.attentions = [SpGraphAttentionLayer(nfeat, nhid, dropout=dropout, alpha=alpha, concat=True) for _ in
                           range(nheads)]
        self.attentions2 = [SpGraphAttentionLayer(nhid * nheads, nhid, dropout=dropout, alpha=alpha, concat=True) for _ in
                            range(nheads)]

        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)
        for i, attention in enumerate(self.attentions2):
            self.add_module('attention_2{}'.format(i), attention)

        self.out_att = [SpGraphAttentionLayer(nhid * nheads, nclass, dropout=dropout, alpha=alpha, concat=True) for _ in
                        range(nheads)]
        for i, att in enumerate(self.out_att):
            self.add_module('att_{}'.format(i), att)

    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj) for att in self.attentions2], dim=1)
        x = F.dropout(x, self.dropout, training=self.training)
        tmp = [att(x, adj) for att in self.out_att]
        output = 0
        for t in tmp:
            output += t
        output = output / self.nheads
        # output = F.tanh(output)
        output = F.elu(output)
        return output