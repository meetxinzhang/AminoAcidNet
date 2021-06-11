# encoding: utf-8
"""
@author: Xin Zhang
@contact: zhangxin@szbl.ac.cn
@file: atom_conv.py
@time: 6/9/21 4:55 PM
@desc:
"""
import torch
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.nn import GCNConv
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree
from torch_sparse import SparseTensor


class AtomConv(torch.nn.Module):
    def __init__(self, n_input, n_neural=16, r_coulomb=1.0):
        super(AtomConv, self).__init__()
        self.r_coulomb = r_coulomb
        self.n_input = n_input
        self.n_neural = n_neural

        self.conv1 = GCNConv(self.n_input, self.n_neural)

    def forward(self, atom, edge_index):
        """
        :param atom: [n_atom, 5], 5: idx serial mol row_in_periodic n_ele
        :param edge_index: [n_atom, 25, 3], 3: idx, distance, bond_bool
        :return:
        """
        x = self.conv1(atom, edge_index)
        x = F.relu(x)

        # return F.log_softmax(x, dim=1)
        return x


class GCNConv(MessagePassing):

    def __init__(self, in_channels, out_channels):
        super(GCNConv, self).__init__(aggr='add')  # "Add" aggregation (Step 5).
        self.lin = torch.nn.Linear(in_channels, out_channels)

    def forward(self, atom, edge_index):
        # x has shape [N, in_channels]
        # edge_index has shape [2, E]

        # Step 1: Add self-loops to the adjacency matrix.
        edge_index, _ = add_self_loops(edge_index, num_nodes=atom.size(0))

        # Step 2: Linearly transform node feature matrix.
        x = self.lin(x)

        # Step 3: Compute normalization.
        row, col = edge_index
        deg = degree(col, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        # Step 4-5: Start propagating messages.
        return self.propagate(edge_index, x=x, norm=norm)

    def message(self, x_j, norm):
        # x_j has shape [E, out_channels]

        # Step 4: Normalize node features.
        return norm.view(-1, 1) * x_j