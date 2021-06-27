import math

import torch
import torch.nn.functional as F
from torch.nn import Linear

from torch_scatter import scatter_add, scatter_max

from torch_geometric.nn import GCNConv
from torch_geometric.utils import add_remaining_self_loops, remove_self_loops, softmax
from torch_geometric.nn.pool.topk_pool import topk


class ASAP_Pooling(torch.nn.Module):

    def __init__(self, in_channels, ratio, dropout_att=0, negative_slope=0.2):
        super(ASAP_Pooling, self).__init__()

        self.in_channels = in_channels
        self.ratio = ratio
        self.negative_slope = negative_slope
        self.dropout_att = dropout_att
        self.lin_q = Linear(in_channels, in_channels)
        self.gat_att = Linear(2 * in_channels, 1)
        self.gnn_score = LEConv(self.in_channels, 1)  # gnn_score: uses LEConv to find cluster fitness scores
        self.gnn_intra_cluster = GCNConv(self.in_channels,
                                         self.in_channels)  # gnn_intra_cluster: uses GCN to account for intra cluster properties, e.g., edge-weights
        self.reset_parameters()

    def reset_parameters(self):
        self.lin_q.reset_parameters()
        self.gat_att.reset_parameters()
        self.gnn_score.reset_parameters()
        self.gnn_intra_cluster.reset_parameters()

    def forward(self, x, edge_index, edge_weight=None, batch=None):
        if batch is None:
            batch = edge_index.new_zeros(x.size(0))

        # NxF
        x = x.unsqueeze(-1) if x.dim() == 1 else x
        # Add Self Loops
        fill_value = 1
        num_nodes = scatter_add(batch.new_ones(x.size(0)), batch, dim=0)
        edge_index, edge_weight = add_remaining_self_loops(edge_index=edge_index, edge_weight=edge_weight,
                                                           fill_value=fill_value, num_nodes=num_nodes.sum())

        N = x.size(0)  # total num of nodes in batch

        # ExF
        x_pool = self.gnn_intra_cluster(x=x, edge_index=edge_index, edge_weight=edge_weight)
        x_pool_j = x_pool[edge_index[1]]
        x_j = x[edge_index[1]]

        # ---Master query formation---
        # NxF
        X_q, _ = scatter_max(x_pool_j, edge_index[0], dim=0)
        # NxF
        M_q = self.lin_q(X_q)
        # ExF
        M_q = M_q[edge_index[0].tolist()]

        score = self.gat_att(torch.cat((M_q, x_pool_j), dim=-1))
        score = F.leaky_relu(score, self.negative_slope)
        score = softmax(score, edge_index[0], num_nodes=num_nodes.sum())

        # Sample attention coefficients stochastically.
        score = F.dropout(score, p=self.dropout_att, training=self.training)
        # ExF
        v_j = x_j * score.view(-1, 1)
        # ---Aggregation---
        # NxF
        out = scatter_add(v_j, edge_index[0], dim=0)

        # ---Cluster Selection
        # Nx1
        fitness = torch.sigmoid(self.gnn_score(x=out, edge_index=edge_index)).view(-1)
        perm = topk(x=fitness, ratio=self.ratio, batch=batch)
        x = out[perm] * fitness[perm].view(-1, 1)

        # ---Maintaining Graph Connectivity
        batch = batch[perm]
        edge_index, edge_weight = graph_connectivity(
            device=x.device,
            perm=perm,
            edge_index=edge_index,
            edge_weight=edge_weight,
            score=score,
            ratio=self.ratio,
            batch=batch,
            N=N)

        return x, edge_index, edge_weight, batch, perm

    def __repr__(self):
        return '{}({}, ratio={})'.format(self.__class__.__name__, self.in_channels, self.ratio)