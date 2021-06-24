# encoding: utf-8
"""
@author: Xin Zhang
@contact: zhangxin@szbl.ac.cn
@file: atom_conv.py
@time: 6/9/21 4:55 PM
@desc:
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def indexing_neighbor(tensor: "(bs, atom_num, dim)", index: "(bs, atom_num, neighbor_num)"):
    """
    Return: (bs, atom_num, neighbor_num, dim)  torch.Size([3, 26670, 15])
    """
    bs = index.size(0)
    id_0 = torch.arange(bs).view(-1, 1, 1)
    tensor_indexed = tensor[id_0, index]
    return tensor_indexed


def get_neighbor_direct_norm(atoms: "(bs, atom_num, 3)", neighbor_index: "(bs, atom_num, neighbor_num)"):
    """
    Return: (bs, atom_num, atom_num, 3)
    """
    pos_neighbors = indexing_neighbor(atoms, neighbor_index)  # [bs, a_n, nei_n, 3]
    neigh_direction = pos_neighbors - atoms.unsqueeze(2)  # [bs, a_n, nei_n, 3] - [bs, a_n, 1, 3]
    neigh_direction_norm = F.normalize(neigh_direction, dim=-1)  # unit vector of distance
    return neigh_direction_norm


# def cos_theta(vectors: "(bs, a_n, nei_n, 3)"):
#     nei_n = vectors.size()[2]
#     assert nei_n % 2 == 0
#
#     fgs4 = torch.split(vectors, 4, dim=2)
#     theta4 = fgs4[0] @ fgs4[1].transpose(2, 3)
#     print(theta4.size())
#
#     fgs2 = torch.split(vectors, 2, dim=2)
#     theta2 = fgs2[0] @ fgs2[1].transpose(2, 3)
#     theta2_2 = fgs2[2] @ fgs2[3].transpose(2, 3)
#     print(theta2.size())
#
#     fgs1 = torch.split(vectors, 1, dim=2)
#     theta1 = fgs1[0] @ fgs1[1].transpose(2, 3)
#     theta1_2 = fgs1[2] @ fgs1[3].transpose(2, 3)
#     theta1_3 = fgs1[4] @ fgs1[5].transpose(2, 3)
#     theta1_4 = fgs1[6] @ fgs1[7].transpose(2, 3)
#     print(theta1.size())


class AtomConv(nn.Module):
    """Extract and recombines structure and chemical elements features from local domain of protein graph
    in batch format, k_size denotes the range of domain.
    :param k_size: int, num of neighbor atoms which are considered
    :param kernel_num, int
    """

    def __init__(self, kernel_num, k_size):
        super(AtomConv, self).__init__()
        self.kernel_num = kernel_num
        self.k_size = k_size

        self.leaky_relu = nn.LeakyReLU(inplace=True)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

        self.angle_weight = nn.Parameter(torch.FloatTensor(kernel_num, k_size))  # k_size must equal neighbor_num
        self.scalar_weight = nn.Parameter(torch.FloatTensor(10, kernel_num))
        self.radius_weight_1 = nn.Parameter(torch.FloatTensor(2, kernel_num))
        self.radius_weight_2 = nn.Parameter(torch.FloatTensor(kernel_num, 1))

        nn.init.uniform_(self.angle_weight, -1, 1)
        nn.init.uniform_(self.scalar_weight, 0, 1)
        nn.init.uniform_(self.radius_weight_1, -1, 1)
        nn.init.uniform_(self.radius_weight_2, -1, 1)

    def forward(self, pos: "(bs, atom_num, 3)",
                atom_fea: "(bs, atom_num, 5)",
                edge_index: "(bs, atom_num, nei_n)",
                edge_fea: "(bs, atom_num, nei_n, 2)",
                atom_mask: "(bs, atom_num)"):
        """
        Return: (bs, atom_num, kernel_num)
        """
        edge_index = edge_index[:, :, 0:self.k_size]
        edge_fea = edge_fea[:, :, 0:self.k_size, :]

        theta, mask_final = self.cos_theta(pos, edge_index, atom_mask)  # [bs, a_n, nei_n, 1]
        fea_cat = self.feature_fusion(atom_fea, edge_index, atom_mask)  # [bs, a_n, nei_n, 10]
        gate = self.message_gating(edge_fea, atom_mask)  # [bs, a_n, nei_n]

        fea_struct = torch.matmul(self.angle_weight, theta).squeeze()  # [bs, a_n, kernel_num]

        fea_elem = torch.matmul(fea_cat, self.scalar_weight)  # [bs, a_n, nei_n, kernel_num]
        fea_elem = torch.mul(gate, fea_elem)  # [bs, a_n, nei_n, kernel_num]

        fea_elem = torch.sum(fea_elem, dim=2).squeeze()  # [bs, a_n, kernel_num]

        interactions = self.leaky_relu(fea_elem + fea_struct)  # [bs, a_n, kernel_num]
        interact_masked = torch.mul(mask_final, interactions)
        return interact_masked

    def cos_theta(self, pos, edge_index, atom_max):
        """
        Embed spatial features
        :return [bs, a_n, nei_n, 1] """
        nei_direct_norm = get_neighbor_direct_norm(pos, edge_index)  # [bs, a_n, nei_n, 3]
        nearest = nei_direct_norm[:, :, 0, :].unsqueeze(2)  # [2, 15, 1, 3]
        else_neigh = nei_direct_norm[:, :, 1:, :]  # [2, 15, nei_n-1, 3]
        theta = else_neigh @ nearest.transpose(2, 3)  # [bs, a_n, nei_n-1, 1]
        cos0_theta = F.pad(theta, [0, 0, 1, 0], value=1)  # cos(0)=1

        mask = atom_max.unsqueeze(-1).repeat(1, 1, self.k_size).unsqueeze(-1)
        mask_final = atom_max.unsqueeze(-1).repeat(1, 1, self.kernel_num)
        return torch.mul(cos0_theta, mask), mask_final

    def feature_fusion(self, fea: "(bs, atom_num, 5)",
                       index: "(bs, atom_num, neighbor_num)",
                       atom_max: "(bs, atom_num)"):
        """Fuse elements features"""
        fea_neigh = indexing_neighbor(fea, index)  # [bs, a_n, nei_n, 5]
        atom_reps_fea = fea.unsqueeze(2).repeat(1, 1, self.k_size, 1)  # [bs, a_n, nei_n, 5]
        fea_cat = torch.cat([atom_reps_fea, fea_neigh], dim=-1)  # [bs, a_n, nei_n, 10]

        mask = atom_max.unsqueeze(-1).repeat(1, 1, self.k_size).unsqueeze(-1)
        return torch.mul(fea_cat, mask)

    def message_gating(self, edge_fea, atom_mask):
        """Generate a gate to select elements features based on distance and bond type"""
        a = self.relu(torch.matmul(edge_fea/1, self.radius_weight_1))  # [bs, a_n, nei_n, kernel_num]
        # g = torch.max(a, dim=3)[0]  # [bs, a_n, nei_n]
        b = self.relu(torch.matmul(a, self.radius_weight_2)).squeeze()  # [bs, a_n, nei_n]

        mask = atom_mask.unsqueeze(-1).repeat(1, 1, self.k_size)
        g = torch.mul(b, mask)
        g = g.unsqueeze(-1).repeat(1, 1, 1, self.kernel_num)  # [bs, a_n, nei_n, kernel_num]
        return self.sigmoid(g)


# def get_neighbor_index(atoms: "(bs, atom_num, 3)", neighbor_num: int):
#     """
#     Return: (bs, atom_num, neighbor_num)
#     """
#     bs, a_n, _ = atoms.size()
#     # device = atoms.device
#     # tensor.transpose(1, 2), transposes only 1 and 2 dim, =tf.transpose(0, 2, 1)
#     inner = torch.bmm(atoms, atoms.transpose(1, 2))  # [bs, a_n, a_n]
#     quadratic = torch.sum(atoms ** 2, dim=2)  # [bs, a_n]
#     # [bs, a_n, a_n] + [bs, 1, a_n] + [bs, a_n, 1]
#     distance = inner * (-2) + quadratic.unsqueeze(1) + quadratic.unsqueeze(2)
#     neighbor_index = torch.topk(distance, k=neighbor_num + 1, dim=-1, largest=False)[1]
#     neighbor_index = neighbor_index[:, :, 1:]
#     return neighbor_index


# class ConvLayer(nn.Module):
#     def __init__(self, in_channel, out_channel, support_num):
#         super().__init__()
#         # arguments:
#         self.in_channel = in_channel
#         self.out_channel = out_channel
#         self.support_num = support_num
#
#         # parameters:
#         self.relu = nn.ReLU(inplace=True)
#         self.weights = nn.Parameter(torch.FloatTensor(in_channel, (support_num + 1) * out_channel))
#         self.bias = nn.Parameter(torch.FloatTensor((support_num + 1) * out_channel))
#         self.directions = nn.Parameter(torch.FloatTensor(3, support_num * out_channel))
#         self.initialize()
#
#     def initialize(self):
#         stdv = 1. / math.sqrt(self.out_channel * (self.support_num + 1))
#         self.weights.data.uniform_(-stdv, stdv)
#         self.bias.data.uniform_(-stdv, stdv)
#         self.directions.data.uniform_(-stdv, stdv)
#
#     def forward(self,
#                 neighbor_index: "(bs, vertice_num, neighbor_index)",
#                 vertices: "(bs, vertice_num, 3)",
#                 feature_map: "(bs, vertice_num, in_channel)"):
#         """
#         Return: output feature map: (bs, vertice_num, out_channel)
#         """
#         bs, vertice_num, neighbor_num = neighbor_index.size()
#         neighbor_direction_norm = get_neighbor_direct_norm(vertices, neighbor_index)
#         support_direction_norm = F.normalize(self.directions, dim=0)
#         theta = neighbor_direction_norm @ support_direction_norm  # (bs, vertice_num, neighbor_num, support_num * out_channel)
#         theta = self.relu(theta)
#         theta = theta.contiguous().view(bs, vertice_num, neighbor_num, -1)
#         # (bs, vertice_num, neighbor_num, support_num * out_channel)
#
#         feature_out = feature_map @ self.weights + self.bias  # (bs, vertice_num, (support_num + 1) * out_channel)
#         feature_center = feature_out[:, :, :self.out_channel]  # (bs, vertice_num, out_channel)
#         feature_support = feature_out[:, :, self.out_channel:]  # (bs, vertice_num, support_num * out_channel)
#
#         # Fuse together - max among product
#         feature_support = indexing_neighbor(feature_support,
#                                             neighbor_index)  # (bs, vertice_num, neighbor_num, support_num * out_channel)
#         activation_support = theta * feature_support  # (bs, vertice_num, neighbor_num, support_num * out_channel)
#         activation_support = activation_support.view(bs, vertice_num, neighbor_num, self.support_num, self.out_channel)
#         activation_support = torch.max(activation_support, dim=2)[0]  # (bs, vertice_num, support_num, out_channel)
#         activation_support = torch.sum(activation_support, dim=2)  # (bs, vertice_num, out_channel)
#         feature_fuse = feature_center + activation_support  # (bs, vertice_num, out_channel)
#         return feature_fuse
#
#
# class PoolLayer(nn.Module):
#     def __init__(self, pooling_rate: int = 4, neighbor_num: int = 4):
#         super().__init__()
#         self.pooling_rate = pooling_rate
#         self.neighbor_num = neighbor_num
#
#     def forward(self,
#                 vertices: "(bs, vertice_num, 3)",
#                 feature_map: "(bs, vertice_num, channel_num)"):
#         """
#         Return:
#             vertices_pool: (bs, pool_vertice_num, 3),
#             feature_map_pool: (bs, pool_vertice_num, channel_num)
#         """
#         bs, vertice_num, _ = vertices.size()
#         neighbor_index = get_neighbor_index(vertices, self.neighbor_num)
#         neighbor_feature = indexing_neighbor(feature_map,
#                                              neighbor_index)  # (bs, vertice_num, neighbor_num, channel_num)
#         pooled_feature = torch.max(neighbor_feature, dim=2)[0]  # (bs, vertice_num, channel_num)
#
#         pool_num = int(vertice_num / self.pooling_rate)
#         sample_idx = torch.randperm(vertice_num)[:pool_num]
#         vertices_pool = vertices[:, sample_idx, :]  # (bs, pool_num, 3)
#         feature_map_pool = pooled_feature[:, sample_idx, :]  # (bs, pool_num, channel_num)
#         return vertices_pool, feature_map_pool


# def test():
#     import time
#     bs = 3
#     atom_n = 6400
#     dim = 3
#     nei_n = 15  # must be double
#     pos = torch.randn(bs, atom_n, dim)
#     neighbor_index = get_neighbor_index(pos, nei_n)
#
#     conv_1 = AtomConv(kernel_num=32, k_size=14)
#     # conv_2 = ConvLayer(in_channel=32, out_channel=64, support_num=3)
#     # pool = PoolLayer(pooling_rate=4, neighbor_num=4)
#
#     # print("Input size: {}".format(pos.size()))
#     # print(neighbor_index)
#     # print(pos)
#     f1 = conv_1(pos, None, neighbor_index, None)
#     # print("f1 shape: {}".format(f1.size()))
#
#     # f2 = conv_2(neighbor_index, pos, f1)
#     # print("f2 shape: {}".format(f2.size()))
#
#     # v_pool, f_pool = pool(pos, f2)
#     # print("pool atom_n shape: {}, f shape: {}".format(v_pool.size(), f_pool.size()))
#
#
# if __name__ == "__main__":
#     test()
