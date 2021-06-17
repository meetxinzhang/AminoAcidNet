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


def get_neighbor_index(atoms: "(bs, atom_num, 3)", neighbor_num: int):
    """
    Return: (bs, atom_num, neighbor_num)
    """
    bs, a_n, _ = atoms.size()
    # device = atoms.device
    # tensor.transpose(1, 2), transposes only 1 and 2 dim, =tf.transpose(0, 2, 1)
    inner = torch.bmm(atoms, atoms.transpose(1, 2))  # [bs, a_n, a_n]
    quadratic = torch.sum(atoms ** 2, dim=2)  # [bs, a_n]
    # [bs, a_n, a_n] + [bs, 1, a_n] + [bs, a_n, 1]
    distance = inner * (-2) + quadratic.unsqueeze(1) + quadratic.unsqueeze(2)
    neighbor_index = torch.topk(distance, k=neighbor_num + 1, dim=-1, largest=False)[1]
    neighbor_index = neighbor_index[:, :, 1:]
    return neighbor_index


# def get_nearest_index(target: "(bs, v1, 3)", source: "(bs, v2, 3)"):
#     """
#     Return: (bs, v1, 1)
#     """
#     inner = torch.bmm(target, source.transpose(1, 2))  # (bs, v1, v2)
#     s_norm_2 = torch.sum(source ** 2, dim=2)  # (bs, v2)
#     t_norm_2 = torch.sum(target ** 2, dim=2)  # (bs, v1)
#     d_norm_2 = s_norm_2.unsqueeze(1) + t_norm_2.unsqueeze(2) - 2 * inner
#     nearest_index = torch.topk(d_norm_2, k=1, dim=-1, largest=False)[1]
#     return nearest_index


def indexing_neighbor(tensor: "(bs, atom_num, dim)", index: "(bs, atom_num, neighbor_num)"):
    """
    Return: (bs, atom_num, neighbor_num, dim)
    """
    bs, _, _ = index.size()
    id_0 = torch.arange(bs).view(-1, 1, 1)
    tensor_indexed = tensor[id_0, index]
    return tensor_indexed


def get_neighbor_direct_norm(atoms: "(bs, atom_num, 3)", neighbor_index: "(bs, atom_num, neighbor_num)"):
    """
    Return: (bs, atom_num, atom_num, 3)
    """
    pos_neighbors = indexing_neighbor(atoms, neighbor_index)  # [bs, a_n, nei_n, 3]
    neighbor_direction = pos_neighbors - atoms.unsqueeze(2)  # [bs, a_n, nei_n, 3] - [bs, a_n, 1, 3]
    neighbor_direction_norm = F.normalize(neighbor_direction, dim=-1)  # unit vector of distance
    return neighbor_direction_norm


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

def cos_theta(vectors: "(bs, a_n, nei_n, 3)"):
    nearest = vectors[:, :, 0, :].unsqueeze(2)
    else_neigh = vectors[:, :, 1:, :]
    # print(nearest.size())
    # print(else_neigh.size())
    theta = else_neigh @ nearest.transpose(2, 3)
    # print(theta.size())
    return theta


class AtomConv(nn.Module):
    """Extract structure features from surface, independent from atom coordinates"""

    def __init__(self, kernel_num, k_size):
        super(AtomConv, self).__init__()
        self.kernel_num = kernel_num
        self.k_size = k_size

        self.relu = nn.ReLU(inplace=True)
        # self.directions = nn.Parameter(torch.FloatTensor(3, k_size * 1))  # linear weight
        self.angle_weights = nn.Parameter(torch.FloatTensor(1, k_size * 1))  # linear weight
        # self.initialize()

    # def initialize(self):
    #     stdv = 1. / math.sqrt(self.k_size * self.kernel_num)
    #     self.directions.data.uniform_(-stdv, stdv)

    def forward(self,
                neighbor_index: "(bs, atom_num, neighbor_num)",
                atoms: "(bs, atom_num, 3)"):
        """
        Return vertices with local feature: (bs, atom_num, kernel_num)
        """
        bs, atom_num, neighbor_num = neighbor_index.size()
        nei_direct_norm = get_neighbor_direct_norm(atoms, neighbor_index)  # [bs, a_n, nei_n, 3]
        theta = cos_theta(nei_direct_norm)  # [bs, a_n, nei_n-1 1]

        # support_direct_norm = F.normalize(self.directions, dim=0)  # (3, s * k)
        # theta = nei_direct_norm @ support_direct_norm  # [bs, atom_num, neighbor_num, s*k]

        theta = self.relu(theta)
        theta = theta.contiguous().view(bs, atom_num, neighbor_num, self.k_size, self.kernel_num)

        # to get the biggest [k_size, kernel_num] in neighbor_num neighbors of each atom
        theta = torch.max(theta, dim=2)[0]  # [bs, atom_num, k_size, kernel_num]

        feature = torch.sum(theta, dim=2)  # [bs, atom_num, kernel_num]
        return feature


class ConvLayer(nn.Module):
    def __init__(self, in_channel, out_channel, support_num):
        super().__init__()
        # arguments:
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.support_num = support_num

        # parameters:
        self.relu = nn.ReLU(inplace=True)
        self.weights = nn.Parameter(torch.FloatTensor(in_channel, (support_num + 1) * out_channel))
        self.bias = nn.Parameter(torch.FloatTensor((support_num + 1) * out_channel))
        self.directions = nn.Parameter(torch.FloatTensor(3, support_num * out_channel))
        self.initialize()

    def initialize(self):
        stdv = 1. / math.sqrt(self.out_channel * (self.support_num + 1))
        self.weights.data.uniform_(-stdv, stdv)
        self.bias.data.uniform_(-stdv, stdv)
        self.directions.data.uniform_(-stdv, stdv)

    def forward(self,
                neighbor_index: "(bs, vertice_num, neighbor_index)",
                vertices: "(bs, vertice_num, 3)",
                feature_map: "(bs, vertice_num, in_channel)"):
        """
        Return: output feature map: (bs, vertice_num, out_channel)
        """
        bs, vertice_num, neighbor_num = neighbor_index.size()
        neighbor_direction_norm = get_neighbor_direct_norm(vertices, neighbor_index)
        support_direction_norm = F.normalize(self.directions, dim=0)
        theta = neighbor_direction_norm @ support_direction_norm  # (bs, vertice_num, neighbor_num, support_num * out_channel)
        theta = self.relu(theta)
        theta = theta.contiguous().view(bs, vertice_num, neighbor_num, -1)
        # (bs, vertice_num, neighbor_num, support_num * out_channel)

        feature_out = feature_map @ self.weights + self.bias  # (bs, vertice_num, (support_num + 1) * out_channel)
        feature_center = feature_out[:, :, :self.out_channel]  # (bs, vertice_num, out_channel)
        feature_support = feature_out[:, :, self.out_channel:]  # (bs, vertice_num, support_num * out_channel)

        # Fuse together - max among product
        feature_support = indexing_neighbor(feature_support,
                                            neighbor_index)  # (bs, vertice_num, neighbor_num, support_num * out_channel)
        activation_support = theta * feature_support  # (bs, vertice_num, neighbor_num, support_num * out_channel)
        activation_support = activation_support.view(bs, vertice_num, neighbor_num, self.support_num, self.out_channel)
        activation_support = torch.max(activation_support, dim=2)[0]  # (bs, vertice_num, support_num, out_channel)
        activation_support = torch.sum(activation_support, dim=2)  # (bs, vertice_num, out_channel)
        feature_fuse = feature_center + activation_support  # (bs, vertice_num, out_channel)
        return feature_fuse


class PoolLayer(nn.Module):
    def __init__(self, pooling_rate: int = 4, neighbor_num: int = 4):
        super().__init__()
        self.pooling_rate = pooling_rate
        self.neighbor_num = neighbor_num

    def forward(self,
                vertices: "(bs, vertice_num, 3)",
                feature_map: "(bs, vertice_num, channel_num)"):
        """
        Return:
            vertices_pool: (bs, pool_vertice_num, 3),
            feature_map_pool: (bs, pool_vertice_num, channel_num)
        """
        bs, vertice_num, _ = vertices.size()
        neighbor_index = get_neighbor_index(vertices, self.neighbor_num)
        neighbor_feature = indexing_neighbor(feature_map,
                                             neighbor_index)  # (bs, vertice_num, neighbor_num, channel_num)
        pooled_feature = torch.max(neighbor_feature, dim=2)[0]  # (bs, vertice_num, channel_num)

        pool_num = int(vertice_num / self.pooling_rate)
        sample_idx = torch.randperm(vertice_num)[:pool_num]
        vertices_pool = vertices[:, sample_idx, :]  # (bs, pool_num, 3)
        feature_map_pool = pooled_feature[:, sample_idx, :]  # (bs, pool_num, channel_num)
        return vertices_pool, feature_map_pool


def test():
    import time
    bs = 2
    atom_n = 15
    dim = 3
    nei_n = 8  # must be double
    pos = torch.randn(bs, atom_n, dim)
    neighbor_index = get_neighbor_index(pos, nei_n)

    s = 3
    conv_1 = AtomConv(kernel_num=32, k_size=s)
    conv_2 = ConvLayer(in_channel=32, out_channel=64, support_num=s)
    pool = PoolLayer(pooling_rate=4, neighbor_num=4)

    # print("Input size: {}".format(pos.size()))
    # print(neighbor_index)
    # print(pos)
    f1 = conv_1(neighbor_index, pos)
    # print("f1 shape: {}".format(f1.size()))

    f2 = conv_2(neighbor_index, pos, f1)
    # print("f2 shape: {}".format(f2.size()))

    v_pool, f_pool = pool(pos, f2)
    # print("pool atom_n shape: {}, f shape: {}".format(v_pool.size(), f_pool.size()))


if __name__ == "__main__":
    test()
