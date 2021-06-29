# encoding: utf-8
"""
@author: Xin Zhang
@contact: zhangxin@szbl.ac.cn
@file: conv.py
@time: 6/9/21 4:55 PM
@desc:
"""
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

    def cos_theta(self, pos, edge_index, atom_mask):
        """
        Embed spatial features
        :return [bs, a_n, nei_n, 1] """
        nei_direct_norm = get_neighbor_direct_norm(pos, edge_index)  # [bs, a_n, nei_n, 3]
        nearest = nei_direct_norm[:, :, 0, :].unsqueeze(2)  # [2, 15, 1, 3]
        else_neigh = nei_direct_norm[:, :, 1:, :]  # [2, 15, nei_n-1, 3]
        theta = else_neigh @ nearest.transpose(2, 3)  # [bs, a_n, nei_n-1, 1]
        cos0_theta = F.pad(theta, [0, 0, 1, 0], value=1)  # cos(0)=1

        mask = atom_mask.unsqueeze(-1).repeat(1, 1, self.k_size).unsqueeze(-1)
        mask_final = atom_mask.unsqueeze(-1).repeat(1, 1, self.kernel_num)
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
        a = self.relu(torch.matmul(edge_fea, self.radius_weight_1))  # [bs, a_n, nei_n, kernel_num]
        # g = torch.max(a, dim=3)[0]  # [bs, a_n, nei_n]
        b = self.relu(torch.matmul(a, self.radius_weight_2)).squeeze()  # [bs, a_n, nei_n]

        mask = atom_mask.unsqueeze(-1).repeat(1, 1, self.k_size)
        g = torch.mul(b, mask)
        g = g.unsqueeze(-1).repeat(1, 1, 1, self.kernel_num)  # [bs, a_n, nei_n, kernel_num]
        return self.sigmoid(g)


class ConvLayer(nn.Module):
    """Extract and recombines structure and chemical elements features from local domain of protein graph
    in batch format, k_size denotes the range of domain.
    :param k_size: int, num of neighbor atoms which are considered
    :param kernel_num, int
    """

    def __init__(self, kernel_num, k_size, in_channels, node_fea_dim):
        super(ConvLayer, self).__init__()
        self.kernel_num = kernel_num
        self.k_size = k_size
        self.in_channels = in_channels
        self.dim = node_fea_dim

        self.leaky_relu = nn.LeakyReLU(inplace=True)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

        self.angle_weight = nn.Parameter(torch.FloatTensor(kernel_num, k_size*in_channels))
        self.scalar_weight = nn.Parameter(torch.FloatTensor(2*in_channels*self.dim, kernel_num))
        self.radius_weight_1 = nn.Parameter(torch.FloatTensor(1, kernel_num))
        self.radius_weight_2 = nn.Parameter(torch.FloatTensor(kernel_num, 1))

        nn.init.uniform_(self.angle_weight, -1, 1)
        nn.init.uniform_(self.scalar_weight, 0, 1)
        nn.init.uniform_(self.radius_weight_1, -1, 1)
        nn.init.uniform_(self.radius_weight_2, -1, 1)

    def forward(self, pos: "(bs, c, n, 3)",
                node_fea: "(bs, c, n, d)",
                node_mask: "(bs, n)"):
        """
        Return: (bs, atom_num, kernel_num)
        """
        # TODO: pos fea first layer
        bs, n = node_mask.size()
        distances, edge_index = self.get_edge(pos)  # [bs, c, n, k_size]

        theta, mask_final = self.cos_theta(pos, edge_index, node_mask)  # [bs, n, k_size, c], [bs, c, n, k_size]
        fea_cat = self.feature_fusion(node_fea, edge_index, node_mask)  # [bs, n, k_size, c, 2*d]
        gate = self.message_gating(distances, node_mask)  # [bs, n, k_size, c]

        theta_channel = theta.view(bs, n, -1)  # [bs, n, k_size*c]
        # [bs, n, kernel_num, 1] to [bs, kernel_num, n, 1]
        fea_struct = torch.matmul(self.angle_weight, theta_channel).permute(0, 2, 1, 3)

        # [bs, n, k_size, c, 2*d] to # [bs, n, k_size, c*2*d]
        fea_gated = torch.mul(gate.unsqueeze(-1), fea_cat).view(bs, n, self.k_size, -1)
        fea_elem = torch.matmul(fea_gated, self.scalar_weight)  # [bs, n, k_size, kernel_num]

        # [bs, n, 1, kernel_num] to [bs, kernel_num, n, 1]
        fea_elem = torch.sum(fea_elem, dim=2).permute(0, 3, 1, 2)

        interactions = self.leaky_relu(fea_elem + fea_struct)  # [bs, kernel_num, n, 1]
        interact_masked = torch.mul(mask_final, interactions)
        return interact_masked

    def get_edge(self, pos: "(bs, c, n, 3)"):
        """
        Return: (bs, atom_num, neighbor_num)
        """
        # device = atoms.device
        # tensor.transpose(1, 2), transposes only 1 and 2 dim, =tf.transpose(0, 2, 1)
        inner = torch.matmul(pos, pos.transpose(2, 3))  # [bs, c, n, n]
        quadratic = torch.sum(pos ** 2, dim=3)  # [bs, c, n]
        # [bs, c, n, n] + [bs, c, 1, n] + [bs, c, n, 1]
        distance = inner * (-2) + quadratic.unsqueeze(2) + quadratic.unsqueeze(3)
        ngh_distance, ngb_index = torch.topk(distance, k=self.k_size + 1, dim=-1, largest=False)  # [bs, c, n, k_size+1]
        # distance reference itself is 0 and should be ignore
        return ngh_distance[:, :, :, 1:], ngb_index[:, :, :, 1:]

    def indexing_neighbor(self, tensor: "(bs, c, n, dim)", index: "(bs, c, n, k_size)"):
        """
        tensor: if pos dim=3, and dim=else for fea, c=1 for first layer
        Return: (bs, c, n, k_size, dim)
        """
        bs = index.size(0)
        id_0 = torch.arange(bs).view(-1, 1, 1, 1)
        tensor_indexed = tensor[id_0, index]
        return tensor_indexed

    def get_neighbor_direct_norm(self, pos: "(bs, c, n, dim)", ngb_index: "(bs, c, n, k_size)"):
        """
        Return: (bs, c, n, k_size, 3)
        """
        pos_ngb = self.indexing_neighbor(pos, ngb_index)  # [bs, c, n, k_size, 3]
        neigh_direction = pos_ngb - pos.unsqueeze(3)  # [bs, c, n, k_size, 3] - [bs, c, n, 1, 3]
        neigh_direction_norm = F.normalize(neigh_direction, dim=-1)  # unit vector of distance
        return neigh_direction_norm

    def cos_theta(self, pos: "(bs, c, n, 3)", edge_index: "(bs, c, n, k_size)", node_mask: "(bs, n)"):
        """
        Embed spatial features
        :return theta_masked [bs, n, k_size, c]
        :return mask_final [bs, kernel_num, n, 1]"""
        nei_direct_norm = self.get_neighbor_direct_norm(pos, edge_index)  # [bs, c, n, k_size, 3]
        nearest = nei_direct_norm[:, :, :, 0, :].unsqueeze(3)  # [bs, c, n, 1, 3]
        else_neigh = nei_direct_norm[:, :, :, 1:, :]  # [bs, c, n, k_size-1, 3]
        theta = else_neigh @ nearest.transpose(2, 3)  # [bs, c, n, k_size-1, 1]
        cos0_theta = F.pad(theta, [0, 0, 0, 1, 0], value=1).squeeze()  # cos(0)=1, [bs, c, n, k_size]

        c = pos.size()[1]
        mask = node_mask.unsqueeze(1).repeat(1, c, 1).unsqueeze(-1)  # [bs, c, n, 1]
        mask_final = node_mask.unsqueeze(1).repeat(1, self.kernel_num, 1).unsqueeze(-1)  #  [bs, kernel_num, n, 1]

        theta_masked = torch.mul(cos0_theta, mask).permute(0, 2, 3, 1)  # [bs, n, k_size, c]
        return theta_masked, mask_final

    def feature_fusion(self, node_fea: "(bs, c, n, d)",
                       index: "(bs, c, n, k_size)",
                       node_mask: "(bs, n)"):
        """Fuse elements features, dim=5 in first layer and 1 for else layers"""
        bs, n = node_mask.size()
        fea_neigh = self.indexing_neighbor(node_fea, index)  # [bs, c, n, k_size, d]
        node_reps_fea = node_fea.unsqueeze(2).repeat(1, 1, 1, self.k_size, 1)  # [bs, c, n, k_size, d]
        # [bs, c, n, k_size, 2*d] to [bs, n, k_size, c, 2*d]
        fea_cat = torch.cat([node_reps_fea, fea_neigh], dim=-1).permute(0, 2, 3, 1, 4)

        mask = node_mask.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        return torch.mul(fea_cat, mask)

    def message_gating(self, distance: "(bs, c, n, k_size)", node_mask: "(bs, n)"):
        """Generate a gate to select elements features based on distance and bond type"""
        a = self.relu(torch.matmul(distance, self.radius_weight_1))  # [bs, c, n, k_size, kernel_num]
        # g = torch.max(a, dim=3)[0]  # [bs, a_n, nei_n]
        b = self.relu(torch.matmul(a, self.radius_weight_2)).squeeze()  # [bs, c, n, k_size]

        mask = node_mask.unsqueeze(1).repeat(1, self.in_channels, 1).unsqueeze(-1).repeat(1, 1, 1, self.k_size)
        g = torch.mul(b, mask)  # [bs, c, n, k_size]
        return self.sigmoid(g.permute(0, 2, 3, 1))

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
#         self.weights.data_engineer.uniform_(-stdv, stdv)
#         self.bias.data_engineer.uniform_(-stdv, stdv)
#         self.directions.data_engineer.uniform_(-stdv, stdv)
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
