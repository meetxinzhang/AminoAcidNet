# encoding: utf-8
"""
@author: Xin Zhang
@contact: zhangxin@szbl.ac.cn
@file: atom_pooling.py
@time: 6/10/21 4:37 PM
@desc:
"""
import torch
import torch.nn as nn
import numpy as np

#
# class AtomPooling(nn.Module):
#     def __init__(self):
#         super(AtomPooling, self).__init__()
#
#     def __call__(self, pos: "(bs, atom_num, 3)",
#                  atom_fea: "(bs, a_n, dim)",
#                  res_idx: "(bs, a_n)",
#                  atom_mask: "(bs, a_n)"):
#
#         pass


def indexing_neighbor(tensor: "(bs, atom_num, dim)", index: "(bs, atom_num, neighbor_num)"):
    """
    Return: (bs, atom_num, neighbor_num, dim)  torch.Size([3, 26670, 15])
    """
    bs = index.size(0)
    id_0 = torch.arange(bs).view(-1, 1, 1)
    tensor_indexed = tensor[id_0, index]
    return tensor_indexed


class MaxPooling(nn.Module):
    def __init__(self, pooling_size: int = 10):
        super(MaxPooling, self).__init__()
        self.pooling_size = pooling_size\


    def forward(self,
                pos: "(bs, atom_num, 3)",
                atom_fea: "(bs, atom_num, channel_num)",
                res_idx: "(bs, atom_num)",
                atom_mask: "(bs, atom_num)"):
        """
        Return:
            vertices_pool: (bs, pool_atom_num, 3),
            feature_map_pool: (bs, pool_atom_num, channel_num)
        """
        bs, atom_num, channel = atom_fea.size()

        fea_block = atom_fea.view(bs, -1, self.pooling_size, channel)

        pos_block = pos.view(bs, -1, self.pooling_size, 3)
        res_block = res_idx.view(bs, -1, self.pooling_size)
        mask_block = atom_mask.view(bs, -1, self.pooling_size)

        

        # neighbor_feature = indexing_neighbor(atom_fea, edge_index)  # [bs, atom_num, neighbor_num, channel_num]
        # pooled_feature = torch.max(neighbor_feature, dim=2)[0]  # [bs, atom_num, channel_num]
        #
        # pool_num = int(atom_num / self.pooling_size)
        # sample_idx = torch.randperm(atom_num)[:pool_num]
        #
        # vertices_pool = pos[:, sample_idx, :]  # (bs, pool_num, 3)
        # feature_map_pool = pooled_feature[:, sample_idx, :]  # (bs, pool_num, channel_num)
        return
