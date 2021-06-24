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


class AtomPooling(nn.Module):
    def __init__(self):
        super(AtomPooling, self).__init__()

    def __call__(self, pos: "(bs, atom_num, 3)",
                 atom_fea: "(bs, a_n, dim)",
                 res_idx: "(bs, a_n)",
                 atom_mask: "(bs, a_n)"):
        pass
