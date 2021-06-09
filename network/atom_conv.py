# encoding: utf-8
"""
@author: Xin Zhang
@contact: zhangxin@szbl.ac.cn
@file: atom_conv.py
@time: 6/9/21 4:55 PM
@desc:
"""
import torch


class AtomConv(torch.nn.Module):
    def __init__(self, r_coulomb=1.0, n_neural=16):
        super(AtomConv, self).__init__()
        self.r_coulomb = r_coulomb
        self.n_neural = n_neural

    def __call__(self, atom, neighbors):
        """
        :param atom: [n_atom, 5], 5: idx serial mol row_in_periodic n_ele
        :param neighbors: [n_atom, 25, 3]
        :return:
        """
        0



