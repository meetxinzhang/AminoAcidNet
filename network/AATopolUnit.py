# encoding: utf-8
"""
@author: Xin Zhang
@contact: zhangxin@szbl.ac.cn
@file: AATopolUnit.py
@time: 5/18/21 4:01 PM
@desc:
"""
import torch_geometric as geo

import torch
from torch.autograd import Variable
from torch_geometric.nn import MessagePassing


def coordinate_parser(struture):

    return [None]


class AATopolUnit(torch.nn.Module):

    def __init__(self, w_size=None, n_scale=3, n_neighbors=50):
        super(AATopolUnit, self).__init__()
        if w_size is None:
            w_size = [19, 3]
        self.w_size = w_size
        self.n_scale = n_scale
        self.n_neighbors = n_neighbors

        # encode properties of amino acid side chain, like charge, polar and hydrophobic,
        # we believed network will learn itself by training
        self.weight_unit = Variable(torch.FloatTensor(self.w_size), requires_grad=True)
        self.conv_f = torch.nn.Conv2d(in_channels=3, )
        self.activate = torch.nn.Sigmoid()
        pass

    def forward(self, structure, neighbors, neighbors_index, atom_amino_idx):
        """
        encode conformation and neighbors features of amino acid
        :param structure: 3D coor of amino acid
        :param neighbors: list [B, n_atom, n_neighbor=50, 43=40+3]
        :param neighbors_index: list  [B, n_atom, n_neighbor=50]
        :param atom_amino_idx [B, n_atom] Mapping from the amino idx to atom idx
        :return:
        """

        motion = ()

        new_structure = torch.bmm(structure, motion)
        # use coordinate to calculate loss for each amino acid weights
        coordinate = coordinate_parser(new_structure)
        output = None

    def F_motion(self, neighbors, neighbor_index, atom_amino_idx):
        """
        Calculate average Force of side chain use neighbor atoms rather than amino acid,
        due to many strong chemical bond inside of amino acids, that to say each amino acid is a entry,
        so we consider the average Force.
        :param neighbors:        [B, n_atom, n_neighbor=50, 43=40+3]
        :param neighbor_index:   [B, n_atom, n_neighbor=50]
        :param atom_amino_idx [B, n_atom] Mapping from the amino idx to atom idx
        :return:
        """

        atoms_non_this = []

    def F_motion_by_conv(self, neighbors, neighbor_index, atom_amino_idx):
        pass


