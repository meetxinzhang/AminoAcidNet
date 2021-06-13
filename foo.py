# encoding: utf-8
"""
@author: Xin Zhang
@contact: zhangxin@szbl.ac.cn
@file: foo.py
@time: 5/19/21 5:12 PM
@desc:
"""
import torch
import json
import h5py
import numpy as np
from data.protein_parser import build_node_edge

# with open('/media/zhangxin/Raid0/dataset/PP/json/1a2k.ent.json', 'r') as file:
#     json_data = json.load(file)
#
#     atoms = json_data['atoms']
#     res_idx = json_data['res_idx']
#     bonds = json_data['bonds']
#     contacts = json_data['contacts']
#
#     build_node_edge(atoms, bonds, contacts)


# data = torch.randn(size=[3, 3])
# samples = 10
# print(data)
#
# file = h5py.File('test.hdf5', 'w')
# dataset = file.create_dataset(name='test_data', shape=(samples, 3, 3), maxshape=(samples, 10, 10), dtype='float', chunks=True)
# dataset[0] = data
#
# dataset.resize(size=(samples, 3, 5))
#
# reader = h5py.File('test.hdf5', 'r')
# output = reader['test_data'][0]
# print(output)
def get_neighbor_index(atoms: "(bs, atom_num, 3)", neighbor_num: int):
    """
    Return: (bs, vertice_num, neighbor_num)
    """
    bs, a_n, _ = atoms.size()
    # device = atoms.device
    # print('T', atoms.transpose(1, 2))
    inner = torch.bmm(atoms, atoms.transpose(1, 2))  # (bs, a_n, a_n)
    # print('inner:', inner)
    quadratic = torch.sum(atoms ** 2, dim=2)  # (bs, a_n)
    # print('quadratic: ', quadratic)
    # print(quadratic.unsqueeze(1))
    distance = inner * (-2) + quadratic.unsqueeze(1) + quadratic.unsqueeze(2)
    neighbor_index = torch.topk(distance, k=neighbor_num + 1, dim=-1, largest=False)[1]
    neighbor_index = neighbor_index[:, :, 1:]
    print(neighbor_index)
    return neighbor_index


atoms = torch.randn(30).reshape((2, 5, 3))
print(atoms, '\n')


get_neighbor_index(atoms, 2)

