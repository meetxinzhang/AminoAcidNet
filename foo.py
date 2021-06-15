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
from baseline.ldk.gcn3d_lkd import indexing_neighbor

tensor = [[[0.2462, -1.3954, 0.0226],
           [0.6297, 0.9197, 0.3977],
           [1.8450, -0.7786, 0.4734],
           [0.0810, -0.7617, 0.5351],
           [1.1971, -0.4210, 1.7760]],

          [[0.1859, 1.6950, 0.1322],
           [-0.6759, -2.1377, -0.6646],
           [0.2580, -0.0107, 0.5895],
           [0.6554, 0.5402, 0.7859],
           [0.1759, 1.4783, -0.2102]]]

index = [[[3, 2, 4],
          [3, 4, 2],
          [4, 3, 0],
          [0, 4, 2],
          [2, 3, 1]],

         [[4, 3, 2],
          [2, 3, 4],
          [3, 4, 0],
          [2, 0, 4],
          [0, 3, 2]]]

tensor = torch.Tensor(tensor)
index = torch.tensor(index)

neighbor_pos = tensor[torch.arange(2).view(-1, 1, 1), index]
# print(neighbor_pos.size())
# print(tensor.unsqueeze(2).size())
# print(neighbor_pos)
#
# print(neighbor_pos - tensor.unsqueeze(2))

a = [[1, 2],
     [3, 4]]

b = [[1, 2],
     [3, 4]]


a = torch.Tensor(a)
b = torch.Tensor(b)
print(a @ b)
print(torch.mm(a, b))
