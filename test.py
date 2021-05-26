# encoding: utf-8
"""
@author: Xin Zhang
@contact: zhangxin@szbl.ac.cn
@file: test.py
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

data = torch.randn(size=[3, 3])
samples = 10
print(data)

file = h5py.File('test.hdf5', 'w')
dataset = file.create_dataset(name='test_data', shape=(samples, 3, 3), maxshape=(samples, 10, 10), dtype='float', chunks=True)
dataset[0] = data

dataset.resize(size=(samples, 3, 5))

reader = h5py.File('test.hdf5', 'r')
output = reader['test_data'][0]
print(output)
