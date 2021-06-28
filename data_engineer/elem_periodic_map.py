# encoding: utf-8
"""
@author: Xin Zhang
@contact: zhangxin@szbl.ac.cn
@file: build_h5.py
@time: 5/25/21 11:34 AM
@desc:
"""

# serial mol row_in_periodic n_ele
atoms_periodic_dic = {
    'H': [1, 1.008, 1, 1],
    'C': [6, 12.011, 2, 4],
    'N': [7, 14.007, 2, 5],
    'O': [8, 15.999, 2, 6],
    'S': [16, 32.06, 3, 6]
}


# where groups20.txt is located, 167 heavy atoms in 20 amino acids
groups20_path = '/preprocess/data_engineer/groups20.txt'

# Create a one-hot encoded feature map for each protein atom
heavy_atom_idx_dic = {}  # dic
with open(groups20_path, 'r') as f:
    data = f.readlines()
    for idx, line in enumerate(data):
        name, _ = line.split(" ")
        heavy_atom_idx_dic[name] = idx
