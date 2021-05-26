# encoding: utf-8
"""
@author: Xin Zhang
@contact: zhangxin@szbl.ac.cn
@file: loader_from_h5.py
@time: 5/26/21 11:40 AM
@desc:
"""

import h5py
import torch
from torch.utils.data import DataLoader


def construct_dataloader_from_h5(filename, batch_size):
    dataset = H5PytorchDataset(filename)
    print('construct dataloader... total: ', dataset.__len__(), ', ', batch_size, ' per batch.')
    return torch.utils.data.DataLoader(H5PytorchDataset(filename),
                                       batch_size=batch_size,
                                       shuffle=True,
                                       num_workers=4)


def collation():
    pass


class H5PytorchDataset(torch.utils.data.Dataset):
    def __init__(self, filename):
        super(H5PytorchDataset, self).__init__()

        self.h5py_file = h5py.File(filename, 'r')
        self.num_samples, self.max_sequence_len = self.h5py_file['primary'].shape

    def __getitem__(self, index):
        atom_fea = torch.Tensor(self.h5py_file['atom_fea'][index])
        atom_3d = torch.Tensor(self.h5py_file['atom_3d'][index])
        edge = torch.Tensor(self.h5py_file['edge'][index])
        res_idx = torch.Tensor(self.h5py_file['res_idx'][index])
        affinity = torch.Tensor(self.h5py_file['affinity'][index])

        return atom_fea, atom_3d, edge, res_idx, affinity

    def __len__(self):
        return self.num_samples
