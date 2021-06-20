# encoding: utf-8
"""
@author: Xin Zhang
@contact: zhangxin@szbl.ac.cn
@file: load_from_pkl.py
@time: 5/12/21 5:46 PM
@desc:
"""

import os
import numpy as np
import platform
import pickle
import random
import glob
import torch
from torch.utils.data import Dataset
from arguments import build_parser
from data.affinity_parser import get_affinity

parser = build_parser()
args = parser.parse_args()
batch_size = args.batch_size
random_seed = args.random_seed
parallel_jobs = args.parallel_jobs


def get_loader(pkl_dir, affinities_path):
    dataset = PickleDataset(pkl_dir, affinities_path)
    print('construct dataloader... total: ', dataset.__len__(), ', ', batch_size, ' per batch.')
    return torch.utils.data.DataLoader(dataset,
                                       batch_size=batch_size,
                                       sampler=None,
                                       collate_fn=collate_padding,
                                       shuffle=True,
                                       num_workers=parallel_jobs)


def get_train_test_validation_sampler(ratio_test, ratio_val):
    pass


def collate_padding(batch):
    """
    Do padding while stack batch in torch data_loader
    :param batch shape=[bs, 6],
        and 6 for tuple: (pos, atom_fea, edge_idx, edge_attr, res_idx, affinity)
        single tensor: [n_atom, 3], [n_atom, 5],  [n_atom, n_nei], [n_atom, m_nei, 2], [n_atom]
    """
    max_n_atom = max([x[0].size(0) for x in batch])  # get max atoms
    n_ngb = batch[0][2].size(1)  # num neighbors are same for all so take the first value
    bs = len(batch)  # Batch size

    # all zeros
    final_pos = torch.zeros(bs, max_n_atom, 3)
    final_atom_fea = torch.zeros(bs, max_n_atom, 5)
    final_edge_idx = torch.zeros(bs, max_n_atom, n_ngb)
    final_edge_attr = torch.zeros(bs, max_n_atom, n_ngb, 2)
    # final_neighbor_map = torch.zreros(bs, max_n_atom, 25, 3)
    final_res_idx = torch.zeros(bs, max_n_atom)
    final_atom_mask = torch.zeros(bs, max_n_atom)
    final_affinity = torch.zeros(bs, 1)

    amino_base_idx = 0  # start number of 1st atom

    batch_protein_ids, amino_crystal = [], 0
    for i, (pos, atom_fea, edge_idx, edge_attr, res_idx, affinity) in enumerate(batch):
        num_atom = atom_fea.size(0)
        """[n_atom, 3], [n_atom, 5],  [n_atom, n_nei], [n_atom, m_nei, 2], [n_atom], 1
        """
        final_pos[i][:num_atom, :] = pos
        final_atom_fea[i][:num_atom, :] = atom_fea
        final_edge_idx[i][:num_atom, :] = edge_idx
        final_edge_attr[i][:num_atom, :, :] = edge_attr
        # renumber amino acids in a batch
        # Final shape of res_idx in a batch:
        # [000 111 22222 00000] base = 2+1
        # [33 4444 55555 33333] base = 5+1
        # [6666 777 888 99 666] base = 9+1
        final_res_idx[i][:num_atom] = res_idx + amino_base_idx  # list + int
        final_res_idx[i][num_atom:] = amino_base_idx
        # torch.max(atom_amino_idx)  to get the number of amino acids
        amino_base_idx += torch.max(res_idx) + 1
        final_affinity[i] = affinity

        final_atom_mask[i][:num_atom] = 1  # donate the ture atom rather a padding
        # new_batch.append([(final_atom_fea, final_pos, final_neighbor_map, final_res_idx, final_atom_mask),
        #                   (final_affinity, pdb_id)])

        amino_crystal += 1

    return [final_pos, final_atom_fea, final_edge_idx, final_edge_attr, final_res_idx], final_affinity


class PickleDataset(Dataset):
    def __init__(self, pkl_dir, affinities_path, sample_rate=1):
        print("Starting pre-processing of raw data...")
        self.sample_rate = sample_rate
        self.affinity_dic = get_affinity(file_path=affinities_path)
        assert os.path.exists(pkl_dir), '{} does not exist!'.format(pkl_dir)
        # glob 查找符合特定规则的文件 full path. "*"匹配0个或多个字符；"?"匹配单个字符；"[]"匹配指定范围内的字符，[0-9]匹配数字。
        self.filepath_list = self.file_filter(glob.glob(pkl_dir + '/*'))  # list['filename', ...]

        random.seed(random_seed)
        random.shuffle(self.filepath_list)

    def __len__(self):
        return len(self.filepath_list)

    def __getitem__(self, idx):
        filepath = self.filepath_list[idx]
        if platform.system() == 'Windows':
            pdb_id = filepath.split('\\')[-1][0:4]
        else:
            pdb_id = filepath.split('/')[-1][0:4]

        with open(filepath, 'rb') as f:
            pos = torch.Tensor(pickle.load(f))
            atom_fea = torch.Tensor(pickle.load(f))
            edge_idx = torch.Tensor(pickle.load(f))
            edge_attr = torch.Tensor(pickle.load(f))
            res_idx = torch.Tensor(pickle.load(f))
        # [n_atom, 3], [n_atom, 5],  [n_atom, n_nei], [n_atom, m_nei, 2], [n_atom], 1
        return pos, atom_fea, edge_idx, edge_attr, res_idx, self.affinity_dic.get(pdb_id)

    def file_filter(self, input_files):
        disallowed_file_endings = (".gitignore", ".DS_Store")
        allowed_file_endings = ".pkl"
        _input_files = input_files[:int(len(input_files) * self.sample_rate)]
        return list(filter(lambda x: not x.endswith(disallowed_file_endings) and x.endswith(allowed_file_endings),
                           _input_files))

