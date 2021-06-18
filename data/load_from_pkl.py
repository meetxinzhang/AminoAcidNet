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

parser = build_parser()
args = parser.parse_args()
batch_size = args.batch_size
random_seed = args.random_seed
parallel_jobs = args.parallel_jobs


def get_loader(pkl_dir):
    dataset = PickleDataset(pkl_dir)
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
    :param batch shape=[bs, 6], [... , [atom_fea, pos, edge_attr, edge_idx, res_idx, affinity]]
    """
    print(np.shape(batch))
    max_n_atom = max([len(x[0]) for x in batch])  # get max atoms
    # A = max([len(x[0][1]) for x in dataset_list])  # max amino in protein
    n_neighbors = np.shape(batch[0][2])[1]  # 25 num neighbors are same for all so take the first value
    bs = len(batch)  # Batch size
    # h_b = batch[0][0][1].size(2)  # 43 edge feature length

    # all zeros
    final_atom_fea = torch.zeros(bs, max_n_atom, 5)
    final_pos = torch.zeros(bs, max_n_atom, 3)
    # final_neighbor_map = torch.zeros(bs, max_n_atom, 25, 3)
    final_edge_attr = torch.zeros(bs, n_neighbors, 2)
    final_edge_idx = torch.zeros(bs, n_neighbors, 1)
    final_res_idx = torch.zeros(bs, max_n_atom)
    final_atom_mask = torch.zeros(bs, max_n_atom)
    final_affinity = torch.zeros(bs, 1)
    new_batch = []

    # final_protein_atom_fea = torch.zeros(bs, max_n_atom)
    # # final_nbr_fea = torch.zeros(bs, max_n_atom, n_neighbors, h_b)
    # final_nbr_fea_idx = torch.zeros(bs, max_n_atom, n_neighbors, dtype=torch.long)
    # final_atom_amino_idx = torch.zeros(bs, max_n_atom)
    # final_atom_mask = torch.zeros(bs, max_n_atom)
    # final_target = torch.zeros(bs, 1)
    amino_base_idx = 0  # start number of 1st atom

    batch_protein_ids, amino_crystal = [], 0
    for i, ((atom_fea, pos, neighbor_map, res_idx), (infinity, pdb_id)) in enumerate(
            batch):
        num_atom = atom_fea.size(0)
        """
                [atom_fea,       [n_atom, 5]
                      pos,       [n_atom, 3]
             neighbor_map,       [n_atom, 25, 3]
                  res_idx,       [n_atom]
                atom_mask]       [n_atom]
        """
        final_atom_fea[i][:num_atom] = atom_fea
        final_pos[i][:num_atom] = pos
        final_neighbor_map[i][:num_atom] = neighbor_map
        # renumber amino acids in a batch
        # Final shape of res_idx in a batch:
        # [000 111 22222 00000] base = 2+1
        # [33 4444 55555 33333] base = 5+1
        # [6666 777 888 99 666] base = 9+1
        final_res_idx[i][:num_atom] = res_idx + amino_base_idx  # list + int
        final_res_idx[i][num_atom:] = amino_base_idx
        # torch.max(atom_amino_idx)  to get the number of amino acids
        amino_base_idx += torch.max(res_idx) + 1
        final_affinity[i] = infinity

        final_atom_mask[i][:num_atom] = 1  # donate the ture atom rather a padding
        # new_batch.append([(final_atom_fea, final_pos, final_neighbor_map, final_res_idx, final_atom_mask),
        #                   (final_affinity, pdb_id)])

        amino_crystal += 1

    return (final_atom_fea, final_pos, final_neighbor_map, final_res_idx, final_atom_mask), final_affinity


class PickleDataset(Dataset):
    def __init__(self, pkl_dir):
        assert os.path.exists(pkl_dir), '{} does not exist!'.format(pkl_dir)

        self.pkl_dir = pkl_dir

        print("Starting pre-processing of raw data...")
        # glob 查找符合特定规则的文件 full path.
        # 匹配符："*", "?", "[]"。"*"匹配0个或多个字符；"?"匹配单个字符；"[]"匹配指定范围内的字符，[0-9]匹配数字。
        filepath_list = glob.glob(self.pkl_dir + '/*')
        self.filepath_list = self.file_filter(filepath_list)  # list['filename', ...]

        random.seed(random_seed)
        random.shuffle(self.filepath_list)

    def __len__(self):
        return len(self.filepath_list)

    def __getitem__(self, idx):
        filepath = self.filepath_list[idx]
        # if platform.system() == 'Windows':
        #     pdb_id = filepath.split('\\')[-1].split('_')[0]
        # else:
        #     pdb_id = filepath.split('/')[-1].split('_')[0]

        with open(filepath, 'rb') as f:
            atom_fea = torch.tensor(pickle.load(f))
            # pos = torch.tensor(pickle.load(f))
            # neighbor_map = torch.tensor(pickle.load(f))
            pos = pickle.load(f)
            edge_idx = pickle.load(f)
            edge_attr = pickle.load(f)
            res_idx = pickle.load(f)
            affinity = pickle.load(f)

        return atom_fea, pos, edge_attr, edge_idx, res_idx, affinity

    def file_filter(self, input_files):
        disallowed_file_endings = (".gitignore", ".DS_Store")
        allowed_file_endings = ".pkl"
        rate = 0.3
        _input_files = input_files[:int(len(input_files) * rate)]
        return list(filter(lambda x: not x.endswith(disallowed_file_endings) and x.endswith(allowed_file_endings),
                           _input_files))

