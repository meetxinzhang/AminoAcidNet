# encoding: utf-8
"""
@author: Xin Zhang
@contact: zhangxin@szbl.ac.cn
@file: load_from_pkl.py
@time: 5/12/21 5:46 PM
@desc:
"""

import os
import platform
import pickle
import random
import glob
import torch
from torch.utils.data import Dataset
from arguments import build_parser
from torch_geometric.data import Data

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
                                       collate_fn=None,
                                       shuffle=True,
                                       num_workers=parallel_jobs)


def collation(batch):
    pass


def get_train_test_validation_sampler(ratio_test, ratio_val):
    pass


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
        if platform.system() == 'Windows':
            pdb_id = filepath.split('\\')[-1].split('_')[0]
        else:
            pdb_id = filepath.split('/')[-1].split('_')[0]

        with open(filepath, 'rb') as f:
            atom_fea = pickle.load(f)
            # atom_3d = pickle.load(f)
            # neighbor_map = pickle.load(f)
            pos = pickle.load(f)
            edge_idx = pickle.load(f)
            edge_attr = pickle.load(f)
            res_idx = pickle.load(f)
            affinity = pickle.load(f)

        return (atom_fea, pos, edge_idx, edge_attr, res_idx), (affinity, pdb_id)

    def file_filter(self, input_files):
        disallowed_file_endings = (".gitignore", ".DS_Store")
        allowed_file_endings = ".pkl"
        rate = 0.3
        _input_files = input_files[:int(len(input_files) * rate)]
        return list(filter(lambda x: not x.endswith(disallowed_file_endings) and x.endswith(allowed_file_endings),
                           _input_files))



# def collate_pool(dataset_list):
#     """
#     Do padding while stack batch in torch data_loader
#     :param dataset_list [batch_size, (protein_atom_fea, nbr_fea_gauss, nbr_fea_idx, atom_amino_idx), (affinity, protein_id)]
#     """
#     max_n_atom = max([x[0][0].size(0) for x in dataset_list])  # get max atoms
#     # A = max([len(x[0][1]) for x in dataset_list])  # max amino in protein
#     n_neighbors = dataset_list[0][0][1].size(1)  # 50 num neighbors are same for all so take the first value
#     B = len(dataset_list)  # Batch size
#     h_b = dataset_list[0][0][1].size(2)  # 43 edge feature length
#     # all zeros
#     atom_fea = torch.zeros(B, max_n_atom, )
#
#     final_protein_atom_fea = torch.zeros(B, max_n_atom)
#     final_nbr_fea = torch.zeros(B, max_n_atom, n_neighbors, h_b)
#     final_nbr_fea_idx = torch.zeros(B, max_n_atom, n_neighbors, dtype=torch.long)
#     final_atom_amino_idx = torch.zeros(B, max_n_atom)
#     final_atom_mask = torch.zeros(B, max_n_atom)
#     final_target = torch.zeros(B, 1)
#     amino_base_idx = 0  # start number of 1st atom
#
#     batch_protein_ids, amino_crystal = [], 0
#     for i, ((protein_atom_fea, nbr_fea, nbr_fea_idx, atom_amino_idx), (target, protein_id)) in enumerate(
#             dataset_list):
#         """
#         [protein_atom_fea,       [n_atom, 1]
#                   nbr_fea,       [n_atom, n_neighbor=50, 43=40+3]
#               nbr_fea_idx,       [n_atom, n_neighbor=50]
#            atom_amino_idx,       [n_atom]
#                 atom_mask]
#         """
#         num_nodes = protein_atom_fea.size(0)
#
#         final_protein_atom_fea[i][:num_nodes] = protein_atom_fea.squeeze()
#         final_nbr_fea[i][:num_nodes] = nbr_fea
#         final_nbr_fea_idx[i][:num_nodes] = nbr_fea_idx
#         # ?  renumber amino acids in a batch
#         # In a batch:
#         # [000 111 22222 00000] base = 2+1
#         # [33 4444 55555 33333] base = 5+1
#         # [6666 777 888 99 666] base = 9+1
#         final_atom_amino_idx[i][:num_nodes] = atom_amino_idx + amino_base_idx  # list + int
#         final_atom_amino_idx[i][num_nodes:] = amino_base_idx
#         # torch.max(atom_amino_idx)  to get the number of amino acids
#         amino_base_idx += torch.max(atom_amino_idx) + 1
#         final_target[i] = target
#         final_atom_mask[i][:num_nodes] = 1  # donate the ture atom rather the padding
#         batch_protein_ids.append(protein_id)
#         # batch_amino_crystal.append([amino_crystal for _ in range(len(amino_target))])
#         amino_crystal += 1
#
#     return (final_protein_atom_fea, final_nbr_fea, final_nbr_fea_idx, final_atom_amino_idx, final_atom_mask), \
#            (final_target, batch_protein_ids)
