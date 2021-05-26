# encoding: utf-8
"""
@author: Xin Zhang
@contact: zhangxin@szbl.ac.cn
@file: loader_from_pkl.py
@time: 5/12/21 5:46 PM
@desc:
"""
import csv
import os
import platform
import pickle
import random
import numpy as np
import torch
import glob
from torch.utils.data import Dataset
from data.affinity_parser import get_affinity
from data.protein_parser import AtomCustomJSONInitializer, GaussianDistance


class ProteinDataset(Dataset):
    def __init__(self, pkl_dir, atom_init_filename, random_seed=123):
        assert os.path.exists(pkl_dir), '{} does not exist!'.format(pkl_dir)

        self.pkl_dir = pkl_dir

        protein_atom_init_file = os.path.join(self.pkl_dir, atom_init_filename)
        assert os.path.exists(protein_atom_init_file), '{} does not exist!'.format(protein_atom_init_file)
        self.filenames, self.affinity_dic = self.get_files_list_PBDbind2019()
        random.seed(random_seed)
        random.shuffle(self.filenames)
        self.ari = AtomCustomJSONInitializer(protein_atom_init_file)
        self.gdf = GaussianDistance(dmin=0, dmax=15, step=0.4)

    def get_files_list_PBDbind2019(self):
        """
        Traversal raw data in data/raw/，and use function process_file() to read data in loop at the same time
        output preprocessed data in the format .hdf5 in data/preprocessed/
        """
        print("Starting pre-processing of raw data...")
        # glob 查找符合特定规则的文件 full path.
        # 匹配符："*", "?", "[]"。"*"匹配0个或多个字符；"?"匹配单个字符；"[]"匹配指定范围内的字符，[0-9]匹配数字。
        files_list = glob.glob(self.pkl_dir + '/*')
        files_filtered_list = self.filter_input_files(files_list)  # list['filename', ...]
        affinity_dic = get_affinity(file_path='/media/zhangxin/Raid0/dataset/PP' + '/index/INDEX_general_PP.2019')
        return files_filtered_list, affinity_dic

    def __len__(self):
        return len(self.pkl_dir) - 2

    def __getitem__(self, idx):
        file_path = self.filenames[idx]
        if platform.system() == 'Windows':
            pdb_id = file_path.split('\\')[-1].replace('.ent.pkl', '').replace('_', '')
        else:
            pdb_id = file_path.split('/')[-1].replace('.ent.pkl', '').replace('_', '')
        affinity = self.affinity_dic.get(pdb_id)  # get affinity

        with open(file_path, 'rb') as f:
            # ['ASP_N', ] -> [index, ], shape=[n_atom, ]
            atoms_index = [self.ari.get_atom_fea(atom) for atom in pickle.load(f)]  # ['ASP_N', ]
            # shape=[n_atom, 1]
            atom_fea = torch.Tensor(np.vstack(atoms_index))  # Atom features (here one-hot encoding is used)
            # shape=[n_atom, n_neighbor=50, edge_fea=5], edge_fea=[atom_idx, distance, x,y,z]
            nbr_fea = pickle.load(f)  # Edge features for each atom in the graph
            # shape=[n_atom, n_neighbor=50] every atom has 50 neighbors
            nbr_fea_idx = torch.LongTensor(pickle.load(f))  # Edge connections that define the graph
            # shape=[n_atom] Mapping that denotes which atom corresponds to which amino residue in the protein graph
            atom_amino_idx = torch.LongTensor(pickle.load(f))
            # string
            protein_id = pickle.load(f)
            # shape=[n_atom, n_neighbor=50, 43=40+3]
            nbr_fea_gauss = torch.Tensor(np.concatenate([self.gdf.expand(nbr_fea[:, :, 0]), nbr_fea[:, :, 1:]],
                                                  axis=2))  # Use Gaussian expansion for edge distance
            affinity = torch.Tensor([float(affinity)])
        return (atom_fea, nbr_fea_gauss, nbr_fea_idx, atom_amino_idx), (affinity, protein_id)

    def filter_input_files(self, input_files):
        disallowed_file_endings = (".gitignore", ".DS_Store")
        allowed_file_endings = ".pkl"
        rate = 0.3
        partical_input_files = input_files[:int(len(input_files) * rate)]
        return list(filter(lambda x: not x.endswith(disallowed_file_endings) and x.endswith(allowed_file_endings),
                           partical_input_files))


def collate_pool(dataset_list):
    """
    padding while stack batch in torch data_loader
    :param dataset_list [batch_size, (protein_atom_fea, nbr_fea_gauss, nbr_fea_idx, atom_amino_idx), (affinity, protein_id)]
    """
    max_n_atom = max([x[0][0].size(0) for x in dataset_list])  # get max atoms
    # A = max([len(x[0][1]) for x in dataset_list])  # max amino in protein
    n_neighbors = dataset_list[0][0][1].size(1)  # 50 num neighbors are same for all so take the first value
    B = len(dataset_list)  # Batch size
    h_b = dataset_list[0][0][1].size(2)  # 43 edge feature length
    # all zeros
    atom_fea = torch.zeros(B, max_n_atom, )


    final_protein_atom_fea = torch.zeros(B, max_n_atom)
    final_nbr_fea = torch.zeros(B, max_n_atom, n_neighbors, h_b)
    final_nbr_fea_idx = torch.zeros(B, max_n_atom, n_neighbors, dtype=torch.long)
    final_atom_amino_idx = torch.zeros(B, max_n_atom)
    final_atom_mask = torch.zeros(B, max_n_atom)
    final_target = torch.zeros(B, 1)
    amino_base_idx = 0  # start number of 1st atom

    batch_protein_ids, amino_crystal = [], 0
    for i, ((protein_atom_fea, nbr_fea, nbr_fea_idx, atom_amino_idx), (target, protein_id)) in enumerate(
            dataset_list):
        """
        [protein_atom_fea,       [n_atom, 1]
                  nbr_fea,       [n_atom, n_neighbor=50, 43=40+3]
              nbr_fea_idx,       [n_atom, n_neighbor=50]
           atom_amino_idx,       [n_atom]
                atom_mask]
        """
        num_nodes = protein_atom_fea.size(0)

        final_protein_atom_fea[i][:num_nodes] = protein_atom_fea.squeeze()
        final_nbr_fea[i][:num_nodes] = nbr_fea
        final_nbr_fea_idx[i][:num_nodes] = nbr_fea_idx
        # ?  renumber amino acids in a batch
        # In a batch:
        # [000 111 22222 00000] base = 2+1
        # [33 4444 55555 33333] base = 5+1
        # [6666 777 888 99 666] base = 9+1
        final_atom_amino_idx[i][:num_nodes] = atom_amino_idx + amino_base_idx  # list + int
        final_atom_amino_idx[i][num_nodes:] = amino_base_idx
        # torch.max(atom_amino_idx)  to get the number of amino acids
        amino_base_idx += torch.max(atom_amino_idx) + 1
        final_target[i] = target
        final_atom_mask[i][:num_nodes] = 1  # donate the ture atom rather the padding
        batch_protein_ids.append(protein_id)
        # batch_amino_crystal.append([amino_crystal for _ in range(len(amino_target))])
        amino_crystal += 1

    return (final_protein_atom_fea, final_nbr_fea, final_nbr_fea_idx, final_atom_amino_idx, final_atom_mask), \
           (final_target, batch_protein_ids)


# backup
# def collate_pool(dataset_list):
#     """
#     padding while stack batch in torch data_loader
#     :param dataset_list [batch_size, (protein_atom_fea, nbr_fea_gauss, nbr_fea_idx, atom_amino_idx), (affinity, protein_id)]
#     """
#     max_n_atom = max([x[0][0].size(0) for x in dataset_list])  # get max atoms
#     # A = max([len(x[0][1]) for x in dataset_list])  # max amino in protein
#     n_neighbors = dataset_list[0][0][1].size(1)  # 50 num neighbors are same for all so take the first value
#     B = len(dataset_list)  # Batch size
#     h_b = dataset_list[0][0][1].size(2)  # 43 edge feature length
#     # all zeros
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