# encoding: utf-8
"""
@author: Xin Zhang
@contact: zhangxin@szbl.ac.cn
@file: h5_build.py
@time: 5/25/21 11:34 AM
@desc:
"""
import os
import json
from tqdm import tqdm
from joblib import Parallel, delayed
import numpy as np
import h5py
from collections import defaultdict as ddict
from data.affinity_parser import get_affinity
from arguments import buildParser

parser = buildParser()
args = parser.parse_args()
jsonpath = args.jsonpath
h5_dir = args.h5_dir
h5_name = args.h5_name
override_h5_dataset = args.override_h5_dataset
parallel_jobs = args.parallel_jobs
max_neighbors = args.max_neighbors

MAX_SEQUENCE_LEN = 1000  # 氨基酸主链的最大队列长度
MAX_DATASET_LEN = 10000  # max length of data, consider this value for your PC memory
global current_n_atom
global file_point
global current_buffer_size
file_point = 0
current_buffer_size = 1
current_n_atom = 3000


def createSortedNeighbors(contacts, bonds, max_neighbors):
    """
    generate the k nearest neighbors for each atom based on distance.

    Parameters
    ----------
    contacts        : list, [index1, index2, distance, x1, y1, z1, x2, y2, z2]
    bonds           : list
    max_neighbors   : int
        Limit for the maximum neighbors to be set for each atom.
    """
    bond_true = 1  # is chemical bonds
    bond_false = 0  # non-chemical bonds
    neighbor_map = ddict(list)  # type list
    dtype = [('index2', int), ('distance', float), ('x1', float), ('y1', float), ('z1', float), ('bool_bond', int)]
    idx = 0

    for contact in contacts:
        if ([contact[0], contact[1]] or [contact[1], contact[0]]) in bonds:  # have bonds with this neighbor
            # index2, distance, x1, y1, z1, bond_bool
            neighbor_map[contact[0]].append((contact[1], contact[2], contact[3], contact[4], contact[5], bond_true))
            # index1, distance, x2, y2, z2, bond_bool
            neighbor_map[contact[1]].append((contact[0], contact[2], contact[6], contact[7], contact[8], bond_true))
        else:
            neighbor_map[contact[0]].append((contact[1], contact[2], contact[3], contact[4], contact[5], bond_false))
            neighbor_map[contact[1]].append((contact[0], contact[2], contact[6], contact[7], contact[8], bond_false))
        idx += 1

    # normalize length of neighbors
    for k, v in neighbor_map.items():  # 返回可遍历的(键, 值) 元组数组
        if len(v) < max_neighbors:
            true_nbrs = np.sort(np.array(v, dtype=dtype), order='distance', kind='mergesort').tolist()[0:len(v)]
            true_nbrs.extend([(0, 0, 0, 0, 0, 0) for _ in range(max_neighbors - len(v))])
            neighbor_map[k] = true_nbrs
        else:
            neighbor_map[k] = np.sort(np.array(v, dtype=dtype), order='distance', kind='mergesort').tolist()[
                              0:max_neighbors]
    return neighbor_map


# def json_to_h5(json_file, affinity):
#     """
#     Writes and dumps the processed pkl file for each pdb file.
#     Saves the target values into the protein_id_prop_file.
#     """
#     # path = datapath + pdb_file
#     # all_json_files = [file for file in os.listdir(path) if isfile(join(path, file)) and file.endswith('.json')]
#     # json_files_path = datapath + 'json'
#     # all_json_files = [json_file for json_file in jsonpath
#     #                   if isfile(join(jsonpath, json_file)) and json_file.endswith('.json')]
#     # print('Processing ', json_file)
#     # for filename in all_json_files:
#     global file_point
#     global current_buffer_size
#     save_filename = json_file.replace('.json', '').replace('.ent', '')
#     json_filepath = jsonpath + json_file
#     with open(json_filepath, 'r') as file:
#         json_data = json.load(file)
#
#         neighbor_map = createSortedNeighbors(json_data['contacts'], json_data['bonds'], max_neighbors)
#         amino_atom_idx = json_data['res_idx']
#         atom_fea = json_data['atoms']
#         nbr_fea_idx = np.array([list(map(lambda x: x[0], neighbor_map[idx])) for idx in range(len(json_data['atoms']))])
#         nbr_fea = np.array([list(map(lambda x: x[1:], neighbor_map[idx])) for idx in range(len(json_data['atoms']))])
#
#         length = max(amino_atom_idx)
#         if length > MAX_SEQUENCE_LEN:
#             # 跳过长度超过MAX_SEQUENCE_LEN的蛋白质
#             print("Dropping protein as length too long:", length)
#             return None
#
#         # 追加空间分配, current_buffer_size 是样本数量, 并且指向栈顶
#         if file_point >= current_buffer_size:
#             current_buffer_size = current_buffer_size + 1
#             dataset_p.resize((current_buffer_size, MAX_SEQUENCE_LEN))
#             dataset_s.resize((current_buffer_size, MAX_SEQUENCE_LEN, 9))
#             dataset_l.resize((current_buffer_size, 1))
#
#         if self.padding:
#             primary_padded = np.zeros(MAX_SEQUENCE_LEN)
#             tertiary_padded = np.zeros((MAX_SEQUENCE_LEN, 9))
#
#             primary_padded[:length] = primary
#             tertiary_padded[:length, :] = tertiary
#         else:
#             primary_padded = primary
#             tertiary_padded = tertiary
#
#         dataset_p[file_point] = primary_padded
#         dataset_s[file_point] = tertiary_padded
#         dataset_l[file_point] = affinity
#
#         show_process_realtime(file_point, num_files, name='trans h5')
#         file_point += 1


# run interface
def make_h5():
    # Generate pickle files for each pdb file - naming convention is <protein name><pdb name>.pkl
    if not os.path.exists(h5_dir):
        os.makedirs(h5_dir)

    # 如果 .hdf5 文件已经存在，选择强制写入还是跳过
    if os.path.isfile(h5_dir + h5_name + '.hdf5'):
        print("Preprocessed file for [" + h5_name + "] already exists.")
        if override_h5_dataset:
            print("force_pre_processing_overwrite flag set to True, overwriting old file...")
            os.remove(h5_dir + h5_name + '.hdf5')
            construct_h5db_from_list()
        else:
            print("Skipping pre-processing...")
    else:
        construct_h5db_from_list()
    print("Completed pre-processing.")


def construct_h5db_from_list():
    all_json_files = [json_file for json_file in os.listdir(jsonpath)
                      if os.path.isfile(os.path.join(jsonpath, json_file)) and json_file.endswith('.json')]
    num_files = len(all_json_files)
    affinity_dic = get_affinity(file_path='/media/zhangxin/Raid0/dataset/PP' + '/index/INDEX_general_PP.2019')

    print('build h5 dataset from' + str(num_files) + ' json files: ')
    file_point = 0
    current_n_protein = 1
    current_n_atom = 3000

    # create output file
    file = h5py.File(h5_dir + h5_name + '.hdf5')

    ds_atom = file.create_dataset(name='atom_fea', shape=(current_n_protein, current_n_atom, 4), dtype='string')
    ds_edge = file.create_dataset(name='edge', shape=(current_n_protein, current_n_atom*max_neighbors, ))

    ds_tertiary = file.create_dataset(name='tertiary', shape=(current_n_protein, MAX_SEQUENCE_LEN, 9), dtype='float')
    ds_label = file.create_dataset(name='label', shape=(current_n_protein, 1), dtype='float')

    for json_file in tqdm(all_json_files):
        pdb_id = json_file.replace('.json', '').replace('.ent', '')
        affinity = affinity_dic.get(pdb_id)  # get affinity
        # Parallel(n_jobs=parallel_jobs)(delayed(json_to_h5)(json_file, affinity))

        json_filepath = jsonpath + json_file
        with open(json_filepath, 'r') as file:
            json_data = json.load(file)

            neighbor_map = createSortedNeighbors(json_data['contacts'], json_data['bonds'], max_neighbors)
            # [n_atom]
            amino_atom_idx = json_data['res_idx']
            # [n_atom, 1]
            atom_name = json_data['atoms']
            # [n_atom, n_neighbor=50]
            nbr_fea_idx = np.array(
                [list(map(lambda x: x[0], neighbor_map[idx])) for idx in range(len(json_data['atoms']))])
            # [n_atom, n_neighbor=50, edge_fea=5]
            nbr_fea = np.array(
                [list(map(lambda x: x[1:], neighbor_map[idx])) for idx in range(len(json_data['atoms']))])

            length = max(amino_atom_idx)
            if length > MAX_SEQUENCE_LEN:
                # 跳过长度超过MAX_SEQUENCE_LEN的蛋白质
                print("Dropping protein as length too long:", length)
                return None

            # 追加空间分配, current_n_protein 是样本数量, 并且指向栈顶
            if file_point >= current_n_protein:
                current_n_protein = current_n_protein + 1
                dataset_p.resize((current_n_protein, MAX_SEQUENCE_LEN))
                dataset_s.resize((current_n_protein, MAX_SEQUENCE_LEN, 9))
                dataset_l.resize((current_n_protein, 1))

            if self.padding:
                primary_padded = np.zeros(MAX_SEQUENCE_LEN)
                tertiary_padded = np.zeros((MAX_SEQUENCE_LEN, 9))

                primary_padded[:length] = primary
                tertiary_padded[:length, :] = tertiary
            else:
                primary_padded = primary
                tertiary_padded = tertiary

            dataset_p[file_point] = primary_padded
            dataset_s[file_point] = tertiary_padded
            dataset_l[file_point] = affinity

            show_process_realtime(file_point, num_files, name='trans h5')
            file_point += 1
