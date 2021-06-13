# encoding: utf-8
"""
@author: Xin Zhang
@contact: zhangxin@szbl.ac.cn
@file: build_h5.py
@time: 5/25/21 11:34 AM
@desc:
"""
import os
import json
from tqdm import tqdm
from joblib import Parallel, delayed
import threading
import numpy as np
import h5py
from data.affinity_parser import get_affinity
from arguments import build_parser
from data.protein_parser import build_node_edge

parser = build_parser()
args = parser.parse_args()

override_h5_dataset = args.override_h5_dataset
parallel_jobs = args.parallel_jobs
max_neighbors = args.max_neighbors

MAX_N_PROTEIN = 9999
MAX_N_ATOM = 99999  # 氨基酸主链的最大队列长度


# global current_n_atom
# global file_point
# global current_buffer_size
# file_point = 0
# current_buffer_size = 1
# current_n_atom = 1


# run interface
def make_h5_dataset(json_dir, h5_dir, h5_name):
    # Generate pickle files for each pdb file - naming convention is <protein name><pdb name>.pkl
    if not os.path.exists(h5_dir):
        os.makedirs(h5_dir)

    # 如果 .hdf5 文件已经存在，选择强制写入还是跳过
    if os.path.isfile(h5_dir + h5_name + '.hdf5'):
        print("Preprocessed file for [" + h5_name + "] already exists.")
        if override_h5_dataset:
            print("force_pre_processing_overwrite flag set to True, overwriting old file...")
            os.remove(h5_dir + h5_name + '.hdf5')
            construct_h5db_from_list(json_dir, h5_dir, h5_name)
        else:
            print("Skipping pre-processing...")
    else:
        construct_h5db_from_list(json_dir, h5_dir, h5_name)
    print("Completed pre-processing.")


def thread_read(json_file):
    json_filepath = args.json_dir + json_file
    with open(json_filepath, 'r') as file:
        json_data = json.load(file)

        atoms = json_data['atoms']  # [n_atom, 1]
        res_idx = json_data['res_idx']  # [n_atom]
        bonds = json_data['bonds']
        contacts = json_data['contacts']
        # [n_atom, 5], [n_atom, 25, 3], [n_atom, 3]
        atom_fea, neighbor_map, atom_3d = build_node_edge(atoms, bonds, contacts)
        n_atom = len(atoms)

    return n_atom, atom_fea, neighbor_map, atom_3d, res_idx


def construct_h5db_from_list(json_dir, h5_dir, h5_name):
    all_json_files = [json_file for json_file in os.listdir(json_dir)
                      if os.path.isfile(os.path.join(json_dir, json_file)) and json_file.endswith('.json')]
    num_files = len(all_json_files)
    affinity_dic = get_affinity(file_path='/media/zhangxin/Raid0/dataset/PP' + '/index/INDEX_general_PP.2019')

    print('build h5 dataset from ' + str(num_files) + ' json files: ')
    file_point = 0
    current_n_protein = 1
    current_n_atom = 1

    # create output file
    file = h5py.File(h5_dir + h5_name + '.hdf5', 'w')
    dataset_atom = file.create_dataset(name='atom_fea', shape=(current_n_protein, current_n_atom, 5),
                                       maxshape=(MAX_N_PROTEIN, MAX_N_ATOM, 5), dtype='float', chunks=True)
    dataset_3d = file.create_dataset(name='atom_3d', shape=(current_n_protein, current_n_atom, 3),
                                     maxshape=(MAX_N_PROTEIN, MAX_N_ATOM, 3), dtype='float', chunks=True)
    dataset_edge = file.create_dataset(name='edge', shape=(current_n_protein, current_n_atom, max_neighbors, 3),
                                       maxshape=(MAX_N_PROTEIN, MAX_N_ATOM, max_neighbors, 3), dtype='float',
                                       chunks=True)
    dataset_res_idx = file.create_dataset(name='res_idx', shape=(current_n_protein, current_n_atom),
                                          maxshape=(MAX_N_PROTEIN, MAX_N_ATOM), dtype='float', chunks=True)
    dataset_affinity = file.create_dataset(name='affinity', shape=(current_n_protein, 1),
                                           maxshape=(MAX_N_PROTEIN, 1), dtype='float', chunks=True)

    print('use ' + str(parallel_jobs) + ' jobs to read json and analysis proteins.')
    for json_file in tqdm(all_json_files):
        pdb_id = json_file.replace('.json', '').replace('.ent', '')
        affinity = affinity_dic.get(pdb_id)  # get affinity

        n_atom, atom_fea, neighbor_map, atom_3d, res_idx = Parallel(n_jobs=parallel_jobs)(delayed(thread_read)(json_file))

        if n_atom > MAX_N_ATOM:
            # 跳过长度超过MAX_SEQUENCE_LEN的蛋白质
            print("Dropping protein as length too long:", n_atom)
            return None
        # 追加空间分配
        if n_atom > current_n_atom:
            current_n_atom = n_atom
            dataset_atom.resize(size=(current_n_protein, current_n_atom, 5))
            dataset_3d.resize(size=(current_n_protein, current_n_atom, 3))
            dataset_edge.resize(size=(current_n_protein, current_n_atom, max_neighbors, 3))
            dataset_res_idx.resize(size=(current_n_protein, current_n_atom))

        # 追加空间分配, file_point start with 0, 并且指向栈顶
        if file_point >= current_n_protein:
            current_n_protein += 1
            dataset_atom.resize(size=(current_n_protein, current_n_atom, 5))
            dataset_3d.resize(size=(current_n_protein, current_n_atom, 3))
            dataset_edge.resize(size=(current_n_protein, current_n_atom, max_neighbors, 3))
            dataset_res_idx.resize(size=(current_n_protein, current_n_atom))
            dataset_affinity.resize(size=(current_n_protein, 1))

        # if args.inputs_padding:
        #     primary_padded = np.zeros(MAX_N_ATOM)
        #     tertiary_padded = np.zeros((MAX_N_ATOM, 9))
        #     atom_fea_padded = np.zeros()
        #
        #     primary_padded[:length] = primary
        #     tertiary_padded[:length, :] = tertiary
        # else:
        #     primary_padded = primary
        #     tertiary_padded = tertiary

        dataset_atom[file_point] = atom_fea
        dataset_3d[file_point] = atom_3d
        dataset_edge[file_point] = neighbor_map
        dataset_res_idx[file_point] = res_idx
        dataset_affinity[file_point] = affinity

        file_point += 1


if __name__ == "__main__":
    make_h5_dataset(json_dir=args.json_dir, h5_dir=args.h5_dir, h5_name=args.h5_name)