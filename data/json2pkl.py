#
import csv, pickle, json, os
import argparse
import numpy as np
from os.path import isfile, join
from subprocess import call
from tqdm import tqdm
from joblib import Parallel, delayed
from collections import defaultdict as ddict

from arguments import buildParser
parser = buildParser()
args = parser.parse_args()

datapath = args.datapath
jsonpath = args.jsonpath
savepath = args.savepath
cpp_executable = args.cpp_executable
groups20_filepath = args.groups20_filepath
parallel_jobs = args.parallel_jobs
get_json_files = args.get_json_files
get_pkl_flies = args.get_pkl_flies
protein_id_prop_file = savepath + 'protein_id_prop.csv'
protein_atom_init_file = savepath + 'protein_atom_init.json'

protein_dirs = os.listdir(datapath)
all_proteins = []
for pdb_file in protein_dirs:
    if pdb_file.endswith('.pdb'):
        all_proteins.append(pdb_file)


def json_cpp_commands(pdb_file):
    """
    Convert a single .pdb file to .json format

    Parameters
    ----------
    pdb_file: The folder containing pdb files for a protein.

    """
    path = datapath + pdb_file
    json_filepath = jsonpath + pdb_file.strip('.pdb') + '.json'
    command = cpp_executable + ' -i ' + path + ' -j ' + json_filepath + ' -d 8.0'
    call(command, shell=True)


if get_json_files:
    print('read pdb files to json files:')
    Parallel(n_jobs=parallel_jobs)(delayed(json_cpp_commands)(pdb_file) for pdb_file in tqdm(all_proteins))

# Create a one-hot encoded feature map for each protein atom
feature_map = {}  # dic
with open(groups20_filepath, 'r') as f:
    data = f.readlines()
    len(data)
    len_amino = sum(1 for row in data)
    for idx, line in enumerate(data):
        a = [0] * len_amino
        a[idx] = 1
        name, _ = line.split(" ")
        feature_map[name] = a

print('--------feature_map------------------------------')
print(feature_map)

with open(protein_atom_init_file, 'w') as f:
    json.dump(feature_map, f)  # convert python obj to json


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


def processDirectory(json_file, max_neighbors, savepath, protein_id_prop_file):
    """
    Writes and dumps the processed pkl file for each pdb file.
    Saves the target values into the protein_id_prop_file.
    """
    # path = datapath + pdb_file
    # all_json_files = [file for file in os.listdir(path) if isfile(join(path, file)) and file.endswith('.json')]
    # json_files_path = datapath + 'json'
    # all_json_files = [json_file for json_file in jsonpath
    #                   if isfile(join(jsonpath, json_file)) and json_file.endswith('.json')]
    # print('Processing ', json_file)
    # for filename in all_json_files:
    save_filename = json_file.replace('.json', '').replace('.ent', '')
    json_filepath = jsonpath + json_file
    with open(json_filepath, 'r') as file:
        json_data = json.load(file)

        neighbor_map = createSortedNeighbors(json_data['contacts'], json_data['bonds'], max_neighbors)
        # [n_atom]
        amino_atom_idx = json_data['res_idx']
        # [n_atom, 1]
        atom_fea = json_data['atoms']
        # [n_atom, n_neighbor=50]
        nbr_fea_idx = np.array([list(map(lambda x: x[0], neighbor_map[idx])) for idx in range(len(json_data['atoms']))])
        # [n_atom, n_neighbor=50, edge_fea=5]
        nbr_fea = np.array([list(map(lambda x: x[1:], neighbor_map[idx])) for idx in range(len(json_data['atoms']))])

    with open(savepath + save_filename + '.pkl', 'wb') as file:
        pickle.dump(atom_fea, file)
        pickle.dump(nbr_fea, file)
        pickle.dump(nbr_fea_idx, file)
        pickle.dump(amino_atom_idx, file)
        pickle.dump(save_filename, file)

    # labels
    # with open(protein_id_prop_file, 'a') as file:
    #     writer = csv.writer(file)
    #     global_target, local_targets = get_targets(directory, filename)
    #     writer.writerow((save_filename, global_target, local_targets))


if get_pkl_flies:
    # Generate pickle files for each pdb file - naming convention is <protein name><pdb name>.pkl
    max_neighbors = 50
    if not os.path.exists(savepath):
        os.makedirs(savepath)

    # create a new file
    with open(protein_id_prop_file, 'w') as f:
        pass

    all_json_files = [json_file for json_file in os.listdir(jsonpath)
                      if isfile(join(jsonpath, json_file)) and json_file.endswith('.json')]

    print('convert pdbs to pkl, generate neighbors :')
    # print(len(all_json_files))
    Parallel(n_jobs=parallel_jobs)(
        delayed(processDirectory)(json_file, max_neighbors, savepath, protein_id_prop_file) for json_file in
        tqdm(all_json_files))
