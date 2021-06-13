#
import json
import os
import pickle
from os.path import isfile, join
from joblib import Parallel, delayed
from tqdm import tqdm
from arguments import build_parser
from data.affinity_parser import get_affinity
from data.protein_parser import build_node_edge

parser = build_parser()
args = parser.parse_args()
parallel_jobs = args.parallel_jobs


def thread_read_write(json_filepath, pkl_filepath, affinity):
    """
    Writes and dumps the processed pkl file for each json file.
    Process json files by data.protein_parser import build_node_edge
    """
    with open(json_filepath, 'r') as file:
        json_data = json.load(file)

        atoms = json_data['atoms']  # [n_atom, 1]
        res_idx = json_data['res_idx']  # [n_atom]
        bonds = json_data['bonds']
        contacts = json_data['contacts']

        # # [n_atom, 5], [n_atom, 25, 3], [n_atom, 3]
        # atom_fea, neighbor_map, pos = build_node_edge(atoms, bonds, contacts, PyG_format=False)

        # [n_atom, 5], [n_atom, 2], [n_atom, 2], [n_atom, 3]
        atom_fea, edge_idx, edge_attr, pos = build_node_edge(atoms, bonds, contacts, PyG_format=True)

    with open(pkl_filepath, 'wb') as file:
        pickle.dump(atom_fea, file)
        # ------- if use neighbor_map format
        # pickle.dump(pos, file)
        # pickle.dump(neighbor_map, file)
        # ------- if use pytorch geometric format
        pickle.dump(pos, file)
        pickle.dump(edge_idx, file)
        pickle.dump(edge_attr, file)

        pickle.dump(res_idx, file)
        pickle.dump(affinity, file)


def make_pickle(json_dir, pkl_dir):
    # Generate pickle files for each pdb file - naming convention is <protein name><pdb name>.pkl
    if not os.path.exists(pkl_dir):
        os.makedirs(pkl_dir)

    affinity_dic = get_affinity(file_path='/media/zhangxin/Raid0/dataset/PP/' + 'index/INDEX_general_PP.2019')

    # python no parallel for calculation but for i/o
    all_protein = []
    for json_file in os.listdir(json_dir):
        json_filepath = join(json_dir, json_file)
        if isfile(json_filepath) and json_file.endswith('.json'):
            pdb_id = str(json_file)[0:4]
            affinity = affinity_dic.get(pdb_id)  # get affinity
            pkl_filepath = pkl_dir + pdb_id + '.pkl'
            all_protein.append((json_filepath, pkl_filepath, affinity))

    print('analysis protein and do i/o from json to pkl use parallel:')
    Parallel(n_jobs=parallel_jobs)(
        delayed(thread_read_write)(json_filepath, pkl_filepath, affinity) for (json_filepath, pkl_filepath, affinity) in
        tqdm(all_protein))


if __name__ == "__main__":
    make_pickle(json_dir='/media/zhangxin/Raid0/dataset/PP/single_complex/bind_sites/json_dir/2/',
                pkl_dir='/media/zhangxin/Raid0/dataset/PP/single_complex/bind_sites/pkl_PyG/2/')
