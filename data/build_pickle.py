#
import json
import os
import pickle
from os.path import isfile, join
from joblib import Parallel, delayed
from tqdm import tqdm
from arguments import build_parser
from data.protein_parser import build_protein_graph

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

        # [n_atom, 3], [n_atom, 5], [n_atom, n_nei], [n_atom, n_nei, 2]
        pos, atom_fea, edge_idx, edge_attr = build_protein_graph(atoms, bonds, contacts, PyG_format=False)
        # print(np.shape(pos), np.shape(edge_idx), np.shape(edge_attr), affinity)

    with open(pkl_filepath, 'wb') as file:
        pickle.dump(pos, file)
        pickle.dump(atom_fea, file)
        pickle.dump(edge_idx, file)
        pickle.dump(edge_attr, file)
        pickle.dump(res_idx, file)


def make_pickle(json_dir, pkl_dir):
    # Generate pickle files for each pdb file - naming convention is <protein name><pdb name>.pkl
    if not os.path.exists(pkl_dir):
        os.makedirs(pkl_dir)

    # python no parallel for calculation but for i/o
    read_write_path = []
    for json_file in os.listdir(json_dir):
        json_filepath = join(json_dir, json_file)
        if isfile(json_filepath) and json_file.endswith('.json'):
            pdb_id = str(json_file)[0:4]
            pkl_filepath = pkl_dir + pdb_id + '.pkl'
            read_write_path.append((json_filepath, pkl_filepath))

    print('analysis protein and do i/o from json to pkl use parallel:')
    Parallel(n_jobs=parallel_jobs)(
        delayed(thread_read_write)(json_filepath, pkl_filepath) for (json_filepath, pkl_filepath) in
        tqdm(read_write_path))


if __name__ == "__main__":
    make_pickle(json_dir='/media/zhangxin/Raid0/dataset/PP/single_complex/bind_sites/json_dir/2/',
                pkl_dir='/media/zhangxin/Raid0/dataset/PP/single_complex/bind_sites/pkl/2/')
