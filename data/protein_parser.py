# encoding: utf-8
"""
@author: Xin Zhang
@contact: zhangxin@szbl.ac.cn
@file: protein_parser.py
@time: 5/13/21 5:15 PM
@desc:
"""
import numpy as np
from collections import defaultdict as ddict
from data.periodic_atom_map import atoms_periodic_dic, heavy_atom_idx_dic
from arguments import build_parser
parser = build_parser()
args = parser.parse_args()

max_neighbors = args.max_neighbors
groups20_filepath = args.groups20_filepath


def create_sorted_neighbors(contacts, bonds, max_neighbors):
    """
    generate the k nearest neighbors for each atom based on distance.

    :param contacts        : list, [index1, index2, distance, x1, y1, z1, x2, y2, z2]
    :param bonds           : list
    :param max_neighbors   : int Limit for the maximum neighbors to be set for each atom.
    :param atom_fea:
    """
    bond_true = 1  # is chemical bonds
    bond_false = 0  # non-chemical bonds
    neighbor_map = ddict(list)  # type list
    atom_3d = {}
    dtype = [('index2', int), ('distance', float), ('bool_bond', int)]

    for contact in contacts:
        # atom 3D: x y z
        atom_3d[contact[0]] = [contact[3], contact[4], contact[5]]
        atom_3d[contact[1]] = [contact[6], contact[7], contact[8]]

        if ([contact[0], contact[1]] or [contact[1], contact[0]]) in bonds:  # have bonds with this neighbor
            # index2, distance, bond_bool
            neighbor_map[contact[0]].append((contact[1], contact[2], bond_true))
            # index1, distance, bond_bool
            neighbor_map[contact[1]].append((contact[0], contact[2], bond_true))
        else:
            neighbor_map[contact[0]].append((contact[1], contact[2], bond_false))
            neighbor_map[contact[1]].append((contact[0], contact[2], bond_false))

    # normalize length of neighbors
    for k, v in neighbor_map.items():  # 返回可遍历的(键, 值) 元组数组
        if len(v) < max_neighbors:
            true_nbrs = np.sort(np.array(v, dtype=dtype), order='distance', kind='mergesort').tolist()[0:len(v)]
            true_nbrs.extend([(0, 0, 0) for _ in range(max_neighbors - len(v))])
            neighbor_map[k] = true_nbrs
        else:
            neighbor_map[k] = np.sort(np.array(v, dtype=dtype), order='distance', kind='mergesort').tolist()[
                              0:max_neighbors]
    return list(neighbor_map.values()), list(atom_3d.values())


def build_node_edge(atoms, bonds, contacts):

    atom_fea = []
    for atom in atoms:
        type_ = atom.split('_')[1][0]
        periodic = atoms_periodic_dic[type_]
        idx = heavy_atom_idx_dic[atom]
        atom_fea.append(np.concatenate([[idx], periodic], axis=0))

    neighbor_map, atom_3d = create_sorted_neighbors(contacts, bonds, max_neighbors)
    # [6776, 5], [6776, 25, 3], [6776, 3], []
    return atom_fea, neighbor_map, atom_3d

