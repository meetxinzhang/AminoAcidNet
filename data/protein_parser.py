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
from data.elem_periodic_map import atoms_periodic_dic, heavy_atom_idx_dic
from arguments import build_parser
parser = build_parser()
args = parser.parse_args()
max_neighbors = args.max_neighbors


def create_sorted_graph(contacts, bonds, max_neighbors):
    """
    generate the k nearest neighbors for each atom based on distance.

    :param contacts        : list, [[index1, index2, distance, x1, y1, z1, x2, y2, z2], ...]
    :param bonds           : list
    :param max_neighbors   : int Limit for the maximum neighbors to be set for each atom.
    :param atom_fea:
    """
    bond_true = 1  # is chemical bonds
    bond_false = 0  # non-chemical bonds
    neighbor_map = ddict(list)  # type list
    pos_map = {}
    # type = [('index2', int), ('distance', float), ('bool_bond', int)]

    for contact in contacts:
        # atom 3D: x y z
        pos_map[contact[0]] = [contact[3], contact[4], contact[5]]
        pos_map[contact[1]] = [contact[6], contact[7], contact[8]]

        if ([contact[0], contact[1]] or [contact[1], contact[0]]) in bonds:  # have bonds with this neighbor
            # index2, distance, bond_bool
            neighbor_map[contact[0]].append([contact[1], contact[2], bond_true])
            # index1, distance, bond_bool
            neighbor_map[contact[1]].append([contact[0], contact[2], bond_true])
        else:
            neighbor_map[contact[0]].append([contact[1], contact[2], bond_false])
            neighbor_map[contact[1]].append([contact[0], contact[2], bond_false])

    edge_idx = []
    edge_attr = []
    pos = []

    # normalize length of neighbors and align with atom
    for i in range(len(pos_map)):
        position = pos_map.get(i)
        neighbors = neighbor_map.get(i)

        if len(neighbors) < max_neighbors:
            # true_nbrs = np.sort(np.array(v, dtype=type), order='distance', kind='mergesort').tolist()[0:len(v)]
            neighbors.sort(key=lambda e: e[1])
            neighbors.extend([[0, 0, 0] for _ in range(max_neighbors - len(neighbors))])
        else:
            # neighbor_map[k] = np.sort(np.array(v, dtype=type), order='distance', kind='mergesort').tolist()[
            #                   0:max_neighbors]
            neighbors.sort(key=lambda e: e[1])
            neighbors = neighbors[0:max_neighbors]

        pos.append(position)
        edge_idx.append(np.array(neighbors)[:, 0].tolist())
        edge_attr.append(np.array(neighbors)[:, 1:].tolist())

    return pos, edge_idx, edge_attr


def create_graph(contacts, bonds):
    bond_true = 1  # is chemical bonds
    bond_false = 0  # non-chemical bonds
    edge_idx = []
    edge_attr = []
    atom_3d = {}

    for contact in contacts:
        atom_3d[contact[0]] = [contact[3], contact[4], contact[5]]
        atom_3d[contact[1]] = [contact[6], contact[7], contact[8]]

        edge_idx.append([contact[0], contact[1]])
        if ([contact[0], contact[1]] or [contact[1], contact[0]]) in bonds:
            edge_attr.append([contact[2], bond_true])
        else:
            edge_attr.append([contact[2], bond_false])
    return edge_idx, edge_attr, list(atom_3d)


def build_node_edge(atoms, bonds, contacts, PyG_format):

    atom_fea = []
    for atom in atoms:
        type_ = atom.split('_')[1][0]
        periodic = atoms_periodic_dic[type_]
        idx = heavy_atom_idx_dic[atom]
        atom_fea.append(np.concatenate([[idx], periodic], axis=0))

    if PyG_format:
        edge_idx, edge_attr, pos = create_graph(contacts, bonds)
    else:
        pos, edge_idx, edge_attr = create_sorted_graph(contacts, bonds, max_neighbors)

    # [a_n, 3], [a_n, 5], [a_n, nei_n], [a_n, nei_n, 2]
    return pos, atom_fea, edge_idx, edge_attr

