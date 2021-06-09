# encoding: utf-8
"""
@author: Xin Zhang
@contact: zhangxin@szbl.ac.cn
@file: pdb_parser.py
@time: 12/8/20 11:36 AM
@desc:

"""

from Bio.PDB.Polypeptide import is_aa
import numpy as np
import os
from Bio.PDB import PDBParser, PDBIO, Select
import platform
from utils.log_output import Logger

logger = Logger(log_path='//output/logs/select_chain.log', is_print=False)


def get_single_complex_type_R(structure, header):
    structs_dic = {}
    for mdl in structure:
        for chain in mdl:
            structs_dic[chain.get_id()] = chain

    rec_id = []
    lig_id = []

    n_mdl = len(header['compound'])
    compound = header['compound']

    last_struct = None
    describ_list = []
    target_chain = []

    # easy method: assume H/L are Heavy/Light chains ---------------------
    if 'H' in structs_dic.keys() and 'L' in structs_dic.keys():
        rec_id.append('H')
        rec_id.append('L')
        lig = 'z'
        for mdl_id, mdl in compound.items():
            chain_list = mdl['chain']
            if 'h' in chain_list or 'l' in chain_list:
                continue
            last_d = 999
            for c in chain_list:
                if c.isalpha():
                    chain = structs_dic.get(c.upper())
                    d1 = calculate_distance(structs_dic['H'], chain)
                    d2 = calculate_distance(structs_dic['L'], chain)
                    d = min(d1, d2)
                    if d < last_d and d != 0:
                        last_d = d
                        lig = c.upper()
            lig_id.append(lig)
        return rec_id, lig_id, n_mdl

    # normal method -------------------------------------------------------
    for mdl_id, mdl in compound.items():
        chain_list = mdl['chain']

        # which chain to select in each model --------
        c_id = 'z'
        if len(target_chain) == 0:
            for c in chain_list:
                if c.isalpha() and c < c_id:
                    c_id = c.upper()
                    last_struct = structs_dic.get(c_id.upper())
        else:
            last_d = 999
            for c in chain_list:
                if c.isalpha():
                    chain = structs_dic.get(c.upper())
                    distance = calculate_distance(last_struct, chain)
                    if distance < last_d and distance != 0:
                        last_d = distance
                        last_struct = chain
                        c_id = c.upper()

        target_chain.append(c_id)
        describ_list.append(mdl['molecule'])

    # print('\n after select by distance:')
    # print('target_chain: ', target_chain)
    # print('describ_list: ', describ_list)
    # whether receptor or ligand ------------
    if n_mdl == 1:
        pass
    if n_mdl == 2:
        rec_id.append(target_chain[0])
        lig_id.append(target_chain[1])
    if n_mdl >= 3:
        for des, c in zip(describ_list, target_chain):
            keywords = 'fab light heavy receptor'
            if any(w in des and w for w in keywords.split()):
                rec_id.append(c)
            else:
                lig_id.append(c)

    return rec_id, lig_id, n_mdl


def calculate_distance(chain1, chain2):
    min_distance = 99
    for residue1 in chain1:
        if is_aa(residue1, standard=True):
            for residue2 in chain2:
                if is_aa(residue2, standard=True):
                    try:
                        atom1 = residue1['CA']
                        atom2 = residue2['CA']
                        distance = np.linalg.norm(atom1.get_coord() - atom2.get_coord())
                        if distance < min_distance:
                            min_distance = distance
                    except KeyError:
                        continue
                else:
                    # print(residue1, residue2, 'non - amino acid !!!!!!!!!!!!!!')
                    continue
        else:
            continue
    # print('average_distance: ', distance/n, chain1, chain2)
    return min_distance


# def get_single_complex_by_ascii(header):
#     """
#     To remove duplicate chains in pdb
#     :param dic:
#     :return:
#     """
#     select_id = []
#     interest_dic = ddict(str)
#     for k, v in header.items():
#         if v not in interest_dic.values():
#             interest_dic[k] = v
#             select_id.append(k)
#         else:
#             for old_k, old_v in list(interest_dic.items()):
#                 if v == old_v:
#                     if k < old_k:
#                         interest_dic.pop(old_k)
#                         interest_dic[k] = v
#                         select_id.remove(old_k)
#                         select_id.append(k)
#     print('\n after select_by_ascii: ')
#     for k, v in interest_dic.items():
#         print(k, v)
#     return select_id

class ChainSelect(Select):
    def __init__(self, target_chain_ids):
        self.target_chain_ids = target_chain_ids

    def accept_chain(self, chain):
        return chain.id in self.target_chain_ids


def save_single_complex_chains(file_path, out_dir):
    p = PDBParser(QUIET=True, get_header=True)

    if platform.system() == 'Windows':
        pdb_id = file_path.split('\\')[-1].replace('.ent.pdb', '')
    else:
        pdb_id = file_path.split('/')[-1].replace('.ent.pdb', '')

    try:
        structure = p.get_structure(file=file_path, id=pdb_id)
        header = p.get_header()
        # trailer = p.get_trailer()
    except ValueError as ve:
        print(ve, file_path)
        raise ValueError

    # print('header: ')
    # for k, v in header.items():
    #     print(k, ' : ', v)
    # print('\nstructure: ')
    # for mdl in structure:
    #     for chain in mdl:
    #         print(mdl.get_id(), chain.get_id())

    # seq_header = lines_reader(file_path)
    # print('\nheader: ')
    # for k, v in seq_header.items():
    #     print(k, v)

    # select_by_ascii(seq_header)
    rec_id, lig_id, n_mdl = get_single_complex_type_R(structure, header)

    # save pdb
    if len(rec_id) != 0 and len(lig_id) != 0:
        io = PDBIO()
        io.set_structure(structure)
        save_dir = out_dir + str(n_mdl) + '/'
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        sava_name = pdb_id + '_'.join(e for e in rec_id) + '_'.join(e for e in lig_id) + '.pdb'
        io.save(file=save_dir + sava_name,
                select=ChainSelect(np.concatenate([rec_id, lig_id], axis=0)))
    logger.write('grep ', rec_id, lig_id, pdb_id, '>', pdb_id + '.pdb', join_time=True)
    logger.flush()


if __name__ == "__main__":
    save_single_complex_chains(file_path='/media/zhangxin/Raid0/dataset/PP/4f0z.ent.pdb', out_dir='')
