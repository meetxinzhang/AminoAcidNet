# encoding: utf-8
"""
@author: Xin Zhang
@contact: zhangxin@szbl.ac.cn
@file: pdb_parser.py
@time: 12/8/20 11:36 AM
@desc: read protein from .pdb
Function parser_reader() return a structure object which defined in BioPython
You can traverse the structure object to obtain all molecular, chains, residues and atoms
Use residue.is_aa(residue) to check whether a residue object which you get is a amino acid
Use the following functions to obtain the corresponding values
a.get_name()       # atom name (spaces stripped, e.g. "CA")
a.get_id()         # id (equals atom name)
a.get_coord()      # atomic coordinates
a.get_vector()     # atomic coordinates as Vector object
a.get_bfactor()    # isotropic B factor
a.get_occupancy()  # occupancy
a.get_altloc()     # alternative location specifier
a.get_sigatm()     # standard deviation of atomic parameters
a.get_siguij()     # standard deviation of anisotropic B factor
a.get_anisou()     # anisotropic B factor
a.get_fullname()   # atom name (with spaces, e.g. ".CA.")
"""

from Bio.PDB import is_aa
import numpy as np
from Bio.PDB.PDBParser import PDBParser
from collections import defaultdict as ddict
import platform

local_add = '/home/zhangxin/Downloads/dataset/proxy/all_structures/raw/'

aa_codes = {
    'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E',  # Amino acid
    'PHE': 'F', 'GLY': 'G', 'HIS': 'H', 'LYS': 'K',
    'ILE': 'I', 'LEU': 'L', 'MET': 'M', 'ASN': 'N',
    'PRO': 'P', 'GLN': 'Q', 'ARG': 'R', 'SER': 'S',
    'THR': 'T', 'VAL': 'V', 'TYR': 'Y', 'TRP': 'W',
    'HOH': 'water'}

# aa_codes = {
#     'ALA': 1, 'CYS': 2, 'ASP': 3, 'GLU': 4,  # Amino acid
#     'PHE': 5, 'GLY': 6, 'HIS': 7, 'LYS': 8,
#     'ILE': 9, 'LEU': 10, 'MET': 11, 'ASN': 12,
#     'PRO': 13, 'GLN': 14, 'ARG': 15, 'SER': 16,
#     'THR': 17, 'VAL': 18, 'TYR': 19, 'TRP': 20}


def lines_reader(file_path):
    """
    :param file_path:
    :return: seq
        attention! this seq includes missing residues, though they aren't in pdb.
    """
    seq = ddict(str)
    for line in open(file_path):
        if line[0:6] == "SEQRES":
            elements = line.split()
            chain_id = elements[2]
            for residue_name in elements[4:]:
                # seq[chain_id].append(aa_codes[residue_name])
                seq[chain_id] += aa_codes[residue_name]
    return seq


def get_sequence(structure):
    seq = ddict(str)
    for model in structure:
        for chain in model:
            chain_id = chain.get_id()
            for residue in chain:
                if is_aa(residue) and residue.resname in aa_codes.keys():
                    # seq[chain_id].append(aa_codes[residue.resname])
                    seq[chain_id] += aa_codes[residue.resname]
    return seq


def selection_GTR(structure, header):
    structs_dic = {}
    for mdl in structure:
        for chain in mdl:
            structs_dic[chain.get_id()] = chain

    rec_id = []
    lig_id = []
    n_mdl = len(header['compound'])
    compound = header['compound']

    if n_mdl <= 1:  # that's impossible in antibody task
        pass
    if n_mdl == 2:
        c_l_r = compound['1']['chain']
        t_c = 'z'
        for e in c_l_r:
            if e.isalpha() and e < t_c:
                t_c = e
        chain1 = structs_dic.get(t_c.upper())
        rec_id.append(t_c.upper())

        c_l_l = compound['2']['chain']
        t_l = 'z'
        last_d = 99
        for c in c_l_l:
            chain2 = structs_dic.get(c.upper())
            length = min(len(chain2), len(chain1))

            distance = calculate_average_distance(chain2, chain1, length)
            if distance < last_d:
                last_d = distance
                t_l = c.upper()
        lig_id.append(t_l)

        for mdl_id, mdl in compound.items():
            c_l_r = mdl['1']['chain']
            t_c = 'z'
            for e in c_l_r:
                if e.isalpha() and e < t_c:
                    t_c = e
            chain1 = structs_dic.get(t_c.upper())
            rec_id.append(t_c.upper())

            c_l_l = mdl['2']['chain']
            t_l = 'z'
            last_d = 99
            for c in c_l_l:
                chain2 = structs_dic.get(c.upper())
                length = min(len(chain2), len(chain1))

                distance = calculate_average_distance(chain2, chain1, length)
                if distance < last_d:
                    last_d = distance
                    t_l = c.upper()
            lig_id.append(t_l)

        return rec_id, lig_id


def calculate_average_distance(chain1, chain2, length):
    distance = None
    for residue1, residue2, idx in zip(chain1, chain2, range(1, length)):
        if idx % 10 == 0:
            try:
                atom1 = residue1['CA']
                atom2 = residue2['CA']
                distance = np.linalg.norm(atom1 - atom2)
            except KeyError:
                continue
    return distance


def select_by_ascii(header):
    """
    To remove duplicate chains in pdb
    :param dic:
    :return:
    """
    select_id = []
    interest_dic = ddict(str)
    for k, v in header.items():
        if v not in interest_dic.values():
            interest_dic[k] = v
            select_id.append(k)
        else:
            for old_k, old_v in list(interest_dic.items()):
                if v == old_v:
                    if k < old_k:
                        interest_dic.pop(old_k)
                        interest_dic[k] = v
                        select_id.remove(old_k)
                        select_id.append(k)
    print('\n after select_by_ascii: ')
    for k, v in interest_dic.items():
        print(k, v)
    return select_id


def samplify_pdb(file_path):
    # https://bioinformatics.stackexchange.com/questions/14101/extract-residue-sequence-from-pdb-file-in-biopython-but-open-to-recommendation
    p = PDBParser(QUIET=True, get_header=True)

    if platform.system() == 'Windows':
        pdb_id = file_path.split('\\')[-1].replace('.ent.pdb', '')
    else:
        pdb_id = file_path.split('/')[-1].replace('.ent.pdb', '')

    try:
        structure = p.get_structure(file=file_path, id=pdb_id)
        header = p.get_header()
        trailer = p.get_trailer()
    except ValueError as ve:
        print(ve, file_path)
        raise ValueError

    print('header: ')
    for k, v in header.items():
        print(k, ' : ', v)
    print('\nstructure: ')
    for mdl in structure:
        for chain in mdl:
            print(mdl.get_id(), chain.get_id())

    seq_header = lines_reader(file_path)

    print('\nheader: ')
    for k, v in seq_header.items():
        print(k, v)

    select_by_ascii(seq_header)
    selection_GTR(structure, header)


samplify_pdb(file_path='/media/zhangxin/Raid0/dataset/PP/6mkz.ent.pdb')
