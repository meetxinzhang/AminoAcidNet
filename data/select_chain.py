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


def remove_duplication(dic):
    interest_chain = set()
    select_chain = []
    for k, v in dic.items():
        if v not in interest_chain:
            interest_chain.add(v)
            select_chain.append(k)
        else:


    print(select_chain)
    print(interest_chain)
    return select_chain


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

    seq1 = get_sequence(structure)
    seq2 = lines_reader(file_path)

    print('1, structure: ')
    for k, v in seq1.items():
        print(k, v)
    print('\n2, header: ')
    for k, v in seq2.items():
        print(k, v)

    print('\n after deduplication: ')
    chain1 = remove_duplication(seq1)
    chain2 = remove_duplication(seq2)


samplify_pdb(file_path='/media/zhangxin/Raid0/dataset/PP/6mi2.ent.pdb')
