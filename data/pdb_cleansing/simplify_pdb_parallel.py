# encoding: utf-8
"""
@author: Xin Zhang
@contact: zhangxin@szbl.ac.cn
@file: simplify_pdb_parallel.py
@time: 6/2/21 4:11 PM
@desc:
"""
import glob
import os

from data.pdb_cleansing.select_chain import save_single_complex_chains
from joblib import Parallel, delayed
from tqdm import tqdm
from Bio.PDB import PDBParser, PDBIO, Select
from arguments import build_parser
parser = build_parser()
args = parser.parse_args()
parallel_jobs = args.parallel_jobs


def process_all_files(dir):
    filepath_list = glob.glob(dir + '/*')
    filepath_list = file_filter(filepath_list)  # list['filepath', ...]


def process_chains(filepath, out_dir):
    rec_id, lig_id, n_mdl = save_single_complex_chains(filepath)

    Parallel(n_jobs=parallel_jobs)(delayed(save_single_complex_chains)(pdb_file, out_dir)
                                   for pdb_file in tqdm(filepath_list))


def process_residues():

    pass


def file_filter(input_files):
    disallowed_file_endings = (".gitignore", ".DS_Store")
    allowed_file_endings = ".pdb"
    rate = 1
    _input_files = input_files[:int(len(input_files) * rate)]
    return list(filter(lambda x: not x.endswith(disallowed_file_endings) and x.endswith(allowed_file_endings),
                       _input_files))


if __name__ == "__main__":
    process_chains(dir='/media/zhangxin/Raid0/dataset/PP/', out_dir='/media/zhangxin/Raid0/dataset/PP/single_complex/')
