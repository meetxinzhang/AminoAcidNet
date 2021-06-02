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

from data.pdb_cleansing.select_chain import simplify_pdb
from joblib import Parallel, delayed
from tqdm import tqdm
from arguments import build_parser
parser = build_parser()
args = parser.parse_args()
parallel_jobs = args.parallel_jobs


def process_all_pdb(dir, out_dir):
    filepath_list = glob.glob(dir + '/*')
    filepath_list = file_filter(filepath_list)  # list['filename', ...]
    Parallel(n_jobs=parallel_jobs)(delayed(simplify_pdb)(pdb_file, out_dir) for pdb_file in tqdm(filepath_list))


def file_filter(input_files):
    disallowed_file_endings = (".gitignore", ".DS_Store")
    allowed_file_endings = ".pdb"
    rate = 0.3
    _input_files = input_files[:int(len(input_files) * rate)]
    return list(filter(lambda x: not x.endswith(disallowed_file_endings) and x.endswith(allowed_file_endings),
                       _input_files))


if __name__ == "__main__":
    process_all_pdb(dir='/media/zhangxin/Raid0/dataset/PP/', out_dir='/media/zhangxin/Raid0/dataset/PP/simplify/')
