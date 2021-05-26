# encoding: utf-8
"""
@author: Xin Zhang
@contact: zhangxin@szbl.ac.cn
@file: pdb2json.py
@time: 5/25/21 11:12 AM
@desc:
"""
import csv, pickle, json, os
import argparse
import numpy as np
from os.path import isfile, join
from subprocess import call
from tqdm import tqdm
from joblib import Parallel, delayed

from arguments import build_parser
parser = build_parser()
args = parser.parse_args()

pdb_dir = args.pdb_dir
jsonpath = args.jsonpath
cpp_executable = args.cpp_executable
parallel_jobs = args.parallel_jobs
get_json_files = args.get_json_files

protein_dirs = os.listdir(pdb_dir)
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
    path = pdb_dir + pdb_file
    json_filepath = jsonpath + pdb_file.strip('.pdb') + '.json'
    command = cpp_executable + ' -i ' + path + ' -j ' + json_filepath + ' -d 8.0'
    call(command, shell=True)


if get_json_files:
    print('read pdb files to json files:')
    Parallel(n_jobs=parallel_jobs)(delayed(json_cpp_commands)(pdb_file) for pdb_file in tqdm(all_proteins))
