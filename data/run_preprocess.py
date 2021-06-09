# encoding: utf-8
"""
@author: Xin Zhang
@contact: zhangxin@szbl.ac.cn
@file: run_preprocess.py
@time: 5/25/21 11:12 AM
@desc:
"""
import os
import glob
import platform

from subprocess import call
from tqdm import tqdm
from joblib import Parallel, delayed

from arguments import build_parser
parser = build_parser()
args = parser.parse_args()

parallel_jobs = args.parallel_jobs


def json_cpp_commands(pdb_path, json_dir):
    """
    Convert a single .pdb file to .json format

    Parameters
    ----------
    pdb_file: The folder containing pdb files for a protein.

    """
    if platform.system() == 'Windows':
        pdb_id = pdb_path.split('\\')[-1].replace('.pdb', '')
    else:
        pdb_id = pdb_path.split('/')[-1].replace('.pdb', '')

    json_path = json_dir + pdb_id + '.json'
    command = '/home/zhangxin/ACS/github/AminoAcidNet/data/preprocess/get_features' + ' -i ' + pdb_path + ' -j ' + json_path + ' -d 8.0'
    call(command, shell=True)


def make_json(pdb_dir, json_dir):
    if not os.path.exists(json_dir):
        os.makedirs(json_dir)
    path_list = glob.glob(pdb_dir + '/*')
    pdb_files = file_filter(path_list)

    Parallel(n_jobs=parallel_jobs)(delayed(json_cpp_commands)(pdb_path, json_dir) for pdb_path in tqdm(pdb_files))


def file_filter(input_files):
    disallowed_file_endings = (".gitignore", ".DS_Store")
    allowed_file_endings = ".pdb"
    rate = 1

    allowed_inputs = []
    for file in input_files:
        if os.path.isfile(file):
            if not file.endswith(disallowed_file_endings) and file.endswith(allowed_file_endings):
                allowed_inputs.append(file)
    return allowed_inputs[:int(len(allowed_inputs) * rate)]


if __name__ == "__main__":
    print('read pdb files to json files:')
    make_json(pdb_dir='/media/zhangxin/Raid0/dataset/PP/single_complex/bind_sites/2/',
              json_dir='/media/zhangxin/Raid0/dataset/PP/single_complex/bind_sites/json_dir/2/')
