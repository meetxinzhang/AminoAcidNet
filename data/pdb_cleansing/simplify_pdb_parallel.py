# encoding: utf-8
"""
@author: Xin Zhang
@contact: zhangxin@szbl.ac.cn
@file: simplify_pdb_parallel.py
@time: 6/2/21 4:11 PM
@desc:
"""
import glob
import platform
from data.pdb_cleansing.select_chain import save_single_complex_chains
from data.pdb_cleansing.select_residues import save_bind_sites
from joblib import Parallel, delayed
from tqdm import tqdm
from arguments import build_parser

parser = build_parser()
args = parser.parse_args()
parallel_jobs = args.parallel_jobs
bind_radius = args.bind_radius


def process_chains(file_dir, out_dir):
    filepath_list = glob.glob(file_dir + '/*')
    filepath_list = file_filter(filepath_list)  # list['filepath', ...]

    Parallel(n_jobs=parallel_jobs)(delayed(save_single_complex_chains)(file_path, out_dir)
                                   for file_path in tqdm(filepath_list))


def process_residues(log_path, file_dir, out_dir):
    filepath_list = glob.glob(file_dir + '/*')
    filepath_list = file_filter(filepath_list)  # list['filepath', ...]

    chain_file_list = []
    for line in open(log_path, 'r'):
        line = line.split()
        pdb_id = line[3]
        rec_chains = [x for x in line[1] if x.isalpha()]
        lig_chains = [x for x in line[2] if x.isalpha()]

        if len(rec_chains) == 0 or len(lig_chains) == 0:
            continue

        for file_path in filepath_list:
            if platform.system() == 'Windows':
                p_id = file_path.split('\\')[-1].replace('.pdb', '')
            else:
                p_id = file_path.split('/')[-1].replace('.pdb', '')
            if pdb_id == p_id:
                chain_file_list.append([rec_chains, lig_chains, file_path])

    Parallel(n_jobs=parallel_jobs)(delayed(save_bind_sites)
                                   (file_path, out_dir, rec_chains, lig_chains, bind_radius=bind_radius)
                                   for [rec_chains, lig_chains, file_path] in tqdm(chain_file_list))


def file_filter(input_files):
    disallowed_file_endings = (".gitignore", ".DS_Store")
    allowed_file_endings = ".pdb"
    rate = 1
    input_files = input_files[:int(len(input_files) * rate)]
    return list(filter(lambda x: not x.endswith(disallowed_file_endings) and x.endswith(allowed_file_endings),
                       input_files))


if __name__ == "__main__":
    # process_chains(file_dir='/media/zhangxin/Raid0/dataset/PP/2/',
    #                out_dir='/media/zhangxin/Raid0/dataset/PP/single_complex/2/')

    process_residues(log_path='//output/logs/select_chain.log',
                     file_dir='/media/zhangxin/Raid0/dataset/PP/single_complex/2/',
                     out_dir='/media/zhangxin/Raid0/dataset/PP/single_complex/bind_sites/2/')
