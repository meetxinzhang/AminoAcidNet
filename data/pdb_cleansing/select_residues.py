# encoding: utf-8
"""
@author: Xin Zhang
@contact: zhangxin@szbl.ac.cn
@file: pdb_parser.py
@time: 6/6/21 4:08 PM
@desc:
   fork from https://github.com/manasa711/Protein-Binding-Site-Visualization
   addition features to save new pdb
"""
import platform
import os
import numpy as np
from collections import defaultdict as ddict
from utils.log_output import Logger

logger = Logger(log_path='/home/zhangxin/ACS/github/Apro/output/logs/select_residues.log', is_print=False)


def get_bind_sites(pdf_file, thres, chain1, chain2):
    file = open(pdf_file, 'r')
    # creating lists with the coordinates of CA atoms from both the chains
    cds1 = []
    cds2 = []

    for line in file:
        line = line.rstrip()
        ch1 = []  # [[a_no, x, y, z, aa_no, aa_name], ...]
        ch2 = []
        if line.startswith("ATOM"):
            atom = line.split()
            if atom[4] == chain1:
                if atom[2] == 'CA':  # only considering the C-Alpha atoms for the computation
                    a_no = atom[1]
                    x = atom[6]
                    y = atom[7]
                    z = atom[8]
                    aa_no = atom[5]
                    aa_name = atom[3]
                    ch1.append(a_no)
                    ch1.append(x)
                    ch1.append(y)
                    ch1.append(z)
                    ch1.append(aa_no)
                    ch1.append(aa_name)
                    cds1.append(ch1)
            elif atom[4] == chain2:
                if atom[2] == 'CA':
                    a_no = atom[1]
                    x = atom[6]
                    y = atom[7]
                    z = atom[8]
                    aa_no = atom[5]
                    aa_name = atom[3]
                    ch2.append(a_no)
                    ch2.append(x)
                    ch2.append(y)
                    ch2.append(z)
                    ch2.append(aa_no)
                    ch2.append(aa_name)
                    cds2.append(ch2)
    file.close()
    # calculating Euclidean Distance between CA atoms of chain 1 and CA atoms of chain2

    bind_sites_1 = []  # list with interface atoms from chain 1
    bind_sites_2 = []  # list with interface atoms from chain 2

    for i in cds1:
        for j in cds2:
            x1 = float(i[1])
            y1 = float(i[2])
            z1 = float(i[3])
            x2 = float(j[1])
            y2 = float(j[2])
            z2 = float(j[3])
            e = ((x1 - x2) ** 2) + ((y1 - y2) ** 2) + ((z1 - z2) ** 2)
            euc = e ** 0.5
            if euc <= thres:
                # op = chain1 + ":" + str(i[5]) + "(" + str(i[4]) + ") interacts with " + chain2 + ":" + str(
                #     j[5]) + "(" + str(j[4]) + ")"
                # print(op)  # prints out the atoms from chain1 which interact with the atoms from chain2
                bind_sites_1.append(int(i[4]))
                bind_sites_2.append(int(j[4]))
    return list(set(bind_sites_1)), list(set(bind_sites_2))


def save_bind_sites(pdb_file, save_dir, thres, rec_chains, lig_chains):
    if platform.system() == 'Windows':
        pdb_id = pdb_file.split('\\')[-1].replace('.pdb', '')
    else:
        pdb_id = pdb_file.split('/')[-1].replace('.pdb', '')

    final_res = ddict(set)
    for cr in rec_chains:
        for cl in lig_chains:
            bs1, bs2 = get_bind_sites(pdb_file, thres, cr, cl)
            for aa in bs1:
                final_res[cr].add(aa)
            for aa in bs2:
                final_res[cl].add(aa)

    # print(list(final_res.values()))

    # save pdb
    for ele in final_res:
        if len(list(ele)) == 0:
            return
    chain_ids = np.concatenate([rec_chains, lig_chains], axis=0)
    print(chain_ids)

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    new_file = open(save_dir + pdb_id + '_bind_site.pdb', 'a')

    for line in open(pdb_file, 'r'):
        # line = line.rstrip()
        if line.startswith("ATOM"):
            elements = line.split()
            chain_id = elements[4]
            aa_idx = int(elements[5])
            if chain_id in chain_ids and aa_idx in list(final_res[chain_id]):
                new_file.write(line)
    new_file.close()

    # logger.write('grep atoms ', pdb_id, '>', pdb_id + '.pdb, atoms: \n', final_rec_bs, '\n', final_lig_bs,
    #              join_time=True)
    # logger.flush()


if __name__ == "__main__":
    save_bind_sites(pdb_file='/media/zhangxin/Raid0/dataset/PP/single_complex/3/5sx5.pdb',
                    save_dir='/media/zhangxin/Raid0/dataset/PP/single_complex/bind_sites/3/',
                    thres=15, rec_chains=['H', 'L'], lig_chains=['N'])
