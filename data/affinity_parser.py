# encoding: utf-8
"""
@author: Xin Zhang
@contact: zhangxin@szbl.ac.cn
@file: affinity_parser.py
@time: 1/19/21 5:21 PM
@desc:
"""
import re


def get_affinity(file_path):
    key_value_pairs = []
    with open(file_path, "r") as f:
        lines = f.readlines()[6:]  # start by line 7
        # line can be like those:
        # 3sgb  1.80  1983  Kd=17.9pM     // 3sgb.pdf (56-mer) TURKEY OVOMUCOID INHIBITOR (OMTKY3), 1.79 x 10-11M
        # 2tgp  1.90  1983  Kd~2.4uM      // 2tgp.pdf (58-mer) TRYPSIN INHIBITOR, 2.4 x 10-6M
        # 1ihs  2.00  1994  Ki=0.3nM      // 1ihs.pdf (21-mer) hirutonin-2 with human a-thrombin, led to Ki=0.3nM
        for line in lines:
            line = line.split()
            id = str(line[0])
            affinity_str = str(line[3])
            v = re.findall(r"\d+\.?\d*", affinity_str)[0]
            unit = affinity_str[-2:]
            if unit == 'uM':
                affinity = float(str(v))
            if unit == 'nM':
                affinity = float(str(v))/1000
            if unit == 'pM':
                affinity = float(str(v))/1000000
            if unit == 'fM':
                affinity = float(str(v))/1000000000

            key_value_pairs.append([id, affinity])

    return dict(key_value_pairs)
