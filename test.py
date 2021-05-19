# encoding: utf-8
"""
@author: Xin Zhang
@contact: zhangxin@szbl.ac.cn
@file: test.py
@time: 5/19/21 5:12 PM
@desc:
"""
import torch
B = 10
a = torch.arange(B).unsqueeze(-1)
print(a)