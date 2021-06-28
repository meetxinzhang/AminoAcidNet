# encoding: utf-8
"""
@author: Xin Zhang
@contact: zhangxin@szbl.ac.cn
@file: foo.py
@time: 5/19/21 5:12 PM
@desc:
"""
import torch
from torch.autograd import Variable

# tensor = [[[0.2462, -1.3954, 0.0226],
#            [0.6297, 0.9197, 0.3977],
#            [1.8450, -0.7786, 0.4734],
#            [0.0810, -0.7617, 0.5351],
#            [1.1971, -0.4210, 1.7760]],
#
#           [[0.1859, 1.6950, 0.1322],
#            [-0.6759, -2.1377, -0.6646],
#            [0.2580, -0.0107, 0.5895],
#            [0.6554, 0.5402, 0.7859],
#            [0.1759, 1.4783, -0.2102]]]
#
# data_engineer = [[[3, 2, 4],
#          [3, 4, 20],
#
#          [1, 0, 0],
#          [0, 100, 2]],
#
#
#         [[5, 3, 2],
#          [2, 56, 4],
#
#          [3, 30, 8],
#          [1, 9, 4]]]
# # (2 4 3)
# data_engineer = torch.tensor(data_engineer)
# index = data_engineer.view(2, 2, 2, 3).contiguous()
#
# s_c = torch.sum(index, dim=3)
# print('sum\n', s_c.size())
# print(s_c)
# print('\n')
#
# idx = torch.max(s_c, dim=2)
# print('max\n')
# print(idx.values)
# print(idx.indices.size())
# print(idx.indices)

# r = torch.gather(index, dim=2, index=idx.indices)
# print(r)

a = torch.randn((2, 4, 3))  # 3 channels
a = Variable(a)
a = a.transpose(1, 2)  # [2, 3, 4]

b = torch.randn(2, 4, 3)  # 3 x, y, z
b_ = b.unsqueeze(2).repeat(1, 1, 3, 1)  # [2, 4, channels, 3]
b_ = b_.transpose(1, 2)  # [2, 3, 4, 3]

print(a)
print(b_.size())
print(b_)

pool = torch.nn.MaxPool1d(kernel_size=2, stride=2, padding=0, return_indices=True)
p = pool(a)

print('result: ')
print(p[0])
print(p[1])

# o = torch.gather(b_, dim=2, index=p[1].unsqueeze(-1).repeat(1, 1, 1, 3))
o = torch.gather(b.transpose(1, 2), dim=2, index=p[1])
print(o)

#
# input = [[[[1, 1, 1],
#            [2, 1, 1],
#            [3, 1, 1],
#            [4, 1, 1]],
#
#           [[5, 1, 1],
#            [6, 1, 1],
#            [7, 1, 1],
#            [8, 1, 1]],
#
#           [[9, 1, 1],
#            [10, 1, 1],
#            [11, 1, 1],
#            [12, 1, 1]],
#
#           [[13, 1, 1],
#            [14, 1, 1],
#            [15, 1, 1],
#            [16, 1, 1]],
#
#           [[17, 1, 1],
#            [18, 1, 1],
#            [19, 1, 1],
#            [20, 1, 1]]],
#
#
#          [[[21, 2, 2],
#            [22, 2, 2],
#            [23, 2, 2],
#            [24, 2, 2]],
#
#           [[25, 2, 2],
#            [26, 2, 2],
#            [27, 2, 2],
#            [28, 2, 2]],
#
#           [[29, 2, 2],
#            [30, 2, 2],
#            [31, 2, 2],
#            [32, 2, 2]],
#
#           [[33, 2, 2],
#            [34, 2, 2],
#            [35, 2, 2],
#            [36, 2, 2]],
#
#           [[37, 2, 2],
#            [38, 2, 2],
#            [39, 2, 2],
#            [40, 2, 2]]]]
