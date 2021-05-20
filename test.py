# encoding: utf-8
"""
@author: Xin Zhang
@contact: zhangxin@szbl.ac.cn
@file: test.py
@time: 5/19/21 5:12 PM
@desc:
"""
import torch
import numpy as np

# weight = torch.FloatTensor([[1, 2.3, 3], [4, 5.1, 6.3]])
# embedding = torch.nn.Embedding.from_pretrained(weight)
# inputs = torch.LongTensor([1, 1, 0, 1])
#
# print(embedding(inputs))


a = torch.randn([2, 3, 2])

index = torch.tensor([[[1, 2],
                       [0, 2],
                       [0, 1]],

                      [[2, 0],
                       [0, 2],
                       [1, 2]]])

b = a[torch.arange(2).unsqueeze(-1), index.view(2, -1)]
c = b.view(2, 3, 2, 2)

# print(a, '\n')
# print(b, '\n')
# print(c, '\n')
# print(torch.arange(2).unsqueeze(-1))
# print(index.view(2, -1))

print(a.unsqueeze(2).expand(2, 3, 2, 2))
