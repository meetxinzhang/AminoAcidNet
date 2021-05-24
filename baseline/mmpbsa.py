# encoding: utf-8
"""
@author: Xin Zhang
@contact: zhangxin@szbl.ac.cn
@file: mmpbsa.py
@time: 5/21/21 2:38 PM
@desc:

plot tool of binding free energy and binding affinities
binding free energies are calculated by MMMPBSA
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from mmpbsa_data import affinities, free_energies


def min_max_normalization(arr):
    return [float(x - np.min(arr)) / (np.max(arr) - np.min(arr)) for x in arr]


def mean_normaliztion(arr):
    return [float(x - arr.mean()) / arr.std() for x in arr]


def sigmoid(arr):
    return 1. / (1 + np.exp(-arr))


def alignment(aff, free):
    x = []
    y = []
    for key, value in aff.items():
        x.append(value)
        y.append(free[key])
    return x, y


x, y = alignment(aff=affinities, free=free_energies)
# x = mean_normaliztion(np.array(x))
# y = mean_normaliztion(np.array(y))

# R
data = pd.DataFrame({'affinity': x, 'free energy': y})
print(data.corr())

# b = plt.scatter(x, y)
# plt.xlabel('affinity (nM)')
# plt.ylabel('binding free energy (kcal/mol)')
# plt.show()
z1 = np.polyfit(x, y, 1)  # 用1次多项式拟合
p1 = np.poly1d(z1)
print(p1)  # 在屏幕上打印拟合多项式

yvals = p1(x)  # 也可以使用yvals=np.polyval(z1,x)
plot1 = plt.plot(x, y, '*', label='original values')
plot2 = plt.plot(x, yvals, 'r', label='polyfit values')
plt.xlabel('x axis')
plt.ylabel('y axis')
plt.legend(loc=4)  # 指定legend的位置,读者可以自己help它的用法
plt.title('polyfitting')
plt.show()
