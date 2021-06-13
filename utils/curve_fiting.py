# encoding: utf-8
"""
@author: Xin Zhang
@contact: zhangxin@szbl.ac.cn
@file: curve_fiting.py
@time: 6/13/21 7:15 PM
@desc:
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


def min_max_normalization(arr):
    return [float(x - np.min(arr)) / (np.max(arr) - np.min(arr)) for x in arr]


def mean_normaliztion(arr):
    return [float(x - arr.mean()) / arr.std() for x in arr]


def sigmoid(arr):
    return 1. / (1 + np.exp(-arr))


# x = [575.95109, 700.34178, 2697.33643, 5041.00635, 5280.24756]
# y = [96.031, 264.93025, 1174.71772, 926.53404, 875.02673]
# x = [57.25959, 94.20663, 340.31325, 183.24334, 98.09858]
# y = [1107.10706, 1145.54858, 1658.20587, 2122.41504, 2167.43831]
x = [62.53568, 223.25964, 204.23645, 95.3541, 71.85524]
y = [1653.4585, 1727.41825, 2342.22705, 2511.97892, 2596.68148]

print('x: ', x)
print('y: ', y)
print('  var', '              std')
print('x', np.var(x), np.std(x))
print('y', np.var(y), np.std(y))

# x = min_max_normalization(x)
# y = min_max_normalization(y)

plt.scatter(x, y)

coeff = np.polyfit(x, y, 3)  # 用3次多项式拟合
func = np.poly1d(coeff)
print('\nfitting func: \n', func)  # 在屏幕上打印拟合多项式

x_range = range(15, 230, 10)
yvals = func(x_range)
y_max = np.max(yvals)
for x, y in zip(x_range, yvals):
    if y == y_max:
        print('\nThe max y and relative x: ', y, x)


plt.plot(x_range, yvals, 'r', label='polyfit values')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend(loc=4)
plt.title('polyfitting')
plt.show()

