# encoding: utf-8
"""
@author: Xin Zhang
@contact: zhangxin@szbl.ac.cn
@file: coding_test.py
@time: 6/10/21 10:55 AM
@desc:
"""
# -*-coding:utf8;-*-
# qpy:3
# qpy:console
import math

print("This is console module")


def fa(a):
    b = 1
    while a != 1:
        b *= a
        a -= 1
    return b


def taylor(x, n):
    a = 1
    count = 1
    for k in range(1, n):
        if count % 2 != 0:
            a -= (x ** (2 * k)) / fa(2 * k)
        else:
            a += (x ** (2 * k)) / fa(2 * k)
        count += 1
    return a


def cos(x):
    return taylor(x, 10)


print(cos(3))
