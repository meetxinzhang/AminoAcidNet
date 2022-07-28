# encoding: utf-8
"""
@author: Xin Zhang
@contact: zhangxin@szbl.ac.cn
@time: 2022/7/28 11:45
@desc:
"""
import numpy as np


def dichotomy_account_in_sorted(sorted_list, key):
    l = 0
    r = len(sorted_list) - 1

    # find the left bound
    while l < r:
        mid = (r + l) // 2
        if sorted_list[mid] < key:
            l = mid + 1
        else:
            r = mid  # involving mid==key, so the r isn't right
    left = l

    # find the right bound
    l = 0
    r = len(sorted_list) - 1
    while l < r:
        mid = (r+l) // 2
        if sorted_list[mid] <= key:
            l = mid + 1
        else:
            l = mid
    right = l

    return right - left


if __name__ == '__main__':
    l = dichotomy_account_in_sorted([1, 2, 3, 3, 3, 4], 1)
    print(l)
