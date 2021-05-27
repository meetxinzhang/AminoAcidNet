# encoding: utf-8
"""
@author: Xin Zhang
@contact: zhangxin@szbl.ac.cn
@file: log_output.py
@time: 12/9/20 10:08 AM
@desc:
"""
from datetime import datetime
import numpy as np


def write_out(*args, end='\n', join_time=False):
    if join_time:
        output_string = datetime.now().strftime('%Y-%m-%d %H:%M:%S') \
                        + ": " + str.join(" ", [str(a) for a in args]) + end
    else:
        output_string = "    " + str.join(" ", [str(a) for a in args]) + end

    # globals() 函数会以字典类型返回当前位置的全部全局变量。
    with open("output/logs/" + "experiment_id" + ".log", "a+") as log:
        log.write(output_string)
        log.flush()
    print(output_string, end="")


def shape_np(s='   ', obj=None):
    try:
        print(str(s) + '  ', np.shape(obj))
    except ValueError:
        pass
