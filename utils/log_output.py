# encoding: utf-8
"""
@author: Xin Zhang
@contact: zhangxin@szbl.ac.cn
@file: log_output.py
@time: 12/9/20 10:08 AM
@desc:
"""
from datetime import datetime
import sys


class Logger(object):
    """Writes both to file and terminal"""

    def __init__(self, log_path, is_print=False, mode='a'):
        self.terminal = sys.stdout
        self.log = open(log_path, mode)
        self.is_print = is_print

    def write(self, *message, end='\n', join_time=False):
        if join_time:
            string = datetime.now().strftime('%Y-%m-%d %H:%M:%S') \
                            + ": " + str.join(" ", [str(a) for a in message]) + end
        else:
            string = "    " + str.join(" ", [str(a) for a in message]) + end

        if self.is_print:
            self.terminal.write(string)
        self.log.write(string)

    def flush(self):
        self.log.flush()
        # this flush method is needed for python 3 compatibility.
        # this handles the flush command by doing nothing.
        # you might want to specify some extra behavior here.
        pass
