# -*- coding: utf-8 -*-
# @Time     : 4/20/2025 20:06
# @Author   : Junyi
# @FileName: Parrot.py
# @Software  : PyCharm
# Observing PEP 8 coding style
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import random
from Reality import Reality

class Parrot:
    def __init__(self, N=10):
        self.N = N
        self.Q_table = np.zeros((2 ** self.N, self.N))

    def aggregate_from_data(self, Q_table_list=None):
        # Stack all Q_tables and take mean along axis 0 (across tables)
        self.Q_table = np.mean(np.stack(Q_table_list), axis=0)


