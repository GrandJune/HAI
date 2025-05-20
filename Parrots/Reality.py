# -*- coding: utf-8 -*-
# @Time     : 4/19/2025 21:09
# @Author   : Junyi
# @FileName: Reality.py
# @Software  : PyCharm
# Observing PEP 8 coding style
import numpy as np


class Reality:
    def __init__(self, N=None, global_peak_value =None, local_peak_values =None):
        self.N = N
        self.global_peak_value  = global_peak_value
        self.global_peak_state = [1] * self.N
        self.global_peak_index = 2 ** self.N - 1
        self.payoff_map = np.zeros(2 ** self.N)
        self.payoff_map[-1] = global_peak_value
        if local_peak_values is not None:
            self.local_peak_values = local_peak_values
            available_indices = np.random.choice(
                range(0, 2 ** self.N - 1), # exclude only the global peak
                size=len(local_peak_values),
                replace=False  # This ensures no index is chosen twice
            )
            for idx, val in zip(available_indices, local_peak_values):
                self.payoff_map[idx] = val
            self.local_peak_indices = available_indices