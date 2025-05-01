# -*- coding: utf-8 -*-
# @Time     : 4/19/2025 21:09
# @Author   : Junyi
# @FileName: Reality.py
# @Software  : PyCharm
# Observing PEP 8 coding style
import numpy as np


class Reality:
    def __init__(self, N=None, global_peak=None, local_peaks=None):
        self.N = N
        self.global_peak = global_peak
        self.payoff_map = np.zeros(2 ** self.N)
        self.payoff_map[-1] = global_peak
        if local_peaks is not None:
            self.local_peaks = local_peaks
            # Get all available indices at once, excluding 0 and 2^N-1
            self.local_peaks = local_peaks
            available_indices = np.random.choice(
                range(1, 2 ** self.N - 1),
                size=len(local_peaks),
                replace=False  # This ensures no index is chosen twice
            )
            for l, local_peak in enumerate(local_peaks):
                self.payoff_map[available_indices[l]] = local_peak
            self.local_peak_indices = available_indices
        self.global_peak_index = 2 ** self.N - 1