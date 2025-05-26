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

    def change(self):
        """
        Introduce turbulence by relocating the global and local peaks.
        """
        total_states = 2 ** self.N

        # Reset the payoff map
        self.payoff_map[:] = 0.0

        # Randomly select a new global peak index
        new_global_index = np.random.choice(total_states)
        self.global_peak_index = new_global_index
        self.global_peak_state = list(map(int, bin(new_global_index)[2:].zfill(self.N)))
        self.payoff_map[new_global_index] = self.global_peak_value

        # Generate available indices excluding the new global peak
        remaining_indices = list(set(range(total_states)) - {new_global_index})

        # Randomly assign new local peak indices
        if hasattr(self, 'local_peak_values') and self.local_peak_values is not None:
            new_local_indices = np.random.choice(
                remaining_indices,
                size=len(self.local_peak_values),
                replace=False
            )
            for idx, val in zip(new_local_indices, self.local_peak_values):
                self.payoff_map[idx] = val
            self.local_peak_indices = new_local_indices
