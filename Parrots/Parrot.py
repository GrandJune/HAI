# -*- coding: utf-8 -*-
# @Time     : 4/20/2025 20:06
# @Author   : Junyi
# @FileName: Parrot.py
# @Software  : PyCharm
# Observing PEP 8 coding style
import numpy as np
from Reality import Reality

class Parrot:
    def __init__(self, N=10, capability=1.0, reality=None):
        if not 0 <= capability <= 1:
            raise ValueError("Capability must be between 0 and 1")
        self.N = N
        self.capability = capability
        self.reality = reality
        self.Q_table = np.zeros((2 ** self.N, self.N))
        if capability == 1:
            self.Q_table.fill(1)
        else:
            # Randomly set capability% of Q_table elements to 1
            num_elements = self.Q_table.size
            num_ones = int(num_elements * capability)
            indices = np.random.choice(num_elements, num_ones, replace=False)
            self.Q_table.flat[indices] = 1

    def suggest(self, current_state=None, accuracy=1.0):
        if current_state is None:
            raise ValueError("State must be provided")
        if not (0.0 <= accuracy <= 1.0):
            raise ValueError("Accuracy must be between 0 and 1")

        cur_state_index = self.binary_list_to_int(current_state)
        row = self.Q_table[cur_state_index]
        candidate_actions = [i for i, q in enumerate(row) if q == 1]
        if not candidate_actions:
            return None  # No valid action available

        # Actions that would bring the current state closer to the global peak
        correct_actions = [i for i in range(self.N) if current_state[i] != self.reality.global_peak[i]]
        final_actions = list(set(candidate_actions) & set(correct_actions))
        inaccurate_actions = list(set(candidate_actions) - set(final_actions))
        # If no intersection
        if not final_actions:
            return None
        # Accurate
        if np.random.rand() < accuracy:
            return np.random.choice(final_actions)
        else:
            return np.random.choice(inaccurate_actions)

    def int_to_binary_list(self, state_index):
        return [int(bit) for bit in format(state_index, f'0{self.N}b')]

    def binary_list_to_int(self, state):
        return int(''.join(map(str, state)), 2)

if __name__ == '__main__':
    parrot = Parrot(N=10, capability=1)
    print(parrot.Q_table)