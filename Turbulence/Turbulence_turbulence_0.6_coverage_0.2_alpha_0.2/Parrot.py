# -*- coding: utf-8 -*-
# @Time     : 4/20/2025 20:06
# @Author   : Junyi
# @FileName: Parrot.py
# @Software  : PyCharm
# Observing PEP 8 coding style
import numpy as np
from Reality import Reality

class Parrot:
    def __init__(self, N=10, coverage=1.0, accuracy=1.0, reality=None):
        if not 0 <= coverage <= 1:
            raise ValueError("Coverage must be between 0 and 1")
        self.N = N
        self.coverage = coverage
        self.reality = reality
        self.Q_table = np.zeros((2 ** self.N, self.N))
        self.accuracy = accuracy
        if coverage == 1:
            self.Q_table.fill(1)
        else:
            # Randomly set capability% of Q_table elements to 1
            num_elements = self.Q_table.size
            num_ones = int(num_elements * coverage)
            indices = np.random.choice(num_elements, num_ones, replace=False)
            self.Q_table.flat[indices] = 1

    def suggest(self, current_state=None):
        if current_state is None:
            raise ValueError("State must be provided")

        cur_state_index = self.binary_list_to_int(current_state)
        row = self.Q_table[cur_state_index]
        candidate_actions = [i for i, q in enumerate(row) if q == 1]
        if not candidate_actions:
            return None  # No valid action available

        # Actions that would bring the current state closer to the global peak
        correct_actions = [i for i in range(self.N) if current_state[i] != self.reality.global_peak_state[i]]
        incorrect_actions = [i for i in range(self.N) if current_state[i] == self.reality.global_peak_state[i]]
        available_correct_actions = list(set(candidate_actions) & set(correct_actions))
        available_incorrect_actions = list(set(candidate_actions) & set(incorrect_actions))
        # Accuracy
        if np.random.rand() < self.accuracy and available_correct_actions:
            return np.random.choice(available_correct_actions)
        elif available_incorrect_actions:
            return np.random.choice(available_incorrect_actions)
        else:
            return None

    def int_to_binary_list(self, state_index):
        return [int(bit) for bit in format(state_index, f'0{self.N}b')]

    def binary_list_to_int(self, state):
        return int(''.join(map(str, state)), 2)

if __name__ == '__main__':
    from Reality import Reality
    reality = Reality(N=10, global_peak_value=1, local_peak_values=[10])
    parrot = Parrot(N=10, accuracy=1, coverage=1.0, reality=reality)
    print(parrot.Q_table)