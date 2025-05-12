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
        self.Q_table = np.zeros((2 ** self.N, self.N))
        if capability == 1:
            self.Q_table.fill(1)
        else:
            # Randomly set capability% of Q_table elements to 1
            num_elements = self.Q_table.size
            num_ones = int(num_elements * capability)
            indices = np.random.choice(num_elements, num_ones, replace=False)
            self.Q_table.flat[indices] = 1



    def suggest(self, state=None):
        """
        For those Q_table(s, a) = 1, randomly suggest an accurate action that leads closer to the global peak.
        :param state: Current state (binary string) to get suggestion for
        :return: Suggested action index based on Q-table values, or None if no suggestion available
        """
        if state is None:
            return None

        state_idx = int(state, 2)
        # Get actions with high Q-values (close to 1) for current state
        valid_actions = np.where(self.Q_table[state_idx] > 0.9)[0]

        if len(valid_actions) > 0:
            # Randomly select one of the valid actions
            return np.random.choice(valid_actions)
        else:
            return None


if __name__ == '__main__':
    parrot = Parrot(N=10, capability=1)
    print(parrot.Q_table)