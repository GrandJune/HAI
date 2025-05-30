# -*- coding: utf-8 -*-
# @Time     : 4/19/2025 21:09
# @Author   : Junyi
# @FileName: Reality.py
# @Software  : PyCharm
# Observing PEP 8 coding style
import numpy as np


class Reality:
    def __init__(self, N=None, global_peak_value =None, local_peak_value =None):
        self.N = N
        self.global_peak_value  = global_peak_value
        self.global_peak_state = [1] * self.N
        self.global_peak_index = 2 ** self.N - 1

        self.local_peak_value = local_peak_value
        self.local_peak_state = [0] * self.N
        self.local_peak_index = 0

        self.payoff_map = np.zeros(2 ** self.N)
        self.payoff_map[-1] = global_peak_value
        self.payoff_map[0] = local_peak_value


    def change(self, likelihood=0.1):
        """
        Introduce turbulence by probabilistically flipping each bit in the global peak state
        based on the given likelihood. The global peak cannot occupy the same state as the fixed local peak (index 0).

        Parameters:
        likelihood (float): Probability that each bit in the global peak is flipped (0 to 1).
        """
        assert 0 <= likelihood <= 1, "Likelihood must be between 0 and 1"

        while True:
            # Probabilistically flip bits in the global peak state
            current_state = self.global_peak_state.copy()
            for i in range(self.N):
                if np.random.rand() < likelihood:
                    current_state[i] = 1 - current_state[i]

            new_global_index = int("".join(map(str, current_state)), 2)

            # Ensure the new global peak is not the same as the local peak
            if new_global_index != 0:
                break  # Valid new global peak

        # Update global peak state and index
        self.global_peak_state = current_state
        self.global_peak_index = new_global_index

        # Reset payoff map
        self.payoff_map[:] = 0
        self.payoff_map[new_global_index] = self.global_peak_value
        self.payoff_map[0] = self.local_peak_value  # Fixed local peak
        self.local_peak_index = 0


if __name__ == "__main__":
    N = 10
    global_peak_value = 50
    local_peak_value = 10

    # Initialize the reality
    reality = Reality(N=N, global_peak_value=global_peak_value, local_peak_value=local_peak_value)

    print("Original global peak state:")
    print("State:", reality.global_peak_state)
    print("Index:", reality.global_peak_index)
    print("-" * 40)

    # Introduce turbulence
    likelihood = 0.2  # Number of bits to flip
    reality.change(likelihood=likelihood)

    print("After change with likelihood =", likelihood)
    print("New global peak state:")
    print("State:", reality.global_peak_state)
    print("Index:", reality.global_peak_index)

    # Verify number of flipped bits
    original_state = [1] * N
    new_state = reality.global_peak_state
    flipped = sum([original_state[i] != new_state[i] for i in range(N)])
    print("Bits flipped:", flipped)

