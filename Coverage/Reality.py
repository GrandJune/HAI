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

    def change(self, intensity=10):
        """
        Introduce turbulence by shifting `intensity` number of bits in the global peak state.
        Also relocate local peaks, ensuring they don't overlap with the new global peak.
        """
        assert 0 <= intensity <= self.N, f"Intensity must be between 0 and {self.N}"

        # Flip `intensity` bits in the current global peak state
        current_state = self.global_peak_state.copy()
        indices_to_flip = np.random.choice(self.N, size=intensity, replace=False)
        for idx in indices_to_flip:
            current_state[idx] = 1 - current_state[idx]  # Flip bit

        # Update new global peak
        new_global_index = int("".join(map(str, current_state)), 2)
        self.global_peak_state = current_state
        self.global_peak_index = new_global_index

        # Reset payoff map
        self.payoff_map[:] = 0.0
        self.payoff_map[new_global_index] = self.global_peak_value

        # Reassign local peaks
        total_states = 2 ** self.N
        remaining_indices = list(set(range(total_states)) - {new_global_index})
        if hasattr(self, 'local_peak_values') and self.local_peak_values is not None:
            new_local_indices = np.random.choice(
                remaining_indices,
                size=len(self.local_peak_values),
                replace=False
            )
            for idx, val in zip(new_local_indices, self.local_peak_values):
                self.payoff_map[idx] = val
            self.local_peak_indices = new_local_indices


if __name__ == "__main__":
    N = 10
    global_peak_value = 50
    local_peak_values = [10, 10, 10]

    # Initialize the reality
    reality = Reality(N=N, global_peak_value=global_peak_value, local_peak_values=local_peak_values)

    print("Original global peak state:")
    print("State:", reality.global_peak_state)
    print("Index:", reality.global_peak_index)
    print("-" * 40)

    # Introduce turbulence
    intensity = 10  # Number of bits to flip
    reality.change(intensity=intensity)

    print("After change with intensity =", intensity)
    print("New global peak state:")
    print("State:", reality.global_peak_state)
    print("Index:", reality.global_peak_index)

    # Verify number of flipped bits
    original_state = [1] * N
    new_state = reality.global_peak_state
    flipped = sum([original_state[i] != new_state[i] for i in range(N)])
    print("Bits flipped:", flipped)
    assert flipped == intensity, f"Expected {intensity} bits to be flipped, but got {flipped}"
    print("✅ Test passed.")

