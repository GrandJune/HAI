# -*- coding: utf-8 -*-
# @Time     : 2/16/2025 20:08
# @Author   : Junyi
# @FileName: Q_learning.py
# @Software  : PyCharm
# Observing PEP 8 coding style
import numpy as np
import matplotlib.pyplot as plt
import random

class Q_learn():
    def __init__(self, N, global_peak, local_peaks, peak_num,):
        self.N = N
        self.Q_tabel = [[0] * (self.N + 1)] * 2 ** self.N
        # each row: one of the 2^N state;
        # each column: reward to flip the N-th element; the (N+1) column is to stay still
        self.reality = [0] * 2 ** self.N  # all states are initialized as zeros
        if peak_num == len(local_peaks) + 1:
            peak_indices = random.sample(range(2 ** self.N), peak_num)
            # the first one is the global peak, while the rest is local peaks
            self.reality[peak_indices[0]] = global_peak
            for i in range(len(local_peaks)):
                self.reality[peak_indices[i + 1]] = local_peaks[i]

    def Q_update(self, state, action, reward, next_state, alpha=0.8, gamma=0.9):
        """
        :param state: current state, int
        :param action: current action
        :param reward: immediate reward received for current action (R in the paper)
        :param next_state: the reaching state after the action, int
        :param alpha: weight for (R + gamma Q')
        :param 1 - alpha: weight for Q
        :param gamma: weight for next state quality;
        :return: updated Q table
        """
        reward = self.reality[next_state]  # the immediate reward associated with next state
        self.Q_tabel[state][action] = (1 - alpha) * self.Q_tabel[state][action] + alpha * (reward + gamma * self.Q_tabel[next_state][action])


if __name__ == '__main__':
    q_learn = Q_learn(N=5,global_peak=50, local_peaks=[10, 20], peak_num=3)
    # print(q_learn.Q_tabel)
    print(q_learn.reality)