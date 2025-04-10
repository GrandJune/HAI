# -*- coding: utf-8 -*-
# @Time     : 4/9/2025 17:53
# @Author   : Junyi
# @FileName: Agent.py
# @Software  : PyCharm
# Observing PEP 8 coding style
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import random

class Agent:
    def __init__(self, N, reality):
        self.N = N
        self.reality = reality
        state_index = np.random.choice(range(0, 2 ** self.N - 2))  # cannot be the peaks!!
        self.state = self.int_to_binary_list(state_index=state_index)
        self.Q_table = None
        self.search_trajectory = []
        # performance measure
        self.max_steps = 10000 # avoid unlimited episodes
        self.steps = 0  # time to peak
        self.payoff = 0 # peak reward, typically 1

    def learn(self, alpha, gamma, Q_table):
        """
        :param state: current state, int
        :param alpha: learning rate (cf. Denrell 2004), 0.2
        :param gamma: emphasis on positional value (cf. Denrell 2004); gamma = 0.9 is best in Denrell 2004
        :return:
        """
        self.Q_table = Q_table
        for steps in range(self.max_steps):
            cur_state_index = self.binary_list_to_int(self.state)
            max_index_list = np.where(self.Q_table[cur_state_index] == np.max(self.Q_table[cur_state_index]))[0]
            action = np.random.choice(max_index_list)
            next_state = self.state.copy()
            if action < self.N:
                next_state[action] = 1 - self.state[action]
            next_state_index = int(''.join(map(str, next_state)), 2)
            # Choose a proper next action (II) best
            next_state_quality = max(self.Q_table[next_state_index])  # equal to zero when next state is peaks
            reward = self.reality[next_state_index]  # equal to non-zero when next state is peaks
            self.Q_table[cur_state_index][action] = ((1 - alpha) * self.Q_table[cur_state_index][action] +
                                                     alpha * (reward + gamma * next_state_quality))
            if reward:  # If we reach a rewarded state, stop learning
                self.steps = steps + 1
                break
            # sequential search
            self.state = next_state.copy()
        # re-initialize the agent
        state_index = np.random.choice(range(0, 2 ** self.N - 2))  # cannot be the peaks!!
        self.state = self.int_to_binary_list(state_index=state_index)

    def int_to_binary_list(self, state_index):
        return [int(bit) for bit in format(state_index, f'0{self.N}b')]

    def binary_list_to_int(self, state):
        return int(''.join(map(str, state)), 2)
