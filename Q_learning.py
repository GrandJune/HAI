# -*- coding: utf-8 -*-
# @Time     : 2/16/2025 20:08
# @Author   : Junyi
# @FileName: Q_learning.py
# @Software  : PyCharm
# Observing PEP 8 coding style
import numpy as np
import matplotlib.pyplot as plt
import random

class Agent():
    def __init__(self, N, global_peak, local_peaks):
        self.N = N
        self.max_value = 2 ** self.N - 1  # Maximum possible state index
        self.bit_length = len(bin(self.max_value)[2:])  # Determine required bit length
        self.Q_tabel = [[0] * (self.N + 1)] * 2 ** self.N
        # each row: one of the 2^N state;
        # each column: reward to flip the N-th element; the (N+1) column is to stay still
        self.reality = [0] * 2 ** self.N  # all states are initialized as zeros
        self.peak_indices = random.sample(range(2 ** self.N), len(local_peaks) + 1)
        # the first one is the global peak, while the rest is local peaks
        self.reality[self.peak_indices[0]] = global_peak
        for i in range(len(local_peaks)):
            self.reality[self.peak_indices[i + 1]] = local_peaks[i]
        self.state = [random.randint(0, 1) for _ in range(self.N)]

    def Q_update(self, action, alpha=0.8, gamma=0.9):
        """
        :param state: current state, int
        :param action: 0 - N, with N representing status quo
        :param reward: immediate reward received for current action (R in the paper)
        :param next_state: the reaching state after the action, int
        :param alpha: weight for (R + gamma Q')
        :param 1 - alpha: weight for Q
        :param gamma: weight for next state quality;
        :return: updated Q table
        """
        if action == self.N:
            cur_state_index = int(''.join(map(str, self.state)), 2)
            next_state_index = int(''.join(map(str, self.state)), 2)
        else:
            next_state = self.state.copy()
            next_state[action] = 1 - self.state[action]
            cur_state_index = int(''.join(map(str, self.state)), 2)
            next_state_index = int(''.join(map(str, next_state)), 2)
            self.state = next_state
        reward = self.reality[next_state_index]  # the immediate reward associated with current state and current action (so it's next state)
        next_state_quality = max(self.Q_tabel[next_state_index])  # can be the quality of current state
        self.Q_tabel[cur_state_index][action] = (1 - alpha) * self.Q_tabel[cur_state_index][action] + alpha * (reward + gamma * next_state_quality)
        return reward

    def search(self, tau):
        """
        Larger Tau: exploration (at 30, random walk);  Smaller Tau: exploitation
        :param tau: temperature regulates how sensitive the probability of choosing a given action is to the estimated Q
        :param state: current state, int
        :return:
        """
        cur_state_index = int(''.join(map(str, self.state)), 2)
        q_row = self.Q_tabel[cur_state_index]
        exp_prob_row = [np.exp(each / tau) for each in q_row]
        prob_row = [each / sum(exp_prob_row) for each in exp_prob_row]
        action = np.random.choice(range(self.N + 1), p=prob_row)
        # print("State: ", self.state, "action: ", action)
        print(self.Q_tabel[self.peak_indices[1]])
        reward = self.Q_update(action)
        return reward

    def int_to_binary_list(self, state):
        return [int(bit) for bit in format(state, f'0{self.bit_length}b')]


    def visualize(self):
        """
         the rows (Y-axis) correspond to the outer list (the number of sublists),
          while the columns (X-axis) correspond to the elements within each sublist.
        :return:
        """
        import matplotlib.pyplot as plt
        # Convert to NumPy array (optional, but helps with indexing)
        data_array = np.array(self.Q_tabel)

        # Plot heatmap
        plt.imshow(data_array, cmap='viridis', aspect='auto')  # 'viridis' gives a nice color gradient
        plt.axhline(y=self.peak_indices[0], color='red', linestyle='--', linewidth=2)  # global peak
        plt.axhline(y=self.peak_indices[1], color='black', linestyle='--', linewidth=2)  # local peak
        plt.colorbar(label="Value")  # Adds a color scale
        plt.xlabel("Actions")
        plt.ylabel("States")
        plt.title("Heatmap Visualization")

        plt.show()

if __name__ == '__main__':
    # 2 ^ 5 = 32 states
    q_agent = Agent(N=10, global_peak=50, local_peaks=[10])
    # print(q_learn.Q_tabel)
    print(q_agent.reality)
    for i in range(2000):
        reward = q_agent.search(tau=10)
        if i % 100 == 0:
            q_agent.visualize()
