# -*- coding: utf-8 -*-
# @Time     : 2/16/2025 20:08
# @Author   : Junyi
# @FileName: Q_learning.py
# @Software  : PyCharm
# Observing PEP 8 coding style
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import random

class Agent():
    def __init__(self, N, global_peak, local_peaks):
        self.N = N
        self.max_value = 2 ** self.N - 1  # Maximum possible state index
        self.bit_length = len(bin(self.max_value)[2:])  # Determine required bit length
        self.Q_tabel = defaultdict(dict)
        for row in range(2 ** self.N):
            for col in range(self.N + 1):
                self.Q_tabel[row][col] = 0
        # each row: one of the 2^N state;
        # each column: reward to flip the N-th element; the (N+1) column is to stay still
        self.reality = [0] * 2 ** self.N  # all states are initialized as zeros
        self.peak_indices = random.sample(range(2 ** self.N), len(local_peaks) + 1)
        # the first one is the global peak, while the rest is local peaks
        self.reality[self.peak_indices[0]] = global_peak
        for i in range(len(local_peaks)):
            self.reality[self.peak_indices[i + 1]] = local_peaks[i]
        self.state = [random.randint(0, 1) for _ in range(self.N)]
        self.max_length = 5000  # make sure an episode can end with peaks

    def learn(self, tau=20, alpha=0.8, gamma=0.9):
        """
        One episode concludes with local or global peaks and update its antecedent Q(s, a).
        Larger Tau: exploration (at 30, random walk);  Smaller Tau: exploitation
        :param tau: temperature regulates how sensitive the probability of choosing a given action is to the estimated Q
        :param state: current state, int
        :param alpha: learning rate (cf. Denrell 2004)
        :param gamma: emphasis on positional value (cf. Denrell 2004)
        :return:
        """
        # avoid unlimited search
        for _ in range(self.max_length):
            cur_state_index = int(''.join(map(str, self.state)), 2)
            # print(self.state, cur_state_index)
            q_row = self.Q_tabel[cur_state_index]
            exp_prob_row = [np.exp(each / tau) for each in q_row]
            prob_row = [each / sum(exp_prob_row) for each in exp_prob_row]
            action = np.random.choice(range(self.N + 1), p=prob_row)

            # taking an appropriate action from next state; based on current beliefs
            next_state = self.state.copy()
            if action < self.N:
                next_state[action] = 1 - self.state[action]
            next_state_index = int(''.join(map(str, next_state)), 2)
            reward = self.reality[next_state_index]
            if reward:
                # update the preceding state quality
                self.Q_tabel[cur_state_index][action] = (1 - alpha) * self.Q_tabel[cur_state_index][action] + alpha * reward
                break
            else:
                # according to (Denrell, 2004), the next state quality/credit is the best state-action pair
                next_state_quality = max(self.Q_tabel[next_state_index])
                self.Q_tabel[cur_state_index][action] = (1 - alpha) * self.Q_tabel[cur_state_index][action] + alpha * (
                            reward + gamma * next_state_quality)
                self.state = next_state

    def perform(self, tau=20.0):
        for _ in range(self.max_length):
            # re-initialize
            self.state = [random.randint(0, 1) for _ in range(self.N)]
            cur_state_index = int(''.join(map(str, self.state)), 2)
            # print(self.state, cur_state_index)
            q_row = self.Q_tabel[cur_state_index]
            exp_prob_row = [np.exp(each / tau) for each in q_row]
            prob_row = [each / sum(exp_prob_row) for each in exp_prob_row]
            action = np.random.choice(range(self.N + 1), p=prob_row)

            next_state = self.state.copy()
            if action < self.N:
                next_state[action] = 1 - self.state[action]
            next_state_index = int(''.join(map(str, next_state)), 2)
            reward = self.reality[next_state_index]
            self.state = next_state
            if reward:
                return reward


    def int_to_binary_list(self, state):
        return [int(bit) for bit in format(state, f'0{self.bit_length}b')]

    def visualize(self):
        """
         the rows (Y-axis) correspond to the outer list (the number of sublists),
          while the columns (X-axis) correspond to the elements within each sublist.
        :return:
        """
        # Convert to NumPy array (optional, but helps with indexing)
        data_array = np.array([[self.Q_tabel[row].get(col, 0) for col in range(self.N + 1)]
                               for row in range(2 ** self.N)], dtype=np.float64)
        # Plot heatmap
        # plt.imshow(data_array, cmap='viridis', aspect='auto')  # 'viridis' gives a nice color gradient
        # plt.axhline(y=self.peak_indices[0], color='red', linestyle='--', linewidth=2)  # global peak
        # plt.axhline(y=self.peak_indices[1], color='black', linestyle='--', linewidth=2)  # local peak
        # plt.colorbar(label="Value")  # Adds a color scale
        # plt.xlabel("Actions")
        # plt.ylabel("States")
        # plt.title("Heatmap Visualization")

        # Use LogNorm to reduce zero prominence
        # plt.imshow(data_array, cmap='magma', norm=mcolors.LogNorm(vmin=0.1, vmax=np.max(data_array) + 1),
        #            aspect='auto')
        # plt.colorbar(label="Value (log scale)")
        # plt.xlabel("Actions")
        # plt.ylabel("States")
        # plt.title("Heatmap with Logarithmic Scaling")

        from matplotlib.colors import LinearSegmentedColormap

        # Define a colormap where zeros are light gray, and values range from blue to red
        colors = [(0.9, 0.9, 0.9), "blue", "red"]  # Light gray → Blue → Red
        cmap = LinearSegmentedColormap.from_list("custom_cmap", colors, N=100)
        plt.axhline(y=self.peak_indices[0], color='red', linestyle='--', linewidth=2)  # global peak
        plt.axhline(y=self.peak_indices[1], color='black', linestyle='--', linewidth=2)  # local peak
        plt.imshow(data_array, cmap=cmap, aspect='auto')
        plt.colorbar(label="Value")
        plt.xlabel("Actions")
        plt.ylabel("States")
        plt.title("Heatmap with Custom Colormap (Soft Zero)")

        plt.show()


if __name__ == '__main__':
    # 2 ^ 5 = 32 states
    # print(q_learn.Q_tabel)
    # print(q_agent.reality)
    percentage_high_across_learning_length, percentage_low_across_learning_length = [], []
    # [50, 100, 150, 200, 250, 300, 350]
    learning_length_list = [100, 200, 300]
    agent_num = 200
    for learning_length in learning_length_list:
        reward_across_agents = []
        for _ in range(agent_num):
            q_agent = Agent(N=10, global_peak=50, local_peaks=[10])
            for _ in range(learning_length):
                q_agent.learn(tau=20, alpha=0.8, gamma=0.9)
            reward = q_agent.perform(tau=20)
            reward_across_agents.append(reward)

        percentage_high = sum([1 if reward == 50 else 0 for reward in reward_across_agents]) / agent_num
        percentage_low = sum([1 if reward == 10 else 0 for reward in reward_across_agents]) / agent_num

        percentage_high_across_learning_length.append(percentage_high)
        percentage_low_across_learning_length.append(percentage_low)

    plt.plot(learning_length_list, percentage_high_across_learning_length, "-", color='red', linewidth=2, label="High Peak")
    plt.plot(learning_length_list, percentage_low_across_learning_length, "--", color='blue', linewidth=2, label="Low Peak")
    # Add labels and title
    plt.xlabel('Performance')
    plt.ylabel('Learning Length')
    plt.title('Performance Implications of Maximization vs. Softmax Strategies')

    # Add grid and legend
    plt.grid(True)
    plt.legend()

    # Show the plot
    plt.show()


