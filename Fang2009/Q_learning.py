# -*- coding: utf-8 -*-
# @Time     : 2/16/2025 20:08
# @Author   : Junyi
# @FileName: Q_learning.py
# @Software  : PyCharm
# Observing PEP 8 coding style
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import random

class Agent:
    def __init__(self, N, high_peak, low_peak):
        self.N = N
        self.max_value = 2 ** self.N - 1  # Maximum possible state index
        self.bit_length = len(bin(self.max_value)[2:])  # Determine required bit length
        self.Q_table = np.zeros((2 ** self.N, self.N + 1))
        # each row: one of the 2^N state;
        # each column: reward to flip the N-th element; the (N+1) column is status quo
        self.reality = [0] * 2 ** self.N  # all states are initialized as zeros
        self.high_peak_index = 2 ** self.N - 1  # 111111
        self.low_peak_index = 0   # 00000
        self.high_peak = high_peak
        self.low_peak = low_peak
        self.reality[self.high_peak_index] = high_peak
        self.reality[self.low_peak_index] = low_peak
        self.state = [random.randint(0, 1) for _ in range(self.N)]
        self.max_length = 100000  # make sure an episode can end with peaks
        self.informed_percentage = 0  # the percentage of states that are informed
        self.performance = 0
        self.steps = 0
        self.search_trajectory = []

    def learn(self, tau=20.0, alpha=0.2, gamma=0.9):
        """
        One episode concludes with local or global peaks and update its antecedent Q(s, a).
        Larger Tau: exploration (at 30, random walk);  Smaller Tau: exploitation
        :param tau: temperature regulates how sensitive the probability of choosing a given action is to the estimated Q
        :param state: current state, int
        :param alpha: learning rate (cf. Denrell 2004)
        :param gamma: emphasis on positional value (cf. Denrell 2004); gamma = 0.9 is best in Denrell 2004
        :return:
        """
        # Initialize one learning episode
        cur_state_index = np.random.choice(range(1, 2 ** self.N - 2)) # cannot be the peaks!!
        self.state = self.int_to_binary_list(state_index = cur_state_index)
        self.search_trajectory = []
        for _ in range(self.max_length):
            cur_state_index = self.binary_list_to_int(self.state)
            q_row = self.Q_table[cur_state_index]
            # q_row -= np.max(q_row)  # prevent numerical overflow and preserve softmax behavior
            exp_prob_row = np.exp(q_row / tau)
            prob_row = exp_prob_row / np.sum(exp_prob_row)
            action = np.random.choice(range(self.N + 1), p=prob_row)
            self.search_trajectory.append([cur_state_index, action])
            # print(self.state, cur_state_index, action)
            # taking an appropriate action from next state; based on current beliefs
            next_state = self.state.copy()
            if action < self.N:
                next_state[action] = 1 - self.state[action]
            next_state_index = int(''.join(map(str, next_state)), 2)
            # Choose a proper next action (I) softmax
            # next_q_row = self.Q_table[next_state_index]
            # next_exp_prob_row = np.exp(next_q_row / tau)
            # next_prob_row = next_exp_prob_row / np.sum(next_exp_prob_row)
            # next_action = np.random.choice(range(self.N + 1), p=next_prob_row)
            # next_state_quality = self.Q_table[next_state_index][next_action]

            # Choose a proper next action (II) best
            next_state_quality = max(self.Q_table[next_state_index])  # equal to zero when next state is peaks
            reward = self.reality[next_state_index]  # equal to non-zero when next state is peaks
            self.Q_table[cur_state_index][action] = ((1 - alpha) * self.Q_table[cur_state_index][action] +
                                                     alpha * (reward + gamma * next_state_quality))
            self.state = next_state.copy()  # within one episode, it is sequential search
            if reward:  # If we reach a rewarded state, stop learning
                break  # this break means that the Q_table for the next_state will not be updated.
        self.informed_percentage = np.count_nonzero(np.any(self.Q_table != 0, axis=1)) / (2 ** self.N)

    def evaluate(self, tau=20.0):
        cur_state_index = np.random.choice(range(1, 2 ** self.N - 2)) # cannot be the peaks!!
        self.state = self.int_to_binary_list(state_index = cur_state_index)
        # print("cur_state_index: ", cur_state_index, self.state)
        self.search_trajectory = []
        for perform_step in range(self.max_length):
            cur_state_index = self.binary_list_to_int(self.state)
            q_row = self.Q_table[cur_state_index]
            # q_row -= np.max(q_row)  # prevent numerical overflow and preserve softmax behavior
            exp_prob_row = np.exp(q_row / tau)
            prob_row = exp_prob_row / np.sum(exp_prob_row)
            action = np.random.choice(range(self.N + 1), p=prob_row)
            self.search_trajectory.append([cur_state_index, action])
            next_state = self.state.copy()
            if action < self.N:
                next_state[action] = 1 - self.state[action]
            next_state_index = int(''.join(map(str, next_state)), 2)
            reward = self.reality[next_state_index]
            self.state = next_state.copy()
            if reward:
                self.performance = reward
                self.steps = perform_step + 1
                break

    # def evaluate_max(self):
    #     self.state = [random.randint(0, 1) for _ in range(self.N)]
    #     for perform_step in range(self.max_length):
    #         cur_state_index = self.binary_list_to_int(self.state)
    #         q_row = self.Q_table[cur_state_index]
    #         action = np.argmax(q_row)
    #         next_state = self.state.copy()
    #         if action < self.N:
    #             next_state[action] = 1 - self.state[action]
    #         next_state_index = int(''.join(map(str, next_state)), 2)
    #         reward = self.reality[next_state_index]
    #         self.state = next_state
    #         if reward:
    #             self.performance = reward
    #             self.steps = perform_step
    #             break

    def int_to_binary_list(self, state_index):
        return [int(bit) for bit in format(state_index, f'0{self.bit_length}b')]

    def binary_list_to_int(self, state):
        return int(''.join(map(str, state)), 2)

    def generate_state_with_hamming_distance(self, orientation_state, hamming_distance):
        """
        Generate a new state that is at a specific Hamming distance from the original state.

        :param original_state: The original state (e.g., global peak).
        :param hamming_distance: The number of bit flips to make.
        :return: A new state with the specified Hamming distance.
        """
        new_state = orientation_state.copy()
        indices = list(range(self.N))
        flip_indices = random.sample(indices, hamming_distance)
        for idx in flip_indices:
            new_state[idx] = 1 - new_state[idx]
        return new_state

    def visualize_1(self):
        # Custom colormap: Light gray for zero, then blue → red for positive values
        colors = [(0.9, 0.9, 0.9), "blue", "red"]
        cmap = mcolors.LinearSegmentedColormap.from_list("custom_cmap", colors, N=100)

        # Use a logarithmic scale to enhance contrast if needed
        norm = mcolors.SymLogNorm(linthresh=1, linscale=0.5, vmin=0, vmax=np.max(self.Q_table))

        plt.figure(figsize=(8, 6))
        plt.imshow(self.Q_table, cmap=cmap, aspect='auto', norm=norm)
        plt.colorbar(label="Q-value")
        plt.xlabel("Actions")
        plt.ylabel("States")
        plt.title("Heatmap of Q-table (Zero-dominant)")
        plt.show()

    def visualize(self, search_trajectory=None):
        # Custom colormap: Light gray for zero, then blue → red for positive values
        colors = [(0.9, 0.9, 0.9), "blue", "red"]
        cmap = mcolors.LinearSegmentedColormap.from_list("custom_cmap", colors, N=100)

        # Use a logarithmic scale to enhance contrast if needed
        norm = mcolors.SymLogNorm(linthresh=1, linscale=0.5, vmin=0, vmax=np.max(self.Q_table))

        plt.figure(figsize=(8, 6))
        plt.imshow(self.Q_table, cmap=cmap, aspect='auto', norm=norm)
        plt.colorbar(label="Q-value")
        plt.xlabel("Actions")
        plt.ylabel("States")
        plt.title("Heatmap of Q-table (Zero-dominant)")

        if search_trajectory:
            states, actions = zip(*search_trajectory)  # Extract state and action sequences

            # Scatter plot to highlight visited (state, action) pairs
            plt.scatter(actions, states, color='yellow', edgecolors='black', s=50, label="Search Trajectory")

            # Draw arrows to show transitions
            for i in range(len(search_trajectory) - 1):
                s1, a1 = search_trajectory[i]  # Start point (current step)
                s2, a2 = search_trajectory[i + 1]  # End point (next step)

                plt.quiver(a1, s1, a2 - a1, s2 - s1, angles='xy', scale_units='xy', scale=1,
                           color='black', alpha=0.8, width=0.005, headlength=4, headwidth=3)

        plt.legend(loc="upper right")
        plt.show()

        # Apply logarithmic scaling for better contrast of large values
        # Q_table_log = np.log1p(self.Q_table)  # log(1 + Q_value) to avoid taking log of zero
        #
        # # Custom colormap with high contrast
        # cmap = plt.cm.hot  # Hot colormap for good contrast
        #
        # plt.figure(figsize=(8, 6))
        # plt.imshow(Q_table_log, cmap=cmap, aspect='auto')
        # plt.colorbar(label="Log-Scaled Q-value")
        # plt.xlabel("Actions")
        # plt.ylabel("States")
        # plt.title("Heatmap with Logarithmic Scaling")
        # plt.show()


if __name__ == '__main__':
    # 2 ^ 5 = 32 states
    # random.seed(0)
    # search_trajectory = [[3, 1], [4, 2], [5, 0], [6, 3]]  # Example trajectory
    reward_list, step_list = [], []
    q_agent = Agent(N=10, high_peak=50, low_peak=10)
    for index in range(50):
        q_agent.learn(tau=20, alpha=0.2, gamma=0.9)
        # print(q_agent.Q_table)
        # break
        # print("Informed: ", q_agent.informed_percentage)
        # if index % 50 == 0:
        #     print(index)
        #     q_agent.visualize_1()
    print(q_agent.Q_table[-1], q_agent.Q_table[0])
    q_agent.visualize_1()
    q_agent.evaluate(tau=20)  # exploration
    q_agent.visualize(search_trajectory=q_agent.search_trajectory)
    print("Exploration: ", q_agent.performance, len(q_agent.search_trajectory), q_agent.steps)

    # q_agent.state = [random.randint(0, 1) for _ in range(5)]
    q_agent.evaluate(tau=0.1)  # exploitation
    q_agent.visualize(search_trajectory=q_agent.search_trajectory)
    # print("before evaluate: ", q_agent.state)
    # q_agent.evaluate(tau=0.1)
    # print("after evaluate: ", q_agent.state)
    print("Exploitation: ", q_agent.performance, len(q_agent.search_trajectory), q_agent.steps)


    # print(q_agent.reality)
    # percentage_high_across_learning_length, percentage_low_across_learning_length = [], []
    # step_across_length = []
    # [50, 100, 150, 200, 250, 300, 350]
    # learning_length_list = [100, 200, 300]
    # agent_num = 50
    # for learning_length in learning_length_list:
    #     reward_across_agents = []
    #     step_across_agents = []
    #     for _ in range(agent_num):
    #         np.random.seed(None)
    #         q_agent = Agent(N=10, global_peak=50, local_peaks=[10])
    #         for _ in range(learning_length):
    #             q_agent.learn(tau=20, alpha=0.8, gamma=0.9)
    #         reward, step = q_agent.perform(tau=20)
    #         reward_across_agents.append(reward)
    #         step_across_agents.append(step)
    #
    #     percentage_high = sum([1 if reward == 50 else 0 for reward in reward_across_agents]) / agent_num
    #     percentage_low = sum([1 if reward == 10 else 0 for reward in reward_across_agents]) / agent_num
    #     ave_step = sum(step_across_agents) / len(step_across_agents)
    #
    #     percentage_high_across_learning_length.append(percentage_high)
    #     percentage_low_across_learning_length.append(percentage_low)
    #     step_across_length.append(ave_step)
    #
    # plt.plot(learning_length_list, percentage_high_across_learning_length, "-", color='k', linewidth=2, label="High Peak")
    # plt.plot(learning_length_list, percentage_low_across_learning_length, "--", color='k', linewidth=2, label="Low Peak")
    # plt.plot(learning_length_list, step_across_length, "-", color='grey', linewidth=2, label="Low Peak")
    # # Add labels and title
    # plt.xlabel('Performance')
    # plt.ylabel('Learning Length')
    # plt.title('Performance Implications of Maximization vs. Softmax Strategies')
    #
    # # Add grid and legend
    # plt.grid(True)
    # plt.legend()
    #
    # # Show the plot
    # plt.show()


