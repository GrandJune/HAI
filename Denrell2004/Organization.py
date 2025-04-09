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
from Agent import Agent

class Organization:
    def __init__(self, N, peak, agent_num):
        self.N = N
        self.agent_num = agent_num
        self.agent_list = []
        self.reality = [0] * 2 ** self.N  # all states are initialized as zeros
        self.peak_index = 2 ** self.N - 1  # 111111
        self.peak = peak
        for _ in range(agent_num):
            agent = Agent(N, reality=self.reality)
            self.agent_list.append(agent)
        self.Q_table = np.zeros((2 ** self.N, self.N + 1))  # akin to organization-level code in March's (1991) paper
        # each row: one of the 2^N state;
        # each column: reward to flip the N-th element; the (N+1) column is status quo

        self.reality[self.peak_index] = peak
        self.informed_percentage = 0  # the percentage of states that are informed
        self.ave_max_q = 0
        self.steps_list = []

    def learn(self, alpha=0.2, gamma=0.9):
        """
        :param state: current state, int
        :param alpha: learning rate (cf. Denrell 2004)
        :param gamma: emphasis on positional value (cf. Denrell 2004); gamma = 0.9 is best in Denrell 2004
        :return:
        """
        steps_across_agents = []
        for agent in self.agent_list:
            agent.learn(alpha=alpha, gamma=gamma, Q_table=self.Q_table)  # individuals rely on the organizational knowledge
            steps_across_agents.append(agent.steps)
            self.Q_table = agent.Q_table  # update the Q table into organizational knowledge
        self.steps_list.append(sum(steps_across_agents) / len(steps_across_agents))

    # def learn_with_lambda(self, alpha=0.2, gamma=0.9, lambda_=5):
    #     """
    #     :param state: current state, int
    #     :param alpha: learning rate (cf. Denrell 2004)
    #     :param gamma: emphasis on positional value (cf. Denrell 2004); gamma = 0.9 is best in Denrell 2004
    #     :return:
    #     """
    #     # Initialize one learning episode
    #     cur_state_index = np.random.choice(range(0, 2 ** self.N - 1)) # cannot be the peaks!!
    #     self.state = self.int_to_binary_list(state_index = cur_state_index)
    #     for steps in range(self.max_length):
    #         cur_state_index = self.binary_list_to_int(self.state)
    #         max_index_list = np.where(self.Q_table[cur_state_index] == np.max(self.Q_table[cur_state_index]))[0]
    #         action = np.random.choice(max_index_list)
    #         self.search_trajectory.append([cur_state_index, action])
    #         next_state = self.state.copy()
    #         if action < self.N:
    #             next_state[action] = 1 - self.state[action]
    #         next_state_index = int(''.join(map(str, next_state)), 2)
    #         # Choose a proper next action (II) best
    #         next_state_quality = max(self.Q_table[next_state_index])  # equal to zero when next state is peaks
    #         reward = self.reality[next_state_index]  # equal to non-zero when next state is peaks
    #         # update the current state action pair
    #         # self.Q_table[cur_state_index][action] = ((1 - alpha) * self.Q_table[cur_state_index][action] +
    #         #                                          alpha * (reward + gamma * next_state_quality))
    #         # update preceding state action pair
    #         if lambda_ > 1:
    #             preceding_state_action_pair = self.search_trajectory[-lambda_:]
    #             while len(preceding_state_action_pair):
    #                 preceding_state, preceding_action = preceding_state_action_pair.pop()
    #                 self.Q_table[preceding_state][preceding_action] = ((1 - alpha) * self.Q_table[preceding_state][preceding_action] +
    #                                                          alpha * (reward + gamma * next_state_quality))
    #                 next_state_quality = max(self.Q_table[preceding_state]) # re-locate the quality to this updated preceding
    #
    #         self.state = next_state.copy()  # within one episode, it is sequential search
    #         # if self.Q_table[cur_state_index][action]:
    #             # print(self.Q_table[cur_state_index][action], alpha, reward, gamma, next_state_quality)
    #         if reward:  # If we reach a rewarded state, stop learning
    #             self.steps = steps + 1
    #             break  # this break means that the Q_table for the next_state will not be updated.
    #     self.informed_percentage = np.count_nonzero(np.any(self.Q_table != 0, axis=1)) / (2 ** self.N)

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
        return [int(bit) for bit in format(state_index, f'0{self.N}b')]

    def binary_list_to_int(self, state):
        return int(''.join(map(str, state)), 2)

    def generate_state_with_hamming_distance(self, orientation_state, hamming_distance):
        """
        Generate a new state that is at a specific Hamming distance from the original state.

        :param original_state: The original state (e.g., global peak).
        :param hamming_distance: The number of bit flips to make.
        :return: A new state with the specified Hamming distance.
        """
        random.seed(None)
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
        # norm = mcolors.SymLogNorm(linthresh=1, linscale=0.5, vmin=0, vmax=np.max(self.Q_table))
        norm = mcolors.SymLogNorm(linthresh=1, linscale=0.5, vmin=0, vmax=1)
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
    N = 10
    peak = 1
    agent_num = 100
    learning_episodes  = 100
    firm = Organization(N=N, peak=peak, agent_num=agent_num)
    for episode in range(learning_episodes):
        firm.learn(alpha=0.2, gamma=0.9)
        if episode % 20 == 0:
            firm.visualize_1()
            print(np.max(firm.Q_table[-2]))
    print(firm.steps_list)



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


    # Learning with lambda
    # Time to Solution as Organizational Performance
    # add lambda, the length of updated preceding state-action pairs
    # lambda_ = 5
    # steps_lambda_5 = []
    # steps_across_episode_agent = []
    # for _ in range(100):
    #     np.random.seed(None)
    #     q_agent = Agent(N=10, peak=1)
    #     steps_across_episode = []
    #     for index in range(100):
    #         q_agent.learn_with_lambda(alpha=0.2, gamma=0, lambda_=lambda_)
    #         steps_across_episode.append(q_agent.steps)
    #     steps_across_episode_agent.append(steps_across_episode)
    # steps_lambda_5 = np.mean(steps_across_episode_agent, axis=0)
    #
    # # Figure 4: Steps across Episode
    # x = range(1, 101)
    # # x = [50, 100, 150]
    # fig, ax = plt.subplots()
    # ax.spines["left"].set_linewidth(1.5)
    # ax.spines["right"].set_linewidth(1.5)
    # ax.spines["top"].set_linewidth(1.5)
    # ax.spines["bottom"].set_linewidth(1.5)
    # plt.plot(x, steps_lambda_5, "k--", label="$\lambda={0}$".format(5))
    # # plt.plot(x, max_performance, "k-v", label="Max")
    # plt.xlabel("Episode", fontweight='bold', fontsize=12)
    # plt.ylabel('Steps', fontweight='bold', fontsize=12)
    # # plt.xticks(x)
    # # ax.set_ylim(0, 0.7)
    # plt.legend(frameon=False, ncol=1, fontsize=12)
    # plt.savefig(r"Steps_across_episode_lambda.png", transparent=True, dpi=300)
    # plt.show()
    # plt.clf()
    # print("Lambda=5: ", steps_lambda_5)