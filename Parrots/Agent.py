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
from Reality import Reality

class Agent:
    def __init__(self, N, global_peak=None, local_peaks=None):
        self.N = N
        self.Q_table = np.zeros((2 ** self.N, self.N))
        self.reality = Reality(N=N, global_peak = global_peak, local_peaks = local_peaks)
        self.state = [random.randint(0, 1) for _ in range(self.N)]
        self.next_action = None
        self.max_step = 10000  # make sure an episode can end with peaks
        self.knowledge = 0  # the percentage of states that are informed
        self.performance = 0
        self.steps = 0
        self.search_trajectory = []

    def learn(self, tau=20.0, alpha=0.8, gamma=0.9):
        """
        One episode concludes with local or global peaks and update its antecedent Q(s, a).
        Larger Tau: exploration (at 30, random walk);  Smaller Tau: exploitation
        :param tau: temperature regulates how sensitive the probability of choosing a given action is to the estimated Q
        :param state: current state, int
        :param alpha: learning rate (cf. Denrell 2004)
        :param gamma: emphasis on positional value (cf. Denrell 2004); gamma = 0.9 is best in Denrell 2004
        :return:
        """
        self.search_trajectory = []
        for perform_step in range(self.max_step):
            cur_state_index = self.binary_list_to_int(self.state)
            q_row = self.Q_table[cur_state_index]
            if self.next_action:
                action = self.next_action
            else:
                exp_prob_row = np.exp(q_row / tau)
                prob_row = exp_prob_row / np.sum(exp_prob_row)
                action = np.random.choice(range(self.N), p=prob_row)
            self.search_trajectory.append([cur_state_index, action])
            next_state = self.state.copy()
            if action < self.N:
                next_state[action] = 1 - self.state[action]
            next_state_index = int(''.join(map(str, next_state)), 2)
            next_q_row = self.Q_table[next_state_index]
            next_exp_prob_row = np.exp(next_q_row / tau)
            next_prob_row = next_exp_prob_row / np.sum(next_exp_prob_row)
            next_action = np.random.choice(range(self.N), p=next_prob_row)
            next_state_quality = self.Q_table[next_state_index][next_action]
            reward = self.reality.payoff_map[next_state_index]  # equal to non-zero when next state is peaks
            # self.Q_table[cur_state_index][action] = ((1 - alpha) * self.Q_table[cur_state_index][action] +
            #                                          alpha * (reward + gamma * next_state_quality))
            if reward:  # peak
                self.performance = reward
                self.steps = perform_step + 1
                self.Q_table[cur_state_index][action] = (1 - alpha) * self.Q_table[cur_state_index][action] + alpha * reward
                # Re-initialize
                self.next_action = None
                self.state = [random.randint(0, 1) for _ in range(self.N)]
                break  # this break means that the Q_table for the next_state (i.e., peak) will not be updated.
            else:  # non-peak
                self.Q_table[cur_state_index][action] = (1 - alpha) * self.Q_table[cur_state_index][action] + alpha * gamma * next_state_quality
                # Sequential search
                self.state = next_state.copy()
                self.next_action = next_action

        self.knowledge = np.count_nonzero(np.any(self.Q_table != 0, axis=1)) / (2 ** self.N)

    def learn_with_parrot(self, tau=20.0, alpha=0.8, gamma=0.9, parrot=None):
        """
        One episode concludes with local or global peaks and update its antecedent Q(s, a).
        Larger Tau: exploration (at 30, random walk);  Smaller Tau: exploitation
        :param tau: temperature regulates how sensitive the probability of choosing a given action is to the estimated Q
        :param state: current state, int
        :param alpha: learning rate (cf. Denrell 2004)
        :param gamma: emphasis on positional value (cf. Denrell 2004); gamma = 0.9 is best in Denrell 2004
        :return:
        """
        self.search_trajectory = []
        # temp_Q_table = np.add(self.Q_table, parrot.Q_table)
        for perform_step in range(self.max_step):
            cur_state_index = self.binary_list_to_int(self.state)
            q_row = self.Q_table[cur_state_index]
            # organic action
            if self.next_action:
                organic_action = self.next_action
            else:
                exp_prob_row = np.exp(q_row / tau)
                prob_row = exp_prob_row / np.sum(exp_prob_row)
                organic_action = np.random.choice(range(self.N), p=prob_row)
            # suggested action
            parrot_action = np.argmax(parrot.Q_table[cur_state_index])

            if self.Q_table[cur_state_index][organic_action] > 0:  # initially only consider the knowledge/confidence of the organic action
            # if the user has the knowledge to rule out the suggestion
            # based on its knowledge, it would choose the organic one.
            # should we include the parrot's knowledge in the decision? should we include user's knowledge in the suggestion?
                final_action = organic_action
            else:
            # if the user has no knowledge to reject the suggestion
                final_action = parrot_action

            self.search_trajectory.append([cur_state_index, final_action])
            next_state = self.state.copy()
            # suggestion may also lead to unexpected positional value; lead to unforeseen considerations
            next_state[final_action] = 1 - self.state[final_action]  # flipping
            next_state_index = int(''.join(map(str, next_state)), 2)
            next_q_row = self.Q_table[next_state_index]
            next_exp_prob_row = np.exp(next_q_row / tau)
            next_prob_row = next_exp_prob_row / np.sum(next_exp_prob_row)
            next_action = np.random.choice(range(self.N), p=next_prob_row)
            next_state_quality = self.Q_table[next_state_index][next_action]
            reward = self.reality.payoff_map[next_state_index]  # equal to non-zero when next state is peaks
            if reward:  # peak
                self.performance = reward
                self.steps = perform_step + 1
                self.Q_table[cur_state_index][final_action] = (1 - alpha) * self.Q_table[cur_state_index][final_action] + alpha * reward
                # Re-initialize
                self.state = [0] * self.N
                self.next_action = None
                break
            else:  # non-peak
                self.Q_table[cur_state_index][final_action] = (1 - alpha) * self.Q_table[cur_state_index][final_action] + alpha * gamma * next_state_quality
                # Sequential search
                self.state = next_state.copy()
                self.next_action = next_action

        self.knowledge = np.count_nonzero(np.any(self.Q_table != 0, axis=1)) / (2 ** self.N)

    def int_to_binary_list(self, state_index):
        return [int(bit) for bit in format(state_index, f'0{self.N}b')]

    def binary_list_to_int(self, state):
        return int(''.join(map(str, state)), 2)

    def generate_state_with_hamming_distance(self, orientation_state, hamming_distance):
        """
        Generate a new state that is at a specific Hamming distance from the original state.
        :param orientation_state: the original state (e.g., global peak).
        :param hamming_distance: the number of bit flips to make.
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
    from Parrot import Parrot
    import time
    t0 = time.time()
    random.seed(None)
    N = 10  # problem dimension
    tau = 20  # temperature parameter
    alpha = 0.8  # learning rate
    gamma = 0.9  # discount factor
    global_peak = 50  # as per (Fang, 2009)
    local_peaks = [10, 10, 10]  # add more local peaks to increase complexity
    agent_num = 50
    learning_length = 50
    parrot = Parrot(N=N)
    # varying learning length
    # := varying the data maturity feeded into parrot
    Q_table_list = []
    organic_performance_list, organic_knowledge_list, organic_steps_list = [], [], []
    for _ in range(agent_num):
        agent = Agent(N=N, global_peak=global_peak, local_peaks=local_peaks)
        for episode in range(learning_length + 1):
            agent.learn(tau=tau, alpha=alpha, gamma=gamma)
        Q_table_list.append(agent.Q_table)
        organic_performance_list.append(agent.performance)
        organic_knowledge_list.append(agent.knowledge)
        organic_steps_list.append(agent.steps)
    organic_performance_list = [1 if each == 50 else 0 for each in
                                organic_performance_list]  # the likelihood of finding global peak
    organic_performance = sum(organic_performance_list) / agent_num
    organic_knowledge = sum(organic_knowledge_list) / agent_num
    organic_steps = sum(organic_steps_list) / agent_num

    parrot.aggregate_from_data(Q_table_list=Q_table_list)
    parrot.visualize_1()
    pair_performance_list, pair_knowledge_list, pair_steps_list = [], [], []
    for _ in range(1):
        pair_agent = Agent(N=N, global_peak=global_peak, local_peaks=local_peaks)
        for episode in range(learning_length + 1):
            pair_agent.learn_with_parrot(tau=tau, alpha=alpha, gamma=gamma, parrot=parrot)
            if episode == learning_length:
                pair_agent.visualize(pair_agent.search_trajectory)
        pair_performance_list.append(pair_agent.performance)
        pair_knowledge_list.append(pair_agent.knowledge)
        pair_steps_list.append(pair_agent.steps)
    pair_performance_list = [1 if each == 50 else 0 for each in
                             pair_performance_list]  # the likelihood of finding global peak
    pair_performance = sum(pair_performance_list) / len(pair_performance_list)
    pair_knowledge = sum(pair_knowledge_list) / len(pair_knowledge_list)
    pair_steps = sum(pair_steps_list) / agent_num
    print(pair_knowledge_list)
    print(organic_knowledge, pair_knowledge, parrot.knowledge)
    t1 = time.time()
    print(time.strftime("%H:%M:%S", time.gmtime(t1 - t0)))  # Duration

