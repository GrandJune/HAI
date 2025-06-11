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
    def __init__(self, N, reality=None):
        self.N = N
        self.index = None
        self.Q_table = np.zeros((2 ** self.N, self.N))
        self.reality = reality
        self.next_action = None
        self.state = None
        self.initialize()
        self.max_step = 10000  # make sure an episode can end with peaks
        self.knowledge = 0  # the percentage of states that are informed
        self.knowledge_quality = 0 # the percentage of accurate actions that are informed; informed accurate actions / accurate actions
        self.performance = 0
        self.steps = 0
        self.search_trajectory = []

    def initialize(self):
        self.next_action = None
        self.state = [random.randint(0, 1) for _ in range(self.N)]
        # self.state = [0 for _ in range(self.N)]

    def learn(self, tau=20.0, alpha=0.8, gamma=0.9, evaluation=False):
        """
        One episode concludes with local or global peaks and update its antecedent Q(s, a).
        Larger Tau: exploration (at 30, random walk);  Smaller Tau: exploitation
        :param tau: temperature regulates how sensitive the probability of choosing a given action is to the estimated Q
        :param state: current state, int
        :param alpha: learning rate (cf. Denrell 2004)
        :param gamma: emphasis on positional value (cf. Denrell 2004); gamma = 0.9 is best in Denrell 2004
        :return:
        """
        self.initialize()
        for perform_step in range(self.max_step):
            cur_state_index = self.binary_list_to_int(self.state)
            q_row = self.Q_table[cur_state_index].copy()
            exp_prob_row = np.exp(q_row / tau)
            prob_row = exp_prob_row / np.sum(exp_prob_row)
            action = np.random.choice(range(self.N), p=prob_row)
            self.search_trajectory.append([cur_state_index, action])
            next_state = self.state.copy()
            if action < self.N:
                next_state[action] = 1 - self.state[action]
            next_state_index = self.binary_list_to_int(next_state)
            next_q_row = self.Q_table[next_state_index]
            next_exp_prob_row = np.exp(next_q_row / tau)
            next_prob_row = next_exp_prob_row / np.sum(next_exp_prob_row)
            next_action = np.random.choice(range(self.N), p=next_prob_row)
            next_state_quality = self.Q_table[next_state_index][next_action]
            reward = self.reality.payoff_map[next_state_index]
            if reward:  # peak
                self.performance = reward
                self.steps = perform_step + 1
                self.Q_table[cur_state_index][action] = (1 - alpha) * self.Q_table[cur_state_index][action] + alpha * reward
                # Re-initialize
                self.initialize()
                break  # this break means that the Q_table for the next_state (i.e., peak) will not be updated.
            else:  # non-peak
                self.Q_table[cur_state_index][action] = (1 - alpha) * self.Q_table[cur_state_index][action] + alpha * gamma * next_state_quality
                # Sequential search
                self.state = next_state.copy()

        if evaluation:
            self.knowledge = np.count_nonzero(np.any(self.Q_table != 0, axis=1)) / (2 ** self.N)
            self.knowledge_quality = self.get_Q_table_quality()

    # def learn_with_parrot_0(self, tau=20.0, alpha=0.8, gamma=0.9, valence=50, parrot=None, evaluation=False):
    #     self.initialize()
    #     for perform_step in range(self.max_step):
    #         cur_state_index = self.binary_list_to_int(self.state)
    #         q_row = self.Q_table[cur_state_index]
    #         # first examine whether AI advice is available
    #         suggested_action = parrot.suggest(self.state)
    #         # if an input of state returns valid guidance, then that state is considered guided, and valence is assigned.
    #         if suggested_action is not None:
    #             q_row[suggested_action] += gamma * valence  # inject a valence bonus; bia the attention toward guidance
    #             # This discounts the valence, treating it as a future-oriented value. it is a promising option
    #             # add gamma; not as much as global peak; because gamma is a discount factor compared to reward
    #         shifted_q = q_row - np.max(q_row)  # avoid overflow
    #         exp_prob_row = np.exp(shifted_q / tau)
    #         prob_row = exp_prob_row / np.sum(exp_prob_row)
    #         action = np.random.choice(range(self.N), p=prob_row)
    #         followed_guidance = (suggested_action is not None and action == suggested_action)
    #         self.search_trajectory.append([cur_state_index, action])
    #         next_state = self.state.copy()
    #         next_state[action] = 1 - self.state[action]  # flipping
    #         next_state_index = self.binary_list_to_int(next_state)
    #         reward = self.reality.payoff_map[next_state_index]  # equal to non-zero when next state is peaks
    #         if reward:  # peak
    #             self.performance = reward
    #             self.steps = perform_step + 1
    #             self.Q_table[cur_state_index][action] = (1 - alpha) * self.Q_table[cur_state_index][action] + alpha * reward
    #             # Re-initialize
    #             self.initialize()
    #             break
    #         elif followed_guidance:
    #             self.Q_table[cur_state_index][action] = (1 - alpha) * self.Q_table[cur_state_index][action] + alpha * gamma * valence
    #         else:  # without guidance from parrot
    #             # the next proper action; to calculate the quality of the next state
    #             next_suggestion = parrot.suggest(next_state)
    #             if next_suggestion is not None:
    #                 next_state_quality = valence
    #             else:
    #                 next_q_row = self.Q_table[next_state_index]
    #                 shifted_next_q = next_q_row - np.max(next_q_row)  # avoid overflow
    #                 next_exp_prob_row = np.exp(shifted_next_q / tau)
    #                 next_prob_row = next_exp_prob_row / np.sum(next_exp_prob_row)
    #                 next_action = np.random.choice(range(self.N), p=next_prob_row)
    #                 next_state_quality = self.Q_table[next_state_index][next_action]
    #             self.Q_table[cur_state_index][action] = (1 - alpha) * self.Q_table[cur_state_index][action] + alpha * gamma * next_state_quality
    #             self.state = next_state.copy()
    #
    #     if evaluation:
    #         self.knowledge = np.count_nonzero(np.any(self.Q_table != 0, axis=1)) / (2 ** self.N)
    #         self.knowledge_quality = self.get_Q_table_quality()

    def learn_with_parrot(self, tau=20.0, alpha=0.8, gamma=0.9, valence=50, parrot=None, evaluation=False):
        self.initialize()
        for perform_step in range(self.max_step):
            cur_state_index = self.binary_list_to_int(self.state)
            q_row = self.Q_table[cur_state_index].copy()  # use a copy to avoid altering base Q-values

            # Check parrot suggestion and boost Q for softmax attention
            suggested_action = parrot.suggest(self.state)
            if suggested_action is not None:
                q_row[suggested_action] += gamma * valence  # temporary boost

            # Sample action from softmax-biased distribution
            shifted_q = q_row - np.max(q_row)
            exp_prob_row = np.exp(shifted_q / tau)
            prob_row = exp_prob_row / np.sum(exp_prob_row)
            action = np.random.choice(range(self.N), p=prob_row)

            # Track whether guidance was followed
            followed_guidance = (suggested_action is not None and action == suggested_action)
            self.search_trajectory.append([cur_state_index, action])

            # Transition to next state
            next_state = self.state.copy()
            next_state[action] = 1 - self.state[action]
            next_state_index = self.binary_list_to_int(next_state)
            reward = self.reality.payoff_map[next_state_index]

            if reward:  # Reached a peak
                self.performance = reward
                self.steps = perform_step + 1
                self.Q_table[cur_state_index][action] = (1 - alpha) * self.Q_table[cur_state_index][
                    action] + alpha * reward
                self.initialize()
                break
            elif followed_guidance:  # Reinforce parrot-following path
                self.Q_table[cur_state_index][action] = (1 - alpha) * self.Q_table[cur_state_index][
                    action] + alpha * gamma * valence
            else:
                # Estimate next state value
                next_suggestion = parrot.suggest(next_state)
                if next_suggestion is not None:
                    next_state_quality = valence
                else:
                    next_q_row = self.Q_table[next_state_index].copy()
                    shifted_next_q = next_q_row - np.max(next_q_row)  # avoid overflow
                    next_exp_prob_row = np.exp(shifted_next_q / tau)
                    next_prob_row = next_exp_prob_row / np.sum(next_exp_prob_row)
                    next_action = np.random.choice(range(self.N), p=next_prob_row)
                    next_state_quality = self.Q_table[next_state_index][next_action]
                self.Q_table[cur_state_index][action] = (1 - alpha) * self.Q_table[cur_state_index][
                    action] + alpha * gamma * next_state_quality

            self.state = next_state.copy()

        if evaluation:
            self.knowledge = np.count_nonzero(np.any(self.Q_table != 0, axis=1)) / (2 ** self.N)
            self.knowledge_quality = self.get_Q_table_quality()


    # def learn_with_fading_valence_parrot(self, tau=20.0, alpha=0.8, gamma=0.9,
    #                              initial_valence=50, parrot=None,
    #                              evaluation=False, decay_rate=0.05, valence_floor=10):
    #     self.initialize()
    #     for perform_step in range(self.max_step):
    #         # Compute current valence via exponential decay
    #         valence = max(initial_valence * np.exp(-decay_rate * perform_step), valence_floor)
    #
    #         cur_state_index = self.binary_list_to_int(self.state)
    #         q_row = self.Q_table[cur_state_index]
    #
    #         # Step 1: Determine the current action
    #         suggested_action = parrot.suggest(self.state)
    #         if suggested_action:
    #             action = suggested_action
    #         else:
    #             if self.next_action:
    #                 action = self.next_action
    #             else:
    #                 exp_prob_row = np.exp(q_row / tau)
    #                 prob_row = exp_prob_row / np.sum(exp_prob_row)
    #                 action = np.random.choice(range(self.N), p=prob_row)
    #
    #         self.search_trajectory.append([cur_state_index, action])
    #         next_state = self.state.copy()
    #         next_state[action] = 1 - self.state[action]  # flip bit
    #         next_state_index = self.binary_list_to_int(next_state)
    #
    #         # Step 2: Determine quality of next state
    #         suggested_next_action = parrot.suggest(next_state)
    #         if suggested_next_action is not None:
    #             self.next_action = suggested_next_action
    #         else:
    #             next_q_row = self.Q_table[next_state_index]
    #             next_exp_prob_row = np.exp(next_q_row / tau)
    #             next_prob_row = next_exp_prob_row / np.sum(next_exp_prob_row)
    #             self.next_action = np.random.choice(range(self.N), p=next_prob_row)
    #             next_state_quality = self.Q_table[next_state_index][self.next_action]
    #
    #         # Step 3: Determine the reward and update Q-table
    #         reward = self.reality.payoff_map[next_state_index]
    #         if reward:
    #             self.performance = reward
    #             self.steps = perform_step + 1
    #             self.Q_table[cur_state_index][action] = (1 - alpha) * self.Q_table[cur_state_index][
    #                 action] + alpha * reward
    #             self.initialize()
    #             break
    #         elif suggested_next_action is not None:
    #             self.Q_table[cur_state_index][action] = (1 - alpha) * self.Q_table[cur_state_index][
    #                 action] + alpha * gamma * valence
    #             self.state = next_state.copy()
    #         else:
    #             self.Q_table[cur_state_index][action] = (1 - alpha) * self.Q_table[cur_state_index][
    #                 action] + alpha * gamma * next_state_quality
    #             self.state = next_state.copy()
    #
    #     if evaluation:
    #         self.knowledge = np.count_nonzero(np.any(self.Q_table != 0, axis=1)) / (2 ** self.N)
    #         self.knowledge_quality = self.get_Q_table_quality()

    def learn_with_dynamic_trust_parrot(self, tau=20.0, alpha=0.8, gamma=0.9,
                                       valence=50, parrot=None, evaluation=False,
                                       trust=1.0):
        self.initialize()
        for perform_step in range(self.max_step):
            cur_state_index = self.binary_list_to_int(self.state)
            q_row = self.Q_table[cur_state_index]

            # Step 1: Decide action with trust modulation
            trust_flag = False
            suggested_action = parrot.suggest(self.state)
            if suggested_action is not None and np.random.rand() < trust:
                action = suggested_action
                trust_flag = True
            else:  # only if no trust, turn to internally recorded next action
                if self.next_action is not None:
                    action = self.next_action
                else:
                    exp_prob_row = np.exp(q_row / tau)
                    prob_row = exp_prob_row / np.sum(exp_prob_row)
                    action = np.random.choice(range(self.N), p=prob_row)

            self.search_trajectory.append([cur_state_index, action])
            next_state = self.state.copy()
            next_state[action] = 1 - self.state[action]
            next_state_index = self.binary_list_to_int(next_state)

            # Step 2: Decide next_action with trust modulation
            suggested_next_action = parrot.suggest(next_state)
            if suggested_next_action is not None and trust_flag is True:
                self.next_action = suggested_next_action
            else:
                next_q_row = self.Q_table[next_state_index]
                next_exp_prob_row = np.exp(next_q_row / tau)
                next_prob_row = next_exp_prob_row / np.sum(next_exp_prob_row)
                self.next_action = np.random.choice(range(self.N), p=next_prob_row)
                next_state_quality = self.Q_table[next_state_index][self.next_action]

            # Step 3: Reward handling and Q update
            reward = self.reality.payoff_map[next_state_index]
            if reward:
                self.performance = reward
                self.steps = perform_step + 1
                self.Q_table[cur_state_index][action] = (1 - alpha) * self.Q_table[cur_state_index][
                    action] + alpha * reward
                self.initialize()
                break
            elif suggested_next_action is not None and trust_flag is True:
                self.Q_table[cur_state_index][action] = (1 - alpha) * self.Q_table[cur_state_index][
                    action] + alpha * gamma * valence
                self.state = next_state.copy()
            else:
                # without trust on external parrots, relying on internal evaluation
                self.Q_table[cur_state_index][action] = (1 - alpha) * self.Q_table[cur_state_index][
                    action] + alpha * gamma * next_state_quality
                self.state = next_state.copy()

        if evaluation:
            self.knowledge = np.count_nonzero(np.any(self.Q_table != 0, axis=1)) / (2 ** self.N)
            self.knowledge_quality = self.get_Q_table_quality()


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

    def get_Q_table_quality(self):
        """
        Summarize the Q-table's quality by calculating the average proportion of positive Q-values
        among accurate actions (actions that flip bits differing from the global peak), across all states.
        """
        global_peak = [1] * self.N
        proportions = []

        for state_index in range(2 ** self.N):
            state = self.int_to_binary_list(state_index)

            # skip if already at the global peak
            if state == global_peak:
                continue

            # accurate actions = dimensions where the state differs from the global peak
            accurate_actions = [i for i in range(self.N) if state[i] != global_peak[i]]
            if not accurate_actions:
                continue

            q_row = self.Q_table[state_index]
            positive_q_count = sum(q_row[a] > 0 for a in accurate_actions)
            proportion = positive_q_count / len(accurate_actions)
            proportions.append(proportion)

        overall_quality = np.mean(proportions) if proportions else 0.0
        return overall_quality

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
    random.seed(None)
    from Reality import Reality
    from Parrot import Parrot
    repeat = 1
    reward_list, step_list = [], []
    reality = Reality(N=10, global_peak_value=50, local_peak_value=10)
    parrot = Parrot(N=10, reality=reality, coverage=1.0, accuracy=1.0)

    agent = Agent(N=10, reality=reality)
    for index in range(300):
        agent.learn_with_parrot(tau=20, alpha=0.8, gamma=0.9, valence=50, parrot=parrot)
        if index % 100 == 0:
            agent.visualize_1()
            flat_list = [item for sublist in agent.Q_table for item in sublist]
            top_10 = sorted(flat_list, reverse=True)[:10]
            print(top_10)

    # agent_2 = Agent(N=10, reality=reality)
    # for _ in range(300):
    #     agent_2.learn(tau=20, alpha=0.8, gamma=0.9)
    # agent_2.get_Q_table_quality()


    # if agent.performance == 50:
    #     global_indicator += 1
    # elif agent.performance == 10:
    #     local_indicator += 1
    # step_list.append(agent.steps)
    # print(agent.Q_table[agent.reality.local_peak_indices[0]], agent.Q_table[agent.reality.local_peak_indices[1]], agent.Q_table[-1])
    # print("Global: ", global_indicator / repeat, "Local: ", local_indicator / repeat)
    # print("Steps: ", sum(step_list) / len(step_list) )

    # aggregation test
    # agent = Agent(N=10, high_peak=50, low_peak=10)
    # for index in range(350):
    #     agent.learn(tau=20, alpha=0.2, gamma=0.9)
    # agent.visualize_1()
    # exploration_performance_list, exploration_step_list = [], []
    # for _ in range(200):
    #     agent.evaluate(tau=20)  # exploration
    #     exploration_performance_list.append(agent.performance)
    #     exploration_step_list.append(agent.steps)
    #     # agent.visualize(search_trajectory=agent.search_trajectory)
    # exploration_performance = sum([1 if reward == 50 else 0 for reward in exploration_performance_list]) / len(exploration_performance_list)
    # exploration_step = sum(exploration_step_list) / len(exploration_step_list)
    # print("Exploration: ", exploration_performance, exploration_step)
    #
    # exploitation_performance_list, exploitation_step_list = [], []
    # for _ in range(200):
    #     agent.evaluate(tau=0.1)  # exploitation
    #     exploitation_performance_list.append(agent.performance)
    #     exploitation_step_list.append(agent.steps)
    #     # agent.visualize(search_trajectory=agent.search_trajectory)
    # exploitation_performance = sum([1 if reward == 50 else 0 for reward in exploitation_performance_list]) / len(exploitation_performance_list)
    # exploitation_step = sum(exploitation_step_list) / len(exploitation_step_list)
    # print("Exploitation: ", exploitation_performance, exploitation_step)


    # print(agent.reality)
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
    #         agent = Agent(N=10, global_peak=50, local_peaks=[10])
    #         for _ in range(learning_length):
    #             agent.learn(tau=20, alpha=0.8, gamma=0.9)
    #         reward, step = agent.perform(tau=20)
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


