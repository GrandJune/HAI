import numpy as np
from Parrot import Parrot
from Reality import Reality
from Agent import Agent

def func(agent_num, learning_episode):
    parrot = Parrot(N=10)
    Q_table_list = []
    for _ in range(agent_num):
        agent = Agent(N=10, global_peak=50, local_peaks=10)
        pair_agent = Agent(N=10, global_peak=50, local_peaks=10)
        pair_agent.state = agent.state.copy()
        for _ in range(learning_episode):
            agent.learn(tau=20, alpha=0.8, gamma=0.9)
            Q_table_list.append(agent.Q_table)
            parrot.aggregate_from_data(Q_table_list=Q_table_list)
            pair_agent.learn_with_parrot(tau=20, alpha=0.8, gamma=0.9, parrot=parrot)
            Q_table_list.append(pair_agent.Q_table)





# Example: Define N
N = 5  # Example value
Q_table = np.zeros((2 ** N, N + 1))

# Modify some rows to be non-zero for demonstration
Q_table[3, 2] = 1
Q_table[7, 3] = 2

# Count non-zero rows
non_zero_rows = np.count_nonzero(np.any(Q_table != 0, axis=1))

# Calculate percentage
percentage = (non_zero_rows / Q_table.shape[0]) * 100

print(f"Percentage of non-zero rows: {percentage:.2f}%")
print(2 / 2 ** 5)
# -*- coding: utf-8 -*-
# @Time     : 3/13/2025 19:44
# @Author   : Junyi
# @FileName: Test.py
# @Software  : PyCharm
# Observing PEP 8 coding style
