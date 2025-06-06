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


import numpy as np

def aggregate_q_bayesian(q_list, mu0=0.0, kappa0=1.0, alpha0=1.0, beta0=1.0):
    """
    Bayesian aggregation of Q-tables using a Normal-Inverse-Gamma conjugate prior.

    Parameters:
    - q_list: list or array-like of shape (T, n, m), where each entry is an (n x m) Q-table.
    - mu0: prior mean for Q values.
    - kappa0: strength of prior mean.
    - alpha0, beta0: shape and scale for the prior on variance.

    Returns:
    - mu_n: posterior mean Q-table of shape (n, m).
    - var_n: posterior variance Q-table of shape (n, m).
    """
    Q = np.stack(q_list, axis=0)  # shape (T, n, m)
    N = Q.shape[0]

    # Sample statistics
    xbar = Q.mean(axis=0)                     # shape (n, m)
    S = np.sum((Q - xbar)**2, axis=0)         # sum of squared deviations

    # Posterior hyperparameters
    kappa_n = kappa0 + N
    mu_n = (kappa0 * mu0 + N * xbar) / kappa_n
    alpha_n = alpha0 + N / 2
    beta_n = beta0 + 0.5 * S + (kappa0 * N * (xbar - mu0)**2) / (2 * kappa_n)

    # Posterior variance (mean of the Inverse-Gamma)
    var_n = beta_n / (alpha_n - 1)

    return mu_n, var_n

# Example usage:
q_tables = [np.random.rand(5, 3) for _ in range(10)]  # 10 tables of size 5x3
aggregated_mean, aggregated_var = aggregate_q_bayesian(q_tables)
average_table = np.mean(q_tables, axis=0)
print("Aggregated mean:")
print(aggregated_mean)
print(average_table)
print("Aggregated variance:")
print(aggregated_var)



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
