# -*- coding: utf-8 -*-
# @Time     : 4/25/2025 20:52
# @Author   : Junyi
# @FileName: Bayesian.py
# @Software  : PyCharm
# Observing PEP 8 coding style
import numpy as np
from scipy.stats import multivariate_normal
from scipy.special import expit  # Sigmoid function


class BayesianQAggregator:
    def __init__(self, num_states, num_actions, prior_mean=None, prior_cov=None):
        self.num_states = num_states
        self.num_actions = num_actions
        self.dim = num_states * num_actions

        # Prior over theta (ground-truth Q-table flattened)
        self.mean = prior_mean if prior_mean is not None else np.zeros(self.dim)
        self.cov = prior_cov if prior_cov is not None else np.eye(self.dim)

        self.Qs = []
        self.rewards = []

    def flatten_Q(self, Q):
        return np.array(Q).flatten()

    def add_observation(self, Q, performance):
        y = 1 if performance >= 50 else 0  # Global peak = 1, Local peak = 0
        self.Qs.append(self.flatten_Q(Q))
        self.rewards.append(y)

    def thompson_sample(self):
        return multivariate_normal.rvs(mean=self.mean, cov=self.cov)

    def update_posterior(self):
        X = np.array(self.Qs)
        y = np.array(self.rewards)

        # Compute likelihood using logistic regression assumption
        theta_sample = self.thompson_sample()
        probs = expit(X @ theta_sample)

        # Calculate gradients for posterior update (Laplace approximation for simplicity)
        W = np.diag(probs * (1 - probs))
        cov_inv = np.linalg.inv(self.cov)

        # Posterior covariance and mean update (Bayesian logistic regression)
        S_inv = cov_inv + X.T @ W @ X
        self.cov = np.linalg.inv(S_inv)
        self.mean = self.cov @ (cov_inv @ self.mean + X.T @ (y - probs))

    def aggregate_Q(self):
        # Sample candidate theta and reshape into Q-table shape
        theta_sample = self.thompson_sample()
        Q_agg = theta_sample.reshape(self.num_states, self.num_actions)
        return Q_agg

# Example Q-tables from human agents (2 states, 2 actions for simplicity)
human_Qs = [
    [[1, 2], [3, 4]],   # Agent 1
    [[5, 1], [2, 3]],   # Agent 2
    [[0, 1], [4, 5]],   # Agent 3
    [[6, 2], [5, 7]]    # Agent 4 (global peak)
]

performances = [10, 10, 10, 50]  # Only Agent 4 reached global peak

# Initialize aggregator
num_states, num_actions = 2, 2
aggregator = BayesianQAggregator(num_states, num_actions)

# Add observations
for Q, perf in zip(human_Qs, performances):
    aggregator.add_observation(Q, perf)

# Update posterior using observed Q-tables and performances
aggregator.update_posterior()

# Generate aggregated Q-table for AI agent
aggregated_Q = aggregator.aggregate_Q()

print("Aggregated Q-table for AI agent:")
print(aggregated_Q)
