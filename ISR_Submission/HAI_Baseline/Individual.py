# -*- coding: utf-8 -*-
# @FileName: Individual.py
# Observing PEP 8 coding style

import numpy as np
from Landscape import Landscape


class Individual:
    def __init__(self, N=None, landscape=None, alpha=0.8,
                 gamma=0.9, tau=20.0):
        """
        :param N: number of binary decisions
        :param landscape: assigned conventional NK fitness landscape
        :param alpha: learning rate for SARSA updates (0 to 1)
        :param gamma: discount factor for future rewards (0 to 1)
        :param tau: softmax temperature; higher means more exploration

        Internal knowledge is the subjective state-action value Q(s, a).
        Only NK fitness feedback is used; no AI assistance is modeled.
        """
        if landscape is None:
            raise ValueError("An NK landscape must be provided.")
        if N is None:
            N = landscape.N
        if N != landscape.N or N < 1:
            raise ValueError("N must match the landscape dimension.")
        if not 0 <= alpha <= 1 or not 0 <= gamma <= 1:
            raise ValueError("alpha and gamma must lie between 0 and 1.")
        if tau <= 0:
            raise ValueError("tau must be positive.")

        self.landscape = landscape
        self.N = N
        self.alpha = alpha       # Weight placed on newly observed feedback.
        self.gamma = gamma       # Importance of subsequent state-action values.
        self.tau = tau           # Learning-phase softmax exploration level.

        # 2^N states x N single-bit-flip actions; initially no learned values.
        self.Q_table = np.zeros((2 ** self.N, self.N))

        # Measurement only: which state-action pairs received SARSA updates.
        # Unlike nonzero Q-values, this also counts updates that remain zero.
        self._updated_pairs = np.zeros(self.Q_table.shape, dtype=bool)
        self.total_learning_steps = 0  # Cumulative learning; never reset by episodes.

        self.reset_state()

    def reset_state(self, state=None) -> None:
        """Restart search without resetting accumulated internal knowledge."""
        if state is None:
            self.state = np.random.choice(["0", "1"], self.N).tolist()
        else:
            self.state = [str(bit) for bit in state]
            if (len(self.state) != self.N or
                    any(bit not in ("0", "1") for bit in self.state)):
                raise ValueError("state must contain exactly N binary digits.")

        self.fitness = self.landscape.query_fitness(state=self.state)
        self.best_state = self.state.copy()
        self.best_fitness = self.fitness

        # Trajectory-level performance measures (reset for each new search).
        self.fitness_across_time = [self.fitness]  # Current objective NK fitness.
        self.best_fitness_across_time = [self.best_fitness]  # Best found so far.
        self.steps = 0

        # Textbook SARSA carries the sampled next action into the next step.
        self._next_action = None
        self._next_action_tau = None
        self._next_action_learning = None

    def _state_index(self, state: list) -> int:
        return int("".join(state), 2)

    def _select_action(self, state: list, tau: float) -> int:
        """Select one bit to flip using a numerically stable softmax."""
        q_row = self.Q_table[self._state_index(state)]
        scaled_values = (q_row - np.max(q_row)) / tau
        exp_values = np.exp(scaled_values)
        probabilities = exp_values / np.sum(exp_values)
        return int(np.random.choice(self.N, p=probabilities))

    def search(self, learning=True, tau=None) -> float:
        """Move once, observe immediate NK fitness, and optionally update Q."""
        if tau is None:
            tau = self.tau
        if tau <= 0:
            raise ValueError("tau must be positive.")

        cur_state_index = self._state_index(self.state)
        if (self._next_action is None or
                self._next_action_tau != tau or
                self._next_action_learning != learning):
            action = self._select_action(state=self.state, tau=tau)
        else:
            action = self._next_action

        next_state = self.state.copy()
        next_state[action] = "1" if next_state[action] == "0" else "0"
        next_state_index = self._state_index(next_state)

        reward = self.landscape.query_fitness(state=next_state)  # R = F(s').
        next_action = self._select_action(state=next_state, tau=tau)

        if learning:
            # SARSA: Q(s,a) <- (1-alpha)Q(s,a) + alpha[R+gamma Q(s',a')].
            old_value = self.Q_table[cur_state_index, action]
            next_value = self.Q_table[next_state_index, next_action]
            self.Q_table[cur_state_index, action] = (
                (1 - self.alpha) * old_value
                + self.alpha * (reward + self.gamma * next_value)
            )
            self._updated_pairs[cur_state_index, action] = True
            self.total_learning_steps += 1

        # Always move, including to lower-fitness states (not hill climbing).
        self.state = next_state
        self.fitness = reward
        if self.fitness > self.best_fitness:
            self.best_fitness = self.fitness
            self.best_state = self.state.copy()

        self.steps += 1
        self.fitness_across_time.append(self.fitness)
        self.best_fitness_across_time.append(self.best_fitness)

        self._next_action = next_action
        self._next_action_tau = tau
        self._next_action_learning = learning
        return self.fitness

    def learn(self, steps=100, reset_state=False) -> None:
        """Accumulate SARSA knowledge for a fixed number of search steps."""
        if reset_state:
            self.reset_state()
        for _ in range(steps):
            self.search(learning=True)

    def evaluate(self, steps=100, tau=0.1, reset_state=True) -> float:
        """Assess best-discovered NK fitness without changing internal knowledge."""
        if reset_state:
            self.reset_state()
        for _ in range(steps):
            self.search(learning=False, tau=tau)
        return self.best_fitness

    @property
    def informed_percentage(self) -> float:
        """Share of states with at least one nonzero learned Q-value."""
        return float(np.mean(np.any(self.Q_table != 0, axis=1)))

    def learning_outcomes(self, compute_accuracy=False, tau=None,
                          tolerance=1e-8, max_iterations=10000) -> dict:
        """Return learning mechanisms, independently of search performance.

        informed_percentage: states with >=1 nonzero Q(s,a), / 2^N.
        state_coverage: states with >=1 experienced SARSA update, / 2^N.
        action_coverage: updated state-action pairs, / (2^N * N).

        When compute_accuracy=True, compare learned Q against the exact
        expected discounted return Q^pi on this NK landscape. The policy pi
        is the CURRENT softmax policy induced by Q and the chosen tau.
        Accuracy = 1 - normalized RMSE (1 at perfect prediction, 0 for the
        all-zero prior; negative values mean worse than the zero prior).
        The conditional score only uses state-action pairs actually updated.

        Exact accuracy requires enumerating all 2^N states; use occasionally,
        rather than after every step of a large-N/many-individual simulation.
        """
        experienced_states = np.any(self._updated_pairs, axis=1)

        results = {
            "total_learning_steps": self.total_learning_steps,
            "informed_percentage": self.informed_percentage,
            "state_coverage": float(np.mean(experienced_states)),
            "action_coverage": float(np.mean(self._updated_pairs)),
        }

        if compute_accuracy:
            results.update(self._q_accuracy(tau=tau,
                                            tolerance=tolerance,
                                            max_iterations=max_iterations))
        return results

    def _q_accuracy(self, tau=None, tolerance=1e-8,
                    max_iterations=10000) -> dict:
        """Compare Q(s,a) with exact on-policy Q^pi (measurement only)."""
        if tau is None:
            tau = self.tau  # Same softmax policy used during learning.
        if tau <= 0:
            raise ValueError("tau must be positive.")
        if self.gamma >= 1:
            raise ValueError("Q^pi requires gamma < 1 for ongoing NK rewards.")
        if not 0 < tolerance < 1 or max_iterations < 1:
            raise ValueError("Invalid tolerance or max_iterations.")

        number_states = 2 ** self.N
        states = np.arange(number_states, dtype=np.int64)
        flip_masks = 1 << np.arange(self.N - 1, -1, -1, dtype=np.int64)
        next_indices = states[:, None] ^ flip_masks[None, :]

        # Reuse the original NK landscape's normalized fitness cache.
        fitness = np.fromiter(
            (self.landscape.fitness_cache[format(i, f"0{self.N}b")]
             for i in range(number_states)),
            dtype=float, count=number_states,
        )
        rewards = fitness[next_indices]  # R(s,a) = F(next state).

        scaled = (self.Q_table - self.Q_table.max(axis=1, keepdims=True)) / tau
        exp_values = np.exp(scaled)
        policy = exp_values / exp_values.sum(axis=1, keepdims=True)

        # Bellman policy evaluation: V^pi(s) = sum_a pi(a|s)
        #                          * [F(s') + gamma * V^pi(s')].
        state_values = np.zeros(number_states)
        for _ in range(max_iterations):
            next_values = np.sum(
                policy * (rewards + self.gamma * state_values[next_indices]),
                axis=1,
            )
            if np.max(np.abs(next_values - state_values)) < tolerance:
                state_values = next_values
                break
            state_values = next_values
        else:
            raise RuntimeError("Policy evaluation did not converge.")

        true_q = rewards + self.gamma * state_values[next_indices]
        squared_error = (self.Q_table - true_q) ** 2

        # Normalization: the all-zero Q-table has accuracy 0 by definition.
        reference_scale = float(np.sqrt(np.mean(true_q ** 2)))
        if reference_scale == 0:
            full_accuracy = 1.0 if not np.any(squared_error) else float("nan")
        else:
            full_accuracy = 1 - float(np.sqrt(np.mean(squared_error))) / reference_scale

        if np.any(self._updated_pairs):
            conditional_error = squared_error[self._updated_pairs]
            conditional_true_q = true_q[self._updated_pairs]
            conditional_scale = float(np.sqrt(np.mean(conditional_true_q ** 2)))
            experienced_accuracy = (
                1 - float(np.sqrt(np.mean(conditional_error))) / conditional_scale
                if conditional_scale > 0 else float("nan")
            )
        else:
            experienced_accuracy = float("nan")  # No experience yet.

        return {
            "q_accuracy": full_accuracy,  # All state-action pairs, incl. unknown.
            "experienced_q_accuracy": experienced_accuracy,  # Updated pairs only.
        }

    def describe(self) -> None:
        print("State: {0}, Fitness: {1}".format(self.state, self.fitness))
        print("Best State: {0}, Best Fitness: {1}".format(
            self.best_state, self.best_fitness))
        print("SARSA Q-table shape: ", self.Q_table.shape)
        print("Learning Outcomes: ", self.learning_outcomes())


if __name__ == '__main__':
    # Test Example
    import time

    t0 = time.time()

    np.random.seed(1000)
    search_iteration = 100

    N = 9
    K = 2

    landscape = Landscape(N=N, K=K)
    individual = Individual(N=N, landscape=landscape,
                            alpha=0.8, gamma=0.9, tau=20.0)

    individual.learn(steps=search_iteration)
    individual.describe()

    # Measure both knowledge coverage and long-run Q-value accuracy.
    print("Learning Outcomes (with accuracy): ",
          individual.learning_outcomes(compute_accuracy=True))

    # Evaluate without updating Q or learning-related measures.
    evaluation_fitness = individual.evaluate(steps=search_iteration,
                                             tau=0.1, reset_state=True)
    print("Evaluation Best Fitness: ", evaluation_fitness)

    t1 = time.time()
    print(time.strftime("%H:%M:%S", time.gmtime(t1 - t0)))
