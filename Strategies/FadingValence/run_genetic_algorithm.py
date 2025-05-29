# -*- coding: utf-8 -*-
# @Time     : 2/16/2025 20:08
# @Author   : Junyi
# @FileName: Q_learning.py
# @Software  : PyCharm
# Observing PEP 8 coding style
import numpy as np
from Agent import Agent
from Parrot import Parrot
import multiprocessing as mp
import time
import copy
from multiprocessing import Semaphore
import pickle
from Reality import Reality


def func(agent_num=None, decay_rate=None, loop=None, return_dict=None, sema=None):
    np.random.seed(None)
    N = 10 # problem dimension
    tau = 20  # temperature parameter
    alpha = 0.8  # learning rate
    gamma = 0.9 # discount factor
    learning_length = 100
    population_size = 100
    valence_bounds = (0, 50)
    mutation_rate = 0.1
    global_peak_value = 50 # as per (Fang, 2009)
    local_peak_value = 10 # add more local peaks to increase complexity
    generation_per_block = 20
    episodes_per_block = 10

    # Initialize reality and parrot; fixed
    reality = Reality(N=N, global_peak_value=global_peak_value, local_peak_value=local_peak_value)
    parrot = Parrot(N=N, reality=reality, coverage=1.0, accuracy=1.0)
    # Initial population and agents
    valence_population = np.random.uniform(valence_bounds[0], valence_bounds[1], population_size)
    agents_list = [Agent(N=N, reality=reality) for _ in range(population_size)]
    # Storage
    q_table_snapshots = [copy.deepcopy(agent.Q_table) for agent in agents_list]
    valence_evolution = []

    for block in range(learning_length // episodes_per_block):
        for generation in range(generation_per_block):
            fitness_list = []
            # Simulate learning for current valence
            for i, agent in enumerate(agents_list):
                agent.Q_table = copy.deepcopy(q_table_snapshots[i])  # Reset Q-table to prior state
                agent.performance = 0
                for _ in range(episodes_per_block):
                    agent.learn_with_parrot(tau=tau, alpha=alpha, gamma=gamma,
                                            valence=valence_population[i], parrot=parrot, evaluation=False)
                fitness_list.append(1 / agent.steps if agent.steps > 0 else 1e6)

            # GA: selection, crossover, mutation
            fitness_array = np.array(fitness_list)
            top_indices = np.argsort(fitness_array)[-population_size // 2:]  # top 50%
            survivors = valence_population[top_indices]

            # Crossover
            new_population = []
            while len(new_population) < population_size:
                parents = np.random.choice(survivors, 2)
                crossover_point = np.random.rand()
                child = crossover_point * parents[0] + (1 - crossover_point) * parents[1]
                new_population.append(child)

            valence_population = np.array(new_population)

            # Mutation
            mutation_mask = np.random.rand(population_size) < mutation_rate
            valence_population[mutation_mask] += np.random.normal(0, 5, size=np.sum(mutation_mask))
            valence_population = np.clip(valence_population, valence_bounds[0], valence_bounds[1])

        # Save the best valence of the block
        best_index = np.argmax(fitness_list)
        best_valence = valence_population[best_index]
        valence_evolution.append(best_valence)

        # Advance Q-table state for agents with best valence found
        for i, agent in enumerate(agents_list):
            agent.Q_table = copy.deepcopy(q_table_snapshots[i])
            for _ in range(episodes_per_block):
                agent.learn_with_parrot(tau=tau, alpha=alpha, gamma=gamma,
                                        valence=best_valence, parrot=parrot, evaluation=False)
            q_table_snapshots[i] = copy.deepcopy(agent.Q_table)

    return_dict[loop] = [pair_performance, pair_knowledge, pair_steps, pair_knowledge_quality]
    sema.release()




if __name__ == '__main__':
    t0 = time.time()
    concurrency = 50
    agent_num = 100
    repetition = 100
    decay_rate_list = [0.01, 0.03, 0.05]
    pair_performance_across_episodes, pair_knowledge_across_episodes, pair_steps_across_episodes, pair_knowledge_quality_across_episodes = [], [], [], []
    for decay_rate in decay_rate_list:
        with mp.Manager() as manager:  # immediate memory cleanup
            jobs = []
            return_dict = manager.dict()
            sema = Semaphore(concurrency)

            for loop in range(repetition):
                sema.acquire()
                p = mp.Process(target=func, args=(agent_num, decay_rate, loop, return_dict, sema))
                jobs.append(p)
                p.start()

            for proc in jobs:
                proc.join()

            results = return_dict.values()
            pair_performance = sum([result[0] for result in results]) / repetition
            pair_knowledge = sum([result[1] for result in results]) / repetition
            pair_steps = sum([result[2] for result in results]) / repetition
            pair_knowledge_quality = sum([result[3] for result in results]) / repetition

        pair_performance_across_episodes.append(pair_performance)
        pair_knowledge_across_episodes.append(pair_knowledge)
        pair_steps_across_episodes.append(pair_steps)
        pair_knowledge_quality_across_episodes.append(pair_knowledge_quality)

    with open("pair_performance_across_decay_valence", 'wb') as out_file:
        pickle.dump(pair_performance_across_episodes, out_file)
    with open("pair_knowledge_across_decay_valence", 'wb') as out_file:
        pickle.dump(pair_knowledge_across_episodes, out_file)
    with open("pair_steps_across_decay_valence", 'wb') as out_file:
        pickle.dump(pair_steps_across_episodes, out_file)
    with open("pair_knowledge_quality_across_decay_valence", 'wb') as out_file:
        pickle.dump(pair_knowledge_quality_across_episodes, out_file)

    t1 = time.time()
    print(time.strftime("%H:%M:%S", time.gmtime(t1 - t0)))  # Duration
    print("Across Valence:", time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(time.time())))  # Complete time
