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
    valence_bounds = (10, 100)
    mutation_rate = 0.1
    global_peak_value = 50 # as per (Fang, 2009)
    local_peak_values = [10] # add more local peaks to increase complexity
    # Initialize reality and parrot; fixed
    reality = Reality(N=N, global_peak_value=global_peak_value, local_peak_values=local_peak_values)
    parrot = Parrot(N=N, reality=reality, coverage=1.0, accuracy=1.0)


    # Begin sequential optimization over episodes
    for episode in range(1, learning_length):
        fitness_scores = []
        agents_list = []
        previous_Q = []
        for index in range(population_size):
            agent = Agent(N=N, reality=reality)
            agents_list.append(agent)
            previous_Q.append(agent.Q_table)
            valence_population = np.random.uniform(valence_bounds[0], valence_bounds[1], population_size)

        for generation in range(20):  # Optional: evolve valence within each episode
            new_fitness = []
            for valence in valence_population:
                test_agent = Agent(N=N, reality=reality)
                test_agent.Q_table = copy.deepcopy(previous_Q)
                test_agent.learn_with_parrot(tau=tau, alpha=alpha, gamma=gamma, valence=valence, parrot=parrot,
                                             evaluation=False)
                steps = test_agent.steps if test_agent.steps > 0 else 10000
                fitness = 1 / steps
                new_fitness.append(fitness)

            # Selection
            sorted_indices = np.argsort(new_fitness)[::-1]
            survivors = valence_population[sorted_indices[:population_size // 2]]

            # Crossover + Mutation
            children = []
            while len(children) < population_size - len(survivors):
                parents = np.random.choice(survivors, 2, replace=False)
                child = np.mean(parents)
                if np.random.rand() < mutation_rate:
                    child += np.random.uniform(-5, 5)
                    child = np.clip(child, valence_bounds[0], valence_bounds[1])
                children.append(child)

            valence_population = np.concatenate([survivors, children])
            fitness_scores = new_fitness

        # Choose best valence and update real agent
        best_index = np.argmax(fitness_scores)
        best_valence = valence_population[best_index]
        best_valences.append(best_valence)

        agent.Q_table = copy.deepcopy(previous_Q)
        agent.learn_with_parrot(tau=tau, alpha=alpha, gamma=gamma, valence=best_valence, parrot=parrot,
                                evaluation=False)
        episode_Q_tables.append(copy.deepcopy(agent.Q_table))


    pair_performance_list, pair_knowledge_list, pair_steps_list, pair_knowledge_quality_list = [], [], [], []
    for _ in range(agent_num):
        pair_agent = Agent(N=N, reality=reality)
        for episode in range(learning_length - 1):
            pair_agent.learn_with_parrot(tau=tau, alpha=alpha, gamma=gamma, parrot=parrot,
                                                        valence=valence, evaluation=False)
        pair_performance_list.append(pair_agent.performance)
        pair_knowledge_list.append(pair_agent.knowledge)
        pair_steps_list.append(pair_agent.steps)
        pair_knowledge_quality_list.append(pair_agent.knowledge_quality)
    pair_performance_list = [1 if each == 50 else 0 for each in pair_performance_list] # the likelihood of finding global peak
    pair_performance = sum(pair_performance_list) / agent_num
    pair_knowledge = sum(pair_knowledge_list) / agent_num
    pair_steps = sum(pair_steps_list) / agent_num
    pair_knowledge_quality = sum(pair_knowledge_quality_list) / agent_num
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
