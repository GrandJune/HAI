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
from multiprocessing import Semaphore
import pickle
from Reality import Reality


def func(agent_num=None, loop=None, return_dict=None, sema=None):
    np.random.seed(None)
    N = 10 # problem dimension
    tau = 20  # temperature parameter
    alpha = 0.8  # learning rate
    gamma = 0.9 # discount factor
    learning_length = 500
    turbulence_freq = 100
    likelihood = 0.1
    global_peak_value = 50 # as per (Fang, 2009)
    local_peak_values = [10] # add more local peaks to increase complexity
    reality = Reality(N=N, global_peak_value=global_peak_value, local_peak_values=local_peak_values)
    parrot = Parrot(N=N, reality=reality, coverage=1.0, accuracy=1.0)

    all_organic_performance = []
    all_organic_knowledge = []
    all_organic_steps = []
    all_organic_knowledge_quality = []
    for _ in range(agent_num):
        organic_performance_across_time, organic_knowledge_across_time, organic_steps_across_time, organic_knowledge_quality_across_time = [], [], [], []
        agent = Agent(N=N, reality=reality)
        for episode in range(learning_length):
            reality.change(likelihood=likelihood)
            agent.reality = reality
            agent.learn(tau=tau, alpha=alpha, gamma=gamma, evaluation=False)
        agent.learn(tau=0.1, alpha=alpha, gamma=gamma, evaluation=True)
        organic_performance_across_time.append(agent.performance)
        organic_performance_across_time = [1 if each == 50 else 0 for each in organic_performance_across_time]
        organic_knowledge_across_time.append(agent.knowledge)
        organic_steps_across_time.append(agent.steps)
        organic_knowledge_quality_across_time.append(agent.knowledge_quality)

        all_organic_performance.append(organic_performance_across_time)
        all_organic_knowledge.append(organic_knowledge_across_time)
        all_organic_steps.append(organic_steps_across_time)
        all_organic_knowledge_quality.append(organic_knowledge_quality_across_time)
    # Convert to NumPy arrays and average across agents (axis=0)
    organic_performance_list = np.mean(all_organic_performance, axis=0)
    organic_knowledge_list = np.mean(all_organic_knowledge, axis=0)
    organic_steps_list = np.mean(all_organic_steps, axis=0)
    organic_knowledge_quality_list = np.mean(all_organic_knowledge_quality, axis=0)

    all_pair_performance = []
    all_pair_knowledge = []
    all_pair_steps = []
    all_pair_knowledge_quality = []
    for _ in range(agent_num):
        pair_agent = Agent(N=N, reality=reality)
        pair_performance_across_time, pair_knowledge_across_time, pair_steps_across_time, pair_knowledge_quality_across_time = [], [], [], []
        for episode in range(learning_length):
            reality.change()
            agent.reality = reality
            parrot.reality = reality
            pair_agent.learn_with_parrot(tau=tau, alpha=alpha, gamma=gamma, parrot=parrot, valence=50, evaluation=False)
        pair_agent.learn_with_parrot(tau=0.1, alpha=alpha, gamma=gamma, parrot=parrot, valence=50, evaluation=True)
        pair_performance_across_time.append(pair_agent.performance)
        pair_performance_across_time = [1 if each == 50 else 0 for each in pair_knowledge_across_time]
        pair_steps_across_time.append(pair_agent.steps)
        pair_knowledge_across_time.append(pair_agent.knowledge)
        pair_knowledge_quality_across_time.append(pair_agent.knowledge_quality)

        all_pair_performance.append(pair_performance_across_time)
        all_pair_knowledge.append(pair_knowledge_across_time)
        all_pair_steps.append(pair_steps_across_time)
        all_pair_knowledge_quality.append(pair_knowledge_quality_across_time)
    # Convert to NumPy arrays and average across agents (axis=0)
    pair_performance_list = np.mean(all_pair_performance, axis=0)
    pair_knowledge_list = np.mean(all_pair_knowledge, axis=0)
    pair_steps_list = np.mean(all_pair_steps, axis=0)
    pair_knowledge_quality_list = np.mean(all_pair_knowledge_quality, axis=0)

    return_dict[loop] = [organic_performance_list, organic_knowledge_list, organic_steps_list, organic_knowledge_quality_list,
                         pair_performance_list, pair_knowledge_list, pair_steps_list, pair_knowledge_quality_list]
    sema.release()


if __name__ == '__main__':
    t0 = time.time()
    concurrency = 50
    agent_num = 200
    repetition = 50
    pair_performance_across_episodes, pair_knowledge_across_episodes, pair_steps_across_episodes, pair_knowledge_quality_across_episodes = [], [], [], []
    with mp.Manager() as manager:  # immediate memory cleanup
        jobs = []
        return_dict = manager.dict()
        sema = Semaphore(concurrency)

        for loop in range(repetition):
            sema.acquire()
            p = mp.Process(target=func, args=(agent_num, loop, return_dict, sema))
            jobs.append(p)
            p.start()

        for proc in jobs:
            proc.join()

        results = return_dict.values()

        # Organic
        organic_performance_across_tasks = [result[0] for result in results]
        organic_performance_across_turbulence = np.mean(organic_performance_across_tasks, axis=0)

        organic_knowledge_across_tasls = [result[1] for result in results]
        organic_knowledge_across_turbulence = np.mean(organic_knowledge_across_tasls, axis=0)

        organic_steps_across_tasks = [result[2] for result in results]
        organic_steps_across_turbulence = np.mean(organic_steps_across_tasks, axis=0)

        organic_knowledge_quality_across_tasks = [result[3] for result in results]
        organic_knowledge_quality_across_turbulence = np.mean(organic_knowledge_quality_across_tasks, axis=0)

        # Pairing
        pair_performance_across_tasks = [result[4] for result in results]
        pair_performance_across_turbulence = np.mean(organic_performance_across_tasks, axis=0)

        pair_knowledge_across_tasls = [result[5] for result in results]
        pair_knowledge_across_turbulence = np.mean(organic_knowledge_across_tasls, axis=0)

        pair_steps_across_tasks = [result[6] for result in results]
        pair_steps_across_turbulence = np.mean(organic_steps_across_tasks, axis=0)

        pair_knowledge_quality_across_tasks = [result[7] for result in results]
        pair_knowledge_quality_across_turbulence = np.mean(organic_knowledge_quality_across_tasks, axis=0)

    with open("organic_performance_across_turbulence", 'wb') as out_file:
        pickle.dump(organic_performance_across_turbulence, out_file)
    with open("organic_knowledge_across_turbulence", 'wb') as out_file:
        pickle.dump(organic_knowledge_across_turbulence, out_file)
    with open("organic_steps_across_turbulence", 'wb') as out_file:
        pickle.dump(organic_steps_across_turbulence, out_file)
    with open("organic_knowledge_quality_across_turbulence", 'wb') as out_file:
        pickle.dump(organic_knowledge_quality_across_turbulence, out_file)

    with open("pair_performance_across_turbulence", 'wb') as out_file:
        pickle.dump(pair_performance_across_turbulence, out_file)
    with open("pair_knowledge_across_turbulence", 'wb') as out_file:
        pickle.dump(pair_knowledge_across_turbulence, out_file)
    with open("pair_steps_across_turbulence", 'wb') as out_file:
        pickle.dump(pair_steps_across_turbulence, out_file)
    with open("pair_knowledge_quality_across_turbulence", 'wb') as out_file:
        pickle.dump(pair_knowledge_quality_across_turbulence, out_file)

    t1 = time.time()
    print(time.strftime("%H:%M:%S", time.gmtime(t1 - t0)))  # Duration
    print("Across Valence:", time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(time.time())))  # Complete time
