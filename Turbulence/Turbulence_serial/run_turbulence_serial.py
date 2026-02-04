# -*- coding: utf-8 -*-
# @Time     : 2/16/2025 20:08
# @Author   : Junyi
# @FileName: Q_learning.py
# @Software  : PyCharm
# Observing PEP 8 coding style
import numpy as np
from Agent_turbulence import Agent
from Parrot import Parrot
import multiprocessing as mp
import time
from multiprocessing import Semaphore
import pickle

from Reality import Reality


def func(agent_num=None, learning_length=None, turbulence_frequency=None, turbulence_intensity=None,
         loop=None, return_dict=None, sema=None):
    np.random.seed(None)
    N = 10 # problem dimension
    tau = 20  # temperature parameter
    alpha = 0.8  # learning rate
    gamma = 0.9 # discount factor
    global_peak_value = 50 # as per (Fang, 2009)
    local_peak_value = 10
    organic_performance_across_agents, organic_steps_across_agents, organic_knowledge_across_agents, organic_knowledge_quality_across_agents = [], [], [], []
    for _ in range(agent_num):
        organic_performance_list, organic_knowledge_list, organic_steps_list, organic_knowledge_quality_list = [], [], [], []
        reality = Reality(N=N, global_peak_value=global_peak_value, local_peak_value=local_peak_value)
        agent = Agent(N=N, reality=reality)
        for episode in range(learning_length):
            if episode % turbulence_frequency == 0:
                reality.change(likelihood=turbulence_intensity)
            agent.learn(tau=tau, alpha=alpha, gamma=gamma, evaluation=True)
            organic_performance_list.append(agent.performance)
            organic_steps_list.append(agent.steps)
            organic_knowledge_list.append(agent.knowledge)
            organic_knowledge_quality_list.append(agent.knowledge_quality)
        organic_performance_list = [1 if each == 50 else 0 for each in organic_performance_list]
        organic_performance_across_agents.append(organic_performance_list)
        organic_steps_across_agents.append(organic_steps_list)
        organic_knowledge_across_agents.append(organic_knowledge_list)
        organic_knowledge_quality_across_agents.append(organic_knowledge_quality_list)

    organic_performance_across_agents = np.array(organic_performance_across_agents)
    organic_steps_across_agents = np.array(organic_steps_across_agents)
    organic_knowledge_across_agents = np.array(organic_knowledge_across_agents)
    organic_knowledge_quality_across_agents = np.array(organic_knowledge_quality_across_agents)

    organic_performance_across_agents = organic_performance_across_agents.mean(axis=0).tolist()
    organic_steps_across_agents = organic_steps_across_agents.mean(axis=0).tolist()
    organic_knowledge_across_agents = organic_knowledge_across_agents.mean(axis=0)
    organic_knowledge_quality_across_agents = np.array(organic_knowledge_quality_across_agents)

    pair_performance_across_agents, pair_steps_across_agents, pair_knowledge_across_agents, pair_knowledge_quality_across_agents = [], [], [], []
    for _ in range(agent_num):
        pair_performance_list, pair_knowledge_list, pair_steps_list, pair_knowledge_quality_list = [], [], [], []
        reality = Reality(N=N, global_peak_value=global_peak_value, local_peak_value=local_peak_value)
        parrot = Parrot(N=N, reality=reality)
        agent = Agent(N=N, reality=reality)
        for episode in range(learning_length):
            if episode % turbulence_frequency == 0:
                reality.change(likelihood=turbulence_intensity)
            agent.learn_with_parrot(tau=tau, alpha=alpha, gamma=gamma, parrot=parrot, evaluation=True)
            pair_performance_list.append(agent.performance)
            pair_steps_list.append(agent.steps)
            pair_knowledge_list.append(agent.knowledge)
            pair_knowledge_quality_list.append(agent.knowledge_quality)
        pair_performance_list = [1 if each == 50 else 0 for each in pair_performance_list]
        pair_performance_across_agents.append(pair_performance_list)
        pair_steps_across_agents.append(pair_steps_list)
        pair_knowledge_across_agents.append(pair_knowledge_list)
        pair_knowledge_quality_across_agents.append(pair_knowledge_quality_list)

    pair_performance_across_agents = np.array(pair_performance_across_agents)
    pair_steps_across_agents = np.array(pair_steps_across_agents)
    pair_knowledge_across_agents = np.array(pair_knowledge_across_agents)
    pair_knowledge_quality_across_agents = np.array(pair_knowledge_quality_across_agents)

    pair_performance_across_agents = pair_performance_across_agents.mean(axis=0).tolist()
    pair_steps_across_agents = pair_steps_across_agents.mean(axis=0).tolist()
    pair_knowledge_across_agents = pair_knowledge_across_agents.mean(axis=0)
    pair_knowledge_quality_across_agents = np.array(pair_knowledge_quality_across_agents)

    return_dict[loop] = [organic_performance_across_agents, organic_steps_across_agents, organic_knowledge_across_agents,
                         organic_knowledge_quality_across_agents, pair_performance_across_agents, pair_steps_across_agents,
                         pair_knowledge_across_agents, pair_knowledge_quality_across_agents]
    sema.release()


if __name__ == '__main__':
    t0 = time.time()
    concurrency = 50
    agent_num = 100
    repetition = 100
    learning_length = 500
    turbulence_intensity = 0.4
    turbulence_frequency = 50
    with mp.Manager() as manager:  # immediate memory cleanup
        jobs = []
        return_dict = manager.dict()
        sema = Semaphore(concurrency)

        for loop in range(repetition):
            sema.acquire()
            p = mp.Process(target=func, args=(agent_num, learning_length, turbulence_frequency, turbulence_intensity, loop, return_dict, sema))
            jobs.append(p)
            p.start()

        for proc in jobs:
            proc.join()

        results = return_dict.values()
        organic_performance = [result[0] for result in results]
        organic_performance = np.array(organic_performance)
        organic_performance = organic_performance.mean(axis=0).tolist()

        organic_steps = [result[1] for result in results]
        organic_steps = np.array(organic_steps)
        organic_steps = organic_steps.mean(axis=0).tolist()

        organic_knowledge = [result[2] for result in results]
        organic_knowledge = np.array(organic_knowledge)
        organic_knowledge = organic_knowledge.mean(axis=0).tolist()

        organic_knowledge_quality = [result[3] for result in results]
        organic_knowledge_quality = np.array(organic_knowledge_quality)
        organic_knowledge_quality = organic_knowledge_quality.mean(axis=0).tolist()

        pair_performance = [result[4] for result in results]
        pair_performance = np.array(pair_performance)
        pair_performance = pair_performance.mean(axis=0).tolist()

        pair_steps = [result[5] for result in results]
        pair_steps = np.array(pair_steps)
        pair_steps = pair_steps.mean(axis=0).tolist()

        pair_knowledge = [result[6] for result in results]
        pair_knowledge = np.array(pair_knowledge)
        pair_knowledge = pair_knowledge.mean(axis=0).tolist()

        pair_knowledge_quality = [result[7] for result in results]
        pair_knowledge_quality = np.array(pair_knowledge_quality)
        pair_knowledge_quality = pair_knowledge_quality.mean(axis=0).tolist()

    with open("organic_performance", 'wb') as out_file:
        pickle.dump(organic_performance, out_file)
    with open("organic_steps", 'wb') as out_file:
        pickle.dump(organic_steps, out_file)
    with open("organic_knowledge", 'wb') as out_file:
        pickle.dump(organic_knowledge, out_file)
    with open("organic_knowledge_quality", 'wb') as out_file:
        pickle.dump(organic_knowledge_quality, out_file)

    with open("pair_performance", 'wb') as out_file:
        pickle.dump(pair_performance, out_file)
    with open("pair_steps", 'wb') as out_file:
        pickle.dump(pair_steps, out_file)
    with open("pair_knowledge", 'wb') as out_file:
        pickle.dump(pair_knowledge, out_file)
    with open("pair_knowledge_quality", 'wb') as out_file:
        pickle.dump(pair_knowledge_quality, out_file)

    t1 = time.time()
    duration = int(t1 - t0)

    days = duration // 86400
    hours = (duration % 86400) // 3600
    minutes = (duration % 3600) // 60
    seconds = duration % 60

    if days > 0:
        print(f"Turbulence Duration: {days}d {hours:02}:{minutes:02}:{seconds:02}")
    else:
        print(f"Duration: {hours:02}:{minutes:02}:{seconds:02}")
    print("Across Episodes", time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(time.time())))  # Complete time