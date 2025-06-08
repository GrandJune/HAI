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


def func(agent_num=None, learning_length=None, loop=None, return_dict=None, sema=None):
    np.random.seed(None)
    N = 10 # problem dimension
    tau = 20  # temperature parameter
    alpha = 0.8  # learning rate
    gamma = 0.9 # discount factor
    global_peak_value = 50 # as per (Fang, 2009)
    local_peak_value = 10
    # turbulence_freq = 50
    turbulence_intensity = 0.6
    reality = Reality(N=N, global_peak_value=global_peak_value, local_peak_value=local_peak_value)
    organic_performance_list, organic_knowledge_list, organic_steps_list,organic_knowledge_quality_list = [], [], [], []
    for _ in range(agent_num):
        agent = Agent(N=N, reality=reality)
        for episode in range(learning_length):
            agent.learn(tau=tau, alpha=alpha, gamma=gamma)
        # after learning: change the reality
        reality.change(likelihood=turbulence_intensity)
        agent.learn(tau=0.1, alpha=alpha, gamma=gamma, evaluation=True)  # evaluation
        organic_performance_list.append(agent.performance)
        organic_knowledge_list.append(agent.knowledge)
        organic_steps_list.append(agent.steps)
        organic_knowledge_quality_list.append(agent.knowledge_quality)
    organic_performance_list = [1 if each == 50 else 0 for each in organic_performance_list]  # the likelihood of finding global peak
    organic_performance = sum(organic_performance_list) / agent_num
    organic_knowledge = sum(organic_knowledge_list) / agent_num
    organic_steps = sum(organic_steps_list) / agent_num
    organic_knowledge_quality = sum(organic_knowledge_quality_list) / agent_num

    reality = Reality(N=N, global_peak_value=global_peak_value, local_peak_value=local_peak_value)
    parrot = Parrot(N=N, reality=reality, coverage=1, accuracy=1.0)
    pair_performance_list, pair_knowledge_list, pair_steps_list, pair_knowledge_quality_list = [], [], [], []
    for _ in range(agent_num):
        pair_agent = Agent(N=N, reality=reality)
        for episode in range(learning_length):
            pair_agent.learn_with_parrot(tau=tau, alpha=alpha, gamma=gamma, parrot=parrot, valence=50)
        # after learning: change the reality
        reality.change(likelihood=turbulence_intensity)
        pair_agent.learn_with_parrot(tau=0.1, alpha=alpha, gamma=gamma, parrot=parrot, valence=50, evaluation=True) # evaluation
        pair_performance_list.append(pair_agent.performance)
        pair_knowledge_list.append(pair_agent.knowledge)
        pair_steps_list.append(pair_agent.steps)
        pair_knowledge_quality_list.append(pair_agent.knowledge_quality)
    pair_performance_list = [1 if each == 50 else 0 for each in pair_performance_list] # the likelihood of finding global peak
    pair_performance = sum(pair_performance_list) / agent_num
    pair_knowledge = sum(pair_knowledge_list) / agent_num
    pair_steps = sum(pair_steps_list) / agent_num
    pair_knowledge_quality = sum(pair_knowledge_quality_list) / agent_num

    return_dict[loop] = [organic_performance, organic_knowledge, organic_steps, organic_knowledge_quality,
                         pair_performance, pair_knowledge, pair_steps, pair_knowledge_quality]
    sema.release()


if __name__ == '__main__':
    t0 = time.time()
    concurrency = 50
    agent_num = 100
    repetition = 50
    learning_length_list = [50, 100, 150, 200, 250, 300, 350]
    organic_performance_across_episodes, organic_knowledge_across_episodes, organic_steps_across_episodes, organic_knowledge_quality_across_episodes = [], [], [], []
    pair_performance_across_episodes, pair_knowledge_across_episodes, pair_steps_across_episodes, pair_knowledge_quality_across_episodes = [], [], [], []
    for learning_length in learning_length_list:
        with mp.Manager() as manager:  # immediate memory cleanup
            jobs = []
            return_dict = manager.dict()
            sema = Semaphore(concurrency)

            for loop in range(repetition):
                sema.acquire()
                p = mp.Process(target=func, args=(agent_num, learning_length, loop, return_dict, sema))
                jobs.append(p)
                p.start()

            for proc in jobs:
                proc.join()

            results = return_dict.values()
            organic_performance = sum([result[0] for result in results]) / repetition
            organic_knowledge = sum([result[1] for result in results]) / repetition
            organic_steps = sum([result[2] for result in results]) / repetition
            organic_knowledge_quality = sum([result[3] for result in results]) / repetition

            pair_performance = sum([result[4] for result in results]) / repetition
            pair_knowledge = sum([result[5] for result in results]) / repetition
            pair_steps = sum([result[6] for result in results]) / repetition
            pair_knowledge_quality = sum([result[7] for result in results]) / repetition

        organic_performance_across_episodes.append(organic_performance)
        organic_knowledge_across_episodes.append(organic_knowledge)
        organic_steps_across_episodes.append(organic_steps)
        organic_knowledge_quality_across_episodes.append(organic_knowledge_quality)

        pair_performance_across_episodes.append(pair_performance)
        pair_knowledge_across_episodes.append(pair_knowledge)
        pair_steps_across_episodes.append(pair_steps)
        pair_knowledge_quality_across_episodes.append(pair_knowledge_quality)

    with open("organic_performance_across_episodes", 'wb') as out_file:
        pickle.dump(organic_performance_across_episodes, out_file)
    with open("organic_knowledge_across_episodes", 'wb') as out_file:
        pickle.dump(organic_knowledge_across_episodes, out_file)
    with open("organic_steps_across_episodes", 'wb') as out_file:
        pickle.dump(organic_steps_across_episodes, out_file)
    with open("organic_knowledge_quality_across_episodes", 'wb') as out_file:
        pickle.dump(organic_knowledge_quality_across_episodes, out_file)

    with open("pair_performance_across_episodes", 'wb') as out_file:
        pickle.dump(pair_performance_across_episodes, out_file)
    with open("pair_knowledge_across_episodes", 'wb') as out_file:
        pickle.dump(pair_knowledge_across_episodes, out_file)
    with open("pair_steps_across_episodes", 'wb') as out_file:
        pickle.dump(pair_steps_across_episodes, out_file)
    with open("pair_knowledge_quality_across_episodes", 'wb') as out_file:
        pickle.dump(pair_knowledge_quality_across_episodes, out_file)

    t1 = time.time()
    duration = int(t1 - t0)

    days = duration // 86400
    hours = (duration % 86400) // 3600
    minutes = (duration % 3600) // 60
    seconds = duration % 60

    if days > 0:
        print(f"Duration: {days}d {hours:02}:{minutes:02}:{seconds:02}")
    else:
        print(f"Duration: {hours:02}:{minutes:02}:{seconds:02}")
    print("Across Episodes", time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(time.time())))  # Complete time