# -*- coding: utf-8 -*-
# @Time     : 2/16/2025 20:08
# @Author   : Junyi
# @FileName: Q_learning.py
# @Software  : PyCharm
# Observing PEP 8 coding style
import numpy as np
from Agent import Agent
from Parrot import Parrot
from Reality import Reality
import multiprocessing as mp
import time
from multiprocessing import Semaphore
import pickle

def func(agent_num=None, learning_length=None, loop=None, return_dict=None, sema=None):
    np.random.seed(None)
    parrot = Parrot()
    knowledge_list = []
    for episode in range(learning_length):
        Q_table_list = []
        for _ in range(agent_num):
            agent = Agent(N=10, global_peak=50, local_peaks=[10, 10, 10])
            agent.learn(tau=20, alpha=0.8, gamma=0.9)
            Q_table_list.append(agent.Q_table)
        parrot.aggregate_from_data(Q_table_list=Q_table_list)
        for _ in range(agent_num):
            pair_agent = Agent(N=10, global_peak=50, local_peaks=[10, 10, 10])
            pair_agent.learn_with_parrot(tau=20, alpha=0.8, gamma=0.9, parrot=parrot)
            if episode == learning_length - 1:
                knowledge_list.append(pair_agent)

    knowledge = pair_agent.informed_percentage
    aligned_state_index = np.random.choice(range(1, 2 ** 10 - 2))  # cannot be the peaks!!
    # Max
    agent.state = agent.int_to_binary_list(state_index=aligned_state_index)
    agent.learn(tau=0.1, alpha=0.8, gamma=0.9)
    max_performance = agent.performance
    max_steps = agent.steps
    # Softmax
    agent.state = agent.int_to_binary_list(state_index=aligned_state_index)
    agent.learn(tau=20, alpha=0.8, gamma=0.9)
    softmax_performance = agent.performance
    softmax_steps = agent.steps

    return_dict[loop] = [max_performance, max_steps, softmax_performance, softmax_steps, knowledge]
    sema.release()


if __name__ == '__main__':
    t0 = time.time()
    concurrency = 50
    repetition = 50
    hyper_repetition = 40
    learning_length_list = [50, 100, 150, 200, 250, 300, 350]
    # learning_length_list = [50, 100, 150]
    max_performance_across_episodes, softmax_performance_across_episodes = [], []
    max_steps_across_episodes, softmax_steps_across_episodes = [], []
    knowledge_across_episodes = []
    for learning_length in learning_length_list:
        max_performance_list, max_steps_list, softmax_performance_list, softmax_steps_list, knowledge_list = [], [], [], [], []
        for hyper_loop in range(hyper_repetition):
            manager = mp.Manager()
            jobs = []
            return_dict = manager.dict()
            sema = Semaphore(concurrency)
            for loop in range(repetition):
                sema.acquire()
                p = mp.Process(target=func, args=(learning_length, loop, return_dict, sema))
                jobs.append(p)
                p.start()
            for proc in jobs:
                proc.join()
            results = return_dict.values()  # Don't need dict index, since it is repetition.
            max_performance_list += [result[0] for result in results]
            max_steps_list += [result[1] for result in results]
            softmax_performance_list += [result[2] for result in results]
            softmax_steps_list += [result[3] for result in results]
            knowledge_list += [result[4] for result in results]

        max_performance = sum([1 if reward == 50 else 0 for reward in max_performance_list]) / len(max_performance_list)
        softmax_performance = sum([1 if reward == 50 else 0 for reward in softmax_performance_list]) / len(softmax_performance_list)

        max_performance_across_episodes.append(max_performance)
        softmax_performance_across_episodes.append(softmax_performance)
        max_steps_across_episodes.append(sum(max_steps_list) / len(max_steps_list))
        softmax_steps_across_episodes.append(sum(softmax_steps_list) / len(softmax_steps_list))
        knowledge_across_episodes.append(sum(knowledge_list) / len(knowledge_list))

    with open("max_performance_across_episodes", 'wb') as out_file:
        pickle.dump(max_performance_across_episodes, out_file)
    with open("softmax_performance_across_episodes", 'wb') as out_file:
        pickle.dump(softmax_performance_across_episodes, out_file)
    with open("max_steps_across_episodes", 'wb') as out_file:
        pickle.dump(max_steps_across_episodes, out_file)
    with open("softmax_steps_across_episodes", 'wb') as out_file:
        pickle.dump(softmax_steps_across_episodes, out_file)
    with open("knowledge_across_episodes", 'wb') as out_file:
        pickle.dump(knowledge_across_episodes, out_file)

    t1 = time.time()
    print(time.strftime("%H:%M:%S", time.gmtime(t1 - t0)))  # Duration
    print("Figure 1:", time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(time.time())))  # Complete time
