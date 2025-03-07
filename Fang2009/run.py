# -*- coding: utf-8 -*-
# @Time     : 2/16/2025 20:08
# @Author   : Junyi
# @FileName: Q_learning.py
# @Software  : PyCharm
# Observing PEP 8 coding style
import numpy as np
from Q_learning import Agent
import multiprocessing as mp
import time
from multiprocessing import Semaphore
import pickle

def func(learning_length=None, loop=None, return_dict=None, sema=None):
    np.random.seed(None)
    q_agent = Agent(N=10, global_peak=50, local_peaks=[10])
    for _ in range(learning_length):
        q_agent.learn(tau=20, alpha=0.8, gamma=0.9)
    reward = q_agent.perform(tau=20)
    return_dict[loop] = [reward]
    sema.release()


if __name__ == '__main__':
    t0 = time.time()
    concurrency = 50
    repetition = 120
    hyper_repetition = 10
    learning_length_list = [50, 100, 150, 200, 250, 300, 350]
    percentage_high_across_learning_length, percentage_low_across_learning_length = [], []
    for learning_length in learning_length_list:
        performance_list = []
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
            performance_list += [result[0] for result in results]

        percentage_high = sum([1 if reward == 50 else 0 for reward in performance_list]) / len(performance_list)
        percentage_low = sum([1 if reward == 10 else 0 for reward in performance_list]) / len(performance_list)

        percentage_high_across_learning_length.append(percentage_high)
        percentage_low_across_learning_length.append(percentage_low)

    with open("highs_across_learning_length", 'wb') as out_file:
        pickle.dump(percentage_high_across_learning_length, out_file)
    with open("lows_across_learning_length", 'wb') as out_file:
        pickle.dump(percentage_low_across_learning_length, out_file)

    t1 = time.time()
    print(time.strftime("%H:%M:%S", time.gmtime(t1 - t0)))  # Duration
    print("Learning Rate:", time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(time.time())))  # Complete time

