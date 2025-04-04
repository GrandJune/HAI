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
    q_agent = Agent(N=10, high_peak=50, low_peak=10)
    for _ in range(learning_length):
        q_agent.learn(tau=20, alpha=0.2, gamma=0.9)
    q_agent.evaluate(tau=20)
    return_dict[loop] = [q_agent.performance, q_agent.steps, q_agent.informed_percentage]
    sema.release()


if __name__ == '__main__':
    t0 = time.time()
    concurrency = 50
    repetition = 50
    hyper_repetition = 80
    learning_length_list = [50, 100, 150, 200, 250, 300, 350]
    # learning_length_list = [50, 100, 150]
    (percentage_high_across_learning_length, percentage_low_across_learning_length,
     steps_across_learning_length, informed_across_learning_length) = [], [], [], []
    for learning_length in learning_length_list:
        performance_list, steps_list, informed_percentage_list = [], [], []
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
            steps_list += [result[1] for result in results]
            informed_percentage_list += [result[2] for result in results]

        percentage_high = sum([1 if reward == 50 else 0 for reward in performance_list]) / len(performance_list)
        # percentage_low = sum([1 if reward == 10 else 0 for reward in performance_list]) / len(performance_list)

        percentage_high_across_learning_length.append(percentage_high)
        # percentage_low_across_learning_length.append(percentage_low)
        steps_across_learning_length.append(sum(steps_list) / len(steps_list))
        informed_across_learning_length.append(sum(informed_percentage_list) / len(informed_percentage_list))

    with open("softmax_performance_across_learning", 'wb') as out_file_1:
        pickle.dump(percentage_high_across_learning_length, out_file_1)
    # with open("lows_softmax_across_learning_length", 'wb') as out_file_2:
    #     pickle.dump(percentage_low_across_learning_length, out_file_2)
    with open("softmax_steps_across_learning", 'wb') as out_file_3:
        pickle.dump(steps_across_learning_length, out_file_3)
    with open("softmax_informed_across_learning", 'wb') as out_file_4:
        pickle.dump(informed_across_learning_length, out_file_4)

    t1 = time.time()
    print(time.strftime("%H:%M:%S", time.gmtime(t1 - t0)))  # Duration
    print("Softmax:", time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(time.time())))  # Complete time

