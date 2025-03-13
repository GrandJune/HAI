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

def func(distance=None, learning_length=None, loop=None, return_dict=None, sema=None):
    np.random.seed(None)
    q_agent = Agent(N=10, high_peak=50, low_peak=10)
    low_peak_state = [0] * 10
    # re-located into certain distance from local peak
    q_agent.state = q_agent.generate_state_with_hamming_distance(orientation_state=low_peak_state, hamming_distance=distance)
    for _ in range(learning_length):
        q_agent.learn(tau=20, alpha=0.2, gamma=0.9)
    q_agent.evaluate(tau=20)
    return_dict[loop] = [q_agent.performance, q_agent.informed_percentage]
    sema.release()


if __name__ == '__main__':
    t0 = time.time()
    concurrency = 50
    repetition = 50
    hyper_repetition = 40
    learning_length = 50
    distance_list = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    percentage_high_across_distance, percentage_low_across_distance, informed_across_distance = [], [], []
    for distance in distance_list:
        performance_list, informed_list = [], []
        for hyper_loop in range(hyper_repetition):
            manager = mp.Manager()
            jobs = []
            return_dict = manager.dict()
            sema = Semaphore(concurrency)
            for loop in range(repetition):
                sema.acquire()
                p = mp.Process(target=func, args=(distance, learning_length, loop, return_dict, sema))
                jobs.append(p)
                p.start()
            for proc in jobs:
                proc.join()
            results = return_dict.values()  # Don't need dict index, since it is repetition.
            performance_list += [result[0] for result in results]
            informed_list += [result[1] for result in results]

        percentage_high = sum([1 if reward == 50 else 0 for reward in performance_list]) / len(performance_list)
        # percentage_low = sum([1 if reward == 10 else 0 for reward in performance_list]) / len(performance_list)

        percentage_high_across_distance.append(percentage_high)
        # percentage_low_across_distance.append(percentage_low)
        informed_across_distance.append(sum(informed_list) / len(informed_list))

    with open("performance_across_distance", 'wb') as out_file_1:
        pickle.dump(percentage_high_across_distance, out_file_1)
    # with open("lows_across_distance", 'wb') as out_file_2:
    #     pickle.dump(percentage_low_across_distance, out_file_2)
    with open("informed_across_distance", 'wb') as out_file_3:
        pickle.dump(informed_across_distance, out_file_3)

    t1 = time.time()
    print(time.strftime("%H:%M:%S", time.gmtime(t1 - t0)))  # Duration
    print("Distance:", time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(time.time())))  # Complete time

