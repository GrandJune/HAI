#!/usr/bin/env py39
# -*- coding: utf-8 -*-
# @Author   : Junyi
# @FileName: Organic_experiential_learning.py
# @Software : PyCharm
# Observing PEP 8 coding style

import numpy as np
from Landscape import Landscape
from Individual import Individual
import multiprocessing as mp
import time
from multiprocessing import Semaphore
import pickle
import copy
import os


# mp version
# One worker generates one NK landscape and tests several independent individuals.
def func(N=None, K=None, individual_reset_num=1, learning_steps_list=None,
         search_iteration=100, evaluation_reset_num=3,
         alpha=0.8, gamma=0.9, tau_learning=1.0, tau_evaluation=0.1,
         compute_accuracy=True, base_seed=1000,
         loop=None, return_dict=None, sema=None):
    """
    Organic experiential learning without AI assistance.

    learning_steps_list: cumulative numbers of SARSA learning steps (focal IV).
    search_iteration: fixed number of steps used to evaluate learned knowledge.
    individual_reset_num: independent learners per NK landscape.
    evaluation_reset_num: independent evaluation searches per learning checkpoint.

    Each individual retains its Q-table across learning checkpoints. Evaluation
    uses a copy, so it does not change the individual's subsequent learning.
    """
    try:
        if base_seed is None:
            np.random.seed(None)
        else:
            # Different landscape repetitions have independent reproducible seeds.
            seed = np.random.SeedSequence(
                [int(base_seed), int(K), int(loop)]).generate_state(1)[0]
            np.random.seed(int(seed))

        landscape = Landscape(N=N, K=K, norm="MaxMin")
        individual_reset_returns = []

        for individual_reset in range(individual_reset_num):
            individual = Individual(N=N, landscape=landscape,
                                    alpha=alpha, gamma=gamma, tau=tau_learning)

            # Use the same evaluation starting states and random draws at each checkpoint.
            evaluation_states = [np.random.choice(["0", "1"], N).tolist()
                                 for _ in range(evaluation_reset_num)]
            evaluation_seeds = np.random.randint(0, 2 ** 32 - 1,
                                                 size=evaluation_reset_num,
                                                 dtype=np.uint32)

            outcomes_across_learning_steps = []
            fitness_across_time_across_learning_steps = []

            for learning_steps in learning_steps_list:
                # Continue learning from the previous checkpoint, without resetting Q.
                additional_steps = learning_steps - individual.total_learning_steps
                individual.learn(steps=additional_steps, reset_state=False)

                # Internal-knowledge measures after the current learning budget.
                knowledge = individual.learning_outcomes(
                    compute_accuracy=compute_accuracy, tau=tau_learning)

                evaluation_best_fitness_list = []
                evaluation_best_rank_list = []
                evaluation_final_fitness_list = []
                evaluation_fitness_gain_list = []
                evaluation_fitness_across_time_list = []

                for evaluation_reset in range(evaluation_reset_num):
                    # Evaluate a copy: no SARSA updates or changes to the learner.
                    # Restore the RNG so evaluation does not affect later training.
                    training_random_state = np.random.get_state()
                    np.random.seed(int(evaluation_seeds[evaluation_reset]))

                    evaluator = copy.copy(individual)
                    evaluator.reset_state(state=evaluation_states[evaluation_reset])
                    initial_fitness = evaluator.fitness
                    evaluator.evaluate(steps=search_iteration,
                                       tau=tau_evaluation, reset_state=False)

                    evaluation_best_fitness_list.append(evaluator.best_fitness)
                    evaluation_best_rank_list.append(
                        landscape.query_fitness_rank(state=evaluator.best_state))
                    evaluation_final_fitness_list.append(evaluator.fitness)
                    evaluation_fitness_gain_list.append(
                        evaluator.best_fitness - initial_fitness)
                    evaluation_fitness_across_time_list.append(
                        evaluator.best_fitness_across_time.copy())

                    np.random.set_state(training_random_state)

                # One result per individual and learning checkpoint.
                outcomes_across_learning_steps.append([
                    np.mean(evaluation_best_fitness_list),
                    np.mean(evaluation_best_rank_list),
                    np.mean(evaluation_final_fitness_list),
                    np.mean(evaluation_fitness_gain_list),
                    individual.best_fitness,                 # Best found in training.
                    individual.fitness,                      # Current training fitness.
                    knowledge["informed_percentage"],        # States with nonzero Q.
                    knowledge["state_coverage"],             # States ever updated.
                    knowledge["action_coverage"],            # State-action pairs updated.
                    knowledge.get("q_accuracy", np.nan),     # Accuracy of entire Q-table.
                    knowledge.get("experienced_q_accuracy", np.nan),
                ])

                # Mean best-so-far fitness over the evaluation search trajectory.
                fitness_across_time_across_learning_steps.append(
                    np.mean(evaluation_fitness_across_time_list, axis=0).tolist())

            individual_reset_returns.append([
                outcomes_across_learning_steps,
                fitness_across_time_across_learning_steps,
            ])

        # All individuals on this repetition share the same NK landscape.
        return_dict[loop] = individual_reset_returns
    finally:
        sema.release()


if __name__ == '__main__':
    import datetime
    now = datetime.datetime.now()
    print(now.strftime("%Y-%m-%d %H:%M:%S"))
    t0 = time.time()

    landscape_iteration = 40
    individual_reset_num = 5     # Independent learners on each landscape.
    search_iteration = 100       # Fixed evaluation search budget.
    evaluation_reset_num = 3     # Repeated evaluations per learning checkpoint.

    N = 10
    K_list = [2, 4, 6, 8, 9]  # NK interdependence (problem complexity).

    # Focal IV: accumulated experiential learning; 0 = no acquired Q-values.
    learning_steps_list = [0, 100, 250, 500]

    alpha = 0.8          # SARSA learning rate.
    gamma = 0.9          # Weight of discounted future fitness rewards.
    tau_learning = 1.0   # Learning-phase softmax temperature.
    tau_evaluation = 0.1 # More exploitative search during evaluation.

    compute_accuracy = True  # Exact Q-value accuracy; expensive for large N.
    base_seed = None        # None for unseeded runs.
    concurrency = 4        # Limit concurrent NK landscapes to available memory.
    output_dir = "Organic_experiential_learning_results"
    os.makedirs(output_dir, exist_ok=True)

    if (not learning_steps_list or learning_steps_list[0] != 0 or
            learning_steps_list != sorted(set(learning_steps_list))):
        raise ValueError("learning_steps_list must be sorted, unique, and begin at 0.")
    if not K_list or any(K < 0 or K >= N for K in K_list):
        raise ValueError("K must be between 0 and N-1.")
    if compute_accuracy and gamma >= 1:
        raise ValueError("Q-value accuracy requires gamma < 1.")

    # Outcomes averaged over landscapes and individuals, separately for each K.
    # Each element contains one value per learning checkpoint.
    evaluation_best_fitness_across_K = []
    evaluation_best_rank_across_K = []
    evaluation_final_fitness_across_K = []
    evaluation_fitness_gain_across_K = []
    training_best_fitness_across_K = []
    training_final_fitness_across_K = []
    informed_percentage_across_K = []
    state_coverage_across_K = []
    action_coverage_across_K = []
    q_accuracy_across_K = []
    experienced_q_accuracy_across_K = []
    evaluation_best_fitness_across_time_across_K = []

    for K in K_list:
        manager = mp.Manager()
        return_dict = manager.dict()
        sema = Semaphore(concurrency)
        jobs = []

        try:
            for loop in range(landscape_iteration):
                # Close finished processes to avoid exhausting OS file handles.
                while True:
                    acquired = sema.acquire(timeout=0.5)

                    for index, proc in jobs[:]:
                        if proc.exitcode is not None:
                            proc.join()
                            exitcode = proc.exitcode
                            proc.close()
                            jobs.remove((index, proc))

                            if exitcode != 0:
                                raise RuntimeError(
                                    "Repetition {0} failed with exit code {1}".format(index, exitcode)
                                )

                    if acquired:
                        break

                p = mp.Process(target=func, args=(
                    N, K, individual_reset_num, learning_steps_list,
                    search_iteration, evaluation_reset_num,
                    alpha, gamma, tau_learning, tau_evaluation,
                    compute_accuracy, base_seed, loop, return_dict, sema))
                p.start()
                jobs.append((loop, p))

            # Collect and close any remaining processes.
            while jobs:
                for index, proc in jobs[:]:
                    if proc.exitcode is not None:
                        proc.join()
                        exitcode = proc.exitcode
                        proc.close()
                        jobs.remove((index, proc))

                        if exitcode != 0:
                            raise RuntimeError(
                                "Repetition {0} failed with exit code {1}".format(index, exitcode)
                            )

                if jobs:
                    time.sleep(0.5)

            if len(return_dict) != landscape_iteration:
                raise RuntimeError("Missing results: expected {0}, received {1}".format(
                    landscape_iteration, len(return_dict)))

            landscape_returns = [return_dict[index] for index in range(landscape_iteration)]
        finally:
            for index, proc in jobs:
                if proc.is_alive():
                    proc.terminate()
            for index, proc in jobs:
                proc.join()
                proc.close()
            manager.shutdown()

        # Flatten landscape repetitions and individual resets for averaging.
        returns = [item for landscape_results in landscape_returns for item in landscape_results]
        if len(returns) != landscape_iteration * individual_reset_num:
            raise RuntimeError("Missing individual resets for K={0}".format(K))

        arr = np.asarray([item[0] for item in returns], dtype=float)
        # shape: (number of individuals, learning checkpoints, 11 measures)
        time_series_arr = np.asarray([item[1] for item in returns], dtype=float)
        # shape: (number of individuals, learning checkpoints, evaluation steps + 1)

        evaluation_best_fitness_across_K.append(np.mean(arr[:, :, 0], axis=0).tolist())
        evaluation_best_rank_across_K.append(np.mean(arr[:, :, 1], axis=0).tolist())
        evaluation_final_fitness_across_K.append(np.mean(arr[:, :, 2], axis=0).tolist())
        evaluation_fitness_gain_across_K.append(np.mean(arr[:, :, 3], axis=0).tolist())
        training_best_fitness_across_K.append(np.mean(arr[:, :, 4], axis=0).tolist())
        training_final_fitness_across_K.append(np.mean(arr[:, :, 5], axis=0).tolist())
        informed_percentage_across_K.append(np.mean(arr[:, :, 6], axis=0).tolist())
        state_coverage_across_K.append(np.mean(arr[:, :, 7], axis=0).tolist())
        action_coverage_across_K.append(np.mean(arr[:, :, 8], axis=0).tolist())

        # Accuracy can be undefined (e.g., no learning for experienced Q accuracy).
        q_accuracy_means = []
        experienced_q_accuracy_means = []
        for learning_index in range(len(learning_steps_list)):
            q_values = arr[:, learning_index, 9]
            q_values = q_values[np.isfinite(q_values)]
            q_accuracy_means.append(np.mean(q_values) if len(q_values) else np.nan)

            experienced_values = arr[:, learning_index, 10]
            experienced_values = experienced_values[np.isfinite(experienced_values)]
            experienced_q_accuracy_means.append(
                np.mean(experienced_values) if len(experienced_values) else np.nan)

        q_accuracy_across_K.append(q_accuracy_means)
        experienced_q_accuracy_across_K.append(experienced_q_accuracy_means)
        evaluation_best_fitness_across_time_across_K.append(
            np.mean(time_series_arr, axis=0).tolist())

        print("K={0} finished".format(K))

    # Save each measure separately, as in Visibility_extent.py.
    # Each pickle contains a list across K for the given learning budget.
    for learning_index, learning_steps in enumerate(learning_steps_list):
        evaluation_best_fitness_list = [row[learning_index] for row in evaluation_best_fitness_across_K]
        evaluation_best_rank_list = [row[learning_index] for row in evaluation_best_rank_across_K]
        evaluation_final_fitness_list = [row[learning_index] for row in evaluation_final_fitness_across_K]
        evaluation_fitness_gain_list = [row[learning_index] for row in evaluation_fitness_gain_across_K]
        training_best_fitness_list = [row[learning_index] for row in training_best_fitness_across_K]
        training_final_fitness_list = [row[learning_index] for row in training_final_fitness_across_K]
        informed_percentage_list = [row[learning_index] for row in informed_percentage_across_K]
        state_coverage_list = [row[learning_index] for row in state_coverage_across_K]
        action_coverage_list = [row[learning_index] for row in action_coverage_across_K]
        q_accuracy_list = [row[learning_index] for row in q_accuracy_across_K]
        experienced_q_accuracy_list = [row[learning_index] for row in experienced_q_accuracy_across_K]
        evaluation_best_fitness_across_time_list = [
            row[learning_index] for row in evaluation_best_fitness_across_time_across_K]

        with open(os.path.join(output_dir, "organic_learning_steps_{0}_evaluation_best_fitness_across_K".format(
                learning_steps)), 'wb') as out_file:
            pickle.dump(evaluation_best_fitness_list, out_file)

        with open(os.path.join(output_dir, "organic_learning_steps_{0}_evaluation_best_rank_across_K".format(
                learning_steps)), 'wb') as out_file:
            pickle.dump(evaluation_best_rank_list, out_file)

        with open(os.path.join(output_dir, "organic_learning_steps_{0}_evaluation_final_fitness_across_K".format(
                learning_steps)), 'wb') as out_file:
            pickle.dump(evaluation_final_fitness_list, out_file)

        with open(os.path.join(output_dir, "organic_learning_steps_{0}_evaluation_fitness_gain_across_K".format(
                learning_steps)), 'wb') as out_file:
            pickle.dump(evaluation_fitness_gain_list, out_file)

        with open(os.path.join(output_dir, "organic_learning_steps_{0}_training_best_fitness_across_K".format(
                learning_steps)), 'wb') as out_file:
            pickle.dump(training_best_fitness_list, out_file)

        with open(os.path.join(output_dir, "organic_learning_steps_{0}_training_final_fitness_across_K".format(
                learning_steps)), 'wb') as out_file:
            pickle.dump(training_final_fitness_list, out_file)

        with open(os.path.join(output_dir, "organic_learning_steps_{0}_informed_percentage_across_K".format(
                learning_steps)), 'wb') as out_file:
            pickle.dump(informed_percentage_list, out_file)

        with open(os.path.join(output_dir, "organic_learning_steps_{0}_state_coverage_across_K".format(
                learning_steps)), 'wb') as out_file:
            pickle.dump(state_coverage_list, out_file)

        with open(os.path.join(output_dir, "organic_learning_steps_{0}_action_coverage_across_K".format(
                learning_steps)), 'wb') as out_file:
            pickle.dump(action_coverage_list, out_file)

        with open(os.path.join(output_dir, "organic_learning_steps_{0}_q_accuracy_across_K".format(
                learning_steps)), 'wb') as out_file:
            pickle.dump(q_accuracy_list, out_file)

        with open(os.path.join(output_dir, "organic_learning_steps_{0}_experienced_q_accuracy_across_K".format(
                learning_steps)), 'wb') as out_file:
            pickle.dump(experienced_q_accuracy_list, out_file)

        with open(os.path.join(output_dir,
                               "organic_learning_steps_{0}_evaluation_best_fitness_across_time_across_K".format(
                                   learning_steps)), 'wb') as out_file:
            pickle.dump(evaluation_best_fitness_across_time_list, out_file)

    # Keep the across-K ordering and experiment settings available for the evaluator.
    settings = {
        "N": N, "K_list": K_list, "learning_steps_list": learning_steps_list,
        "landscape_iteration": landscape_iteration,
        "individual_reset_num": individual_reset_num,
        "search_iteration": search_iteration,
        "evaluation_reset_num": evaluation_reset_num,
        "alpha": alpha, "gamma": gamma,
        "tau_learning": tau_learning, "tau_evaluation": tau_evaluation,
        "compute_accuracy": compute_accuracy, "base_seed": base_seed,
    }
    with open(os.path.join(output_dir, "organic_learning_settings"), 'wb') as out_file:
        pickle.dump(settings, out_file)

    t1 = time.time()
    now = datetime.datetime.now()
    print(now.strftime("%Y-%m-%d %H:%M:%S"))
    print("Organic Experiential Learning:",
          time.strftime("%H:%M:%S", time.gmtime(t1 - t0)))
