#!/usr/bin/env py39
# -*- coding: utf-8 -*-
# @Author   : Junyi
# @FileName: Organic_experiential_learning.py
# @Software : PyCharm
# Observing PEP 8 coding style

import copy
import csv
import datetime
import multiprocessing as mp
import os
import pickle
import time
from multiprocessing import Semaphore

import numpy as np
from Landscape import Landscape
from Individual import Individual


# One row per landscape, individual reset, and cumulative learning checkpoint.
# Repeated evaluation searches are averaged within each row (not independent runs).
RAW_COLUMNS = [
    "N", "K", "landscape_repetition", "individual_reset", "learning_steps",
    "evaluation_search_steps", "evaluation_reset_num", "alpha", "gamma",
    "tau_learning", "tau_evaluation", "training_best_fitness",
    "training_final_fitness", "evaluation_best_fitness",
    "evaluation_best_rank", "evaluation_final_fitness", "evaluation_fitness_gain",
    "informed_percentage", "state_coverage", "action_coverage",
    "q_accuracy", "experienced_q_accuracy",
]

# Evaluation measures (DVs), then internal-knowledge measures (mechanisms).
OUTCOME_COLUMNS = [
    "evaluation_best_fitness", "evaluation_best_rank",
    "evaluation_final_fitness", "evaluation_fitness_gain",
    "training_best_fitness", "training_final_fitness",
    "informed_percentage", "state_coverage", "action_coverage",
    "q_accuracy", "experienced_q_accuracy",
]


def evaluate_individual(individual, landscape, search_iteration, tau,
                        initial_state, evaluation_seed):
    """Test acquired knowledge without changing the learner or its RNG stream.

    A shallow copy shares the READ-ONLY Q-table during evaluation, while
    reset_state() creates a new trajectory on the copy. The same starting
    states and random draws are reused at every learning checkpoint.
    """
    training_random_state = np.random.get_state()
    try:
        np.random.seed(int(evaluation_seed))
        evaluator = copy.copy(individual)
        evaluator.reset_state(state=initial_state)
        best_fitness = evaluator.evaluate(steps=search_iteration,
                                          tau=tau, reset_state=False)
        return {
            "best_fitness": best_fitness,
            "best_rank": landscape.query_fitness_rank(state=evaluator.best_state),
            "final_fitness": evaluator.fitness,
            "fitness_gain": best_fitness - evaluator.fitness_across_time[0],
        }, evaluator.best_fitness_across_time.copy()
    finally:
        # Evaluation must not alter which actions are sampled in later learning.
        np.random.set_state(training_random_state)


# mp version: one worker generates ONE NK landscape and repeats individuals on it.
def func(N=None, K=None, individual_reset_num=1,
         learning_steps_list=None, search_iteration=100,
         evaluation_reset_num=3, alpha=0.8, gamma=0.9,
         tau_learning=1.0, tau_evaluation=0.1,
         compute_accuracy=False, base_seed=1000,
         loop=None, return_dict=None, sema=None):
    """Organic experiential learning, with no AI or external knowledge.

    Focal IV: cumulative learning_steps (number of SARSA updates).
    K varies the NK landscape's interdependence; other parameters are fixed.
    One learner continues accumulating Q-values across learning checkpoints.
    At each checkpoint, evaluation starts afresh and never updates the Q-table.
    """
    try:
        # Deterministic independent seed for each (K, landscape repetition).
        # Set base_seed=None for nondeterministic runs (like the original script).
        if base_seed is None:
            np.random.seed(None)
        else:
            worker_seed = np.random.SeedSequence(
                [int(base_seed), int(K), int(loop)]).generate_state(1)[0]
            np.random.seed(int(worker_seed))

        landscape = Landscape(N=N, K=K, norm="MaxMin")
        individual_reset_returns = []

        for individual_reset in range(individual_reset_num):
            # New experiential learner; share the same NK landscape.
            individual = Individual(N=N, landscape=landscape,
                                    alpha=alpha, gamma=gamma, tau=tau_learning)

            # Fixed test starts and random streams across learning checkpoints.
            test_states = [np.random.choice(["0", "1"], N).tolist()
                           for _ in range(evaluation_reset_num)]
            test_seeds = np.random.randint(0, 2 ** 32 - 1,
                                           size=evaluation_reset_num,
                                           dtype=np.uint32)

            for learning_steps in learning_steps_list:
                # Checkpoints are cumulative: avoid re-learning earlier steps.
                incremental_steps = learning_steps - individual.total_learning_steps
                individual.learn(steps=incremental_steps, reset_state=False)

                # Learning measures use the training softmax policy.
                learning_results = individual.learning_outcomes(
                    compute_accuracy=compute_accuracy, tau=tau_learning)

                evaluation_returns = []
                evaluation_trajectories = []
                for evaluation_reset in range(evaluation_reset_num):
                    results, trajectory = evaluate_individual(
                        individual=individual,
                        landscape=landscape,
                        search_iteration=search_iteration,
                        tau=tau_evaluation,
                        initial_state=test_states[evaluation_reset],
                        evaluation_seed=test_seeds[evaluation_reset])
                    evaluation_returns.append(results)
                    evaluation_trajectories.append(trajectory)

                # Performance is measured out-of-sample, under a fixed search budget.
                record = {
                    "N": N,
                    "K": K,
                    "landscape_repetition": loop,
                    "individual_reset": individual_reset,
                    "learning_steps": learning_steps,
                    "evaluation_search_steps": search_iteration,
                    "evaluation_reset_num": evaluation_reset_num,
                    "alpha": alpha,
                    "gamma": gamma,
                    "tau_learning": tau_learning,
                    "tau_evaluation": tau_evaluation,
                    "training_best_fitness": individual.best_fitness,
                    "training_final_fitness": individual.fitness,
                    "evaluation_best_fitness": float(np.mean(
                        [item["best_fitness"] for item in evaluation_returns])),
                    "evaluation_best_rank": float(np.mean(
                        [item["best_rank"] for item in evaluation_returns])),
                    "evaluation_final_fitness": float(np.mean(
                        [item["final_fitness"] for item in evaluation_returns])),
                    "evaluation_fitness_gain": float(np.mean(
                        [item["fitness_gain"] for item in evaluation_returns])),
                    "informed_percentage": learning_results["informed_percentage"],
                    "state_coverage": learning_results["state_coverage"],
                    "action_coverage": learning_results["action_coverage"],
                    "q_accuracy": learning_results.get("q_accuracy", np.nan),
                    "experienced_q_accuracy": learning_results.get(
                        "experienced_q_accuracy", np.nan),
                }

                # Mean best-fitness search trajectory across repeated evaluations.
                mean_trajectory = np.mean(
                    np.asarray(evaluation_trajectories, dtype=float), axis=0).tolist()
                individual_reset_returns.append((record, mean_trajectory))

        return_dict[loop] = individual_reset_returns
    finally:
        if sema is not None:
            sema.release()


def aggregate_results(rows, trajectories, K, learning_steps_list):
    """Summarize repetitions, retaining individual records for future analysis."""
    summary_rows = []
    trajectory_means = {}

    for learning_steps in learning_steps_list:
        indices = [i for i, record in enumerate(rows)
                   if record["learning_steps"] == learning_steps]
        selected = [rows[i] for i in indices]
        summary = {"N": rows[0]["N"], "K": K,
                   "learning_steps": learning_steps, "n_individual_runs": len(selected)}

        for measure in OUTCOME_COLUMNS:
            values = np.asarray([record[measure] for record in selected], dtype=float)
            # Experienced accuracy is undefined with zero learning steps.
            valid = values[np.isfinite(values)]
            summary[measure + "_mean"] = (float(np.mean(valid)) if valid.size
                                             else float("nan"))
            summary[measure + "_std"] = (float(np.std(valid, ddof=1))
                                            if valid.size > 1 else float("nan"))

        summary_rows.append(summary)
        trajectory_means[learning_steps] = np.mean(
            np.asarray([trajectories[i] for i in indices], dtype=float),
            axis=0).tolist()

    return summary_rows, trajectory_means


def run_experiment(N, K_list, landscape_iteration, individual_reset_num,
                   learning_steps_list, search_iteration, evaluation_reset_num,
                   alpha, gamma, tau_learning, tau_evaluation,
                   compute_accuracy, concurrency, base_seed, output_dir):
    """Run multiple NK landscapes in parallel, and save results after each K."""
    if not 1 <= N <= 24:
        raise ValueError("N must be between 1 and 24 (dense Q-table grows as 2^N*N).")
    if (not K_list or any(K < 0 or K >= N for K in K_list)):
        raise ValueError("Each K must lie between 0 and N-1.")
    if (not learning_steps_list or learning_steps_list[0] != 0 or
            learning_steps_list != sorted(set(learning_steps_list)) or
            any(step < 0 for step in learning_steps_list)):
        raise ValueError("learning_steps_list must be sorted, unique, and begin at 0.")
    if (landscape_iteration < 1 or individual_reset_num < 1 or
            evaluation_reset_num < 1 or search_iteration < 1 or concurrency < 1):
        raise ValueError("All iteration counts and concurrency must be positive.")
    if compute_accuracy and gamma >= 1:
        raise ValueError("Exact infinite-horizon Q accuracy requires gamma < 1.")

    os.makedirs(output_dir, exist_ok=True)
    raw_path = os.path.join(output_dir, "organic_learning_raw.csv")
    summary_path = os.path.join(output_dir, "organic_learning_summary.csv")
    summary_columns = ["N", "K", "learning_steps", "n_individual_runs"]
    for measure in OUTCOME_COLUMNS:
        summary_columns.extend([measure + "_mean", measure + "_std"])

    with open(raw_path, "w", newline="") as file:
        csv.DictWriter(file, fieldnames=RAW_COLUMNS).writeheader()
    with open(summary_path, "w", newline="") as file:
        csv.DictWriter(file, fieldnames=summary_columns).writeheader()

    for K in K_list:
        manager = mp.Manager()
        return_dict = manager.dict()
        sema = Semaphore(concurrency)
        jobs = []

        try:
            for loop in range(landscape_iteration):
                # Reap finished workers promptly to avoid exhausting OS handles.
                while True:
                    acquired = sema.acquire(timeout=0.5)
                    for index, proc in jobs[:]:
                        if proc.exitcode is not None:
                            proc.join()
                            exitcode = proc.exitcode
                            proc.close()
                            jobs.remove((index, proc))
                            if exitcode != 0:
                                if acquired:
                                    sema.release()
                                raise RuntimeError(
                                    "K={0}, landscape repetition {1} failed (exit {2})."
                                    .format(K, index, exitcode))
                    if acquired:
                        break

                proc = mp.Process(target=func, args=(
                    N, K, individual_reset_num, learning_steps_list,
                    search_iteration, evaluation_reset_num, alpha, gamma,
                    tau_learning, tau_evaluation, compute_accuracy,
                    base_seed, loop, return_dict, sema))
                try:
                    proc.start()
                except Exception:
                    sema.release()
                    raise
                jobs.append((loop, proc))

            while jobs:
                for index, proc in jobs[:]:
                    if proc.exitcode is not None:
                        proc.join()
                        exitcode = proc.exitcode
                        proc.close()
                        jobs.remove((index, proc))
                        if exitcode != 0:
                            raise RuntimeError(
                                "K={0}, landscape repetition {1} failed (exit {2})."
                                .format(K, index, exitcode))
                if jobs:
                    time.sleep(0.25)

            if len(return_dict) != landscape_iteration:
                raise RuntimeError(
                    "Missing landscape results for K={0}: expected {1}, got {2}."
                    .format(K, landscape_iteration, len(return_dict)))

            landscape_returns = [return_dict[index]
                                 for index in range(landscape_iteration)]
        finally:
            for index, proc in jobs:
                if proc.is_alive():
                    proc.terminate()
            for index, proc in jobs:
                proc.join()
                proc.close()
            manager.shutdown()

        # Flatten landscape -> individual -> learning-checkpoint repetitions.
        returns = [item for landscape_runs in landscape_returns
                   for item in landscape_runs]
        expected = landscape_iteration * individual_reset_num * len(learning_steps_list)
        if len(returns) != expected:
            raise RuntimeError("Expected {0} results for K={1}; got {2}."
                               .format(expected, K, len(returns)))

        raw_rows = [item[0] for item in returns]
        trajectories = [item[1] for item in returns]
        summary_rows, mean_trajectories = aggregate_results(
            raw_rows, trajectories, K, learning_steps_list)

        # Save after each complexity K so completed configurations are retained.
        with open(raw_path, "a", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=RAW_COLUMNS)
            writer.writerows(raw_rows)
        with open(summary_path, "a", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=summary_columns)
            writer.writerows(summary_rows)
        with open(os.path.join(output_dir,
                               "organic_learning_evaluation_trajectories_K_{0}.pkl".format(K)), "wb") as file:
            pickle.dump(mean_trajectories, file)

        for row in summary_rows:
            print("K={0:>2}, learning={1:>6}: evaluation fitness={2:.4f}, "
                  "state coverage={3:.4f}, Q accuracy={4:.4f}".format(
                      K, row["learning_steps"],
                      row["evaluation_best_fitness_mean"],
                      row["state_coverage_mean"],
                      row["q_accuracy_mean"]))

    return raw_path, summary_path


if __name__ == '__main__':
    now = datetime.datetime.now()
    print(now.strftime("%Y-%m-%d %H:%M:%S"))
    t0 = time.time()

    # Experiment: cumulative experiential learning before a fixed evaluation.
    # 0 steps = an entirely uninformed Q-table; later checkpoints share one learner.
    # With N=10 the Q-table contains 10,240 entries; include later checkpoints.
    learning_steps_list = [0, 100, 250, 500, 1000, 2000, 5000, 10000]

    # Conventional NK landscape; K is task complexity, not the focal IV.
    N = 10
    K_list = [0, 2, 4, 6, 8, 9]

    landscape_iteration = 40      # Independent NK landscapes per K.
    individual_reset_num = 5     # Independent learners on each landscape.
    search_iteration = 100       # Fixed evaluation search budget (all conditions).
    evaluation_reset_num = 3     # Fresh test starts per learner/checkpoint.

    alpha = 0.8   # SARSA learning rate (Fang et al. baseline).
    gamma = 0.9   # Weight of discounted future NK fitness rewards.
    tau_learning = 1.0    # Exploratory training softmax; calibrate for NK reward scale.
    tau_evaluation = 0.1  # More exploitative, fixed for all learning budgets.

    # Exact Q^pi accuracy is expensive at high N; coverage is always measured.
    compute_accuracy = True

    concurrency = 4      # Keep low: one dense (2^N x N) Q-table per worker.
    base_seed = 1000     # Reproducible landscapes/learners; None for random seeds.
    output_dir = "Organic_experiential_learning_results"

    raw_path, summary_path = run_experiment(
        N=N, K_list=K_list, landscape_iteration=landscape_iteration,
        individual_reset_num=individual_reset_num,
        learning_steps_list=learning_steps_list,
        search_iteration=search_iteration,
        evaluation_reset_num=evaluation_reset_num,
        alpha=alpha, gamma=gamma,
        tau_learning=tau_learning, tau_evaluation=tau_evaluation,
        compute_accuracy=compute_accuracy, concurrency=concurrency,
        base_seed=base_seed, output_dir=output_dir)

    t1 = time.time()
    print("Saved raw results:", raw_path)
    print("Saved summary:", summary_path)
    print("Organic Experiential Learning:",
          time.strftime("%H:%M:%S", time.gmtime(t1 - t0)))
