import numpy as np
import time
import sys
import tracemalloc
import matplotlib.pyplot as plt
import os

algorithms_path = r".."
if algorithms_path not in sys.path:
    sys.path.append(algorithms_path)

from algorithms.ACO.TSP.aco import *
from algorithms.PSO.PSO_discrete import pso_discrete_tsp as pso_track
from algorithms.ABC.ABC import abc_track     
from algorithms.FA.firefly_algorithm import FireflyAlgorithm as FA
from algorithms.CS.cuckoo_search import CuckooSearch
from algorithms.BFS.bfs import bfs_tsp
from algorithms.DFS.dfs import dfs_tsp


# -------------------- Hàm chi phí TSP --------------------
def tsp_cost(path, distance_matrix):
    total = 0 
    n = len(path)
    for i in range(n):
        total += distance_matrix[path[i-1], path[i]]  
    return total

# ------------------- Wrapper thuật toán -------------------
def run_aco_tsp(distance_matrix, num_ants=20, num_iterations=100):
    graph = Graph(distance_matrix)
    aco = ACO(graph, num_ants=num_ants, num_iterations=num_iterations,
              alpha=1, beta=2, q0=0.9, rho=0.1, phi=0.1)
    best_path, best_cost, history, _ = aco.run()
    return best_path, best_cost, history

def run_pso_tsp(distance_matrix, n_iter=100, n_particles=40):
    best_path, best_cost, history = pso_track(
        distance_matrix=distance_matrix,
        n_particles=n_particles,
        n_iter=n_iter,
        w=0.5, c1=1.5, c2=1.5, seed=42,
    )
    return best_path, best_cost, history

def run_abc_tsp(distance_matrix, n_iter=100, n_bees=40):
    dim = len(distance_matrix)
    def obj_fn(x):
        perm = np.argsort(x)
        return tsp_cost(perm, distance_matrix)
    history_positions, best_scores, best_pos, _, _ = abc_track(
        obj_fn=obj_fn, n_iter=n_iter, n_bees=n_bees,
        dim=dim, bound=(0, dim-1), seed=42
    )
    best_path = np.argsort(best_pos)
    return best_path, best_scores[-1], best_scores.tolist()

def run_fa_tsp(distance_matrix, n_fireflies=30, max_gen=100):
    dim = len(distance_matrix)
    def obj_fn(x):
        perm = np.argsort(x)
        return tsp_cost(perm, distance_matrix)
    fa = FA(objective_fn=obj_fn, dim=dim,
            n_fireflies=n_fireflies, max_gen=max_gen,
            alpha=0.5, beta0=1.0, gamma=0.01,
            lb=0, ub=dim-1, alpha_decay=0.97,
            problem_type="continuous", seed=42)
    best_position, best_cost, convergence_curve = fa.optimize(verbose=False)
    best_path = np.argsort(best_position)
    return best_path, best_cost, convergence_curve.tolist()

def run_cs_tsp(distance_matrix, n_nests=50, n_iter=100):
    dim = len(distance_matrix)
    def obj_fn(x):
        perm = np.argsort(x)
        return tsp_cost(perm, distance_matrix)
    cs = CuckooSearch(
        obj_func=obj_fn, dim=dim, dtype='permutation',
        n_nests=n_nests, pa=0.25, alpha=0.01, beta=1.5, rng=42
    )
    out = cs.optimize(n_iter=n_iter)
    return np.argsort(out['best']), out['best_f'], out['history'].tolist()

def run_bfs_tsp(distance_matrix):
    best_path, best_cost = bfs_tsp(distance_matrix)
    return best_path, best_cost, None  # không có history

def run_dfs_tsp(distance_matrix):
    best_path, best_cost = dfs_tsp(distance_matrix)
    return best_path, best_cost, None  # không có history


def main():
    distance_matrix = np.loadtxt(os.path.join("..", "data", "discrete", "dantzig42_d.txt"))

    algorithms = {
        "ACO": run_aco_tsp,
        "PSO": run_pso_tsp,
        "ABC": run_abc_tsp,
        "FA": run_fa_tsp,
        "CS": run_cs_tsp,
        "BFS": run_bfs_tsp,
        "DFS": run_dfs_tsp,
    }

    colors = ['r','g','b','c','m','y','k']
    plt.figure(figsize=(12,7))

    peak_memories = []  # lưu peak memory từng thuật toán
    names = []

    for (name, func), color in zip(algorithms.items(), colors):
        print(f"Running {name}...")
        tracemalloc.start()
        start_time = time.time()
        best_path, best_cost, history = func(distance_matrix)
        elapsed = time.time() - start_time
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        peak_kb = peak / 1024
        peak_memories.append(peak_kb)
        names.append(name)

        print(f"{name}: best cost={best_cost:.2f}, time={elapsed:.6f}s, peak memory={peak_kb:.2f} KB")

        # -----------------------
        # Vẽ hội tụ theo thời gian
        # -----------------------
        if history is None or len(history) == 0:
            if name == "BFS":
                plt.scatter(elapsed, best_cost, label=name, color=color, s=50, marker='o')
            elif name == "DFS":
                plt.scatter(elapsed + 1e-5, best_cost, label=name, color=color, s=50, marker='^')
        else:
            history = list(history)
            n_iter = len(history)
            times = np.linspace(0, elapsed, n_iter)
            plt.plot(times, history, label=name, color=color)

    plt.xlabel("Time (s)")
    plt.ylabel("Best distance")
    plt.title("So sánh hội tụ TSP theo thời gian - 7 thuật toán")
    plt.legend()
    plt.grid(True)
    plt.show()

    # -----------------------
    # Vẽ sơ đồ cột đo bộ nhớ
    # -----------------------
    plt.figure(figsize=(10,6))
    plt.bar(names, peak_memories, color=colors)
    plt.ylabel("Peak memory (KB)")
    plt.title("So sánh bộ nhớ sử dụng của 7 thuật toán TSP")
    plt.grid(axis='y')
    plt.show()


if __name__ == "__main__":
    main()