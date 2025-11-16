from __future__ import annotations

import os
import time
import json
from typing import List, Dict, Any, Optional, Tuple

import numpy as np

# ensure imports from project
import sys
_project_root = os.path.dirname(os.path.dirname(__file__))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from cuckoo_search import make_tsp_obj, CuckooSearch
from tools.plotting import plot_convergence, boxplot_finals, plot_success_rate


def make_counted_tsp(distance_matrix: np.ndarray):
    base_obj = make_tsp_obj(distance_matrix)
    state = {"count": 0, "vals": []}

    def obj(perm: np.ndarray) -> float:
        state["count"] += 1
        v = float(base_obj(perm))
        state["vals"].append(v)
        return v

    obj.state = state
    return obj, state


def best_so_far_from_vals(vals: List[float]) -> np.ndarray:
    arr = np.asarray(vals)
    return np.minimum.accumulate(arr)


def random_tsp_instances(n_nodes: int, seed: int = 0) -> np.ndarray:
    rng = np.random.RandomState(seed)
    pts = rng.rand(n_nodes, 2)
    dmat = np.sqrt(((pts[:, None, :] - pts[None, :, :]) ** 2).sum(axis=2))
    return dmat


def swap_neighbor(perm: np.ndarray, rng: np.random.RandomState) -> np.ndarray:
    p = perm.copy()
    i, j = rng.randint(0, p.size, size=2)
    p[i], p[j] = p[j], p[i]
    return p


def hill_climbing_tsp(obj_func, dim: int, rng: int, max_evals: int) -> Dict[str, Any]:
    rng = np.random.RandomState(rng)
    perm = np.arange(dim)
    rng.shuffle(perm)
    f = obj_func(perm.copy())
    best_perm = perm.copy()
    best_f = f
    history_vals = list(obj_func.state['vals'])
    evals = obj_func.state['count']
    while evals < max_evals:
        cand = swap_neighbor(perm, rng)
        fnew = obj_func(cand.copy())
        evals = obj_func.state['count']
        if fnew < f:
            perm = cand
            f = fnew
            if f < best_f:
                best_f = f
                best_perm = perm.copy()
        history_vals.append(best_f)
    return {"best": best_perm, "best_f": float(best_f), "history": best_so_far_from_vals(obj_func.state['vals']), "evals": evals}


def simulated_annealing_tsp(obj_func, dim: int, rng: int, max_evals: int, t0: float = 1.0, cooling: float = 0.995) -> Dict[str, Any]:
    rng = np.random.RandomState(rng)
    perm = np.arange(dim)
    rng.shuffle(perm)
    f = obj_func(perm.copy())
    best_perm = perm.copy()
    best_f = f
    T = t0
    evals = obj_func.state['count']
    while evals < max_evals:
        cand = swap_neighbor(perm, rng)
        fnew = obj_func(cand.copy())
        evals = obj_func.state['count']
        delta = fnew - f
        if delta < 0 or rng.rand() < np.exp(-delta / max(1e-12, T)):
            perm = cand
            f = fnew
            if f < best_f:
                best_f = f
                best_perm = perm.copy()
        T *= cooling
    return {"best": best_perm, "best_f": float(best_f), "history": best_so_far_from_vals(obj_func.state['vals']), "evals": evals}


def order_crossover(p1: np.ndarray, p2: np.ndarray, rng: np.random.RandomState) -> np.ndarray:
    n = p1.size
    a, b = sorted(rng.choice(n, size=2, replace=False))
    child = -np.ones(n, dtype=int)
    child[a:b+1] = p1[a:b+1]
    fill = [x for x in p2 if x not in child]
    idx = 0
    for i in range(n):
        if child[i] == -1:
            child[i] = fill[idx]
            idx += 1
    return child


def genetic_tsp(obj_func, dim: int, rng: int, max_evals: int, pop_size: int = 50, crossover_rate: float = 0.9, mutation_rate: float = 0.1) -> Dict[str, Any]:
    rng = np.random.RandomState(rng)
    pop = np.array([rng.permutation(dim) for _ in range(pop_size)])
    fitness = np.array([obj_func(ind.copy()) for ind in pop])
    evals = obj_func.state['count']
    best_idx = int(np.argmin(fitness))
    best = pop[best_idx].copy()
    best_f = float(fitness[best_idx])
    while evals < max_evals:
        new_pop = np.empty_like(pop)
        for i in range(pop_size):
            # tournament selection
            a, b = rng.randint(0, pop_size, size=2)
            p1 = pop[a] if fitness[a] < fitness[b] else pop[b]
            a, b = rng.randint(0, pop_size, size=2)
            p2 = pop[a] if fitness[a] < fitness[b] else pop[b]
            if rng.rand() < crossover_rate:
                child = order_crossover(p1, p2, rng)
            else:
                child = p1.copy()
            # mutation: swap
            if rng.rand() < mutation_rate:
                i1, i2 = rng.randint(0, dim, size=2)
                child[i1], child[i2] = child[i2], child[i1]
            new_pop[i] = child
        pop = new_pop
        fitness = np.array([obj_func(ind.copy()) for ind in pop])
        evals = obj_func.state['count']
        cur_best_idx = int(np.argmin(fitness))
        if fitness[cur_best_idx] < best_f:
            best_f = float(fitness[cur_best_idx])
            best = pop[cur_best_idx].copy()
    return {"best": best, "best_f": float(best_f), "history": best_so_far_from_vals(obj_func.state['vals']), "evals": evals}


def run_tsp_experiment(n_nodes: int = 20, n_runs: int = 5, max_evals: int = 5000, results_dir: str = "results_TSP", save_plots: bool = True):
    os.makedirs(results_dir, exist_ok=True)
    plots_dir = None
    if save_plots:
        plots_dir = os.path.join(results_dir, 'plots')
        os.makedirs(plots_dir, exist_ok=True)

    dmat = random_tsp_instances(n_nodes, seed=0)
    obj_factory = lambda: make_counted_tsp(dmat)

    algos = [
        ("HillClimb", hill_climbing_tsp),
        ("SimAnneal", simulated_annealing_tsp),
        ("Genetic", genetic_tsp),
        ("Cuckoo", None),
    ]

    results: Dict[str, Any] = {}
    for name, func in algos:
        histories = []
        finals = []
        times = []
        for run in range(n_runs):
            seed = 1000 + run
            obj, state = obj_factory()
            start = time.perf_counter()
            if name == "Cuckoo":
                cs = CuckooSearch(obj_func=obj, dim=n_nodes, dtype='permutation', n_nests=60, rng=seed)
                out = cs.optimize(n_iter=min(1000, int(max_evals // 10)), verbose=False)
                elapsed = time.perf_counter() - start
                evals = obj.state['count']
                history = best_so_far_from_vals(obj.state['vals'])
                final = out['best_f']
            else:
                out = func(obj, n_nodes, seed, max_evals)
                elapsed = 0.0  # time inside func not measured precisely
                evals = out['evals']
                history = out['history']
                final = out['best_f']

            histories.append(history)
            finals.append(final)
            times.append(elapsed)
            # save run
            run_record = {
                'algorithm': name,
                'seed': seed,
                'n_nodes': n_nodes,
                'final': final,
                'evals': evals,
                'time_s': elapsed,
                'history': history.tolist(),
            }
            with open(os.path.join(results_dir, f"run_{name}_n{n_nodes}_s{seed}.json"), 'w') as f:
                json.dump(run_record, f)
            print(f"{name} run {run+1}/{n_runs}: evals={evals}, final={final:.6g}")

        results[name] = {"histories": histories, "finals": finals, "times": times}
        # plots
        if save_plots:
            plot_convergence(histories, f"Convergence: {name} (TSP {n_nodes})", savepath=os.path.join(plots_dir, f"convergence_{name}_n{n_nodes}.png"))
        else:
            plot_convergence(histories, f"Convergence: {name} (TSP {n_nodes})")

    # combined plots
    boxplot_finals({k: v['finals'] for k, v in results.items()}, title=f"Final tour length (TSP {n_nodes})", savepath=os.path.join(plots_dir, f"boxplot_finals_n{n_nodes}.png") if save_plots else None)
    # derive numeric thresholds from final results (25/50/75 percentiles)
    all_finals = np.concatenate([np.asarray(v['finals']) for v in results.values()]) if results else np.array([])
    if all_finals.size > 0:
        thr_vals = tuple(np.percentile(all_finals, [25, 50, 75]))
    else:
        thr_vals = (1e-3,)

    plot_success_rate({k: v['histories'] for k, v in results.items()}, thresholds=thr_vals, title=f"Success rate (TSP {n_nodes})", save_dir=plots_dir if save_plots else None)

    return results


if __name__ == '__main__':
    run_tsp_experiment(n_nodes=20, n_runs=5, max_evals=1000, save_plots=True)
