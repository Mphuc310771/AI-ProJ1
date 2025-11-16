from __future__ import annotations

import os
import sys
import time
import json
import argparse
from datetime import datetime
from typing import List, Dict, Any, Optional

import numpy as np

# Ensure project root (parent of experiments) is on sys.path so packages import correctly
_project_root = os.path.dirname(os.path.dirname(__file__))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from benchmarks.rastrigin import make_counted_rastrigin
from optimizers.hill_climbing import HillClimbing
from optimizers.simulated_annealing import SimulatedAnnealing
from optimizers.genetic_algorithm import GeneticAlgorithm
from cuckoo_search import CuckooSearch
from tools.plotting import plot_convergence, boxplot_finals, plot_success_rate, boxplot_memory



def best_so_far_from_vals(vals: List[float]) -> np.ndarray:
    vals = np.asarray(vals)
    return np.minimum.accumulate(vals)


def cliff_delta(a: List[float], b: List[float]) -> float:
    """Compute Cliff's delta (effect size) between two arrays."""
    a = np.asarray(a)
    b = np.asarray(b)
    n, m = len(a), len(b)
    if n == 0 or m == 0:
        return 0.0
    # count comparisons
    greater = 0
    lesser = 0
    for x in a:
        greater += np.sum(x > b)
        lesser += np.sum(x < b)
    return float((greater - lesser) / (n * m))

def mann_whitney_u_np(a: List[float], b: List[float]) -> float:
    """Mann–Whitney U (numpy-only). Trả về U-statistic."""
    a = np.asarray(a)
    b = np.asarray(b)
    n1, n2 = len(a), len(b)
    all_vals = np.concatenate([a, b])
    ranks = all_vals.argsort().argsort() + 1
    r1 = ranks[:n1].sum()
    U1 = r1 - n1 * (n1 + 1) / 2
    return float(U1)


def run_experiment(dim: int = 5, n_runs: int = 5, max_evals: int = 10000, thresholds: Optional[List[float]] = None, results_dir: str = "results_rastrigin", save_plots: bool = False):
    if thresholds is None:
        thresholds = [15 ,10 , 5]

    algos = [
        ("HillClimb", HillClimbing, {"step_scale": 0.2}),
        ("SimAnneal", SimulatedAnnealing, {"t0": 1.0, "cooling": 0.995, "step_scale": 0.2}),
        ("Genetic", GeneticAlgorithm, {"pop_size": 50}),
        ("Cuckoo", CuckooSearch, {"n_nests": 50, "alpha": 0.3, "pa": 0.25}),
    ]

    os.makedirs(results_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    summary_rows = []
    all_histories = {}
    finals_dict = {}
    mem_dict = {}

    plots_dir = None
    if save_plots:
        plots_dir = os.path.join(results_dir, 'plots')
        os.makedirs(plots_dir, exist_ok=True)

    for name, Alg, params in algos:
        print(f"Running {name} ...")
        histories = []
        finals = []
        times = []
        evals_to_thr = {thr: [] for thr in thresholds}
        mem_estimates = []

        for run in range(n_runs):
            seed = 1000 + run
            obj, state = make_counted_rastrigin(dim)
            bounds = (-5.12, 5.12)
            start = time.perf_counter()


            if name == "Cuckoo":
                cs = CuckooSearch(obj_func=obj, dim=dim, bounds=bounds, n_nests=params.get('n_nests', 50), pa=params.get('pa', 0.25), alpha=params.get('alpha', 0.3), rng=seed)
                out = cs.optimize(n_iter=min(1000, int(max_evals // max(1, params.get('n_nests', 50)))), verbose=False)
                elapsed = time.perf_counter() - start
                evals = obj.state['count']
                history = best_so_far_from_vals(obj.state['vals'])
                # memory estimate
                mem_bytes = params.get('n_nests', 50) * dim * 8
            else:
                alg = Alg(obj_func=obj, dim=dim, bounds=bounds, rng=seed, **params)
                out = alg.run(max_evals=max_evals)
                elapsed = out.get('time', 0.0)
                evals = obj.state['count']
                history = best_so_far_from_vals(obj.state['vals'])
                # attempt to estimate memory usage from attributes
                if hasattr(alg, 'pop_size'):
                    mem_bytes = int(getattr(alg, 'pop_size') * dim * 8)
                else:
                    mem_bytes = dim * 8

            histories.append(history)
            finals.append(float(out['best_f']))
            times.append(float(elapsed))
            mem_estimates.append(int(mem_bytes))

            # compute evals to thresholds
            for thr in thresholds:
                idx = np.where(history <= thr)[0]
                evals_to_thr[thr].append(int(idx[0] + 1) if idx.size > 0 else None)

            # save per-run JSON
            run_record: Dict[str, Any] = {
                'algorithm': name,
                'seed': int(seed),
                'dim': int(dim),
                'params': params,
                'final_best_f': float(out['best_f']),
                'evals_used': int(evals),
                'time_s': float(elapsed),
                'memory_estimate_bytes': int(mem_bytes),
                'evals_to_thresholds': {str(k): v for k, v in evals_to_thr.items()},
                'history': history.tolist(),
            }
            run_fn = os.path.join(results_dir, f"run_{name}_d{dim}_s{seed}.json")
            with open(run_fn, 'w') as f:
                json.dump(run_record, f)

            print(f"  run {run+1}/{n_runs}: evals={evals}, best_f={finals[-1]:.6g}, time={times[-1]:.3f}s")

        # aggregate per-algorithm
        results_row = {
            'algorithm': name,
            'dim': dim,
            'n_runs': n_runs,
            'final_mean': float(np.mean(finals)),
            'final_std': float(np.std(finals)),
            'time_mean_s': float(np.mean(times)),
            'time_std_s': float(np.std(times)),
            'memory_mean_bytes': int(np.mean(mem_estimates)),
        }
        # add success rates per threshold
        for thr in thresholds:
            reached = sum(1 for v in evals_to_thr[thr] if v is not None)
            results_row[f'success_rate_thr_{thr}'] = float(reached) / float(n_runs)

        summary_rows.append(results_row)
        all_histories[name] = histories
        finals_dict[name] = finals
        mem_dict[name] = mem_estimates

        # per-algo plots
        if save_plots:
            plot_convergence(histories, f"Convergence: {name} (Rastrigin {dim}D)", savepath=os.path.join(plots_dir, f"convergence_{name}_d{dim}.png"))
        else:
            plot_convergence(histories, f"Convergence: {name} (Rastrigin {dim}D)")

    # save aggregated summary
    summary_csv = os.path.join(results_dir, f"summary_{timestamp}.csv")
    import pandas as pd
    df = pd.DataFrame(summary_rows)
    df.to_csv(summary_csv, index=False)
    print(f"Saved summary to {summary_csv}")

    # combined plots
    if save_plots:
        boxplot_finals(finals_dict, title=f"Final fitness (Rastrigin {dim}D)", savepath=os.path.join(plots_dir, f"boxplot_finals_d{dim}.png"))
        plot_success_rate(all_histories, thresholds=thresholds, title=f"Success rate (Rastrigin {dim}D)", save_dir=plots_dir)
    else:
        boxplot_finals(finals_dict, title=f"Final fitness (Rastrigin {dim}D)")
        plot_success_rate(all_histories, thresholds=thresholds, title=f"Success rate (Rastrigin {dim}D)")

    # pairwise statistical tests
    print('\nPairwise Mann-Whitney U tests (p-value) and Cliff\'s delta:')
    names = list(finals_dict.keys())
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a = finals_dict[names[i]]
            b = finals_dict[names[j]]
            U = mann_whitney_u_np(a, b)
            # p-value không tính vì không dùng scipy, đặt None
            pval = None
            delta = cliff_delta(a, b)
            print(f"{names[i]} vs {names[j]}: U={U:.3f}, p={pval}, cliff_delta={delta:.4f}")

    # Memory summary and combined plot
    # Pretty-print memory summary using best unit
    def _format_bytes_stats(arr: np.ndarray):
        maxv = arr.max()
        if maxv >= 1024 ** 3:
            scale = 1024 ** 3
            unit = 'GiB'
        elif maxv >= 1024 ** 2:
            scale = 1024 ** 2
            unit = 'MiB'
        elif maxv >= 1024:
            scale = 1024
            unit = 'KiB'
        else:
            scale = 1
            unit = 'bytes'
        return (arr.mean() / scale, arr.std() / scale, arr.min() / scale, arr.max() / scale, unit)

    print('\nMemory usage summary (per-algorithm):')
    for name, mems in mem_dict.items():
        if len(mems) == 0:
            continue
        arr = np.array(mems)
        mean, std, mn, mx, unit = _format_bytes_stats(arr)
        print(f"  {name}: mean={mean:.2f} {unit}, std={std:.2f} {unit}, min={mn:.2f} {unit}, max={mx:.2f} {unit}")

    if save_plots:
        try:
            boxplot_memory(mem_dict, title=f"Peak memory usage (Rastrigin {dim}D)", savepath=os.path.join(plots_dir, f"memory_boxplot_d{dim}.png"))
        except Exception:
            print('Could not save memory plot')
    else:
        try:
            boxplot_memory(mem_dict, title=f"Peak memory usage (Rastrigin {dim}D)")
        except Exception:
            print('Could not plot memory; ensure matplotlib is available')



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dim', type=int, default=5)
    parser.add_argument('--n_runs', type=int, default=5)
    parser.add_argument('--max_evals', type=int, default=5000)
    parser.add_argument('--results_dir', type=str, default='results_rastrigin')
    parser.add_argument('--save-plots', action='store_true', dest='save_plots')
    args = parser.parse_args()
    run_experiment(dim=args.dim, n_runs=args.n_runs, max_evals=args.max_evals, results_dir=args.results_dir, save_plots=args.save_plots)
