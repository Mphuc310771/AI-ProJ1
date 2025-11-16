from __future__ import annotations
import os
import sys
import time
import json
from typing import Any, Dict, List, Tuple
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def _ensure_project_path():
    # Add Project_Swarm/algorithms parent to sys.path so local modules import
    _project_root = os.path.dirname(os.path.dirname(__file__))
    if _project_root not in sys.path:
        sys.path.insert(0, _project_root)


_ensure_project_path()

import importlib.util

# helper to load module by path if present
def _load_module_from_path(path: str, module_name: str):
    if not os.path.exists(path):
        return None
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None:
        return None
    mod = importlib.util.module_from_spec(spec)
    loader = spec.loader
    assert loader is not None
    try:
        loader.exec_module(mod)
        return mod
    except ImportError:
        # Fallback: some modules use relative imports (e.g. "from .base import ...")
        # which fail when loading a file outside a package. Attempt to rewrite
        # leading relative imports by stripping a single leading dot and
        # execute the modified source with the module directory on sys.path.
        try:
            import re

            with open(path, 'r', encoding='utf-8') as fh:
                src = fh.read()

            # remove single leading dots from "from .name import" and "import .name"
            src_fixed = re.sub(r'from\s+\.(?=\w)', 'from ', src)
            src_fixed = re.sub(r'import\s+\.(?=\w)', 'import ', src_fixed)

            module_dir = os.path.dirname(path)
            inserted = False
            if module_dir not in sys.path:
                sys.path.insert(0, module_dir)
                inserted = True

            # execute modified source in module namespace
            code = compile(src_fixed, path, 'exec')
            exec(code, mod.__dict__)
            # set some metadata
            mod.__file__ = path
            mod.__package__ = None
            # register in sys.modules
            sys.modules[module_name] = mod
            if inserted:
                try:
                    sys.path.remove(module_dir)
                except Exception:
                    pass
            return mod
        except Exception:
            # give up
            raise


def _find_file(root: str, filename: str) -> str | None:
    """Recursively search `root` for a file named `filename` and return its path or None."""
    for dirpath, _, files in os.walk(root):
        if filename in files:
            return os.path.join(dirpath, filename)
    return None

# try to locate modules in the algorithms folder
PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
ALG_ROOT = os.path.join(PROJECT_ROOT, 'algorithms')

available_algos: Dict[str, bool] = {}

# HillClimbing: search for any hill_climbing.py under algorithms/
HillClimbing = None
hc_path = _find_file(ALG_ROOT, 'hill_climbing.py')
mod = _load_module_from_path(hc_path, 'hcmod') if hc_path else None
if mod is not None and hasattr(mod, 'HillClimbing'):
    HillClimbing = getattr(mod, 'HillClimbing')
    available_algos['HillClimb'] = True

# CuckooSearch: search for cuckoo_search.py
CuckooSearch = None
cs_path = _find_file(ALG_ROOT, 'cuckoo_search.py')
mod = _load_module_from_path(cs_path, 'csmod') if cs_path else None
if mod is not None and hasattr(mod, 'CuckooSearch'):
    CuckooSearch = getattr(mod, 'CuckooSearch')
    available_algos['Cuckoo'] = True

# ACOR (search for acor.py)
ACOR = None
acor_path = _find_file(ALG_ROOT, 'acor.py')
mod = _load_module_from_path(acor_path, 'acor_mod') if acor_path else None
if mod is not None and hasattr(mod, 'ACOR'):
    ACOR = getattr(mod, 'ACOR')
    available_algos['ACOR'] = True

# PSO (search for PSO.py or pso.py)
pso_track = None
pso_path = _find_file(ALG_ROOT, 'PSO.py') or _find_file(ALG_ROOT, 'pso.py')
mod = _load_module_from_path(pso_path, 'psomod') if pso_path else None
if mod is not None and hasattr(mod, 'pso_track'):
    pso_track = getattr(mod, 'pso_track')
    available_algos['PSO'] = True

# ABC (search for ABC.py or abc.py)
abc_track = None
abc_path = _find_file(ALG_ROOT, 'ABC.py') or _find_file(ALG_ROOT, 'abc.py')
mod = _load_module_from_path(abc_path, 'abcmod') if abc_path else None
if mod is not None and hasattr(mod, 'abc_track'):
    abc_track = getattr(mod, 'abc_track')
    available_algos['ABC'] = True

# Firefly (search for firefly_algorithm.py or firefly.py)
FireflyAlgorithm = None
fa_path = _find_file(ALG_ROOT, 'firefly_algorithm.py') or _find_file(ALG_ROOT, 'firefly.py')
mod = _load_module_from_path(fa_path, 'famod') if fa_path else None
if mod is not None and hasattr(mod, 'FireflyAlgorithm'):
    FireflyAlgorithm = getattr(mod, 'FireflyAlgorithm')
    available_algos['FA'] = True


def rastrigin(x: np.ndarray) -> float:
    A = 10.0
    x = np.asarray(x)
    return float(A * x.size + np.sum(x ** 2 - A * np.cos(2 * np.pi * x)))


def rastrigin_vec(X: np.ndarray):
    """Vectorized rastrigin: accepts shape (n, d) or (d,) and returns array or scalar as appropriate."""
    X = np.asarray(X)
    A = 10.0
    if X.ndim == 1:
        d = X.shape[0]
        return float(A * d + np.sum(X ** 2 - A * np.cos(2 * np.pi * X)))
    else:
        d = X.shape[-1]
        return A * d + np.sum(X ** 2 - A * np.cos(2 * np.pi * X), axis=-1)


def run_single_trial(algo_name: str, dim: int, seed: int, budget: int) -> Dict[str, Any]:
    """Run a single trial of algorithm `algo_name` on Rastrigin with `dim`, return dict with history, final, time, memory_bytes."""
    rng = np.random.RandomState(seed)
    bounds = (-5.12, 5.12)
    result: Dict[str, Any] = {'algorithm': algo_name, 'seed': seed, 'dim': dim}

    if algo_name == 'HillClimb' and 'HillClimb' in available_algos:
        # Use Optimizer API: run(max_evals)
        alg = HillClimbing(obj_func=rastrigin, dim=dim, bounds=bounds, rng=seed, step_scale=0.1)
        t0 = time.perf_counter()
        out = alg.run(max_evals=budget)
        elapsed = time.perf_counter() - t0
        history = out.get('history', [])
        final = out.get('best_f', float('nan'))
        mem_bytes = dim * 8

        result.update({'history': np.asarray(history).tolist(), 'final': float(final), 'time_s': float(elapsed), 'memory_bytes': int(mem_bytes)})
        return result

    if algo_name == 'Cuckoo' and 'Cuckoo' in available_algos:
        n_nests = 50
        n_iter = max(1, int(budget // max(1, n_nests)))
        cs = CuckooSearch(obj_func=rastrigin, dim=dim, bounds=bounds, n_nests=n_nests, pa=0.25, alpha=0.3, rng=seed)
        t0 = time.perf_counter()
        out = cs.optimize(n_iter=n_iter, verbose=False)
        elapsed = time.perf_counter() - t0
        history = out.get('history', np.array([])).tolist()
        final = float(out.get('best_f', float('nan')))
        mem_bytes = int(n_nests * dim * 8)
        result.update({'history': history, 'final': final, 'time_s': elapsed, 'memory_bytes': mem_bytes})
        return result

    if algo_name == 'PSO' and 'PSO' in available_algos:
        n_particles = 40
        n_iter = max(1, int(budget // n_particles))
        t0 = time.perf_counter()
        # PSO pso_track signature: pso_track(func, n_iter, n_particles, dim, bound=...)
        out = pso_track(rastrigin_vec, n_iter=n_iter, n_particles=n_particles, dim=dim, bound=bounds, seed=seed)
        # support multiple return formats defensively
        if isinstance(out, (tuple, list)):
            if len(out) >= 2:
                # common tuple: (history_positions, best_scores, best_pos, iters, fe)
                if len(out) == 5:
                    hist_pos, best_scores, best_pos, iters_ran, fe = out
                elif len(out) == 4:
                    hist_pos, best_scores, best_pos, iters_ran = out
                    fe = n_particles * int(iters_ran)
                else:
                    # try to find best_scores in the tuple
                    # assume second element
                    hist_pos = out[0] if len(out) > 0 else None
                    best_scores = out[1] if len(out) > 1 else out
                    best_pos = out[2] if len(out) > 2 else None
                    iters_ran = int(len(best_scores)) if hasattr(best_scores, '__len__') else 0
                    fe = n_particles * int(iters_ran)
            else:
                # single-item tuple
                best_scores = out[0]
                hist_pos = None
                best_pos = None
                iters_ran = int(len(best_scores)) if hasattr(best_scores, '__len__') else 0
                fe = n_particles * int(iters_ran)
        else:
            # single array-like or scalar returned
            best_scores = out
            hist_pos = None
            best_pos = None
            iters_ran = int(len(best_scores)) if hasattr(best_scores, '__len__') else 0
            fe = n_particles * int(iters_ran)

        # coerce scalar -> 1-D array to avoid subscript errors
        best_scores = np.atleast_1d(np.asarray(best_scores))
        elapsed = time.perf_counter() - t0
        history = best_scores.tolist() if best_scores is not None else []
        final = float(best_scores[-1]) if best_scores.size > 0 else float('nan')
        mem_bytes = int(n_particles * dim * 8)
        result.update({'history': history, 'final': final, 'time_s': elapsed, 'memory_bytes': mem_bytes, 'fe': int(fe)})
        return result

    if algo_name == 'ACOR' and 'ACOR' in available_algos:
        ac = ACOR(dim=dim, K=10, ants=40, xi=0.85, iterations=max(10, int(budget // 50)))
        t0 = time.perf_counter()
        sol = ac.optimize()
        elapsed = time.perf_counter() - t0
        history = getattr(ac, 'best_history', [])
        final = float(history[-1]) if len(history) > 0 else float('nan')
        mem_bytes = int(ac.K * dim * 8) if hasattr(ac, 'K') else dim * 8
        result.update({'history': history, 'final': final, 'time_s': elapsed, 'memory_bytes': mem_bytes})
        return result

    if algo_name == 'ABC' and 'ABC' in available_algos:
        n_food = 40
        n_iter = max(1, int(budget // n_food))
        t0 = time.perf_counter()
        # abc_track expects keyword 'obj_fn' for the objective
        out = abc_track(obj_fn=rastrigin_vec, n_iter=n_iter, n_bees=n_food, dim=dim, bound=bounds, seed=seed)
        # out: history_positions, best_scores, best_pos, iterations_ran, fe
        if isinstance(out, (list, tuple)) and len(out) >= 2:
            _, best_scores, *_ = out
        else:
            best_scores = out
        elapsed = time.perf_counter() - t0
        best_scores = np.atleast_1d(np.asarray(best_scores))
        history = best_scores.tolist() if best_scores is not None else []
        final = float(best_scores[-1]) if best_scores.size > 0 else float('nan')
        mem_bytes = int(n_food * dim * 8)
        # fe may be missing; try to extract
        fe = int(out[-1]) if isinstance(out, (list, tuple)) and len(out) >= 5 else int(n_food * int(max(1, n_iter)))
        result.update({'history': history, 'final': final, 'time_s': elapsed, 'memory_bytes': mem_bytes, 'fe': int(fe)})
        return result

    if algo_name == 'FA' and 'FA' in available_algos:
        n_fireflies = 40
        n_iter = max(1, int(budget // n_fireflies))
        # FireflyAlgorithm signature: FireflyAlgorithm(objective_fn, dim, n_fireflies=..., max_gen=..., lb=..., ub=...)
        fa = FireflyAlgorithm(objective_fn=rastrigin, dim=dim, n_fireflies=n_fireflies, max_gen=n_iter, lb=bounds[0], ub=bounds[1])
        t0 = time.perf_counter()
        best_pos, best_val, curve = fa.optimize(verbose=False)
        elapsed = time.perf_counter() - t0
        history = np.asarray(curve).tolist() if hasattr(curve, '__iter__') else [float(curve)]
        final = float(best_val)
        mem_bytes = int(n_fireflies * dim * 8)
        result.update({'history': history, 'final': final, 'time_s': elapsed, 'memory_bytes': mem_bytes})
        return result

    raise RuntimeError(f"Algorithm {algo_name} not implemented or not available in this environment")


def run_comparison(algorithms: List[str], dim: int = 5, runs: int = 5, budget: int = 2000, results_dir: str = 'results_compare'):
    os.makedirs(results_dir, exist_ok=True)
    all_histories: Dict[str, List[List[float]]] = {a: [] for a in algorithms}
    finals: Dict[str, List[float]] = {a: [] for a in algorithms}
    times: Dict[str, List[float]] = {a: [] for a in algorithms}
    mems: Dict[str, List[int]] = {a: [] for a in algorithms}

    for a in algorithms:
        if a not in available_algos and a not in ('HillClimb', 'Cuckoo'):
            print(f"Warning: algorithm {a} not available; skipping")
            continue
        print(f"Running algorithm: {a}")
        for r in range(runs):
            seed = 1000 + r
            try:
                res = run_single_trial(a, dim=dim, seed=seed, budget=budget)
            except Exception as e:
                print(f"  Error running {a}: {e}")
                break
            all_histories[a].append(res['history'])
            finals[a].append(res['final'])
            times[a].append(res['time_s'])
            mems[a].append(res['memory_bytes'])
            with open(os.path.join(results_dir, f"run_{a}_d{dim}_s{seed}.json"), 'w') as f:
                json.dump(res, f)

    # Combined convergence: pad and compute mean
    T = 0
    for a, hs in all_histories.items():
        for h in hs:
            T = max(T, len(h))

    padded = {}
    for a, hs in all_histories.items():
        if len(hs) == 0:
            continue
        mats = np.zeros((len(hs), T))
        for i, h in enumerate(hs):
            L = len(h)
            mats[i, :L] = h
            if L < T:
                mats[i, L:] = h[-1]
        padded[a] = mats

    plt.figure(figsize=(10, 6))
    for a, mat in padded.items():
        mean = np.mean(mat, axis=0)
        std = np.std(mat, axis=0)
        x = np.arange(mean.size)
        plt.plot(x, mean, label=a)
        plt.fill_between(x, mean - std, mean + std, alpha=0.12)
    plt.xlabel('Iteration')
    plt.ylabel('Best objective (lower better)')
    plt.title(f'Convergence comparison (Rastrigin d={dim})')
    plt.legend()
    plt.grid(alpha=0.3)
    conv_path = os.path.join(results_dir, f'convergence_compare_d{dim}.png')
    plt.savefig(conv_path, dpi=180, bbox_inches='tight')
    plt.close()
    print('Saved convergence comparison to', conv_path)

    # time bar chart
    algs = [a for a in algorithms if len(times.get(a, [])) > 0]
    means = [np.mean(times[a]) for a in algs]
    stds = [np.std(times[a]) for a in algs]
    plt.figure(figsize=(8, 4))
    sns.barplot(x=algs, y=means)
    for i, v in enumerate(means):
        plt.text(i, v + 1e-6, f"{v:.3f}", ha='center', va='bottom')
    plt.ylabel('Time (s)')
    plt.title('Average runtime per algorithm')
    plt.grid(axis='y', alpha=0.3)
    tpath = os.path.join(results_dir, f'time_compare_d{dim}.png')
    plt.savefig(tpath, dpi=180, bbox_inches='tight')
    plt.close()
    print('Saved time comparison to', tpath)

    # memory bar chart
    means_m = [np.mean(mems[a]) if len(mems[a])>0 else 0 for a in algs]
    plt.figure(figsize=(8, 4))
    sns.barplot(x=algs, y=means_m)
    for i, v in enumerate(means_m):
        plt.text(i, v + 1e-6, f"{int(v)}", ha='center', va='bottom')
    plt.ylabel('Memory estimate (bytes)')
    plt.title('Estimated memory per algorithm')
    plt.grid(axis='y', alpha=0.3)
    mpath = os.path.join(results_dir, f'memory_compare_d{dim}.png')
    plt.savefig(mpath, dpi=180, bbox_inches='tight')
    plt.close()
    print('Saved memory comparison to', mpath)

    # boxplot of finals
    plt.figure(figsize=(8, 5))
    data = [finals[a] for a in algs]
    sns.boxplot(data=data)
    plt.xticks(ticks=np.arange(len(algs)), labels=algs)
    plt.ylabel('Final best')
    plt.title('Final solution distribution across runs')
    bpath = os.path.join(results_dir, f'finals_boxplot_d{dim}.png')
    plt.savefig(bpath, dpi=180, bbox_inches='tight')
    plt.close()
    print('Saved finals boxplot to', bpath)

    return {'histories': all_histories, 'finals': finals, 'times': times, 'mems': mems, 'padded': padded}


if __name__ == '__main__':
    algos = ['HillClimb', 'ACOR', 'PSO', 'ABC', 'FA', 'Cuckoo']
    outdir = os.path.join(os.path.dirname(__file__), '..', 'results_compare_continuous')
    outdir = os.path.abspath(outdir)
    res = run_comparison(algorithms=algos, dim=5, runs=10, budget=300, results_dir=outdir)
    print('Done. Results saved to', outdir)
