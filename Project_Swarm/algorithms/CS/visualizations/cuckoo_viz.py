from __future__ import annotations

import os
import json
from typing import Any, Dict, List, Tuple

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

# ensure project imports work
import sys
_root = os.path.dirname(os.path.dirname(__file__))
if _root not in sys.path:
    sys.path.insert(0, _root)

from cuckoo_search import CuckooSearch, rastrigin, make_tsp_obj
from benchmarks.rastrigin import make_counted_rastrigin
from tools.plotting import plot_convergence, boxplot_finals


def _ensure_dir(d: str):
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)


def run_cuckoo_rastrigin(dim: int = 2, n_runs: int = 5, n_iter: int = 500, n_nests: int = 50,
                         pa: float = 0.25, alpha: float = 0.01, beta: float = 1.5,
                         results_dir: str = 'results_viz') -> Dict[str, Any]:
    """Run CuckooSearch on Rastrigin multiple times and produce convergence & 3D surface (if dim==2).

    Returns a dict with histories and finals and saves PNGs to `results_dir`.
    """
    _ensure_dir(results_dir)
    histories = []
    finals = []
    for run in range(n_runs):
        obj, state = make_counted_rastrigin(dim)
        cs = CuckooSearch(obj_func=obj, dim=dim, bounds=(-5.12, 5.12),
                          n_nests=n_nests, pa=pa, alpha=alpha, beta=beta,
                          dtype='continuous', rng=42 + run)
        out = cs.optimize(n_iter=n_iter, verbose=False)
        histories.append(out['history'].tolist())
        finals.append(float(out['best_f']))
        print(f"Rastrigin run {run+1}/{n_runs} best={out['best_f']:.6g}")

    # convergence plot
    plot_convergence(histories, f"Cuckoo Rastrigin (dim={dim})", savepath=os.path.join(results_dir, f"cuckoo_rastrigin_conv_d{dim}.png"))

    # 3D surface if 2D
    if dim == 2:
        xx = np.linspace(-5.12, 5.12, 200)
        X, Y = np.meshgrid(xx, xx)
        Z = np.empty_like(X)
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                Z[i, j] = rastrigin(np.array([X[i, j], Y[i, j]]))

        fig = plt.figure(figsize=(10, 6))
        ax = fig.add_subplot(121, projection='3d')
        ax.plot_surface(X, Y, Z, cmap='viridis', linewidth=0, antialiased=False, alpha=0.9)
        ax.set_title('Rastrigin surface')
        ax.set_xlabel('x0')
        ax.set_ylabel('x1')

        ax2 = fig.add_subplot(122)
        cs = ax2.contourf(X, Y, Z, levels=50, cmap='viridis')
        fig.colorbar(cs, ax=ax2)
        ax2.set_title('Rastrigin contour')
        ax2.set_xlabel('x0')
        ax2.set_ylabel('x1')
        fig.tight_layout()
        fig.savefig(os.path.join(results_dir, f"rastrigin_surface_d2.png"), dpi=150)
        plt.close(fig)

    # boxplot of final bests
    boxplot_finals({'Cuckoo': finals}, title=f'Final best (Rastrigin d={dim})', savepath=os.path.join(results_dir, f'cuckoo_rastrigin_finals_d{dim}.png'))

    # save raw results
    with open(os.path.join(results_dir, f'cuckoo_rastrigin_results_d{dim}.json'), 'w') as f:
        json.dump({'histories': histories, 'finals': finals}, f)

    return {'histories': histories, 'finals': finals}


def sensitivity_rastrigin(param_name: str, values: List[float], dim: int = 5, n_runs: int = 5,
                          n_iter: int = 500, results_dir: str = 'results_viz') -> None:
    """Perform 1-D sensitivity analysis of a single CuckooSearch parameter (pa, alpha, beta, n_nests).

    Saves a boxplot of final fitness vs parameter value.
    """
    assert param_name in ('pa', 'alpha', 'beta', 'n_nests')
    _ensure_dir(results_dir)
    all_finals = []
    labels = []
    for v in values:
        finals = []
        for run in range(n_runs):
            obj, state = make_counted_rastrigin(dim)
            kwargs = dict(n_nests=50, pa=0.25, alpha=0.01, beta=1.5)
            kwargs[param_name] = v
            cs = CuckooSearch(obj_func=obj, dim=dim, bounds=(-5.12, 5.12), dtype='continuous', rng=100 + run, **kwargs)
            out = cs.optimize(n_iter=n_iter, verbose=False)
            finals.append(float(out['best_f']))
        all_finals.append(finals)
        labels.append(str(v))
        print(f"Param {param_name}={v} -> median final {np.median(finals):.6g}")

    # boxplot
    plt.figure(figsize=(8, 5))
    sns.boxplot(data=all_finals)
    plt.xticks(ticks=np.arange(len(labels)), labels=labels)
    plt.xlabel(param_name)
    plt.ylabel('Final best')
    plt.title(f'Sensitivity of Cuckoo on Rastrigin (dim={dim}) w.r.t {param_name}')
    plt.grid(alpha=0.3)
    plt.savefig(os.path.join(results_dir, f'sensitivity_rastrigin_{param_name}.png'), dpi=150, bbox_inches='tight')
    plt.close()


def run_cuckoo_tsp(n_nodes: int = 20, n_runs: int = 5, n_iter: int = 500, n_nests: int = 60,
                   pa: float = 0.25, results_dir: str = 'results_viz') -> Dict[str, Any]:
    """Run CuckooSearch on a random TSP instance and visualize convergence and best tours.
    Saves plots and JSON results in `results_dir`.
    """
    _ensure_dir(results_dir)
    rng = np.random.RandomState(2)
    pts = rng.rand(n_nodes, 2)
    dmat = np.sqrt(((pts[:, None, :] - pts[None, :, :]) ** 2).sum(axis=2))
    obj = make_tsp_obj(dmat)

    histories = []
    finals = []
    best_tours: List[np.ndarray] = []
    for run in range(n_runs):
        cs = CuckooSearch(obj_func=obj, dim=n_nodes, dtype='permutation', n_nests=n_nests, pa=pa, rng=10 + run)
        out = cs.optimize(n_iter=n_iter, verbose=False)
        histories.append(out['history'].tolist())
        finals.append(float(out['best_f']))
        best_tours.append(out['best'].astype(int))
        print(f"TSP run {run+1}/{n_runs} best={out['best_f']:.6g}")

    # convergence
    plot_convergence(histories, f"Cuckoo TSP (n={n_nodes})", savepath=os.path.join(results_dir, f'cuckoo_tsp_conv_n{n_nodes}.png'))

    # plot best tour of first run
    tour = best_tours[0]
    x = pts[:, 0]
    y = pts[:, 1]
    tour_x = np.concatenate([x[tour], x[[tour[0]]]])
    tour_y = np.concatenate([y[tour], y[[tour[0]]]])
    plt.figure(figsize=(6, 6))
    plt.scatter(x, y, c='k')
    for i, (xx, yy) in enumerate(zip(x, y)):
        plt.text(xx, yy, str(i), color='blue')
    plt.plot(tour_x, tour_y, '-o')
    plt.title('Best tour (run 1)')
    plt.savefig(os.path.join(results_dir, f'cuckoo_tsp_besttour_n{n_nodes}.png'), dpi=150, bbox_inches='tight')
    plt.close()

    # boxplot of finals
    boxplot_finals({'Cuckoo': finals}, title=f'Final tour length (TSP n={n_nodes})', savepath=os.path.join(results_dir, f'cuckoo_tsp_finals_n{n_nodes}.png'))

    with open(os.path.join(results_dir, f'cuckoo_tsp_results_n{n_nodes}.json'), 'w') as f:
        json.dump({'histories': histories, 'finals': finals}, f)

    return {'histories': histories, 'finals': finals, 'pts': pts, 'best_tours': best_tours}


def sensitivity_tsp(param_name: str, values: List[Any], n_nodes: int = 20, n_runs: int = 5,
                    n_iter: int = 500, results_dir: str = 'results_viz') -> None:
    assert param_name in ('n_nests', 'pa')
    _ensure_dir(results_dir)
    all_finals = []
    labels = []
    for v in values:
        finals = []
        for run in range(n_runs):
            rng = np.random.RandomState(2)
            pts = rng.rand(n_nodes, 2)
            dmat = np.sqrt(((pts[:, None, :] - pts[None, :, :]) ** 2).sum(axis=2))
            obj = make_tsp_obj(dmat)
            kwargs = dict(n_nests=60, pa=0.25)
            kwargs[param_name] = v
            cs = CuckooSearch(obj_func=obj, dim=n_nodes, dtype='permutation', rng=200 + run, **kwargs)
            out = cs.optimize(n_iter=n_iter, verbose=False)
            finals.append(float(out['best_f']))
        all_finals.append(finals)
        labels.append(str(v))
        print(f"TSP param {param_name}={v} median final {np.median(finals):.6g}")

    plt.figure(figsize=(8, 5))
    sns.boxplot(data=all_finals)
    plt.xticks(ticks=np.arange(len(labels)), labels=labels)
    plt.xlabel(param_name)
    plt.ylabel('Final tour length')
    plt.title(f'Sensitivity of Cuckoo on TSP w.r.t {param_name}')
    plt.grid(alpha=0.3)
    plt.savefig(os.path.join(results_dir, f'sensitivity_tsp_{param_name}.png'), dpi=150, bbox_inches='tight')
    plt.close()


if __name__ == '__main__':
    # quick smoke: run a few visualizations
    _outdir = 'results_viz'
    _ensure_dir(_outdir)
    print('Running Cuckoo on Rastrigin (2D) for demo...')
    run_cuckoo_rastrigin(dim=2, n_runs=4, n_iter=300, results_dir=_outdir)
    print('Running Cuckoo on TSP (demo)...')
    run_cuckoo_tsp(n_nodes=20, n_runs=3, n_iter=300, results_dir=_outdir)
