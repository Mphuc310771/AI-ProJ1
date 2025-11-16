from __future__ import annotations

import os
import json
from typing import Any, Dict, List, Tuple
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

# ------------------------------------------------------------
# Ensure folder always inside the same directory as this file
# ------------------------------------------------------------
_base_dir = os.path.dirname(os.path.abspath(__file__))
_default_results_dir = os.path.join(_base_dir, "results_viz")

def _ensure_dir(d: str | None):
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)


# project imports
from cuckoo_search import CuckooSearch, rastrigin, make_tsp_obj
from benchmarks.rastrigin import make_counted_rastrigin
from tools.plotting import plot_convergence, boxplot_finals


# ------------------------------------------------------------
# Surface plot
# ------------------------------------------------------------
def show_2d_surface(func_2d, xlim=(-5.12, 5.12), ylim=(-5.12, 5.12),
                    resolution=200, title='Objective Function'):
    x = np.linspace(*xlim, resolution)
    y = np.linspace(*ylim, resolution)
    X, Y = np.meshgrid(x, y)

    pts = np.stack([X, Y], axis=-1)
    Z = np.apply_along_axis(func_2d, -1, pts)

    fig = plt.figure(figsize=(10, 6))

    ax = fig.add_subplot(121, projection="3d")
    ax.plot_surface(X, Y, Z, cmap="viridis", linewidth=0, antialiased=False, alpha=0.9)
    ax.set_title(f"{title} Surface")

    ax2 = fig.add_subplot(122)
    cs = ax2.contourf(X, Y, Z, levels=50, cmap="viridis")
    fig.colorbar(cs, ax=ax2)
    ax2.set_title(f"{title} Contour")

    plt.tight_layout()
    plt.show()


# ------------------------------------------------------------
# Utility
# ------------------------------------------------------------
def _half_convergence_iter(history: List[float]) -> int:
    h = np.asarray(history)
    if h.size == 0:
        return 0
    init = float(h[0])
    final = float(h[-1])
    if init == final:
        return int(h.size)
    target = init - 0.5 * (init - final)
    idx = np.where(h <= target)[0]
    return int(idx[0] + 1) if idx.size > 0 else int(h.size)


# ------------------------------------------------------------
# Rastrigin (continuous)
# ------------------------------------------------------------
def run_cuckoo_rastrigin(
    dim: int = 4,
    n_runs: int = 5,
    n_iter: int = 500,
    n_nests: int = 50,
    pa: float = 0.25,
    alpha: float = 0.01,
    beta: float = 1.5,
    results_dir: str | None = None,
) -> Dict[str, Any]:

    if results_dir is None:
        results_dir = _default_results_dir
    _ensure_dir(results_dir)

    histories = []
    finals = []

    for run in range(n_runs):
        obj, _ = make_counted_rastrigin(dim)

        cs = CuckooSearch(
            obj_func=obj,
            dim=dim,
            bounds=(-5.12, 5.12),
            n_nests=n_nests,
            pa=pa,
            alpha=alpha,
            beta=beta,
            dtype="continuous",
            rng=42 + run,
        )
        out = cs.optimize(n_iter=n_iter, verbose=False)
        histories.append(out["history"].tolist())
        finals.append(float(out["best_f"]))
        print(f"Rastrigin run {run+1}/{n_runs} best={out['best_f']:.6g}")

    plot_convergence(
        histories,
        f"Cuckoo Rastrigin (dim={dim})",
        savepath=os.path.join(results_dir, f"cuckoo_rastrigin_conv_d{dim}.png"),
    )

    # 3D surface for dim=2
    if dim == 2:
        xx = np.linspace(-5.12, 5.12, 200)
        X, Y = np.meshgrid(xx, xx)
        Z = np.zeros_like(X)
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                Z[i, j] = rastrigin(np.array([X[i, j], Y[i, j]]))

        fig = plt.figure(figsize=(10, 6))
        ax = fig.add_subplot(121, projection="3d")
        ax.plot_surface(X, Y, Z, cmap="viridis", alpha=0.9)

        ax2 = fig.add_subplot(122)
        cs = ax2.contourf(X, Y, Z, levels=50, cmap="viridis")
        fig.colorbar(cs, ax=ax2)

        fig.tight_layout()
        fig.savefig(os.path.join(results_dir, "rastrigin_surface_d2.png"), dpi=150)
        plt.close(fig)

    boxplot_finals(
        {"Cuckoo": finals},
        title=f"Final best (Rastrigin d={dim})",
        savepath=os.path.join(results_dir, f"cuckoo_rastrigin_finals_d{dim}.png"),
    )

    with open(os.path.join(results_dir, f"cuckoo_rastrigin_results_d{dim}.json"), "w") as f:
        json.dump({"histories": histories, "finals": finals}, f)

    return {"histories": histories, "finals": finals}


# ------------------------------------------------------------
# Sensitivity: Rastrigin
# ------------------------------------------------------------
def sensitivity_rastrigin(
    param_name: str,
    values: List[float],
    dim: int = 5,
    n_runs: int = 5,
    n_iter: int = 500,
    results_dir: str | None = None,
):
    if param_name == "nest":
        param_name = "n_nests"
    assert param_name in ("pa", "alpha", "beta", "n_nests")

    if results_dir is None:
        results_dir = _default_results_dir
    _ensure_dir(results_dir)

    all_finals = []
    labels = []

    for v in values:
        finals = []
        histories = []

        for run in range(n_runs):
            obj, _ = make_counted_rastrigin(dim)
            kwargs = dict(n_nests=50, pa=0.25, alpha=0.01, beta=1.5)
            kwargs[param_name] = v

            cs = CuckooSearch(
                obj_func=obj,
                dim=dim,
                bounds=(-5.12, 5.12),
                dtype="continuous",
                rng=100 + run,
                **kwargs,
            )
            out = cs.optimize(n_iter=n_iter, verbose=False)
            finals.append(float(out["best_f"]))
            histories.append(out["history"].tolist())

        all_finals.append(finals)
        labels.append(str(v))
        print(f"Param {param_name}={v} → median {np.median(finals):.6g}")

    plt.figure(figsize=(8, 5))
    sns.boxplot(data=all_finals)
    plt.xticks(ticks=np.arange(len(labels)), labels=labels)
    plt.xlabel(param_name)
    plt.ylabel("Final best")
    plt.title(f"Sensitivity on Rastrigin w.r.t. {param_name}")
    plt.grid(alpha=0.3)
    plt.savefig(os.path.join(results_dir, f"sensitivity_rastrigin_{param_name}.png"), dpi=150)
    plt.close()


# ------------------------------------------------------------
# TSP (permutation)
# ------------------------------------------------------------
def run_cuckoo_tsp(
    n_nodes: int = 20,
    n_runs: int = 5,
    n_iter: int = 500,
    n_nests: int = 60,
    pa: float = 0.25,
    results_dir: str | None = None,
) -> Dict[str, Any]:

    if results_dir is None:
        results_dir = _default_results_dir
    _ensure_dir(results_dir)

    rng = np.random.RandomState(2)
    pts = rng.rand(n_nodes, 2)
    dmat = np.sqrt(((pts[:, None, :] - pts[None, :, :]) ** 2).sum(axis=2))
    obj = make_tsp_obj(dmat)

    histories = []
    finals = []
    best_tours = []

    for run in range(n_runs):
        cs = CuckooSearch(
            obj_func=obj,
            dim=n_nodes,
            dtype="permutation",
            n_nests=n_nests,
            pa=pa,
            rng=10 + run,
        )
        out = cs.optimize(n_iter=n_iter, verbose=False)
        histories.append(out["history"].tolist())
        finals.append(float(out["best_f"]))
        best_tours.append(out["best"].astype(int))

        print(f"TSP run {run+1}/{n_runs} best={out['best_f']:.6g}")

    plot_convergence(
        histories,
        f"Cuckoo TSP (n={n_nodes})",
        savepath=os.path.join(results_dir, f"cuckoo_tsp_conv_n{n_nodes}.png"),
    )

    # plot best tour
    tour = best_tours[0]
    x, y = pts[:, 0], pts[:, 1]
    tx = np.concatenate([x[tour], [x[tour[0]]]])
    ty = np.concatenate([y[tour], [y[tour[0]]]])

    plt.figure(figsize=(6, 6))
    plt.scatter(x, y, c="k")
    for i, (xx, yy) in enumerate(zip(x, y)):
        plt.text(xx, yy, str(i), color="blue")
    plt.plot(tx, ty, "-o")
    plt.title("Best tour (run 1)")
    plt.savefig(os.path.join(results_dir, f"cuckoo_tsp_besttour_n{n_nodes}.png"), dpi=150)
    plt.close()

    boxplot_finals(
        {"Cuckoo": finals},
        title=f"Final tour length (TSP n={n_nodes})",
        savepath=os.path.join(results_dir, f"cuckoo_tsp_finals_n{n_nodes}.png"),
    )

    with open(os.path.join(results_dir, f"cuckoo_tsp_results_n{n_nodes}.json"), "w") as f:
        json.dump({"histories": histories, "finals": finals}, f)

    return {"histories": histories, "finals": finals, "pts": pts, "best_tours": best_tours}


# ------------------------------------------------------------
# Sensitivity: TSP
# ------------------------------------------------------------
def sensitivity_tsp(
    param_name: str,
    values: List[Any],
    n_nodes: int = 20,
    n_runs: int = 5,
    n_iter: int = 500,
    results_dir: str | None = None,
):

    if param_name == "nest":
        param_name = "n_nests"
    assert param_name in ("n_nests", "pa", "alpha")

    if results_dir is None:
        results_dir = _default_results_dir
    _ensure_dir(results_dir)

    all_finals = []
    labels = []

    for v in values:
        finals = []
        histories = []

        for run in range(n_runs):
            rng = np.random.RandomState(2)
            pts = rng.rand(n_nodes, 2)
            dmat = np.sqrt(((pts[:, None, :] - pts[None, :, :]) ** 2).sum(axis=2))
            obj = make_tsp_obj(dmat)

            kwargs = dict(n_nests=60, pa=0.25, alpha=0.01)
            kwargs[param_name] = v

            cs = CuckooSearch(
                obj_func=obj,
                dim=n_nodes,
                dtype="permutation",
                rng=200 + run,
                **kwargs,
            )
            out = cs.optimize(n_iter=n_iter, verbose=False)
            finals.append(float(out["best_f"]))
            histories.append(out["history"].tolist())

        all_finals.append(finals)
        labels.append(str(v))
        print(f"TSP param {param_name}={v} median={np.median(finals):.6g}")

    plt.figure(figsize=(8, 5))
    sns.boxplot(data=all_finals)
    plt.xticks(ticks=np.arange(len(labels)), labels=labels)
    plt.xlabel(param_name)
    plt.ylabel("Final tour length")
    plt.title(f"Sensitivity of Cuckoo on TSP w.r.t {param_name}")
    plt.grid(alpha=0.3)
    plt.savefig(os.path.join(results_dir, f"sensitivity_tsp_{param_name}.png"), dpi=150)
    plt.close()


# ------------------------------------------------------------
# MAIN
# ------------------------------------------------------------
if __name__ == "__main__":
    results_dir = None  # để các hàm tự chọn _default_results_dir

    show_2d_surface(rastrigin, title="Rastrigin Function")

    print("Running Cuckoo on Rastrigin (2D) ...")
    run_cuckoo_rastrigin(dim=2, n_runs=5, n_iter=300, results_dir=results_dir)

    print("Running Cuckoo on TSP ...")
    run_cuckoo_tsp(n_nodes=15, n_runs=5, n_iter=300, results_dir=results_dir)

    print("Performing sensitivity analysis...")
    alpha_vals = [0.01, 0.1, 0.5]
    nests_vals = [10, 25, 50]
    pa_vals = [0.1, 0.4, 0.6]

    sensitivity_rastrigin("alpha", alpha_vals, dim=2, n_runs=3, n_iter=300, results_dir=results_dir)
    sensitivity_rastrigin("n_nests", nests_vals, dim=2, n_runs=3, n_iter=300, results_dir=results_dir)
    sensitivity_rastrigin("pa", pa_vals, dim=5, n_runs=2, n_iter=300, results_dir=results_dir)

    sensitivity_tsp("alpha", alpha_vals, n_nodes=15, n_runs=3, n_iter=300, results_dir=results_dir)
    sensitivity_tsp("n_nests", nests_vals, n_nodes=15, n_runs=3, n_iter=300, results_dir=results_dir)
    sensitivity_tsp("pa", pa_vals, n_nodes=15, n_runs=3, n_iter=300, results_dir=results_dir)
