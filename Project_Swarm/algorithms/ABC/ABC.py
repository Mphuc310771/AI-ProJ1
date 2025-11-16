#!/usr/bin/env python3


import argparse
import os
import csv
import numpy as np
import matplotlib.pyplot as plt

# -------------------------
# (1) Hàm mục tiêu (vectorized)
# -------------------------
def rastrigin(X):
    """Vectorized Rastrigin on last axis. X shape (..., d) or (d,)"""
    X = np.asarray(X)
    A = 10.0
    if X.ndim == 1:
        d = X.shape[0]
        return A * d + np.sum(X**2 - A * np.cos(2 * np.pi * X))
    else:
        d = X.shape[-1]
        return A * d + np.sum(X**2 - A * np.cos(2 * np.pi * X), axis=-1)

def sphere(X):
    """Vectorized Sphere (sum of squares) on last axis"""
    X = np.asarray(X)
    if X.ndim == 1:
        return np.sum(X**2)
    else:
        return np.sum(X**2, axis=-1)

# map name -> function and recommended bounds
OBJ_MAP = {
    "rastrigin": {"fn": rastrigin, "bound": (-5.12, 5.12)},
    "sphere": {"fn": sphere, "bound": (-5.12, 5.12)},
}

# -------------------------
# (2) Một run ABC 
# -------------------------
def abc_track(
    obj_fn,
    n_iter=100,
    n_bees=40,
    dim=2,
    bound=(-5.12, 5.12),
    seed=None,
    employed_phi_scale=1.0,
    onlooker_step_scale=0.5,
    scout_prob=0.1,
    early_stop_eps=None,
):
    """
    Run one ABC trial for objective `obj_fn`.
    Returns:
      history_positions: ndarray (T_r, N, d)
      best_scores: ndarray (T_r,)
      best_pos: ndarray (d,)
      iterations_ran: int
      fe_count: int
    """
    if seed is not None:
        np.random.seed(int(seed))

    bmin, bmax = bound
    bees = np.random.uniform(bmin, bmax, (n_bees, dim))
    fitness = obj_fn(bees)  # vectorized
    trial = np.zeros(n_bees, dtype=int)

    best_idx = int(np.argmin(fitness))
    best_pos = bees[best_idx].copy()
    best_fitness = float(fitness[best_idx])

    history_positions = []
    best_scores = []
    fe = n_bees  # initial evaluations for initialization

    for it in range(n_iter):
        # Employed phase
        for i in range(n_bees):
            k = np.random.randint(n_bees - 1)
            if k >= i:
                k += 1
            phi = np.random.uniform(-employed_phi_scale, employed_phi_scale, size=dim)
            candidate = bees[i] + phi * (bees[i] - bees[k])
            candidate = np.clip(candidate, bmin, bmax)
            val = obj_fn(candidate)
            fe += 1
            if val < fitness[i]:
                bees[i] = candidate
                fitness[i] = val
                trial[i] = 0
            else:
                trial[i] += 1

        # Onlooker phase
        probs = 1.0 / (1.0 + fitness)
        s = np.sum(probs)
        if s <= 0:
            probs = np.ones_like(probs) / n_bees
        else:
            probs = probs / s

        cumsum = np.cumsum(probs)
        for _ in range(n_bees):
            r = np.random.rand()
            idx = int(np.searchsorted(cumsum, r, side='right'))
            idx = min(max(idx, 0), n_bees - 1)
            i = idx
            k = np.random.randint(n_bees - 1)
            if k >= i:
                k += 1
            step = np.random.uniform(-onlooker_step_scale, onlooker_step_scale, size=dim)
            candidate = bees[i] + step
            candidate = np.clip(candidate, bmin, bmax)
            val = obj_fn(candidate)
            fe += 1
            if val < fitness[i]:
                bees[i] = candidate
                fitness[i] = val
                trial[i] = 0
            else:
                trial[i] += 1

        # Scout phase (probabilistic + based on trial)
        if np.random.rand() < scout_prob:
            idx_max = int(np.argmax(trial))
            bees[idx_max] = np.random.uniform(bmin, bmax, dim)
            fitness[idx_max] = obj_fn(bees[idx_max])
            fe += 1
            trial[idx_max] = 0

        # update best
        cur_best_idx = int(np.argmin(fitness))
        cur_best_val = float(fitness[cur_best_idx])
        if cur_best_val < best_fitness:
            best_fitness = cur_best_val
            best_pos = bees[cur_best_idx].copy()

        history_positions.append(bees.copy())
        best_scores.append(best_fitness)

        # early stopping
        if (early_stop_eps is not None) and (best_fitness <= early_stop_eps):
            break

    history_positions = np.array(history_positions)
    best_scores = np.array(best_scores)
    iterations_ran = history_positions.shape[0]
    return history_positions, best_scores, best_pos, iterations_ran, fe

# -------------------------
# (3) Run multiple trials and save results (per problem)
# -------------------------
def run_experiments_for_problem(
    problem_name="rastrigin",
    runs=30,
    n_iter=100,
    n_bees=40,
    dim=10,
    seed0=0,
    early_stop_eps=None,
    outdir="results",
    employed_phi_scale=1.0,
    onlooker_step_scale=0.5,
    scout_prob=0.1,
):
    os.makedirs(outdir, exist_ok=True)
    obj_fn = OBJ_MAP[problem_name]["fn"]
    bound = OBJ_MAP[problem_name]["bound"]

    all_best_scores = []
    finals = []
    final_positions = []
    iters_ran = []
    fes = []
    seeds = []

    for r in range(runs):
        seed = seed0 + r
        seeds.append(int(seed))
        hist, best_scores, best_pos, iters, fe = abc_track(
            obj_fn=obj_fn,
            n_iter=n_iter,
            n_bees=n_bees,
            dim=dim,
            bound=bound,
            seed=seed,
            employed_phi_scale=employed_phi_scale,
            onlooker_step_scale=onlooker_step_scale,
            scout_prob=scout_prob,
            early_stop_eps=early_stop_eps,
        )
        all_best_scores.append(best_scores)
        finals.append(float(best_scores[-1]))
        final_positions.append(best_pos)
        iters_ran.append(int(iters))
        fes.append(int(fe))
        print(f"[{problem_name}] Run {r+1}/{runs} seed={seed} final_best={best_scores[-1]:.6e} iters={iters} FE={fe}")

    # pad best_scores to matrix R x T
    max_len = max(arr.shape[0] for arr in all_best_scores)
    padded = np.zeros((runs, max_len))
    for i, arr in enumerate(all_best_scores):
        L = arr.shape[0]
        padded[i, :L] = arr
        if L < max_len:
            padded[i, L:] = arr[-1]

    # Save outputs with problem-specific filenames
    np.save(os.path.join(outdir, f"{problem_name}_best_scores.npy"), padded)
    np.save(os.path.join(outdir, f"{problem_name}_final_positions.npy"), np.array(final_positions))

    csv_path = os.path.join(outdir, f"{problem_name}_final_bests.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["run", "seed", "final_best", "iterations", "FE"])
        for i in range(runs):
            writer.writerow([i, seeds[i], finals[i], iters_ran[i], fes[i]])

    # summary
    mean_val = float(np.mean(finals))
    std_val = float(np.std(finals, ddof=1))
    median_val = float(np.median(finals))
    min_val = float(np.min(finals))
    max_val = float(np.max(finals))
    summary_csv = os.path.join(outdir, f"{problem_name}_summary_stats.csv")
    with open(summary_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        writer.writerow(["mean_final_best", mean_val])
        writer.writerow(["std_final_best", std_val])
        writer.writerow(["median_final_best", median_val])
        writer.writerow(["min_final_best", min_val])
        writer.writerow(["max_final_best", max_val])

    stats = {
        "final_bests": np.array(finals),
        "best_scores_matrix": padded,
        "final_positions": np.array(final_positions),
        "seeds": np.array(seeds),
        "iters_ran": np.array(iters_ran),
        "fes": np.array(fes),
    }
    print(f"\nSaved {problem_name} results to: {outdir}")
    return stats

# -------------------------
# (4) Plotting utilities (single and combined)
# -------------------------
def plot_convergence_matrix(best_scores_matrix, label=None, color=None):
    R, T = best_scores_matrix.shape
    mean = np.mean(best_scores_matrix, axis=0)
    std = np.std(best_scores_matrix, axis=0, ddof=1 if R>1 else 0)
    x = np.arange(T)
    plt.plot(x, mean, linewidth=2.4, label=label, color=color, zorder=10)
    plt.fill_between(x, mean - std, mean + std, alpha=0.25, color=color)

def save_convergence_plot_single(best_scores_matrix, outpath, title=None, sample_runs=6, color=None, use_logscale=True):
    plt.figure(figsize=(9,5))
    R = best_scores_matrix.shape[0]
    if sample_runs > 0:
        idx = np.linspace(0, R-1, min(sample_runs, R)).astype(int)
        for i in idx:
            plt.plot(np.arange(best_scores_matrix.shape[1]), best_scores_matrix[i], linewidth=1.0, alpha=0.25, color='gray')

    plot_convergence_matrix(best_scores_matrix, label=title or "Mean best", color=color)

    plt.xlabel("Iteration")
    plt.ylabel("Best fitness (lower is better)")
    if title:
        plt.title(title)
    if use_logscale:
        plt.yscale("log")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()
    print("Saved:", outpath)

def save_convergence_plot_combined(matrices_dict, outpath, sample_runs=6, use_logscale=True):
    plt.figure(figsize=(9,5))
    colors = ['#2c7fb8', '#fdae61', '#7fc97f', '#beaed4']
    for i, (label, mat) in enumerate(matrices_dict.items()):
        R = mat.shape[0]
        if sample_runs > 0:
            idx = np.linspace(0, R-1, min(sample_runs, R)).astype(int)
            for j in idx:
                plt.plot(np.arange(mat.shape[1]), mat[j], linewidth=0.8, alpha=0.18, color='gray')
        color = colors[i % len(colors)]
        plot_convergence_matrix(mat, label=label, color=color)
    plt.xlabel("Iteration")
    plt.ylabel("Best fitness (lower is better)")
    if use_logscale:
        plt.yscale("log")
    plt.title("Convergence comparison")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()
    print("Saved combined plot:", outpath)

def plot_best_solution_surface(best_pos, obj_name, bound, outpath, title=None):
    os.makedirs(os.path.dirname(outpath) or ".", exist_ok=True)
    if len(best_pos) != 2:
        print("⚠️ plot_best_solution_surface supports dim=2 only.")
        return
    bmin, bmax = bound
    x = np.linspace(bmin, bmax, 300)
    y = np.linspace(bmin, bmax, 300)
    X, Y = np.meshgrid(x, y)
    if obj_name == "rastrigin":
        Z = rastrigin(np.stack([X, Y], axis=-1))
    else:
        Z = sphere(np.stack([X, Y], axis=-1))

    plt.figure(figsize=(6,5))
    cp = plt.contourf(X, Y, Z, levels=80, cmap="viridis")
    plt.colorbar(cp, label="Function value")
    plt.scatter(best_pos[0], best_pos[1], color="red", s=80, edgecolors="black", label="Best solution")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title(title or f"{obj_name.capitalize()} best solution (ABC)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()
    print("Saved:", outpath)

# -------------------------
# (5) CLI
# -------------------------
def parse_args():
    p = argparse.ArgumentParser()
    # default changed to "both" so it runs both problems if user supplies no --problem
    p.add_argument("--problem", type=str, choices=["rastrigin", "sphere", "both"], default="both",
                   help="Problem to run: 'rastrigin', 'sphere' or 'both' (default: both)")
    p.add_argument("--runs", type=int, default=30)
    p.add_argument("--n_iter", type=int, default=100)
    p.add_argument("--n_bees", type=int, default=40)
    p.add_argument("--dim", type=int, default=10)
    p.add_argument("--seed0", type=int, default=0)
    p.add_argument("--early_stop_eps", type=float, default=None)
    p.add_argument("--outdir", type=str, default="results")
    p.add_argument("--employed_phi_scale", type=float, default=1.0)
    p.add_argument("--onlooker_step_scale", type=float, default=0.5)
    p.add_argument("--scout_prob", type=float, default=0.1)
    p.add_argument("--sample_runs_plot", type=int, default=6)
    p.add_argument("--use_logscale", action="store_true")
    return p.parse_args()

def main():
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    to_run = []
    if args.problem == "both":
        to_run = ["rastrigin", "sphere"]
    else:
        to_run = [args.problem]

    stats_map = {}
    for prob in to_run:
        stats = run_experiments_for_problem(
            problem_name=prob,
            runs=args.runs,
            n_iter=args.n_iter,
            n_bees=args.n_bees,
            dim=args.dim,
            seed0=args.seed0,
            early_stop_eps=args.early_stop_eps,
            outdir=args.outdir,
            employed_phi_scale=args.employed_phi_scale,
            onlooker_step_scale=args.onlooker_step_scale,
            scout_prob=args.scout_prob,
        )
        stats_map[prob] = stats
        # Save single convergence plot
        save_convergence_plot_single(
            stats["best_scores_matrix"],
            outpath=os.path.join(args.outdir, f"{prob}_convergence.png"),
            title=f"{prob.capitalize()} convergence (dim={args.dim}, N={args.n_bees})",
            sample_runs=args.sample_runs_plot,
            color=None,
            use_logscale=args.use_logscale,
        )
        # If dim == 2, save best-solution surface
        if args.dim == 2:
            best_idx = int(np.argmin(stats["final_bests"]))
            best_pos = np.load(os.path.join(args.outdir, f"{prob}_final_positions.npy"))[best_idx]
            plot_best_solution_surface(best_pos, prob, OBJ_MAP[prob]["bound"],
                                       outpath=os.path.join(args.outdir, f"{prob}_best_solution_surface.png"),
                                       title=f"{prob.capitalize()} best solution (ABC)")

    # If ran both, create combined convergence plot
    if "rastrigin" in stats_map and "sphere" in stats_map:
        matrices = {
            "Rastrigin": stats_map["rastrigin"]["best_scores_matrix"],
            "Sphere": stats_map["sphere"]["best_scores_matrix"],
        }
        save_convergence_plot_combined(matrices,
                                      outpath=os.path.join(args.outdir, "convergence_both.png"),
                                      sample_runs=args.sample_runs_plot,
                                      use_logscale=args.use_logscale)
    print("All done. Outputs in:", args.outdir)

if __name__ == "__main__":
    main()
