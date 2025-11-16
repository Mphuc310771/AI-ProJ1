#!/usr/bin/env python3
"""

Usage:
    python pso_multi_funcs.py --runs 30 --n_particles 40 --n_iter 100 --dim 10
"""
import os
import csv
import argparse
import numpy as np
import matplotlib.pyplot as plt

# -------------------------
# (1) Objective functions (vectorized on last axis)
# -------------------------
def rastrigin(X):
    """X: (..., d) or (d,) -> scalar or array"""
    X = np.asarray(X)
    A = 10.0
    if X.ndim == 1:
        d = X.shape[0]
        return A * d + np.sum(X**2 - A * np.cos(2 * np.pi * X))
    else:
        d = X.shape[-1]
        return A * d + np.sum(X**2 - A * np.cos(2 * np.pi * X), axis=-1)

def sphere(X):
    """Sphere function: f(x) = sum(x^2). Vectorized on last axis."""
    X = np.asarray(X)
    if X.ndim == 1:
        return float(np.sum(X**2))
    else:
        return np.sum(X**2, axis=-1).astype(float)

# -------------------------
# (2) PSO single run (generic objective)
# -------------------------
def pso_track(func,
              n_iter=100,
              n_particles=40,
              dim=2,
              bound=(-5.12, 5.12),
              w=0.7, c1=1.5, c2=1.5,
              vel_scale=None,
              seed=None,
              early_stop_eps=None):
    """
    Run one PSO trial on objective `func`.
    Returns: history_positions (T_r, N, d), best_scores (T_r,), best_pos (d,), iters_ran, fe_count
    """
    if seed is not None:
        np.random.seed(int(seed))

    bmin, bmax = bound
    if vel_scale is None:
        vel_scale = 0.1 * (bmax - bmin)

    # init
    positions = np.random.uniform(bmin, bmax, (n_particles, dim))
    velocities = np.random.uniform(-vel_scale, vel_scale, (n_particles, dim))

    pbest_pos = positions.copy()
    # evaluate vectorized
    pbest_val = func(positions)
    fe = n_particles

    gidx = int(np.argmin(pbest_val))
    gpos = pbest_pos[gidx].copy()
    gval = float(pbest_val[gidx])

    history_positions = []
    best_scores = []

    for t in range(n_iter):
        r1 = np.random.rand(n_particles, dim)
        r2 = np.random.rand(n_particles, dim)

        velocities = (w * velocities
                      + c1 * r1 * (pbest_pos - positions)
                      + c2 * r2 * (gpos - positions))
        positions = positions + velocities
        positions = np.clip(positions, bmin, bmax)

        vals = func(positions)
        fe += n_particles

        # update pbest
        better_mask = vals < pbest_val
        if np.any(better_mask):
            pbest_pos[better_mask] = positions[better_mask]
            pbest_val[better_mask] = vals[better_mask]

        # update gbest
        idx = int(np.argmin(pbest_val))
        if pbest_val[idx] < gval:
            gval = float(pbest_val[idx])
            gpos = pbest_pos[idx].copy()

        history_positions.append(positions.copy())
        best_scores.append(gval)

        if (early_stop_eps is not None) and (gval <= early_stop_eps):
            break

    history_positions = np.array(history_positions)
    best_scores = np.array(best_scores)
    iters_ran = history_positions.shape[0]
    return history_positions, best_scores, gpos, iters_ran, fe

# -------------------------
# (3) Run experiments (multiple runs) for one function
# -------------------------
def run_experiments_for(func, func_name,
                        runs=30, n_iter=100, n_particles=40, dim=10,
                        bound=(-5.12,5.12), w=0.7, c1=1.5, c2=1.5,
                        seed0=0, early_stop_eps=None, outdir="results"):
    out_dir = os.path.join(outdir, func_name)
    os.makedirs(out_dir, exist_ok=True)

    all_best_scores = []
    finals = []
    final_positions = []
    iters_ran = []
    fes = []
    seeds = []

    for r in range(runs):
        seed = seed0 + r
        seeds.append(int(seed))
        hist, best_scores, best_pos, iters, fe = pso_track(
            func,
            n_iter=n_iter,
            n_particles=n_particles,
            dim=dim,
            bound=bound,
            w=w, c1=c1, c2=c2,
            seed=seed,
            early_stop_eps=early_stop_eps
        )
        all_best_scores.append(best_scores)
        finals.append(float(best_scores[-1]))
        final_positions.append(best_pos)
        iters_ran.append(int(iters))
        fes.append(int(fe))
        print(f"[{func_name}] Run {r+1}/{runs} seed={seed} final_best={best_scores[-1]:.6e} iters={iters} FE={fe}")

    # pad to same length (pad with last value)
    max_len = max(arr.shape[0] for arr in all_best_scores)
    padded = np.zeros((runs, max_len))
    for i, arr in enumerate(all_best_scores):
        L = arr.shape[0]
        padded[i, :L] = arr
        if L < max_len:
            padded[i, L:] = arr[-1]

    # Save outputs
    np.save(os.path.join(out_dir, f"{func_name}_best_scores.npy"), padded)
    np.save(os.path.join(out_dir, f"{func_name}_final_positions.npy"), np.array(final_positions))

    csv_path = os.path.join(out_dir, f"{func_name}_final_bests.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["run", "seed", "final_best", "iterations", "FE"])
        for i in range(runs):
            writer.writerow([i, seeds[i], finals[i], iters_ran[i], fes[i]])

    # summary stats
    mean_val = float(np.mean(finals))
    std_val = float(np.std(finals, ddof=1))
    median_val = float(np.median(finals))
    min_val = float(np.min(finals))
    max_val = float(np.max(finals))
    summary_csv = os.path.join(out_dir, f"{func_name}_summary_stats.csv")
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
        "out_dir": out_dir
    }
    print(f"[{func_name}] Saved results to: {out_dir}")
    return stats

# -------------------------
# (4) Plot combined convergence (two functions, twin axes)
# -------------------------
def plot_combined_convergence(statsA, nameA, statsB, nameB, outpath, sample_runs=6, use_logscaleA=False, use_logscaleB=False):
    A = statsA["best_scores_matrix"]
    B = statsB["best_scores_matrix"]
    # pad to same length if necessary (should already be padded)
    T = max(A.shape[1], B.shape[1])
    def pad_to(mat, T):
        if mat.shape[1] == T:
            return mat
        out = np.zeros((mat.shape[0], T))
        out[:, :mat.shape[1]] = mat
        out[:, mat.shape[1]:] = mat[:, -1][:, None]
        return out
    A = pad_to(A, T)
    B = pad_to(B, T)

    meanA = np.mean(A, axis=0); stdA = np.std(A, axis=0, ddof=1 if A.shape[0]>1 else 0)
    meanB = np.mean(B, axis=0); stdB = np.std(B, axis=0, ddof=1 if B.shape[0]>1 else 0)
    x = np.arange(T)

    fig, axL = plt.subplots(figsize=(10,5))
    axR = axL.twinx()

    colorA = "#1f77b4"
    colorB = "#d62728"

    R1 = A.shape[0]
    if sample_runs > 0:
        idxA = np.linspace(0, R1-1, min(sample_runs, R1)).astype(int)
        for i in idxA:
            axL.plot(x, A[i], color=colorA, alpha=0.18, linewidth=1)

    R2 = B.shape[0]
    if sample_runs > 0:
        idxB = np.linspace(0, R2-1, min(sample_runs, R2)).astype(int)
        for i in idxB:
            axR.plot(x, B[i], color=colorB, alpha=0.10, linewidth=1)

    lA, = axL.plot(x, meanA, color=colorA, linewidth=2.4, label=f"{nameA} Mean")
    axL.fill_between(x, meanA - stdA, meanA + stdA, color=colorA, alpha=0.25, label=f"{nameA} ± Std")

    lB, = axR.plot(x, meanB, color=colorB, linewidth=2.4, label=f"{nameB} Mean")
    axR.fill_between(x, meanB - stdB, meanB + stdB, color=colorB, alpha=0.18, label=f"{nameB} ± Std")

    axL.set_xlabel("Iteration")
    axL.set_ylabel(f"{nameA} (lower is better)", color=colorA)
    axR.set_ylabel(f"{nameB} (lower is better)", color=colorB)

    if use_logscaleA:
        axL.set_yscale("log")
    if use_logscaleB:
        axR.set_yscale("log")

    lines = [lA, lB]
    labels = [l.get_label() for l in lines]
    axL.legend(lines, labels, loc="upper right", fontsize=9)
    axL.grid(alpha=0.3)
    fig.suptitle(f"{nameA} vs {nameB} — Convergence (Mean ± Std)")
    plt.tight_layout(rect=[0,0,1,0.96])
    os.makedirs(os.path.dirname(outpath) or ".", exist_ok=True)
    plt.savefig(outpath, dpi=220)
    plt.show()
    print("Saved combined plot:", outpath)

# -------------------------
# (5) CLI + main
# -------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--runs", type=int, default=30)
    p.add_argument("--n_iter", type=int, default=100)
    p.add_argument("--n_particles", type=int, default=40)
    p.add_argument("--dim", type=int, default=10)
    p.add_argument("--bound", type=float, nargs=2, default=[-5.12, 5.12])
    p.add_argument("--w", type=float, default=0.7)
    p.add_argument("--c1", type=float, default=1.5)
    p.add_argument("--c2", type=float, default=1.5)
    p.add_argument("--seed0", type=int, default=0)
    p.add_argument("--early_stop_eps", type=float, default=None)
    p.add_argument("--outdir", type=str, default="results")
    p.add_argument("--sample_runs_plot", type=int, default=6)
    p.add_argument("--logscale", action="store_true", help="use logscale for both axes")
    return p.parse_args()

def main():
    args = parse_args()

    # run for Sphere
    stats_sphere = run_experiments_for(
        func=sphere,
        func_name="sphere",
        runs=args.runs,
        n_iter=args.n_iter,
        n_particles=args.n_particles,
        dim=args.dim,
        bound=tuple(args.bound),
        w=args.w, c1=args.c1, c2=args.c2,
        seed0=args.seed0,
        early_stop_eps=args.early_stop_eps,
        outdir=args.outdir
    )

    # run for Rastrigin
    stats_rastrigin = run_experiments_for(
        func=rastrigin,
        func_name="rastrigin",
        runs=args.runs,
        n_iter=args.n_iter,
        n_particles=args.n_particles,
        dim=args.dim,
        bound=tuple(args.bound),
        w=args.w, c1=args.c1, c2=args.c2,
        seed0=args.seed0 + args.runs,  # offset seeds so independent
        early_stop_eps=args.early_stop_eps,
        outdir=args.outdir
    )

    # per-function convergence images
    plot_combined_convergence(
        stats_sphere, "Sphere",
        stats_rastrigin, "Rastrigin",
        outpath=os.path.join(args.outdir, "combined_convergence_sphere_rastrigin.png"),
        sample_runs=args.sample_runs_plot,
        use_logscaleA=args.logscale,
        use_logscaleB=args.logscale
    )

if __name__ == "__main__":
    main()
