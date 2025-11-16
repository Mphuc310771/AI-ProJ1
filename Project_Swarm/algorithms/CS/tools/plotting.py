from __future__ import annotations

import os
import re
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter


# --- helper ---
def ensure_dir(path: str):
    if path and not os.path.exists(path):
        os.makedirs(path)


def safe_filename(name: str) -> str:
    """Convert any title into a valid safe filename"""
    name = name.strip().replace(" ", "_")
    name = re.sub(r"[^A-Za-z0-9_\-\.]", "", name)
    if len(name) == 0:
        name = "figure"
    return name


# ======================================================
# 1. Convergence Plot
# ======================================================
def plot_convergence(histories, title, xlabel='Evaluations', ylabel='Best objective',
                     savepath: str | None = None, savedir: str | None=None):

    ensure_dir(savedir)

    if savepath is None:
        fname = safe_filename(title) + ".png"
        savepath = os.path.join(savedir, fname)

    # Align histories
    max_len = max(len(h) for h in histories)
    S = np.zeros((len(histories), max_len))

    for i, h in enumerate(histories):
        arr = np.asarray(h)
        if len(arr) < max_len:
            arr = np.concatenate([arr, np.full(max_len - len(arr), arr[-1])])
        S[i] = arr

    mean = np.mean(S, axis=0)
    std = np.std(S, axis=0)
    x = np.arange(len(mean))

    plt.figure(figsize=(8,5))
    plt.plot(x, mean, label='Mean')
    plt.fill_between(x, mean-std, mean+std, alpha=0.3, label='Mean ± Std')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(alpha=0.3)

    plt.savefig(savepath, dpi=150, bbox_inches='tight')
    plt.close()


# ======================================================
# 2. Final fitness boxplot
# ======================================================
def boxplot_finals(finals_dict, title='Final Fitness (per algorithm)',
                   savepath: str | None = None, savedir: str | None=None):

    ensure_dir(savedir)

    if savepath is None:
        fname = safe_filename(title) + ".png"
        savepath = os.path.join(savedir, fname)

    names = list(finals_dict.keys())
    data = [finals_dict[n] for n in names]

    plt.figure(figsize=(8,5))
    plt.boxplot(data, labels=names, showmeans=True)
    plt.ylabel('Final best objective')
    plt.title(title)
    plt.grid(alpha=0.3)

    plt.savefig(savepath, dpi=150, bbox_inches='tight')
    plt.close()


# ======================================================
# 3. Success rate plot
# ======================================================
def plot_success_rate(histories_dict, thresholds=(1e-3,),
                      title='Success rate vs evals', save_dir: str | None=None):

    ensure_dir(save_dir)
 
    for thr in thresholds:
        plt.figure(figsize=(8,5))

        for name, runs in histories_dict.items():
            max_len = max(len(h) for h in runs)
            counts = np.zeros(max_len, dtype=int)

            for h in runs:
                arr = np.asarray(h)
                reached = np.where(arr <= thr)[0]
                if reached.size > 0:
                    counts[reached[0]:] += 1

            frac = counts / float(len(runs))
            plt.plot(np.arange(len(frac)), frac, label=name)

        plt.xlabel('Evaluations')
        plt.ylabel('Success rate')
        plt.title(f"{title} (thr={thr})")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.ylim(-0.02, 1.02)

        # filename SAFE
        thr_str = re.sub(r"[^A-Za-z0-9]", "", f"{thr:.0e}")
        fname = safe_filename(f"success_rate_thr_{thr_str}") + ".png"

        plt.savefig(os.path.join(save_dir, fname), dpi=150, bbox_inches='tight')
        plt.close()


# ======================================================
# 4. Memory boxplot
# ======================================================
def boxplot_memory(mem_dict: dict, title='Peak memory usage (bytes)',
                   savepath: str | None = None, savedir: str | None=None):

    ensure_dir(savedir)

    if savepath is None:
        fname = safe_filename(title) + ".png"
        savepath = os.path.join(savedir, fname)

    names = list(mem_dict.keys())
    filtered_names, data = [], []

    for n in names:
        vals = [v for v in (mem_dict.get(n, []) or []) if v is not None]
        if len(vals) > 0:
            filtered_names.append(n)
            data.append(vals)

    if not data:
        print("No memory data to plot.")
        return

    plt.figure(figsize=(8,5))
    plt.boxplot(data, labels=filtered_names, showmeans=True)
    plt.ylabel('Memory')
    plt.title(title)
    plt.grid(alpha=0.3)

    max_val = max(max(vals) for vals in data)
    if max_val >= 1024**3:
        scale, unit = 1024**3, "GiB"
    elif max_val >= 1024**2:
        scale, unit = 1024**2, "MiB"
    elif max_val >= 1024:
        scale, unit = 1024, "KiB"
    else:
        scale, unit = 1, "bytes"

    ax = plt.gca()
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"{x/scale:.2f} {unit}"))
    plt.ylabel(f"Memory ({unit})")

    plt.savefig(savepath, dpi=150, bbox_inches='tight')
    plt.close()
