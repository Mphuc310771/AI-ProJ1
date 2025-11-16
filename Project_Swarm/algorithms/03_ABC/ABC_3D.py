#!/usr/bin/env python3

import argparse
import math
import numpy as np
import matplotlib.pyplot as plt
import random
import time

# ---------------- objective functions ----------------
def objective_pointwise(fn, xs, ys, zs):
    if fn == "sphere":
        return xs*xs + ys*ys + zs*zs
    if fn == "rastrigin":
        A = 10.0
        return 3*A + (xs*xs - A*np.cos(2*np.pi*xs)) + (ys*ys - A*np.cos(2*np.pi*ys)) + (zs*zs - A*np.cos(2*np.pi*zs))
    if fn == "rosenbrock":
        a = xs; b = ys; c = zs
        return 100.0*(b - a*a)**2 + (a - 1.0)**2 + 100.0*(c - b*b)**2 + (b - 1.0)**2
    return xs*xs + ys*ys + zs*zs

def fitness_from_value(vals):
    return 1.0 / (1.0 + np.maximum(0.0, vals))

# ---------------- ABC algorithm ----------------
def init_population(pop, bound, fn, seed=None):
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)
    xs = np.random.uniform(-bound, bound, size=pop)
    ys = np.random.uniform(-bound, bound, size=pop)
    zs = np.random.uniform(-bound, bound, size=pop)
    vals = objective_pointwise(fn, xs, ys, zs)
    fits = fitness_from_value(vals)
    trials = np.zeros(pop, dtype=int)
    return xs, ys, zs, vals, fits, trials

def abc_step(xs, ys, zs, vals, fits, trials, params):
    N = xs.size
    # employed
    for i in range(N):
        k = i
        while k == i:
            k = np.random.randint(0, N)
        phi = np.random.uniform(-1.0, 1.0)
        newx = xs[i] + phi * (xs[i] - xs[k])
        newy = ys[i] + phi * (ys[i] - ys[k])
        newz = zs[i] + phi * (zs[i] - zs[k])
        b = params['bound']
        newx = np.clip(newx, -b, b)
        newy = np.clip(newy, -b, b)
        newz = np.clip(newz, -b, b)
        newval = objective_pointwise(params['fn'], np.array([newx]), np.array([newy]), np.array([newz]))[0]
        if newval < vals[i]:
            xs[i], ys[i], zs[i], vals[i] = newx, newy, newz, newval
            fits[i] = fitness_from_value(np.array([newval]))[0]
            trials[i] = 0
        else:
            trials[i] += 1

    # onlooker
    total_fit = fits.sum()
    probs = fits / total_fit if total_fit > 0 else np.ones(N) / N
    for _ in range(N):
        r = np.random.rand()
        acc = 0.0
        idx = 0
        for j, p in enumerate(probs):
            acc += p
            if r <= acc:
                idx = j
                break
        k = idx
        while k == idx:
            k = np.random.randint(0, N)
        phi = np.random.uniform(-1.0, 1.0)
        newx = xs[idx] + phi * (xs[idx] - xs[k])
        newy = ys[idx] + phi * (ys[idx] - ys[k])
        newz = zs[idx] + phi * (zs[idx] - zs[k])
        b = params['bound']
        newx = np.clip(newx, -b, b)
        newy = np.clip(newy, -b, b)
        newz = np.clip(newz, -b, b)
        newval = objective_pointwise(params['fn'], np.array([newx]), np.array([newy]), np.array([newz]))[0]
        if newval < vals[idx]:
            xs[idx], ys[idx], zs[idx], vals[idx] = newx, newy, newz, newval
            fits[idx] = fitness_from_value(np.array([newval]))[0]
            trials[idx] = 0
        else:
            trials[idx] += 1

    # scout
    mask = trials >= params['limit']
    if mask.any():
        count = mask.sum()
        xs[mask] = np.random.uniform(-params['bound'], params['bound'], size=count)
        ys[mask] = np.random.uniform(-params['bound'], params['bound'], size=count)
        zs[mask] = np.random.uniform(-params['bound'], params['bound'], size=count)
        newvals = objective_pointwise(params['fn'], xs[mask], ys[mask], zs[mask])
        vals[mask] = newvals
        fits[mask] = fitness_from_value(newvals)
        trials[mask] = 0

    return xs, ys, zs, vals, fits, trials

# ---------------- run + visualization + stopping ----------------
def run(params):
    xs, ys, zs, vals, fits, trials = init_population(params['pop'], params['bound'], params['fn'], seed=params.get('seed'))
    best_idx = int(np.argmin(vals))
    best_val = float(vals[best_idx])
    best_pos = (float(xs[best_idx]), float(ys[best_idx]), float(zs[best_idx]))

    plt.ion()
    fig = plt.figure(figsize=(9,7))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlim(-params['bound'], params['bound'])
    ax.set_ylim(-params['bound'], params['bound'])
    ax.set_zlim(-params['bound'], params['bound'])
    ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
    ax.set_box_aspect([1,1,1])
    cmap = plt.get_cmap('viridis')

    sizes = 20 + 300 * fits
    colors = cmap(fits.clip(0,1))
    sc = ax.scatter(xs, ys, zs, s=sizes, c=colors, depthshade=True)
    best_sc = ax.scatter([best_pos[0]], [best_pos[1]], [best_pos[2]], marker='X', s=220, c='yellow', edgecolors='k')
    title = ax.set_title(f"ABC 3D — iter 0   best {best_val:.6e}")

    iter_count = 0
    start_time = time.time()
    last_improve_iter = 0
    last_improve_val = best_val

    try:
        while True:
            if not plt.fignum_exists(fig.number):
                print("Window closed by user.")
                break

            for _ in range(params['speed']):
                xs, ys, zs, vals, fits, trials = abc_step(xs, ys, zs, vals, fits, trials, params)
                iter_count += 1

            # update best
            idx = int(np.argmin(vals))
            cur_best_val = float(vals[idx])
            if cur_best_val < best_val:
                best_val = cur_best_val
                best_pos = (float(xs[idx]), float(ys[idx]), float(zs[idx]))
                last_improve_iter = iter_count
                last_improve_val = best_val

            # update scatter
            sc._offsets3d = (xs, ys, zs)
            sizes = 20 + 300 * fits
            sc.set_sizes(sizes)
            sc.set_facecolor(cmap(fits.clip(0,1)))
            best_sc._offsets3d = ([best_pos[0]], [best_pos[1]], [best_pos[2]])

            if iter_count % 5 == 0:
                title.set_text(f"ABC 3D — iter {iter_count}   best {best_val:.6e}")

            plt.draw()
            plt.pause(0.02)

            # stopping conditions
            if params['max_iter'] is not None and iter_count >= params['max_iter']:
                reason = f"Reached max_iter={params['max_iter']}"
                break
            if params['tol'] is not None and best_val <= params['tol']:
                reason = f"Best value {best_val:.6e} ≤ tol={params['tol']}"
                break
            if params['no_improve_limit'] is not None and (iter_count - last_improve_iter) >= params['no_improve_limit']:
                reason = f"No improvement in last {params['no_improve_limit']} iters"
                break
            if params['time_limit'] is not None and (time.time() - start_time) >= params['time_limit']:
                reason = f"Time limit exceeded ({params['time_limit']} s)"
                break

        elapsed = time.time() - start_time
        print("\nStopped. Reason:", reason)
        print(f"Iterations: {iter_count}, elapsed {elapsed:.2f}s")
        print(f"Best value: {best_val:.6e} at pos ({best_pos[0]:+.6f}, {best_pos[1]:+.6f}, {best_pos[2]:+.6f})")

        plt.ioff()
        plt.show()

    except KeyboardInterrupt:
        print("\nInterrupted by user. Exiting.")
        plt.ioff()
        plt.show()

# ---------------- CLI ----------------
def parse_args():
    p = argparse.ArgumentParser(description="ABC 3D with stopping conditions (default max_iter=200)")
    p.add_argument('--pop', type=int, default=60, help='number of food sources')
    p.add_argument('--limit', type=int, default=30, help='trials before scout')
    p.add_argument('--bound', type=float, default=10.0, help='coordinate bound ±')
    p.add_argument('--speed', type=int, default=1, help='ABC iterations per GUI frame')
    p.add_argument('--fn', type=str, default='rastrigin', choices=['sphere','rastrigin','rosenbrock'], help='objective function')
    p.add_argument('--seed', type=int, default=None, help='random seed (optional)')
    p.add_argument('--max_iter', type=int, default=150, help='maximum ABC iterations (default 200)')
    p.add_argument('--tol', type=float, default=None, help='stop if best ≤ tol (None = disabled)')
    p.add_argument('--no_improve_limit', type=int, default=None, help='stop if no improvement for N iters (None = disabled)')
    p.add_argument('--time_limit', type=float, default=None, help='stop if elapsed time (seconds) exceeds this (None = disabled)')
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    params = {
        'pop': args.pop,
        'limit': args.limit,
        'bound': args.bound,
        'speed': max(1, args.speed),
        'fn': args.fn,
        'seed': args.seed,
        'max_iter': None if args.max_iter <= 0 else args.max_iter,
        'tol': args.tol,
        'no_improve_limit': None if (args.no_improve_limit is None or args.no_improve_limit <= 0) else args.no_improve_limit,
        'time_limit': args.time_limit
    }
    run(params)
