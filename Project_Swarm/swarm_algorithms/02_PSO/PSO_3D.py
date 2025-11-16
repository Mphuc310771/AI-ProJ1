"""
Improved PSO 3D visualization (more readable / aesthetic)
- clearer colors (fitness colormap), larger markers, gold global-best star
- soft fading trails, rotating camera, grid and tighter axes
- option to choose objective via command line and save animation

Run: python pso_3d.py
Requirements: numpy, matplotlib
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (3D projection side-effect)
from matplotlib.animation import FuncAnimation
from matplotlib.colors import Normalize


class PSO3D:
    def __init__(self, obj_func, bounds, n_particles=30, w=0.7, c1=1.5, c2=1.5,
                 max_iter=200, vel_clamp=None, seed=None, trail_length=12):
        """
        Simple, well-documented PSO in 3D. Keeps history for trails.
        obj_func: function accepting (n_particles, 3) and returning (n_particles,)
        bounds: list of (min, max) pairs for three dims
        """
        assert len(bounds) == 3
        self.obj_func = obj_func
        self.bounds = np.array(bounds, dtype=float)
        self.n_particles = int(n_particles)
        self.w = float(w)
        self.c1 = float(c1)
        self.c2 = float(c2)
        self.max_iter = int(max_iter)
        self.vel_clamp = vel_clamp
        self.trail_length = int(trail_length)

        self.rng = np.random.default_rng(seed)

        mins = self.bounds[:, 0]
        maxs = self.bounds[:, 1]
        span = maxs - mins

        # Initialize positions + velocities in sensible ranges
        self.pos = mins + self.rng.random((self.n_particles, 3)) * span
        self.vel = (self.rng.random((self.n_particles, 3)) - 0.5) * span * 0.08

        vals = self.obj_func(self.pos)
        self.pbest_pos = self.pos.copy()
        self.pbest_val = vals.copy()

        best_idx = np.argmin(self.pbest_val)
        self.gbest_pos = self.pbest_pos[best_idx].copy()
        self.gbest_val = float(self.pbest_val[best_idx])

        self.history = [self.pos.copy()]

    def step(self):
        r1 = self.rng.random((self.n_particles, 3))
        r2 = self.rng.random((self.n_particles, 3))

        cognitive = self.c1 * r1 * (self.pbest_pos - self.pos)
        social = self.c2 * r2 * (self.gbest_pos - self.pos)

        self.vel = self.w * self.vel + cognitive + social

        if self.vel_clamp is not None:
            vmax = float(self.vel_clamp)
            np.clip(self.vel, -vmax, vmax, out=self.vel)

        self.pos += self.vel

        # enforce bounds
        for d in range(3):
            low, high = self.bounds[d]
            self.pos[:, d] = np.clip(self.pos[:, d], low, high)

        vals = self.obj_func(self.pos)

        improved = vals < self.pbest_val
        if np.any(improved):
            self.pbest_val[improved] = vals[improved]
            self.pbest_pos[improved] = self.pos[improved]

        best_idx = np.argmin(self.pbest_val)
        if self.pbest_val[best_idx] < self.gbest_val:
            self.gbest_val = float(self.pbest_val[best_idx])
            self.gbest_pos = self.pbest_pos[best_idx].copy()

        self.history.append(self.pos.copy())
        if len(self.history) > self.trail_length:
            self.history.pop(0)


# --- Example objective functions ---

def sphere(x):
    return np.sum(x ** 2, axis=1)


def rastrigin(x):
    A = 10
    n = x.shape[1]
    return A * n + np.sum(x ** 2 - A * np.cos(2 * np.pi * x), axis=1)


def ackley(x):
    # smooth multimodal function
    a = 20
    b = 0.2
    c = 2 * np.pi
    n = x.shape[1]
    sum_sq = np.sum(x ** 2, axis=1)
    sum_cos = np.sum(np.cos(c * x), axis=1)
    return -a * np.exp(-b * np.sqrt(sum_sq / n)) - np.exp(sum_cos / n) + a + np.e


# --- Visualization / animation ---

def animate_pso(pso: PSO3D, func_name='objective', save_file=None, rotate=True):
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')

    # nicer background and grid
    # try to set 3D pane colors; different matplotlib versions expose different attributes
    try:
        ax.w_xaxis.set_pane_color((0.98, 0.98, 0.98, 1.0))
        ax.w_yaxis.set_pane_color((0.98, 0.98, 0.98, 1.0))
        ax.w_zaxis.set_pane_color((0.98, 0.98, 0.98, 1.0))
    except Exception:
        # fallback: set axes facecolor where available, otherwise set figure background
        try:
            ax.xaxis.set_pane_color((0.98, 0.98, 0.98, 1.0))
            ax.yaxis.set_pane_color((0.98, 0.98, 0.98, 1.0))
        except Exception:
            ax.set_facecolor((0.98, 0.98, 0.98))
        fig.patch.set_facecolor((1.0, 1.0, 1.0))
    ax.grid(True, linestyle=':', linewidth=0.5)

    mins = pso.bounds[:, 0]
    maxs = pso.bounds[:, 1]
    ax.set_xlim(mins[0], maxs[0])
    ax.set_ylim(mins[1], maxs[1])
    ax.set_zlim(mins[2], maxs[2])
    ax.set_xlabel('x', fontsize=11)
    ax.set_ylabel('y', fontsize=11)
    ax.set_zlabel('z', fontsize=11)
    ax.set_title(f'PSO 3D — minimizing {func_name}', fontsize=14)

    # Initial values and color mapping (lower fitness = better)
    init_vals = pso.obj_func(pso.pos)
    norm = Normalize(vmin=np.min(init_vals), vmax=np.max(init_vals))

    scat = ax.scatter(pso.pos[:, 0], pso.pos[:, 1], pso.pos[:, 2],
                      c=init_vals, cmap='viridis', norm=norm,
                      s=90, edgecolors='k', linewidths=0.5)

    # global-best marker (gold star)
    gbest_scatter = ax.scatter([pso.gbest_pos[0]], [pso.gbest_pos[1]], [pso.gbest_pos[2]],
                               s=260, marker='*', c='gold', edgecolors='k', linewidths=1.2)

    # trails: one Line3D per particle
    lines = []
    hist = np.array(pso.history)  # (t, n, 3)
    tlen = hist.shape[0]
    for i in range(pso.n_particles):
        xs = hist[:, i, 0]
        ys = hist[:, i, 1]
        zs = hist[:, i, 2]
        line, = ax.plot(xs, ys, zs, linewidth=1.6, alpha=0.45)
        lines.append(line)

    # colorbar for fitness
    mappable = plt.cm.ScalarMappable(cmap='viridis', norm=norm)
    mappable.set_array([])
    cbar = fig.colorbar(mappable, ax=ax, fraction=0.03, pad=0.1)
    cbar.set_label('fitness (lower is better)')

    iter_text = ax.text2D(0.02, 0.95, '', transform=ax.transAxes, fontsize=11)

    # Keep references mutable so update() can reassign
    state = {'scat': scat, 'gbest': gbest_scatter}

    def update(frame):
        pso.step()

        vals = pso.obj_func(pso.pos)

        # update color mapping range gradually to keep contrast
        current_min = min(np.min(vals), norm.vmin)
        current_max = max(np.max(vals), norm.vmax)
        norm.vmin = current_min
        norm.vmax = current_max

        # remove and redraw scatter to update colors cleanly (3D scatter doesn't support set_array reliably)
        state['scat'].remove()
        scat_new = ax.scatter(pso.pos[:, 0], pso.pos[:, 1], pso.pos[:, 2],
                              c=vals, cmap='viridis', norm=norm,
                              s=90, edgecolors='k', linewidths=0.5)
        state['scat'] = scat_new

        # update global best marker
        state['gbest'].remove()
        state['gbest'] = ax.scatter([pso.gbest_pos[0]], [pso.gbest_pos[1]], [pso.gbest_pos[2]],
                                     s=260, marker='*', c='gold', edgecolors='k', linewidths=1.2)

        # update trails with soft fading: older points less opaque
        hist = np.array(pso.history)  # (t, n, 3)
        tlen = hist.shape[0]
        # draw each particle's history
        for i, line in enumerate(lines):
            xs = hist[:, i, 0]
            ys = hist[:, i, 1]
            zs = hist[:, i, 2]
            line.set_data(xs, ys)
            line.set_3d_properties(zs)
            # fade based on trail length (visual effect)
            line.set_alpha(0.18 + 0.82 * (np.linspace(0, 1, tlen)[-1]))

        # rotate camera slowly for a dynamic view
        if rotate:
            azim = 30 + frame * 0.6
            elev = 30
            ax.view_init(elev=elev, azim=azim % 360)

        iter_text.set_text(f'iter: {frame+1}/{pso.max_iter}   gbest: {pso.gbest_val:.6f}')

        # update colorbar (workaround) by resetting mappable
        mappable.set_clim(norm.vmin, norm.vmax)

        return [state['scat'], state['gbest'], *lines, iter_text]

    anim = FuncAnimation(fig, update, frames=pso.max_iter, interval=100, blit=False)

    if save_file:
        try:
            anim.save(save_file, fps=20)
            print(f"Saved animation to {save_file}")
        except Exception as e:
            print("Warning: could not save animation:", e)

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PSO 3D visualization (improved)')
    parser.add_argument('--func', choices=['rastrigin', 'sphere', 'ackley'], default='rastrigin')
    parser.add_argument('--particles', type=int, default=40)
    parser.add_argument('--iters', type=int, default=200)
    parser.add_argument('--save', type=str, default=None, help='filename to save animation (mp4/gif)')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    bounds = [(-5.12, 5.12), (-5.12, 5.12), (-5.12, 5.12)]
    func_map = {'rastrigin': rastrigin, 'sphere': sphere, 'ackley': ackley}

    objective = func_map[args.func]

    pso = PSO3D(obj_func=objective,
                bounds=bounds,
                n_particles=args.particles,
                w=0.7,
                c1=1.5,
                c2=1.5,
                max_iter=args.iters,
                vel_clamp=1.0,
                seed=args.seed,
                trail_length=14)

    animate_pso(pso, func_name=args.func, save_file=args.save, rotate=True)
