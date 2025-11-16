# Vẽ 3D

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
from firefly_algorithm import FireflyAlgorithm
# from mpl_toolkits.mplot3d import Axes3D


# =============================
# PARAMETERS
# =============================
POINT_COLOR = "#fff8a0"   # vàng nhạt (core)
GLOW_COLOR = "#ffcc33"    # vàng đậm (glow)
SAVE_VIDEO = True
VIDEO_NAME = "firefly_3d.mp4"

N_FIREFLIES = 300
MAX_GEN = 30


# =============================
# OBJECTIVE FUNCTION
# =============================
def sphere3d(x):
    return np.sum(x**2)


# =============================
# UPDATE
# =============================
def update_fireflies(pop, brightness, alpha, beta0, gamma, lb, ub):
    n, dim = pop.shape
    new_pop = pop.copy()
    for i in range(n):
        for j in range(n):
            if brightness[j] > brightness[i]:
                r = np.linalg.norm(pop[i] - pop[j])
                beta = beta0 * np.exp(-gamma * r**2)
                rand = alpha * (np.random.rand(dim) - 0.5)
                new_pop[i] = pop[i] + beta * (pop[j] - pop[i]) + rand
    return np.clip(new_pop, lb, ub)




# =============================
# MAIN
# =============================
def main():
    algo = FireflyAlgorithm(
        objective_fn=sphere3d,
        dim=3, n_fireflies=N_FIREFLIES, max_gen=MAX_GEN,
        alpha=0.3, beta0=1.0, gamma=0.03,
        lb=-10, ub=10,
        problem_type="continuous",
        seed=12
    )

    dim = algo.dim
    n = algo.n_fireflies
    lb, ub = algo.lb, algo.ub
    alpha, beta0, gamma = algo.alpha, algo.beta0, algo.gamma

    pop = np.random.uniform(lb, ub, (n, dim))

    # --------------------------
    # Setup 3D figure
    # --------------------------
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    fig.patch.set_facecolor("#000000")
    ax.set_facecolor("#000000")
    ax.grid(False)

    for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
        axis.pane.fill = False
        axis.pane.set_edgecolor("#333333")

    ax.set_xlabel("X", color="white")
    ax.set_ylabel("Y", color="white")
    ax.set_zlabel("Z", color="white")
    ax.tick_params(colors="white")

    writer = None
    if SAVE_VIDEO:
        writer = FFMpegWriter(fps=7)
        writer.setup(fig, VIDEO_NAME, dpi=150)

    # Two layers: core + glow
    scat_core = ax.scatter([], [], [], s=0, c=POINT_COLOR, alpha=1.0)
    scat_glow = ax.scatter([], [], [], s=0, c=GLOW_COLOR, alpha=0.05)

    # --------------------------
    # Animation loop
    # --------------------------
    for gen in range(MAX_GEN + 1):

        # Compute fitness & brightness
        fitness = np.array([sphere3d(p) for p in pop])
        brightness = 1 / (1 + fitness)   # 0 → weak, 1 → strong

        x = pop[:, 0]
        y = pop[:, 1]
        z = pop[:, 2]

        # =============================
        # MULTI-LAYER BLOOM GLOW EFFECT
        # =============================

        # Core (sharp bright center)
        core_sizes = 20 + 500 * brightness
        scat_core._offsets3d = (x, y, z)
        scat_core.set_sizes(core_sizes)
        core_alpha = 0.1 + 0.5 * brightness  # sáng mạnh → đậm hơn
        color_core = np.zeros((n, 4))
        color_core[:, 0:3] = [1.0, 1.0, 0.7]  # RGB
        color_core[:, 3] = core_alpha  # A

        scat_core.set_facecolors(color_core)

        colors = np.zeros((n, 4))
        for i in range(n):
            b = brightness[i]
            # vàng → trắng
            colors[i] = [
                1.0,
                0.85 + 0.15 * b,
                0.2 + 0.8 * b,
                0.6 + 0.4 * b  # càng sáng alpha càng cao
            ]

        scat_core.set_facecolors(colors)

        # Glow (big, soft, bloom)
        glow_sizes = 1500 * (brightness ** 0.8) + 100
        scat_glow._offsets3d = (x, y, z)
        scat_glow.set_sizes(glow_sizes)

        # Alpha of glow also follows brightness → sáng hơn thì ánh sáng lan rộng
        scat_glow.set_alpha(0.1 + 0.25 * brightness)

        # --------------------------
        # Titles & limits
        # --------------------------
        ax.set_title(
            f"Firefly Algorithm — Bloom Lighting — Gen {gen}",
            color="white", fontsize=14
        )
        ax.set_xlim(lb, ub)
        ax.set_ylim(lb, ub)
        ax.set_zlim(lb, ub)

        if writer:
            writer.grab_frame()

        plt.pause(0.05)

        # Update swarm
        pop = update_fireflies(pop, brightness, alpha, beta0, gamma, lb, ub)

    if writer:
        writer.finish()

    plt.show()


if __name__ == "__main__":
    main()