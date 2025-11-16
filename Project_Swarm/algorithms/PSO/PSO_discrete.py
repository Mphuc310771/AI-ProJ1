import numpy as np
import random
import copy

def pso_discrete_tsp(distance_matrix, n_particles=40, n_iter=200, w=0.5, c1=1.5, c2=1.5, seed=None):
    """
    Discrete PSO cho TSP.
    distance_matrix: ma trận khoảng cách (n x n)
    Trả về: best_path, best_cost, history
    """
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)

    n_cities = distance_matrix.shape[0]
    particles = [np.random.permutation(n_cities) for _ in range(n_particles)]
    fitness = np.array([tsp_cost(p, distance_matrix) for p in particles])

    pbest_pos = [p.copy() for p in particles]
    pbest_fit = fitness.copy()

    gbest_idx = np.argmin(pbest_fit)
    gbest_pos = pbest_pos[gbest_idx].copy()
    gbest_fit = pbest_fit[gbest_idx]

    history = [gbest_fit]

    def apply_swap(perm, swap):
        perm = perm.copy()
        for i, j in swap:
            perm[i], perm[j] = perm[j], perm[i]
        return perm

    def get_swap_sequence(p1, p2):
        """Trả về danh sách swap để biến p1 thành p2"""
        p1 = p1.copy()
        swap_seq = []
        for i in range(len(p1)):
            if p1[i] != p2[i]:
                j = np.where(p1 == p2[i])[0][0]
                swap_seq.append((i, j))
                p1[i], p1[j] = p1[j], p1[i]
        return swap_seq


    for t in range(n_iter):
        for i in range(n_particles):
            swap_pbest = get_swap_sequence(particles[i], pbest_pos[i])
            swap_gbest = get_swap_sequence(particles[i], gbest_pos)

            new_perm = particles[i].copy()
            n_swap_p = int(c1 * len(swap_pbest))
            n_swap_g = int(c2 * len(swap_gbest))
            if n_swap_p > 0:
                swap_sel = random.sample(swap_pbest, min(n_swap_p, len(swap_pbest)))
                new_perm = apply_swap(new_perm, swap_sel)
            if n_swap_g > 0:
                swap_sel = random.sample(swap_gbest, min(n_swap_g, len(swap_gbest)))
                new_perm = apply_swap(new_perm, swap_sel)

            if random.random() < w:
                a, b = np.random.choice(n_cities, 2, replace=False)
                new_perm[a], new_perm[b] = new_perm[b], new_perm[a]

            particles[i] = new_perm
            fitness[i] = tsp_cost(new_perm, distance_matrix)

            if fitness[i] < pbest_fit[i]:
                pbest_pos[i] = new_perm.copy()
                pbest_fit[i] = fitness[i]

        gbest_idx = np.argmin(pbest_fit)
        if pbest_fit[gbest_idx] < gbest_fit:
            gbest_pos = pbest_pos[gbest_idx].copy()
            gbest_fit = pbest_fit[gbest_idx]

        history.append(gbest_fit)

    return gbest_pos, gbest_fit, history

def tsp_cost(path, distance_matrix):
    total = 0
    n = len(path)
    for i in range(n):
        total += distance_matrix[path[i-1], path[i]] 
    return total
