from __future__ import annotations

import numpy as np
from typing import Dict, Any, Optional
from .base import Optimizer


class GeneticAlgorithm(Optimizer):
    name = "GeneticAlgorithm"

    def __init__(self, obj_func, dim, bounds, rng=None, pop_size: int = 50, crossover_rate: float = 0.9, mutation_rate: float = None, tournament_k: int = 3, **params):
        super().__init__(obj_func, dim, bounds, rng, **params)
        self.pop_size = int(pop_size)
        self.crossover_rate = float(crossover_rate)
        self.tournament_k = int(tournament_k)
        self.mutation_rate = float(mutation_rate) if mutation_rate is not None else 1.0 / max(1, dim)

    def _tournament(self, pop, fitnesses):
        idxs = self.rng.randint(0, self.pop_size, size=self.tournament_k)
        best = idxs[np.argmin(fitnesses[idxs])]
        return pop[best]

    def _run(self, max_evals: int, max_iters: Optional[int], callback) -> Dict[str, Any]:
        low, high = self.bounds
        low = np.asarray(low)
        high = np.asarray(high)
        if low.shape == ():
            low = np.full(self.dim, float(low))
            high = np.full(self.dim, float(high))

        pop = self.rng.uniform(low, high, size=(self.pop_size, self.dim))
        # record bytes used by population array (numpy nbytes)
        try:
            self.last_pop_nbytes = int(pop.nbytes)
        except Exception:
            self.last_pop_nbytes = int(self.pop_size * self.dim * 8)
        fitnesses = np.array([self.obj(ind.copy()) for ind in pop])
        evals = self.pop_size
        best_idx = int(np.argmin(fitnesses))
        best_x = pop[best_idx].copy()
        best_f = float(fitnesses[best_idx])
        history = [best_f]

        while evals < max_evals:
            new_pop = np.empty_like(pop)
            for i in range(self.pop_size):
                # selection
                parent1 = self._tournament(pop, fitnesses).copy()
                parent2 = self._tournament(pop, fitnesses).copy()
                # crossover
                if self.rng.rand() < self.crossover_rate:
                    # uniform crossover
                    mask = self.rng.rand(self.dim) < 0.5
                    child = parent1.copy()
                    child[mask] = parent2[mask]
                else:
                    child = parent1.copy()
                # mutation (gaussian)
                for d in range(self.dim):
                    if self.rng.rand() < self.mutation_rate:
                        sigma = 0.1 * (high[d] - low[d])
                        child[d] += self.rng.normal(scale=sigma)
                child = np.clip(child, low, high)
                new_pop[i] = child
            pop = new_pop
            try:
                self.last_pop_nbytes = int(pop.nbytes)
            except Exception:
                self.last_pop_nbytes = int(self.pop_size * self.dim * 8)
            # evaluate
            fitnesses = np.array([self.obj(ind.copy()) for ind in pop])
            evals += self.pop_size
            cur_best = float(np.min(fitnesses))
            if cur_best < best_f:
                best_idx = int(np.argmin(fitnesses))
                best_f = float(fitnesses[best_idx])
                best_x = pop[best_idx].copy()
            history.append(best_f)
            if callback is not None:
                callback(evals, best_x, best_f)
        return {"best_x": best_x, "best_f": best_f, "history": history, "evals": evals}
