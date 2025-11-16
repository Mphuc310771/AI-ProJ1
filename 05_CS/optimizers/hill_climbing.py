from __future__ import annotations

import numpy as np
from typing import Dict, Any, Optional
from .base import Optimizer


class HillClimbing(Optimizer):
    name = "HillClimbing"

    def __init__(self, obj_func, dim, bounds, rng=None, step_scale: float = 0.1, **params):
        super().__init__(obj_func, dim, bounds, rng, **params)
        self.step_scale = float(step_scale)

    def _run(self, max_evals: int, max_iters: Optional[int], callback) -> Dict[str, Any]:
        # simple stochastic hill-climbing with restarts until max_evals
        best_x = None
        best_f = float("inf")
        history = []

        # start with a random point
        low, high = self.bounds
        low = np.asarray(low)
        high = np.asarray(high)
        if low.shape == ():
            low = np.full(self.dim, float(low))
            high = np.full(self.dim, float(high))

        # current solution
        x = self.rng.uniform(low, high, size=self.dim)
        f = float(self.obj(x.copy()))
        history.append(f)
        evals = 1
        if f < best_f:
            best_f = f
            best_x = x.copy()

        while evals < max_evals:
            # propose neighbor
            sigma = self.step_scale * (high - low)
            x_new = x + self.rng.normal(scale=sigma, size=self.dim)
            x_new = np.clip(x_new, low, high)
            f_new = float(self.obj(x_new.copy()))
            evals += 1
            if f_new < f:
                x = x_new
                f = f_new
                if f_new < best_f:
                    best_f = f_new
                    best_x = x.copy()
            else:
                # with small prob do a random restart to escape plateaus
                if self.rng.rand() < 0.001:
                    x = self.rng.uniform(low, high, size=self.dim)
                    f = float(self.obj(x.copy()))
                    evals += 1
            history.append(best_f)
            if callback is not None:
                callback(evals, best_x, best_f)
        return {"best_x": best_x, "best_f": best_f, "history": history, "evals": evals}
