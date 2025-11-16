from __future__ import annotations

import numpy as np
from typing import Dict, Any, Optional
from .base import Optimizer


class SimulatedAnnealing(Optimizer):
    name = "SimulatedAnnealing"

    def __init__(self, obj_func, dim, bounds, rng=None, t0: float = 1.0, cooling: float = 0.995, step_scale: float = 0.2, **params):
        super().__init__(obj_func, dim, bounds, rng, **params)
        self.t = float(t0)
        self.cooling = float(cooling)
        self.step_scale = float(step_scale)

    def _run(self, max_evals: int, max_iters: Optional[int], callback) -> Dict[str, Any]:
        low, high = self.bounds
        low = np.asarray(low)
        high = np.asarray(high)
        if low.shape == ():
            low = np.full(self.dim, float(low))
            high = np.full(self.dim, float(high))

        x = self.rng.uniform(low, high, size=self.dim)
        f = float(self.obj(x.copy()))
        best_x = x.copy()
        best_f = f
        history = [best_f]
        evals = 1

        while evals < max_evals:
            sigma = self.step_scale * (high - low)
            x_new = x + self.rng.normal(scale=sigma, size=self.dim)
            x_new = np.clip(x_new, low, high)
            f_new = float(self.obj(x_new.copy()))
            evals += 1
            delta = f_new - f
            if delta < 0 or self.rng.rand() < np.exp(-delta / max(1e-12, self.t)):
                x = x_new
                f = f_new
                if f < best_f:
                    best_f = f
                    best_x = x.copy()
            # cool down
            self.t *= self.cooling
            history.append(best_f)
            if callback is not None:
                callback(evals, best_x, best_f)
        return {"best_x": best_x, "best_f": best_f, "history": history, "evals": evals}
