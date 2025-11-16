from __future__ import annotations

import numpy as np
from typing import Callable, Dict, List, Optional, Tuple


class HillClimbing:
    """Simple stochastic hill-climbing optimizer with fixed-step Gaussian moves.

    API:
      HillClimbing(obj_func, dim, bounds, rng=None, step_scale=0.1)
      .run(max_evals) -> dict with keys 'history' (list of best-so-far values) and 'best_f'
    """

    def __init__(
        self,
        obj_func: Callable[[np.ndarray], float],
        dim: int,
        bounds: Tuple[float, float] = (-5.12, 5.12),
        rng: Optional[int | np.random.RandomState] = None,
        step_scale: float = 0.1,
    ) -> None:
        self.obj = obj_func
        self.dim = int(dim)
        self.lb, self.ub = float(bounds[0]), float(bounds[1])
        self.step_scale = float(step_scale)
        if isinstance(rng, np.random.RandomState):
            self.rng = rng
        else:
            self.rng = np.random.RandomState(None if rng is None else int(rng))

        self.best_x: Optional[np.ndarray] = None
        self.best_f: float = float('inf')
        self.best_history: List[float] = []

    def _random_solution(self) -> np.ndarray:
        return self.rng.uniform(self.lb, self.ub, size=(self.dim,))

    def run(self, max_evals: int = 1000) -> Dict[str, object]:
        # initialize
        x = self._random_solution()
        fx = float(self.obj(x))
        self.best_x = x.copy()
        self.best_f = fx
        self.best_history = [self.best_f]

        evals = 1
        while evals < max_evals:
            # propose a gaussian step
            step = self.rng.normal(loc=0.0, scale=self.step_scale, size=(self.dim,))
            x_new = x + step
            # clip to bounds
            x_new = np.clip(x_new, self.lb, self.ub)
            fx_new = float(self.obj(x_new))
            evals += 1
            # accept if better
            if fx_new < fx:
                x, fx = x_new, fx_new
            # update best
            if fx < self.best_f:
                self.best_f = fx
                self.best_x = x.copy()
            self.best_history.append(self.best_f)

        return {"history": self.best_history, "best_f": float(self.best_f), "best_x": (self.best_x.tolist() if self.best_x is not None else None)}


def _rastrigin(x: np.ndarray) -> float:
    A = 10.0
    x = np.asarray(x)
    return float(A * x.size + np.sum(x ** 2 - A * np.cos(2 * np.pi * x)))


def demo_run():
    hc = HillClimbing(obj_func=_rastrigin, dim=5, bounds=(-5.12, 5.12), rng=42, step_scale=0.1)
    out = hc.run(max_evals=500)
    print("best_f:", out['best_f'])


if __name__ == '__main__':
    demo_run()
