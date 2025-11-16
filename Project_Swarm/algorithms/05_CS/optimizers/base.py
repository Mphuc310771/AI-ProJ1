from __future__ import annotations

import time
from typing import Callable, Optional, Dict, Any
import numpy as np


class Optimizer:
    """Base optimizer interface.

    Concrete optimizers should inherit and implement `_step` or `run`.
    """

    name = "base"

    def __init__(self, obj_func: Callable[[np.ndarray], float], dim: int, bounds=None, rng: Optional[int] = None, **params):
        self.obj = obj_func
        self.dim = int(dim)
        self.bounds = bounds
        self.params = params
        self.rng = np.random.RandomState(rng)

    def run(self, max_evals: int = 10000, max_iters: Optional[int] = None, callback=None) -> Dict[str, Any]:
        """Run optimization until `max_evals` or `max_iters` reached.

        Returns a dict with keys: `best_x`, `best_f`, `history` (list of best_f after each evaluation),
        `evals`, `time`.
        """
        t0 = time.perf_counter()
        res = self._run(max_evals=max_evals, max_iters=max_iters, callback=callback)
        res.setdefault("time", time.perf_counter() - t0)
        return res

    def _run(self, max_evals: int, max_iters: Optional[int], callback) -> Dict[str, Any]:
        raise NotImplementedError()
