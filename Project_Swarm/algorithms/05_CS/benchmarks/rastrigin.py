from __future__ import annotations

import numpy as np
from typing import Callable, Tuple


def rastrigin(x: np.ndarray) -> float:
    x = np.asarray(x)
    return float(10 * x.size + np.sum(x ** 2 - 10 * np.cos(2 * np.pi * x)))


def make_counted_rastrigin(dim: int, bounds: Tuple[float, float] = (-5.12, 5.12)) -> Tuple[Callable[[np.ndarray], float], dict]:
    """Return (obj_func, state) where state contains 'count' and 'vals' list.

    The returned obj_func has attribute `state` referencing the dict.
    """
    base = lambda x: rastrigin(x)
    state = {"count": 0, "vals": []}

    def obj(x):
        state["count"] += 1
        v = base(x)
        state["vals"].append(v)
        return v

    obj.state = state
    obj.dim = dim
    obj.bounds = bounds
    return obj, state
