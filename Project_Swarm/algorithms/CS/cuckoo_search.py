"""
Cuckoo Search (NumPy-only implementation)

Tính năng:
- Triển khai thuật toán Cuckoo Search (CS) cho tối ưu hóa liên tục, nhị phân và rời rạc (permutation, integer/graph-coloring).
- Chỉ dùng NumPy (không dùng SciPy / scikit-learn...).
- Mã mô-đun, có docstring chi tiết, hỗ trợ seed, trả về lịch sử, callback.
- Bao gồm các hàm mục tiêu chuẩn: Sphere, Rastrigin, Rosenbrock, Ackley.
- Bao gồm ví dụ cho TSP (permutation), Knapsack (binary), Graph Coloring (integer/coloring).

Sử dụng:
- Gọi CuckooSearch(obj_func, dim, dtype=..., **kwargs) rồi optimize(n_iter).
- Đối với permutation: cung cấp `initializer='random_permutation'` và `bounds` không cần.
- Đối với integer (coloring): cung cấp một tuple `bounds=(min_val, max_val)`.

Tệp này có ví dụ chạy nhanh ở cuối ("__main__").

"""

from __future__ import annotations

import numpy as np
import seaborn as sns
from typing import Callable, Optional, Tuple, Dict, Any
import matplotlib.pyplot as plt
  

def _mantegna_levy_rng(rng: np.random.RandomState, beta: float, size: int) -> np.ndarray:
    """Sinh bước Levy theo Mantegna dùng rng để đảm bảo reproducibility."""
    from math import gamma, sin, pi

    sigma_u = (
        (gamma(1 + beta) * sin(pi * beta / 2.0))
        / (gamma((1 + beta) / 2.0) * beta * 2 ** ((beta - 1.0) / 2.0))
    ) ** (1.0 / beta)
    u = rng.normal(0, sigma_u, size=size)
    v = rng.normal(0, 1.0, size=size)
    step = u / (np.abs(v) ** (1.0 / beta))
    return step


class CuckooSearch:
    """Cuckoo Search optimizer supporting multiple encodings.

    dtype options:
      - 'continuous' : real-valued vectors (requires bounds=(low,high))
      - 'binary'     : bitstrings (0/1)
      - 'permutation': permutations (for TSP)
      - 'integer'    : integer vectors with bounds (min,max) per dimension (useful for coloring)

    obj_func should accept a NumPy array representing a candidate and return a scalar (to minimize).
    """

    def __init__(
        self,
        obj_func: Callable[[np.ndarray], float],
        dim: int,
        bounds: Optional[Any] = None,
        n_nests: int = 25,
        pa: float = 0.25,
        alpha: float = 0.01,
        beta: float = 1.5,
        dtype: str = "continuous",
        rng: Optional[Any] = None,
        initializer: Optional[str] = None,
    ) -> None:
        if dtype not in ("continuous", "binary", "permutation", "integer"):
            raise ValueError("dtype must be one of 'continuous','binary','permutation','integer'")

        self.obj_func = obj_func
        self.dim = int(dim)
        self.bounds = bounds
        self.n_nests = int(n_nests)
        self.pa = float(pa)
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.dtype = dtype
        self.initializer = initializer

        if rng is None or isinstance(rng, int):
            self.rng = np.random.RandomState(rng)
        else:
            self.rng = rng

        self.nests = None
        self.fvals = None
        self.best_idx = None
        self.best = None
        self.best_f = np.inf

    # ----------------- Initialization -----------------
    def _init_nests(self) -> None:
        if self.dtype == "continuous":
            if self.bounds is None:
                raise ValueError("bounds must be provided for continuous problems")
            low, high = self.bounds
            low = np.asarray(low)
            high = np.asarray(high)
            if low.shape == ():
                low = np.full(self.dim, float(low))
                high = np.full(self.dim, float(high))
            self.nests = self.rng.uniform(low, high, size=(self.n_nests, self.dim))
        elif self.dtype == "binary":
            self.nests = self.rng.randint(0, 2, size=(self.n_nests, self.dim)).astype(np.int8)
        elif self.dtype == "permutation":
            self.nests = np.array([self._random_permutation() for _ in range(self.n_nests)])
        elif self.dtype == "integer":
            # bounds can be (min_val, max_val) scalars or arrays
            if self.bounds is None:
                raise ValueError("bounds must be provided for integer problems")
            lo, hi = self.bounds
            lo = np.asarray(lo)
            hi = np.asarray(hi)
            if lo.shape == ():
                lo = np.full(self.dim, int(lo))
                hi = np.full(self.dim, int(hi))
            arr = np.empty((self.n_nests, self.dim), dtype=np.int32)
            for i in range(self.dim):
                arr[:, i] = self.rng.randint(lo[i], hi[i] + 1, size=self.n_nests)
            self.nests = arr

        self.fvals = np.array([self._eval(i) for i in range(self.n_nests)])
        self.best_idx = int(np.argmin(self.fvals))
        self.best = self.nests[self.best_idx].copy()
        self.best_f = float(self.fvals[self.best_idx])
        # record approximate bytes used by nests array (numpy nbytes when available)
        try:
            self.last_nests_nbytes = int(self.nests.nbytes)
        except Exception:
            # fallback estimate
            self.last_nests_nbytes = int(self.n_nests * self.dim * 8)

    def _random_permutation(self) -> np.ndarray:
        perm = np.arange(self.dim)
        self.rng.shuffle(perm)
        return perm.copy()

    def _eval(self, idx_or_candidate) -> float:
        if isinstance(idx_or_candidate, int):
            x = self.nests[int(idx_or_candidate)]
        else:
            x = np.asarray(idx_or_candidate)
        return float(self.obj_func(x.copy()))

    # ----------------- Move operators -----------------
    def _levy_step(self, size: int) -> np.ndarray:
        return _mantegna_levy_rng(self.rng, self.beta, size)

    def _continuous_move(self, x: np.ndarray, best: np.ndarray) -> np.ndarray:
        step = self._levy_step(self.dim)
        x_new = x + self.alpha * step * (x - best)
        low, high = self.bounds
        low = np.asarray(low)
        high = np.asarray(high)
        if low.shape == ():
            low = np.full(self.dim, float(low))
            high = np.full(self.dim, float(high))
        x_new = np.clip(x_new, low, high)
        return x_new

    def _binary_move(self, x: np.ndarray) -> np.ndarray:
        step = self._levy_step(self.dim)
        probs = 1.0 / (1.0 + np.exp(-np.abs(step)))
        flips = (self.rng.rand(self.dim) < probs).astype(np.int8)
        return x ^ flips

    def _permutation_move(self, perm: np.ndarray, best_perm: np.ndarray) -> np.ndarray:
        # two simple permutation-specific moves:
        #  - swap two indices with probability from Levy
        #  - partially reorder according to best
        p = perm.copy()
        # swap count proportional to |levy|
        step = self._levy_step(self.dim)
        k = max(1, int(np.mean(np.abs(step)) * self.dim))
        for _ in range(k):
            i, j = self.rng.randint(0, self.dim, size=2)
            p[i], p[j] = p[j], p[i]
        # sometimes apply part of best: take a segment from best and insert
        if self.rng.rand() < 0.2:
            a, b = sorted(self.rng.choice(self.dim, size=2, replace=False))
            seg = best_perm[a:b+1]
            # remove seg elements from p and insert at a
            remaining = [v for v in p if v not in set(seg)]
            p = np.array(remaining[:a] + list(seg) + remaining[a:])
        return p

    def _integer_move(self, x: np.ndarray, best: np.ndarray) -> np.ndarray:
        step = self._levy_step(self.dim)
        delta = np.round(self.alpha * step).astype(np.int32)
        x_new = x + delta * (x - best)
        lo, hi = self.bounds
        lo = np.asarray(lo)
        hi = np.asarray(hi)
        if lo.shape == ():
            lo = np.full(self.dim, int(lo))
            hi = np.full(self.dim, int(hi))
        x_new = np.minimum(np.maximum(x_new, lo), hi)
        return x_new.astype(np.int32)

    # ----------------- Abandon -----------------
    def _abandon(self) -> None:
        n_abandon = int(np.floor(self.pa * self.n_nests))
        if n_abandon <= 0:
            return
        idxs = self.rng.choice(self.n_nests, size=n_abandon, replace=False)
        for idx in idxs:
            if self.dtype == "continuous":
                low, high = self.bounds
                low = np.asarray(low)
                high = np.asarray(high)
                if low.shape == ():
                    low = np.full(self.dim, float(low))
                    high = np.full(self.dim, float(high))
                self.nests[idx] = self.rng.uniform(low, high, size=self.dim)
            elif self.dtype == "binary":
                self.nests[idx] = self.rng.randint(0, 2, size=self.dim)
            elif self.dtype == "permutation":
                self.nests[idx] = self._random_permutation()
            elif self.dtype == "integer":
                lo, hi = self.bounds
                lo = np.asarray(lo)
                hi = np.asarray(hi)
                if lo.shape == ():
                    lo = np.full(self.dim, int(lo))
                    hi = np.full(self.dim, int(hi))
                for d in range(self.dim):
                    self.nests[idx, d] = self.rng.randint(lo[d], hi[d] + 1)
            self.fvals[idx] = self._eval(int(idx))

    # ----------------- Optimize -----------------
    def optimize(
        self,
        n_iter: int = 1000,
        verbose: bool = False,
        callback: Optional[Callable[[int, np.ndarray, float], None]] = None,
    ) -> Dict[str, Any]:
        self._init_nests()
        history = []
        for it in range(1, int(n_iter) + 1):
            for i in range(self.n_nests):
                x = self.nests[i]
                if self.dtype == "continuous":
                    new = self._continuous_move(x, self.best)
                elif self.dtype == "binary":
                    new = self._binary_move(x)
                elif self.dtype == "permutation":
                    new = self._permutation_move(x, self.best)
                elif self.dtype == "integer":
                    new = self._integer_move(x, self.best)
                fnew = float(self.obj_func(new.copy()))
                if fnew < self.fvals[i]:
                    self.nests[i] = new
                    self.fvals[i] = fnew
                    if fnew < self.best_f:
                        self.best_f = fnew
                        self.best = self.nests[i].copy()

            self._abandon()

            cur_best_idx = int(np.argmin(self.fvals))
            if self.fvals[cur_best_idx] < self.best_f:
                self.best_f = float(self.fvals[cur_best_idx])
                self.best = self.nests[cur_best_idx].copy()

            history.append(self.best_f)
            if callback is not None:
                callback(it, self.best.copy(), self.best_f)
            if verbose and (it % max(1, int(n_iter / 10)) == 0 or it == 1):
                print(f"Iter {it}/{n_iter} best_f={self.best_f:.6g}")

        return {"best": self.best, "best_f": self.best_f, "history": np.array(history)}


# ----------------- Các hàm mục tiêu tiêu chuẩn -----------------

def sphere(x: np.ndarray) -> float:
    return float(np.sum(x ** 2))


def rastrigin(x: np.ndarray) -> float:
    return float(10 * x.size + np.sum(x ** 2 - 10 * np.cos(2 * np.pi * x)))


def rosenbrock(x: np.ndarray) -> float:
    return float(np.sum(100.0 * (x[1:] - x[:-1] ** 2) ** 2 + (x[:-1] - 1.0) ** 2))


def ackley(x: np.ndarray) -> float:
    a = 20
    b = 0.2
    c = 2 * np.pi
    n = x.size
    s1 = np.sum(x ** 2)
    s2 = np.sum(np.cos(c * x))
    return float(-a * np.exp(-b * np.sqrt(s1 / n)) - np.exp(s2 / n) + a + np.e)


# ----------------- Bài toán rời rạc: TSP, KP, Graph Coloring helpers -----------------

# TSP: obj takes permutation (array of node indices) and returns total tour length

def make_tsp_obj(distance_matrix: np.ndarray) -> Callable[[np.ndarray], float]:
    def obj(perm: np.ndarray) -> float:
        perm = perm.astype(int)
        n = perm.size
        dist = 0.0
        for i in range(n - 1):
            dist += distance_matrix[perm[i], perm[i + 1]]
        dist += distance_matrix[perm[-1], perm[0]]
        return float(dist)

    return obj


# Knapsack: binary vector x indicating items taken. obj returns negative value (we minimize), so -total_value if weight<=W else large penalty

def make_knap_obj(values: np.ndarray, weights: np.ndarray, capacity: float, penalty: float = 1e6) -> Callable[[np.ndarray], float]:
    def obj(x: np.ndarray) -> float:
        x = x.astype(int)
        total_w = float(np.dot(weights, x))
        total_v = float(np.dot(values, x))
        if total_w > capacity:
            return float(penalty + (total_w - capacity))
        return float(-total_v)

    return obj


# Graph coloring: integer vector colors[v] in [0, K-1]. Objective: number of edge conflicts (minimize). Secondary objective: number of colors used.

def make_graph_coloring_obj(edges: np.ndarray, K: int) -> Callable[[np.ndarray], float]:
    # edges: array shape (m,2) of undirected edges
    def obj(colors: np.ndarray) -> float:
        colors = colors.astype(int)
        conflicts = 0
        for u, v in edges:
            if colors[int(u)] == colors[int(v)]:
                conflicts += 1
        # also penalize using many colors (optional): fraction of used colors
        used = len(np.unique(colors))
        return float(conflicts + 0.01 * used)

    return obj


# ----------------- Ví dụ chạy nhanh -----------------
if __name__ == "__main__":
    
    def plot_convergence_with_std(histories, title, xlabel='Iteration', ylabel='Best objective'):
        """Vẽ convergence plot với mean ± std"""
        histories = np.array(histories)
        mean_hist = np.mean(histories, axis=0)
        std_hist = np.std(histories, axis=0)
        
        plt.figure(figsize=(10, 6))
        iterations = np.arange(len(mean_hist))
        
        # Vẽ đường mean
        sns.lineplot(x=iterations, y=mean_hist, label='Mean')
        
        # Vẽ vùng sai số (mean ± std)
        plt.fill_between(iterations, 
                         mean_hist - std_hist, 
                         mean_hist + std_hist, 
                         alpha=0.3, label='Mean ± Std')
        
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()
    n_runs = 5
    
    # Continuous test: Rastrigin in 5D (chạy 30 lần)
    print("--- Continuous: Rastrigin (5D) ---")
    dim = 5
    rastrigin_histories = []
    
    for run in range(n_runs):
        print(f"Run {run+1}/{n_runs}")
        cs = CuckooSearch(obj_func=rastrigin, dim=dim, bounds=(-5.12, 5.12), 
                        n_nests=100, pa=0.25, alpha=0.3, beta=1.5, 
                        dtype='continuous', rng=42+run)
        out = cs.optimize(n_iter=500, verbose=False)
        rastrigin_histories.append(out['history'])
        print(f"  best_f: {out['best_f']:.6f}")
    
    plot_convergence_with_std(rastrigin_histories, 'Convergence plot (Rastrigin)')

    # TSP example: small random points (chạy 5 lần)
    print("\n--- Discrete: TSP (permutation) ---")
    n_nodes = 20
    pts = np.random.RandomState(2).rand(n_nodes, 2)
    dmat = np.sqrt(((pts[:, None, :] - pts[None, :, :]) ** 2).sum(axis=2))
    tsp_obj = make_tsp_obj(dmat)
    tsp_histories = []
    
    for run in range(n_runs):
        print(f"Run {run+1}/{n_runs}")
        cs_tsp = CuckooSearch(obj_func=tsp_obj, dim=n_nodes, dtype='permutation', 
                             n_nests=60, pa=0.3, alpha=0.5, beta=1.5, rng=1+run)
        out_tsp = cs_tsp.optimize(n_iter=500, verbose=False)
        tsp_histories.append(out_tsp['history'])
        print(f"  best_f: {out_tsp['best_f']:.6f}")
    
    plot_convergence_with_std(tsp_histories, 'Convergence plot (TSP)')

    # Knapsack example (chạy 5 lần)
    print("\n--- Discrete: Knapsack (binary) ---")
    rng_seed = np.random.RandomState(1)
    n_items = 50
    values = rng_seed.randint(10, 100, size=n_items)
    weights = rng_seed.randint(1, 50, size=n_items)
    capacity = int(0.3 * weights.sum())
    kp_obj = make_knap_obj(values, weights, capacity)
    kp_histories = []
    
    for run in range(n_runs):
        print(f"Run {run+1}/{n_runs}")
        cs_kp = CuckooSearch(obj_func=kp_obj, dim=n_items, dtype='binary', 
                            n_nests=90, pa=0.25, alpha=0.5, beta=1.5, rng=2+run)
        out_kp = cs_kp.optimize(n_iter=500, verbose=False)
        kp_histories.append(out_kp['history'])
        print(f"  best_f: {out_kp['best_f']:.6f}")
    
    plot_convergence_with_std(kp_histories, 'Convergence plot (Knapsack)')

    # Graph coloring example (chạy 5 lần)
    print("\n--- Discrete: Graph Coloring ---")
    rng_graph = np.random.RandomState(3)
    n = 30
    p = 0.08
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if rng_graph.rand() < p:
                edges.append((i, j))
    edges = np.array(edges, dtype=int)
    K = 4
    gc_obj = make_graph_coloring_obj(edges, K)
    gc_histories = []
    
    for run in range(n_runs):
        print(f"Run {run+1}/{n_runs}")
        cs_gc = CuckooSearch(obj_func=gc_obj, dim=n, bounds=(0, K - 1), dtype='integer', 
                            n_nests=80, pa=0.25, alpha=0.7, beta=1.5, rng=4+run)
        out_gc = cs_gc.optimize(n_iter=500, verbose=False)
        gc_histories.append(out_gc['history'])
        print(f"  best_f: {out_gc['best_f']:.6f}")
    
    plot_convergence_with_std(gc_histories, 'Convergence plot (Graph Coloring)')