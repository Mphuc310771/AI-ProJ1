import numpy as np
from solution import Solution

class SolutionArchive:
    def __init__(self, K, dim, objective):
        self.K = K
        self.dim = dim
        self.objective = objective

        self.solutions = []
        for _ in range(K):
            x = np.random.uniform(objective.lower, objective.upper, dim)
            f = objective.rastrigin(x)
            self.solutions.append(Solution(x, f))

        self.sort()

        idxs = np.arange(K)
        q = 0.1
        self.weights = 1 / (np.sqrt(2 * np.pi) * q * K) * np.exp(-idxs**2 / (2 * (q * K)**2))
        self.weights /= np.sum(self.weights)

    def sort(self):
        self.solutions.sort(key=lambda s: s.f)

    def get_mean(self, i):
        return self.solutions[i].x

    def get_sigma(self, i):
        X = np.array([s.x for s in self.solutions])
        mean_i = self.solutions[i].x
        return np.mean(np.abs(X - mean_i), axis=0)
