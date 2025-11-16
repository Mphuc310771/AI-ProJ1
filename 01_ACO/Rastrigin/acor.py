import numpy as np
from objective_function import ObjectiveFunction
from solution import Solution
from solution_archive import SolutionArchive

class ACOR:
    def __init__(self, dim=2, K=10, ants=20, xi=0.85, iterations=200):
        self.dim = dim
        self.K = K
        self.ants = ants
        self.xi = xi
        self.iter = iterations

        self.obj = ObjectiveFunction(dim)
        self.archive = SolutionArchive(K, dim, self.obj)

        self.best_history = []

    def sample_new_solution(self):
        i = np.random.choice(self.K, p=self.archive.weights)

        mean = self.archive.get_mean(i)
        sigma = self.archive.get_sigma(i)

        x_new = np.random.normal(mean, self.xi * sigma)
        x_new = np.clip(x_new, self.obj.lower, self.obj.upper)

        f_new = self.obj.rastrigin(x_new)
        return Solution(x_new, f_new)

    def optimize(self):
        for _ in range(self.iter):
            new_solutions = [self.sample_new_solution() for _ in range(self.ants)]

            merged = self.archive.solutions + new_solutions
            merged.sort(key=lambda s: s.f)
            self.archive.solutions = merged[:self.K]

            self.best_history.append(self.archive.solutions[0].f)

        return self.archive.solutions[0]
