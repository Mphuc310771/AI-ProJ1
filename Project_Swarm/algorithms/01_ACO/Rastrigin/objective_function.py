import numpy as np

class ObjectiveFunction:
    def __init__(self, dim, lower=-5.12, upper=5.12):
        self.dim = dim
        self.lower = lower
        self.upper = upper

    def rastrigin(self, x, A=10):
        x = np.asarray(x)
        return A * self.dim + np.sum(x**2 - A * np.cos(2 * np.pi * x))