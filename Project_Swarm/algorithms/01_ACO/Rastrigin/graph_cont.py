import numpy as np

class SearchSpace:
    def __init__(self, lower_bound=-5.12, upper_bound=5.12, grid_size=50):
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.grid_size = grid_size

        self.x_vals = np.linspace(lower_bound, upper_bound, grid_size)
        self.y_vals = np.linspace(lower_bound, upper_bound, grid_size)
        self.X, self.Y = np.meshgrid(self.x_vals, self.y_vals)

        self.Z = self.evaluate(self.X, self.Y)

        self.pheromones = np.ones((grid_size, grid_size))

    def evaluate(self, x, y, A=10):
        return A * 2 + (x**2 - A * np.cos(2 * np.pi * x)) + (y**2 - A * np.cos(2 * np.pi * y))
