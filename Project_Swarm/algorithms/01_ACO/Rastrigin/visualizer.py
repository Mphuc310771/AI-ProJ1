import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class Visualizer3D:
    def __init__(self, objective, resolution=100):
        self.obj = objective
        self.resolution = resolution

    def plot_surface_with_points(self, points):
        if self.obj.dim != 2:
            raise ValueError("3D visualization only works with dim=2.")

        x = np.linspace(self.obj.lower, self.obj.upper, self.resolution)
        y = np.linspace(self.obj.lower, self.obj.upper, self.resolution)
        X, Y = np.meshgrid(x, y)

        Z = 20 + (X**2 - 10 * np.cos(2 * np.pi * X)) + (Y**2 - 10 * np.cos(2 * np.pi * Y))

        fig = plt.figure(figsize=(12, 7))
        ax = fig.add_subplot(111, projection="3d")

        ax.plot_surface(X, Y, Z, cmap="viridis", alpha=0.7)

        px = [p.x[0] for p in points]
        py = [p.x[1] for p in points]
        pz = [p.f for p in points]

        ax.scatter(px, py, pz, color="red", s=50)
        ax.set_title("ACOR Sampling on Rastrigin 2D")

        plt.show()
