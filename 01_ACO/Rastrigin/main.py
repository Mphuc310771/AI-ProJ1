from acor import ACOR
from visualizer import Visualizer3D
from utils import plot_convergence

if __name__ == "__main__":
    acor = ACOR(dim=2, K=15, ants=30, xi=0.85, iterations=100)
    best = acor.optimize()

    print("Best solution:", best.x)
    print("Best value:", best.f)

    viz = Visualizer3D(acor.obj)
    viz.plot_surface_with_points(acor.archive.solutions)

    plot_convergence(acor.best_history)
