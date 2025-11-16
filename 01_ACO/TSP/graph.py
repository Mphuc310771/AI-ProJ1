import numpy as np

class Graph:
    def __init__(self, distances):
        self.distances = distances
        self.num_nodes = len(distances)
        self.pheromones = np.ones_like(distances, dtype=float)

        heuristics = np.zeros_like(distances, dtype=float)
        heuristics[distances > 0] = 1.0 / distances[distances > 0]
        self.heuristic = heuristics