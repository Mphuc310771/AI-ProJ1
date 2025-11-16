import numpy as np
import matplotlib.pyplot as plt
from ant import Ant

class ACO:
    def __init__(self, graph, num_ants, num_iterations, alpha, beta, q0, rho, phi):
        self.graph = graph
        self.num_ants = num_ants
        self.num_iterations = num_iterations

        self.alpha = alpha
        self.beta = beta
        self.q0 = q0
        self.decay = rho
        self.phi = phi
        self.rho = rho

        avg_distance = np.mean(self.graph.distances[self.graph.distances > 0])
        self.tau0 = 1.0 / (graph.num_nodes * avg_distance)

        self.graph.pheromones = np.full(self.graph.distances.shape, self.tau0, dtype=float)

        self.best_distance_history = []
        self.run_history = []

    def run(self):
        best_path = None
        best_distance = np.inf

        for iteration in range(self.num_iterations):
            ants = [Ant(self.graph, self.alpha, self.beta, self.q0, self.phi, self.tau0) for _ in range(self.num_ants)]
            iteration_best_distance = np.inf
            iteration_best_path = None
            for ant in ants:
                ant.complete_path()
                if ant.total_distance < iteration_best_distance:
                    iteration_best_distance = ant.total_distance
                    iteration_best_path = ant.path
            
            if iteration_best_distance < best_distance:
                best_distance = iteration_best_distance
                best_path = iteration_best_path

            self.offline_pheromone_update(best_path, best_distance)
            self.best_distance_history.append(best_distance)

            history_snapshot = {
                'iteration': iteration,
                'best_distance': best_distance,
                'best_path': best_path.copy(), # Phải copy
                'pheromones': self.graph.pheromones.copy() # Phải copy
            }
            self.run_history.append(history_snapshot)
        return best_path, best_distance, self.best_distance_history, self.run_history

    def offline_pheromone_update(self, best_path, best_distance):
        best_path_mask = np.zeros_like(self.graph.pheromones, dtype=bool)

        for i in range(len(best_path)):
            from_node = best_path[i]
            to_node = best_path[(i + 1) % len(best_path)] 
            best_path_mask[from_node][to_node] = True


        delta_tau = 1.0 / best_distance
        pheromones = self.graph.pheromones
        pheromones[best_path_mask] = (1 - self.rho) * pheromones[best_path_mask] + self.rho * delta_tau
        self.graph.pheromones = pheromones


# import numpy as np
# from ant import Ant

# class ACO:
#     def __init__(self, graph, num_ants, num_iterations, alpha, beta, rho, Q):
#         self.graph = graph
#         self.num_ants = num_ants
#         self.num_iterations = num_iterations

#         self.alpha = alpha
#         self.beta = beta
#         self.rho = rho      # evaporation
#         self.Q = Q          # pheromone constant

#         self.best_distance_history = []

#     def run(self):
#         best_distance = np.inf
#         best_path = None

#         for iteration in range(self.num_iterations):
#             ants = [Ant(self.graph, self.alpha, self.beta) for _ in range(self.num_ants)]

#             for ant in ants:
#                 ant.complete_path()

#             distances = np.array([ant.total_distance for ant in ants])
#             iteration_best_idx = np.argmin(distances)
#             iteration_best_ant = ants[iteration_best_idx]

#             if iteration_best_ant.total_distance < best_distance:
#                 best_distance = iteration_best_ant.total_distance
#                 best_path = iteration_best_ant.path.copy()

#             self.evaporate_pheromones()
#             self.deposit_pheromones(ants)

#             self.best_distance_history.append(best_distance)
#             # print(f"Iteration {iteration+1}/{self.num_iterations} — Best: {best_distance:.4f}")

#         return best_path, best_distance, self.best_distance_history

#     def evaporate_pheromones(self):
#         self.graph.pheromones *= (1 - self.rho)

#     def deposit_pheromones(self, ants):
#         for ant in ants:
#             delta = self.Q / ant.total_distance
#             for i in range(len(ant.path) - 1):
#                 a = ant.path[i]
#                 b = ant.path[i+1]
#                 self.graph.pheromones[a][b] += delta
#             a = ant.path[-1]
#             b = ant.path[0]
#             self.graph.pheromones[a][b] += delta
