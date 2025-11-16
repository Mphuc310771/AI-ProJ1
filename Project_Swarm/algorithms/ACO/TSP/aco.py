import numpy as np
import matplotlib.pyplot as plt

class Graph:
    def __init__(self, distances):
        self.distances = distances
        self.num_nodes = len(distances)
        self.pheromones = np.ones_like(distances, dtype=float)

        heuristics = np.zeros_like(distances, dtype=float)
        heuristics[distances > 0] = 1.0 / distances[distances > 0]
        self.heuristic = heuristics

class Ant:
    def __init__(self, graph, alpha, beta, q0, phi, tau0):
        self.graph = graph

        # Parameters
        self.alpha = alpha
        self.beta = beta
        self.q0 = q0
        self.phi = phi
        self.tau0 = tau0

        self.current_node = np.random.randint(graph.num_nodes)
        self.path = [self.current_node]
        self.total_distance = 0.0
        self.unvisited_nodes = set(range(graph.num_nodes)) - {self.current_node}

    def select_next_node(self):
        unvisited_list = list(self.unvisited_nodes)

        pher = self.graph.pheromones[self.current_node][unvisited_list]
        heur = self.graph.heuristic[self.current_node][unvisited_list]

        q = np.random.rand()

        if q <= self.q0:
            desirability = pher * (heur ** self.beta)
            next_node = unvisited_list[np.argmax(desirability)]

        else:
            desirability = (pher ** self.alpha) * (heur ** self.beta)
            probabilities = desirability / np.sum(desirability)
            next_node = np.random.choice(unvisited_list, p=probabilities)

        return next_node

    def move(self):
        if not self.unvisited_nodes:
            return

        next_node = self.select_next_node()
        prev = self.current_node

        new_tau = (1 - self.phi) * self.graph.pheromones[prev][next_node] + self.phi * self.tau0
        self.graph.pheromones[prev][next_node] = new_tau
        self.graph.pheromones[next_node][prev] = new_tau   # ACS dùng đồ thị vô hướng

        self.total_distance += self.graph.distances[prev][next_node]

        self.current_node = next_node
        self.unvisited_nodes.remove(next_node)
        self.path.append(next_node)

    def complete_path(self):
        while self.unvisited_nodes:
            self.move()

        start = self.path[0]
        last = self.path[-1]
        self.total_distance += self.graph.distances[last][start]




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