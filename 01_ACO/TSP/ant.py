import numpy as np

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


# import numpy as np

# class Ant:
#     def __init__(self, graph, alpha, beta):
#         self.graph = graph
#         self.alpha = alpha
#         self.beta = beta

#         self.current_node = np.random.randint(graph.num_nodes)
#         self.path = [self.current_node]
#         self.total_distance = 0

#         self.unvisited_nodes = set(range(graph.num_nodes))
#         self.unvisited_nodes.remove(self.current_node)

#     def select_next_node(self):
#         unvisited_list = list(self.unvisited_nodes)

#         pheromone = self.graph.pheromones[self.current_node][unvisited_list]
#         heuristic = self.graph.heuristic[self.current_node][unvisited_list]

#         tau = pheromone ** self.alpha
#         eta = heuristic ** self.beta
#         desirability = tau * eta

#         probabilities = desirability / np.sum(desirability)
#         return np.random.choice(unvisited_list, p=probabilities)

#     def move(self):
#         next_node = self.select_next_node()

#         self.total_distance += self.graph.distances[self.current_node][next_node]
#         self.path.append(next_node)
#         self.unvisited_nodes.remove(next_node)
#         self.current_node = next_node

#     def complete_path(self):
#         while self.unvisited_nodes:
#             self.move()

#         start_node = self.path[0]
#         self.total_distance += self.graph.distances[self.current_node][start_node]
