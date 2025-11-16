import numpy as np
import matplotlib.pyplot as plt
from aco import *

distance_file = "..\\..\\..\\data\\discrete\\dantzig42_d.txt"

distance_matrix = np.loadtxt(distance_file)

num_cities = distance_matrix.shape[0]

print(f"Số thành phố: {num_cities}")
print(f"Kích thước distance matrix: {distance_matrix.shape}")

graph = Graph(distance_matrix)

aco = ACO(
    graph,
    num_ants=20,
    num_iterations=100,
    alpha=1,
    beta=2,
    q0=0.9,
    rho=0.1,
    phi=0.1
)

best_path, best_distance, history, run_history = aco.run()


print("\n===== KẾT QUẢ ACO =====")
print("Best distance (ACO):", best_distance)
print("Best path:", best_path)

plt.figure(figsize=(10, 6))
plt.plot(history, label="Best distance theo iteration")
plt.xlabel("Iteration")
plt.ylabel("Distance")
plt.title(f"Biểu đồ hội tụ ACO cho {num_cities} thành phố (Dantzig42)")
plt.legend()
plt.grid(True)
plt.show()
