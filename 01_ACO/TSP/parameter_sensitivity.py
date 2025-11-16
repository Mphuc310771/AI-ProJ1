import numpy as np
import matplotlib.pyplot as plt
from ant import Ant
from graph import Graph
from aco import ACO

distance_file = r"dantzig42_d.txt"

distance_matrix = np.loadtxt(distance_file)

num_cities = distance_matrix.shape[0]

graph = Graph(distance_matrix)

def test_param(param_name, values):
    """
    Chạy ACO với 1 tham số thay đổi.
    Không in ra màn hình, chỉ trả về kết quả.
    """
    best_distances = []
    histories = []

    for v in values:
        # thiết lập tham số mặc định
        kwargs = {
            "num_ants": 50,
            "num_iterations": 400,
            "alpha": 1,
            "beta": 2,
            "q0": 0.9,
            "rho": 0.1,
            "phi": 0.1
        }
        kwargs[param_name] = v

        aco_test = ACO(graph, **kwargs)
        _, best_dist, hist, _ = aco_test.run()

        best_distances.append(best_dist)
        histories.append(hist)

    return best_distances, histories


# =========================================================
# 6 bộ giá trị cần phân tích
# =========================================================
param_config = {
    "alpha": [0.5, 1, 1.5, 2, 3],
    "beta": [1, 2, 3, 4, 5],
    "rho": [0.01, 0.05, 0.1, 0.2, 0.3],
    "phi": [0, 0.05, 0.1, 0.2, 0.3],
    "q0": [0.5, 0.7, 0.9, 1.0],
    "num_ants": [10, 30, 50, 70, 100]
}

best_params = {}  # lưu tham số tối ưu
sensitivity_results = {}  # để vẽ sau

for param_name, values in param_config.items():
    best_list, histories = test_param(param_name, values)

    sensitivity_results[param_name] = (values, best_list)

    best_index = np.argmin(best_list)
    best_params[param_name] = values[best_index]

fig, axes = plt.subplots(3, 2, figsize=(14, 12))
axes = axes.ravel()

param_names_ordered = ["alpha", "beta", "rho", "phi", "q0", "num_ants"]

for i, pname in enumerate(param_names_ordered):
    values, best_distances = sensitivity_results[pname]

    axes[i].plot(values, best_distances, marker='o')
    axes[i].set_title(f"Sensitivity of {pname}")
    axes[i].set_xlabel(pname)
    axes[i].set_ylabel("Best Distance")
    axes[i].grid(True)

plt.tight_layout()
plt.show()


print("\n=== CHẠY ACO VỚI THAM SỐ TỐI ƯU ===")
print(best_params)

aco_final = ACO(
    graph,
    num_ants=best_params["num_ants"],
    num_iterations=500,
    alpha=best_params["alpha"],
    beta=best_params["beta"],
    q0=best_params["q0"],
    rho=best_params["rho"],
    phi=best_params["phi"]
)

best_path_opt, best_dist_opt, hist_opt, _ = aco_final.run()

print("\nBest distance (tối ưu):", best_dist_opt)
print("Best path:", best_path_opt)

# plot convergence
plt.figure(figsize=(10, 5))
plt.plot(hist_opt)
plt.title("Convergence với tham số tối ưu")
plt.xlabel("Iteration")
plt.ylabel("Distance")
plt.grid(True)
plt.show()
