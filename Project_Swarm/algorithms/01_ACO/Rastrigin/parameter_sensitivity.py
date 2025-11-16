import numpy as np
import matplotlib.pyplot as plt
from acor import ACOR
from utils import plot_convergence

def run_once(dim=2, K=15, ants=30, xi=0.85, iterations=100):
    """Chạy ACOR một lần và trả về best value."""
    acor = ACOR(dim=dim, K=K, ants=ants, xi=xi, iterations=iterations)
    best = acor.optimize()
    return best.f, acor.best_history


def sensitivity_analysis():
    default_params = {
        "dim": 2,
        "K": 15,
        "ants": 30,
        "xi": 0.85,
        "iterations": 100
    }

    param_ranges = {
        "K": [5, 10, 15, 20, 30],
        "ants": [10, 20, 30, 40, 50],
        "xi": [0.5, 0.7, 0.85, 0.95],
        "iterations": [50, 100, 150, 200]
    }

    results = {}

    for param, values in param_ranges.items():
        print(f"\n=== Phân tích tham số {param} ===")
        results[param] = []

        for v in values:
            # tạo bản sao để chỉnh
            params = default_params.copy()
            params[param] = v

            best_value, history = run_once(
                dim=params["dim"],
                K=params["K"],
                ants=params["ants"],
                xi=params["xi"],
                iterations=params["iterations"]
            )

            print(f"{param} = {v}, best = {best_value:.6f}")
            results[param].append((v, best_value, history))

        print(f"--- Kết thúc phân tích {param} ---")

    return results


def visualize_results(results):
    """
    Hiển thị tất cả biểu đồ hội tụ của tất cả tham số
    trong cùng một cửa sổ (figure).
    Mỗi tham số = 1 subplot.
    """

    num_params = len(results)
    plt.figure(figsize=(12, 3 * num_params))

    for idx, (param, runs) in enumerate(results.items(), 1):
        plt.subplot(num_params, 1, idx)

        for value, best, history in runs:
            plt.plot(history, label=f"{param}={value}")

        plt.title(f"Convergence curves for parameter: {param}")
        plt.xlabel("Iteration")
        plt.ylabel("Best-so-far value")
        plt.grid(True)
        plt.legend()

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    results = sensitivity_analysis()
    visualize_results(results)
