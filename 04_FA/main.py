from firefly_algorithm import FireflyAlgorithm
import numpy as np
import matplotlib.pyplot as plt


# ================================================================
#  EXAMPLE CODE (Continuous + Discrete)
# ================================================================
def run_examples():
    # --- Continuous Example: Sphere Function ---
    def sphere(x):
        return np.sum(np.square(x))

    fa_cont = FireflyAlgorithm(
        objective_fn=sphere,
        dim=10,
        n_fireflies=30,
        max_gen=100,
        alpha=0.5,
        beta0=1.0,
        gamma=0.01,
        lb=-10,
        ub=10,
        alpha_decay=0.97,
        problem_type="continuous",
        seed=42
    )

    best_pos, best_val, curve = fa_cont.optimize(verbose=True)
    print("\n✅ Continuous Optimization Result:")
    print("Best fitness:", best_val)
    print("Best position:", best_pos[:10], "\n")

    # --- Discrete Example: Knapsack Problem ---
    values = np.random.randint(10, 200, 15)
    weights = np.random.randint(5, 40, 15)
    capacity = 100
    dim = len(values)  # =15

    def knapsack_value(sol):
        sol = np.asarray(sol).astype(int)  # đảm bảo là numpy array 0/1
        if sol.shape[0] != dim:
            raise ValueError(f"Dimension mismatch: sol has length {sol.shape[0]}, expected {dim}")
        total_weight = np.sum(sol * weights)
        if total_weight > capacity:
            return 0
        return int(np.sum(sol * values))

    # Khi tạo FA: dùng dim tương ứng
    fa_disc = FireflyAlgorithm(
        objective_fn=knapsack_value,
        dim=dim,
        n_fireflies=30,
        max_gen=100,
        alpha=0.3, beta0=1.0, gamma=1.0,
        alpha_decay=0.97,
        problem_type="discrete",
        seed=7
    )

    best_sol, best_val2, curve2 = fa_disc.optimize(verbose=True)
    print("🔥 Discrete Optimization Result:")
    print("Best value:", best_val2)
    print("Best solution:", best_sol, "\n")

    # --- Plot convergence ---
    fig, ax1 = plt.subplots(figsize=(8, 5))

    # Trục y1: Continuous
    color = 'tab:blue'
    ax1.set_xlabel('Generation')
    ax1.set_ylabel('Continuous - Sphere', color=color)
    ax1.plot(curve, color=color)
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid(True)

    # Trục y2: Discrete
    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    color = 'tab:red'
    ax2.set_ylabel('Discrete - Knapsack', color=color)
    ax2.plot(curve2, color=color)
    ax2.tick_params(axis='y', labelcolor=color)

    plt.title("Firefly Algorithm - Convergence Curve")
    plt.show()



# ================================================================
#  PARAMETER SENSITIVITY ANALYSIS
# ================================================================
def sensitivity_analysis():
    print("\n==============================")
    print("  PARAMETER SENSITIVITY TEST")
    print("==============================\n")

    def sphere(x):
        return np.sum(np.square(x))

    # Three values for each parameter
    alpha_values = [0.2, 0.5, 0.8]
    beta_values = [0.5, 1.0, 1.5]
    gamma_values = [0.001, 0.01, 0.1]

    test_id = 1

    # ====================================================
    #  VÒNG FOR THEO GAMMA — MỖI GAMMA SINH RA 1 FIGURE
    # ====================================================
    for gamma in gamma_values:

        print("\n=======================================")
        print(f"   RUNNING GROUP FOR gamma = {gamma}")
        print("=======================================\n")

        curves = []   # list of (curve, label)

        for alpha in alpha_values:
            for beta0 in beta_values:

                print(f"\n===== TEST #{test_id} =====")
                print(f"alpha={alpha}, beta0={beta0}, gamma={gamma}")

                fa = FireflyAlgorithm(
                    objective_fn=sphere,
                    dim=10,
                    n_fireflies=30,
                    max_gen=100,
                    alpha=alpha,
                    beta0=beta0,
                    gamma=gamma,
                    lb=-10,
                    ub=10,
                    problem_type="continuous",
                    seed=42
                )

                best_pos, best_fitness, curve = fa.optimize(verbose=False)

                print(f"Best fitness = {best_fitness:.6f}")
                print("Best position:", best_pos[:10])

                label = f"a={alpha}, b={beta0}, g={gamma}"
                curves.append((curve, label))

                test_id += 1

        # ====================================================
        #  TẠO FIGURE CHO TỪNG GIÁ TRỊ GAMMA
        # ====================================================
        plt.figure(figsize=(10, 6))
        for curve, label in curves:
            plt.plot(curve, label=label)

        plt.title(f"Sensitivity Analysis - gamma = {gamma}")
        plt.xlabel("Generation")
        plt.ylabel("Best Fitness")
        plt.grid(True)
        plt.legend(fontsize=8)
        plt.tight_layout()
        plt.show(block=False)

    plt.show()




# ================================================================
#  MAIN MENU
# ================================================================
def main():

    print("\n==============================")
    print("        FIRELY ALGORITHM")
    print("==============================")
    print("1. Run original examples (continuous + knapsack)")
    print("2. Run parameter sensitivity analysis")
    print("==============================")

    choice = input("Choose option (1/2): ").strip()

    if choice == "1":
        run_examples()

    elif choice == "2":
        sensitivity_analysis()

    else:
        print("Invalid choice.")


# ================================================================
#  ENTRY POINT
# ================================================================
if __name__ == "__main__":
    main()















