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
#  READ DATASET FILE
# ================================================================
def load_datasets(filename="datasets.txt"):
    datasets = []
    lines = []

    with open(filename, "r") as f:
        lines = [line.strip() for line in f if line.strip() and not line.startswith("#")]

    i = 0
    while i < len(lines):
        parts = lines[i].split()

        # -------- Sphere test case --------
        if parts[0].upper() == "SPHERE":
            dim = int(parts[1])
            datasets.append({"type": "sphere", "dim": dim})
            i += 1

        # -------- Knapsack test case --------py
        elif parts[0].upper() == "KNAPSACK":
            dim = int(parts[1])

            values = list(map(int, lines[i + 1].split()[1:]))
            weights = list(map(int, lines[i + 2].split()[1:]))

            cap_parts = lines[i + 3].split()
            capacity = int(cap_parts[1])

            datasets.append({
                "type": "knapsack",
                "dim": dim,
                "values": np.array(values),
                "weights": np.array(weights),
                "capacity": capacity
            })

            i += 4

        else:
            i += 1

    return datasets

# ================================================================
#  RUN ALL TEST CASES FROM FILE
# ================================================================
def run_test_cases():

    print("\n==============================")
    print("   RUNNING DATASET EXAMPLES")
    print("==============================\n")

    datasets = load_datasets("datasets.txt")

    for idx, data in enumerate(datasets, start=1):
        print(f"\n===== TEST CASE #{idx} =====")

        # ====================================================
        #  SPHERE TEST CASE
        # ====================================================
        if data["type"] == "sphere":

            dim = data["dim"]
            print(f"→ Sphere function | dim = {dim}")

            def sphere(x):
                return np.sum(np.square(x))

            fa = FireflyAlgorithm(
                objective_fn=sphere,
                dim=dim,
                n_fireflies=30,
                max_gen=100,
                alpha=0.5,
                beta0=1.0,
                gamma=0.01,
                lb=-10,
                ub=10,
                problem_type="continuous",
                seed=42
            )

            best_pos, best_val, curve = fa.optimize(verbose=True)
            print("Best fitness:", best_val)
            print("Best solution:", best_pos[:10])

        # ====================================================
        #  KNAPSACK TEST CASE
        # ====================================================
        elif data["type"] == "knapsack":

            values = data["values"]
            weights = data["weights"]
            capacity = data["capacity"]
            dim = data["dim"]

            print(f"→ Knapsack | dim = {dim}")

            def knapsack_value(sol):
                sol = np.asarray(sol).astype(int)
                total_weight = np.sum(sol * weights)
                return 0 if total_weight > capacity else int(np.sum(sol * values))

            fa = FireflyAlgorithm(
                objective_fn=knapsack_value,
                dim=dim,
                n_fireflies=30,
                max_gen=100,
                alpha=0.3,
                beta0=1.0,
                gamma=1.0,
                problem_type="discrete",
                seed=7
            )

            best_sol, best_val, curve = fa.optimize(verbose=True)
            print("Best fitness:", best_val)
            print("Best solution:", best_sol)

        print("\n----------------------------------------")



# ================================================================
#  MAIN MENU
# ================================================================
def main():

    print("\n==============================")
    print("        FIRELY ALGORITHM")
    print("==============================")
    print("1. Run original examples (continuous + knapsack)")
    print("2. Run parameter sensitivity analysis")
    print("3. Run test cases")
    print("==============================")

    choice = input("Choose option (1/2/3): ").strip()

    if choice == "1":
        run_examples()

    elif choice == "2":
        sensitivity_analysis()

    elif choice == "3":
        run_test_cases()

    else:
        print("Invalid choice.")


# ================================================================
#  ENTRY POINT
# ================================================================
if __name__ == "__main__":
    main()















