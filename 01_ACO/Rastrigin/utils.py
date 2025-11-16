import matplotlib.pyplot as plt

def plot_convergence(history):
    plt.figure(figsize=(8, 4))
    plt.plot(history)
    plt.title("ACOR Convergence Curve")
    plt.xlabel("Iteration")
    plt.ylabel("Best-so-far f(x)")
    plt.grid(True)
    plt.show()
