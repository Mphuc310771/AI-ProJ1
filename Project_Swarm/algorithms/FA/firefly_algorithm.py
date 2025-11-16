import numpy as np


class FireflyAlgorithm:
    """
    Firefly Algorithm (FA)
    ----------------------
    Phiên bản tổng quát hỗ trợ:
        - Continuous Optimization (Sphere)
        - Discrete Optimization (Knapsack)

    Attributes
    ----------
    problem_type : str
        'continuous' hoặc 'discrete'
    objective_fn : callable
        Hàm mục tiêu cần tối ưu hóa
    dim : int
        Số chiều không gian tìm kiếm
    n_fireflies : int
        Số lượng đom đóm (population size)
    max_gen : int
        Số thế hệ (iterations)
    alpha : float
        Mức độ ngẫu nhiên (randomness)
    beta0 : float
        Độ hấp dẫn ban đầu
    gamma : float
        Hệ số hấp thụ ánh sáng
    lb, ub : float
        Giới hạn dưới/trên (chỉ áp dụng cho bài toán liên tục)
    alpha_decay : float
        Hệ số giảm alpha theo thời gian
    seed : int | None
        Seed để tái lập kết quả (reproducibility)
    """

    def __init__(self, objective_fn, dim, n_fireflies=30, max_gen=100,
                 alpha=0.5, beta0=1.0, gamma=0.01,
                 lb=-10, ub=10, alpha_decay=0.97,
                 problem_type="continuous", seed=None):

        self.objective_fn = objective_fn
        self.dim = dim
        self.n_fireflies = n_fireflies
        self.max_gen = max_gen
        self.alpha = alpha
        self.beta0 = beta0
        self.gamma = gamma
        self.lb = lb
        self.ub = ub
        self.alpha_decay = alpha_decay
        self.problem_type = problem_type
        self.seed = seed

        if seed is not None:
            np.random.seed(seed)

    # -------------------------------------------------------------------------
    # 1. Initialize population
    # -------------------------------------------------------------------------
    def initialize_population(self):
        if self.problem_type == "continuous":
            return np.random.uniform(self.lb, self.ub, (self.n_fireflies, self.dim))
        elif self.problem_type == "discrete":
            return np.random.randint(0, 2, (self.n_fireflies, self.dim))
        else:
            raise ValueError("problem_type phải là 'continuous' hoặc 'discrete'.")

    # -------------------------------------------------------------------------
    # 2. Movement rules
    # -------------------------------------------------------------------------
    def move(self, xi, xj):
        """Cách di chuyển giữa hai đom đóm tùy theo loại problem."""
        if self.problem_type == "continuous":
            # khoảng cách Euclidean
            r = np.linalg.norm(xi - xj)
            beta = self.beta0 * np.exp(-self.gamma * r ** 2)
            random_step = self.alpha * (np.random.rand(self.dim) - 0.5) * (self.ub - self.lb)
            new_pos = xi + beta * (xj - xi) + random_step
            return np.clip(new_pos, self.lb, self.ub)

        elif self.problem_type == "discrete":
            # khoảng cách Hamming
            r = np.sum(xi != xj)
            beta = self.beta0 * np.exp(-self.gamma * r ** 2)
            # copy bit từ j sang i
            mask = np.random.rand(self.dim) < beta
            new_solution = np.where(mask, xj, xi)
            # random flip
            flip_mask = np.random.rand(self.dim) < self.alpha
            new_solution = np.logical_xor(new_solution, flip_mask).astype(int)
            return new_solution

    # -------------------------------------------------------------------------
    # 3. Optimize process
    # -------------------------------------------------------------------------
    def optimize(self, verbose=False):
        """Chạy quá trình tối ưu hóa và trả về kết quả."""
        fireflies = self.initialize_population()
        fitness = np.array([self.objective_fn(ff) for ff in fireflies])

        # Trong continuous: minimize; trong discrete (VD knapsack): maximize
        if self.problem_type == "continuous":
            brightness = -fitness
            best_idx = np.argmin(fitness)
            best_fitness = fitness[best_idx]
        else:
            brightness = fitness
            best_idx = np.argmax(fitness)
            best_fitness = fitness[best_idx]

        best_firefly = fireflies[best_idx].copy()
        convergence_curve = [best_fitness]

        for gen in range(self.max_gen):
            for i in range(self.n_fireflies):
                for j in range(self.n_fireflies):
                    if brightness[j] > brightness[i]:
                        new_pos = self.move(fireflies[i], fireflies[j])
                        new_fit = self.objective_fn(new_pos)

                        # Điều kiện tốt hơn tùy loại problem
                        if (self.problem_type == "continuous" and new_fit < fitness[i]) or \
                                (self.problem_type == "discrete" and new_fit > fitness[i]):
                            fireflies[i] = new_pos
                            fitness[i] = new_fit
                            brightness[i] = new_fit if self.problem_type == "discrete" else -new_fit

            # Cập nhật best toàn cục
            if self.problem_type == "continuous":
                gen_best_idx = np.argmin(fitness)
                if fitness[gen_best_idx] < best_fitness:
                    best_fitness = fitness[gen_best_idx]
                    best_firefly = fireflies[gen_best_idx].copy()
            else:
                gen_best_idx = np.argmax(fitness)
                if fitness[gen_best_idx] > best_fitness:
                    best_fitness = fitness[gen_best_idx]
                    best_firefly = fireflies[gen_best_idx].copy()

            # Giảm alpha để hội tụ
            self.alpha *= self.alpha_decay
            convergence_curve.append(best_fitness)

            if verbose and (gen % 10 == 0 or gen == self.max_gen - 1):
                print(f"Gen {gen + 1}/{self.max_gen} | Best fitness: {best_fitness:.6f}")

        return best_firefly, best_fitness, np.array(convergence_curve)







