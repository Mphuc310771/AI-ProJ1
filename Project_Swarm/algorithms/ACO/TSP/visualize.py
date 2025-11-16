import tkinter as tk
from tkinter import ttk, messagebox
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import threading
import time
from aco import *


class ACOVisualizer:
    def __init__(self, master):
        self.master = master
        self.master.title("ACO TSP Visualizer - Ant Colony Optimization for Traveling Salesman Problem")
        self.master.geometry("1600x800")
        
        # Variables
        self.num_cities_var = tk.IntVar(value=20)
        self.num_ants_var = tk.IntVar(value=30)
        self.iterations_var = tk.IntVar(value=100)
        self.animation_speed_var = tk.IntVar(value=300)
        
        self.is_running = False
        self.is_paused = False
        self.cities = None
        self.aco = None
        self.best_path = None
        self.best_distance = np.inf
        self.run_history = None
        self.current_iteration = 0
        
        # Create UI
        self.create_left_panel()
        self.create_right_panel()
        
    def create_left_panel(self):
        """Tạo panel bên trái cho nhập liệu"""
        self.left_frame = ttk.Frame(self.master, width=350)
        self.left_frame.pack(side=tk.LEFT, fill=tk.BOTH, padx=15, pady=15)
        
        # Title
        title_label = ttk.Label(
            self.left_frame, 
            text="ACO TSP Solver", 
            font=("Arial", 16, "bold")
        )
        title_label.pack(pady=15)
        
        # Separator
        ttk.Separator(self.left_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=10)
        
        # Number of Ants
        ttk.Label(self.left_frame, text="Số lượng Kiến:", font=("Arial", 11, "bold")).pack(anchor=tk.W, pady=(10, 2))
        ants_frame = ttk.Frame(self.left_frame)
        ants_frame.pack(fill=tk.X, pady=5)
        ants_entry = ttk.Entry(ants_frame, textvariable=self.num_ants_var, width=8)
        ants_entry.pack(side=tk.LEFT, padx=5)
        ants_scale = ttk.Scale(
            ants_frame, 
            from_=5, 
            to=100, 
            orient=tk.HORIZONTAL,
            variable=self.num_ants_var,
            command=lambda v: self.num_ants_var.set(int(float(v))) if self.num_ants_var.get() != int(float(v)) else None
        )
        ants_scale.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        
        # Number of Cities
        ttk.Label(self.left_frame, text="Số lượng Thành phố:", font=("Arial", 11, "bold")).pack(anchor=tk.W, pady=(10, 2))
        cities_frame = ttk.Frame(self.left_frame)
        cities_frame.pack(fill=tk.X, pady=5)
        cities_entry = ttk.Entry(cities_frame, textvariable=self.num_cities_var, width=8)
        cities_entry.pack(side=tk.LEFT, padx=5)
        cities_scale = ttk.Scale(
            cities_frame, 
            from_=5, 
            to=50, 
            orient=tk.HORIZONTAL,
            variable=self.num_cities_var,
            command=lambda v: self.num_cities_var.set(int(float(v))) if self.num_cities_var.get() != int(float(v)) else None
        )
        cities_scale.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        
        # Number of Iterations
        ttk.Label(self.left_frame, text="Số lượng Iterations:", font=("Arial", 11, "bold")).pack(anchor=tk.W, pady=(10, 2))
        iter_frame = ttk.Frame(self.left_frame)
        iter_frame.pack(fill=tk.X, pady=5)
        iter_entry = ttk.Entry(iter_frame, textvariable=self.iterations_var, width=8)
        iter_entry.pack(side=tk.LEFT, padx=5)
        iter_scale = ttk.Scale(
            iter_frame, 
            from_=10, 
            to=300, 
            orient=tk.HORIZONTAL,
            variable=self.iterations_var,
            command=lambda v: self.iterations_var.set(int(float(v))) if self.iterations_var.get() != int(float(v)) else None
        )
        iter_scale.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        
        # Animation Speed
        ttk.Label(self.left_frame, text="Tốc độ Animation (ms):", font=("Arial", 11, "bold")).pack(anchor=tk.W, pady=(10, 2))
        speed_frame = ttk.Frame(self.left_frame)
        speed_frame.pack(fill=tk.X, pady=5)
        speed_entry = ttk.Entry(speed_frame, textvariable=self.animation_speed_var, width=8)
        speed_entry.pack(side=tk.LEFT, padx=5)
        speed_scale = ttk.Scale(
            speed_frame, 
            from_=50, 
            to=2000, 
            orient=tk.HORIZONTAL,
            variable=self.animation_speed_var,
            command=lambda v: self.animation_speed_var.set(int(float(v))) if self.animation_speed_var.get() != int(float(v)) else None
        )
        speed_scale.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        
        # Separator
        ttk.Separator(self.left_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=15)
        
        # Buttons
        button_frame = ttk.Frame(self.left_frame)
        button_frame.pack(fill=tk.X, pady=10)
        
        self.start_button = ttk.Button(
            button_frame, 
            text="▶ Bắt đầu", 
            command=self.start_algorithm
        )
        self.start_button.pack(fill=tk.X, pady=3)
        
        self.pause_button = ttk.Button(
            button_frame, 
            text="⏸ Tạm dừng", 
            command=self.pause_algorithm,
            state=tk.DISABLED
        )
        self.pause_button.pack(fill=tk.X, pady=3)
        
        self.resume_button = ttk.Button(
            button_frame, 
            text="⏯ Tiếp tục", 
            command=self.resume_algorithm,
            state=tk.DISABLED
        )
        self.resume_button.pack(fill=tk.X, pady=3)
        
        self.reset_button = ttk.Button(
            button_frame, 
            text="🔄 Đặt lại", 
            command=self.reset_algorithm
        )
        self.reset_button.pack(fill=tk.X, pady=3)
        
        # Separator
        ttk.Separator(self.left_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=15)
        
        # Statistics
        ttk.Label(self.left_frame, text="Thông tin chi tiết:", font=("Arial", 11, "bold")).pack(anchor=tk.W)
        
        self.info_frame = ttk.Frame(self.left_frame)
        self.info_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        scrollbar = ttk.Scrollbar(self.info_frame, orient=tk.VERTICAL)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.info_text = tk.Text(
            self.info_frame, 
            height=20, 
            width=40, 
            font=("Courier", 9),
            yscrollcommand=scrollbar.set
        )
        self.info_text.pack(fill=tk.BOTH, expand=True)
        scrollbar.config(command=self.info_text.yview)
        
        self.update_info_panel()
        
    def create_right_panel(self):
        """Tạo panel bên phải cho hiển thị"""
        self.right_frame = ttk.Frame(self.master)
        self.right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=15, pady=15)
        
        # Canvas cho matplotlib
        self.fig = plt.Figure(figsize=(10, 8), dpi=100)
        self.ax = self.fig.add_subplot(111)
        
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.right_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Control frame
        control_frame = ttk.Frame(self.right_frame)
        control_frame.pack(fill=tk.X, pady=10)
        
        ttk.Label(control_frame, text="Iteration:", font=("Arial", 10, "bold")).pack(side=tk.LEFT, padx=10)
        self.iteration_label = ttk.Label(control_frame, text="0/0", font=("Arial", 11, "bold"), foreground="blue")
        self.iteration_label.pack(side=tk.LEFT, padx=5)
        
        ttk.Label(control_frame, text="Best Distance:", font=("Arial", 10, "bold")).pack(side=tk.LEFT, padx=(20, 0))
        self.distance_label = ttk.Label(control_frame, text="N/A", font=("Arial", 11, "bold"), foreground="red")
        self.distance_label.pack(side=tk.LEFT, padx=5)
        
        # Progress bar
        self.progress = ttk.Progressbar(self.right_frame, orient=tk.HORIZONTAL, mode='determinate')
        self.progress.pack(fill=tk.X, pady=5)
        
        # Draw initial canvas
        self.draw_initial_canvas()
        
    def update_info_panel(self):
        """Cập nhật bảng thông tin"""
        self.info_text.config(state=tk.NORMAL)
        self.info_text.delete(1.0, tk.END)
        
        info = f"""
━━━ Tham số cài đặt ━━━
Kiến: {self.num_ants_var.get()}
Thành phố: {self.num_cities_var.get()}
Iterations: {self.iterations_var.get()}
Tốc độ: {self.animation_speed_var.get()} ms

━━━ ACO Parameters ━━━
Alpha (α): 1.0
Beta (β): 2.0
Q0: 0.9
Rho (ρ): 0.1
Phi (φ): 0.1

━━━ Giải thích thuật toán ━━━

▸ Kiến (Ants):
  Các agent di chuyển ngẫu 
  nhiên qua các thành phố

▸ Pheromone:
  Hóa chất để ghi dấu vết
  Con đường tốt → nhiều 
  pheromone

▸ Heuristic (β):
  Ưu tiên đường ngắn hơn

▸ Mục tiêu:
  Tìm đường đi qua tất cả
  thành phố, về điểm đầu,
  với tổng quãng đường tối
  thiểu

━━━ Chú thích hình ━━━
● Đỏ: Thành phố
━ Xanh: Đường tốt nhất
░ Xanh nhạt: Pheromone
  (Đậm = nhiều pheromone)
        """
        
        self.info_text.insert(tk.END, info.strip())
        self.info_text.config(state=tk.DISABLED)
        
    def draw_initial_canvas(self):
        """Vẽ canvas ban đầu"""
        self.ax.clear()
        self.ax.set_title("Chờ bắt đầu...", fontsize=14, fontweight='bold')
        self.ax.set_xlabel("X (km)", fontsize=10)
        self.ax.set_ylabel("Y (km)", fontsize=10)
        self.ax.grid(True, alpha=0.3, linestyle='--')
        self.fig.tight_layout()
        self.canvas.draw()
        
    def start_algorithm(self):
        """Bắt đầu thuật toán"""
        if self.is_running:
            return
        
        # Validate
        if self.num_ants_var.get() < 5 or self.num_cities_var.get() < 5 or self.iterations_var.get() < 1:
            messagebox.showerror("Lỗi", "Các tham số không hợp lệ!")
            return
        
        self.is_running = True
        self.is_paused = False
        self.current_iteration = 0
        
        self.start_button.config(state=tk.DISABLED)
        self.pause_button.config(state=tk.NORMAL)
        self.reset_button.config(state=tk.DISABLED)
        
        thread = threading.Thread(target=self.run_aco, daemon=True)
        thread.start()
        
    def pause_algorithm(self):
        """Tạm dừng thuật toán"""
        self.is_paused = True
        self.pause_button.config(state=tk.DISABLED)
        self.resume_button.config(state=tk.NORMAL)
        
    def resume_algorithm(self):
        """Tiếp tục thuật toán"""
        self.is_paused = False
        self.resume_button.config(state=tk.DISABLED)
        self.pause_button.config(state=tk.NORMAL)
        
    def reset_algorithm(self):
        """Đặt lại thuật toán"""
        self.is_running = False
        self.is_paused = False
        self.current_iteration = 0
        self.best_distance = np.inf
        
        self.start_button.config(state=tk.NORMAL)
        self.pause_button.config(state=tk.DISABLED)
        self.resume_button.config(state=tk.DISABLED)
        self.reset_button.config(state=tk.NORMAL)
        
        self.iteration_label.config(text="0/0")
        self.distance_label.config(text="N/A")
        self.progress.config(value=0)
        
        self.draw_initial_canvas()
        self.update_info_panel()
        
    def run_aco(self):
        """Chạy thuật toán ACO"""
        try:
            # Generate cities
            self.cities = np.random.uniform(0, 100, (self.num_cities_var.get(), 2))
            
            # Calculate distances
            distances = self.calculate_distances(self.cities)
            
            # Initialize ACO
            graph = Graph(distances)
            self.aco = ACO(
                graph=graph,
                num_ants=self.num_ants_var.get(),
                num_iterations=self.iterations_var.get(),
                alpha=1.0,
                beta=2.0,
                q0=0.9,
                rho=0.1,
                phi=0.1
            )
            
            # Run ACO
            best_path, best_distance, history, self.run_history = self.aco.run()
            
            # Animate iterations
            for iteration in range(len(self.run_history)):
                # Check pause
                while self.is_paused and self.is_running:
                    time.sleep(0.1)
                
                if not self.is_running:
                    break
                
                self.current_iteration = iteration
                snapshot = self.run_history[iteration]
                
                # Update visualization
                self.master.after(0, lambda it=iteration, snap=snapshot: 
                                 self.visualize_iteration(it, snap))
                
                # Wait for animation speed
                time.sleep(self.animation_speed_var.get() / 1000.0)
            
            if self.is_running:
                self.is_running = False
                self.start_button.config(state=tk.NORMAL)
                self.pause_button.config(state=tk.DISABLED)
                self.resume_button.config(state=tk.DISABLED)
                self.reset_button.config(state=tk.NORMAL)
                messagebox.showinfo("Hoàn tất", f"Thuật toán hoàn tất!\n\nĐường đi tốt nhất: {best_distance:.2f}")
                
        except Exception as e:
            messagebox.showerror("Lỗi", f"Lỗi: {str(e)}")
            self.is_running = False
            self.start_button.config(state=tk.NORMAL)
            self.pause_button.config(state=tk.DISABLED)
            
    def visualize_iteration(self, iteration, snapshot):
        """Visualize một iteration"""
        if not self.is_running:
            return

        # Clear only the dynamic axes content (we redraw base every time)
        self.ax.clear()

        # Draw all cities
        self.ax.scatter(
            self.cities[:, 0], 
            self.cities[:, 1], 
            c='red', 
            s=150, 
            marker='o',
            zorder=3,
            edgecolors='darkred',
            linewidth=1.5,
            label='Thành phố'
        )
        
        # Add city indices
        for i, city in enumerate(self.cities):
            self.ax.annotate(
                str(i), 
                (city[0], city[1]), 
                xytext=(3, 3), 
                textcoords='offset points',
                fontsize=7,
                fontweight='bold'
            )
        
        # Draw pheromone intensity
        if snapshot['pheromones'] is not None:
            pheromones = snapshot['pheromones'].copy()
            np.fill_diagonal(pheromones, 0)
            
            max_pheromone = np.max(pheromones)
            if max_pheromone > 0:
                # Draw edges with pheromone intensity
                for i in range(len(self.cities)):
                    for j in range(i + 1, len(self.cities)):
                        phe = pheromones[i, j]
                        if phe > max_pheromone * 0.01:
                            intensity = phe / max_pheromone
                            alpha = 0.15 + intensity * 0.35
                            width = 0.5 + intensity * 1.5
                            
                            self.ax.plot(
                                [self.cities[i, 0], self.cities[j, 0]],
                                [self.cities[i, 1], self.cities[j, 1]],
                                color=plt.cm.Blues(0.3 + intensity * 0.6),
                                linewidth=width,
                                alpha=alpha,
                                zorder=1
                            )
        # Prepare best path and animate changes from previous iteration
        best_path = snapshot['best_path']
        best_distance = snapshot['best_distance']

        # Compute edge sets (undirected) for diffing
        current_edges = self.edges_from_path(best_path) if best_path is not None else set()
        prev_edges = getattr(self, 'prev_best_edges', set())

        added_edges = current_edges - prev_edges
        removed_edges = prev_edges - current_edges

        # Store for next iteration
        self.prev_best_edges = current_edges

        # Draw a faint previous best path if exists (so user sees transition)
        if prev_edges:
            prev_path_coords = []
            # try to reconstruct an ordered prev path for nicer visuals when available
            try:
                prev_path = list(snapshot.get('best_path', best_path)) if snapshot.get('best_path') is not None else best_path
                prev_path_coords = [self.cities[c] for c in prev_path]
                prev_x = [p[0] for p in prev_path_coords]
                prev_y = [p[1] for p in prev_path_coords]
                self.ax.plot(prev_x, prev_y, color='gray', linewidth=1.2, alpha=0.35, zorder=1)
            except Exception:
                pass

        # Draw final best path lightly (will be emphasized by animation)
        if best_path is not None:
            path_x = [self.cities[city, 0] for city in best_path]
            path_y = [self.cities[city, 1] for city in best_path]
            self.ax.plot(path_x, path_y, color='green', linewidth=2.5, alpha=0.35, zorder=2)

        # Kick off animated transition highlighting added/removed edges
        self.animate_path_change(added_edges, removed_edges, best_path, best_distance)
        
        # Set labels and title
        self.ax.set_title(
            f'Iteration {iteration + 1}/{self.iterations_var.get()} | '
            f'Kiến: {self.num_ants_var.get()}, Thành phố: {self.num_cities_var.get()}',
            fontsize=12,
            fontweight='bold'
        )
        self.ax.set_xlabel("X (km)", fontsize=10)
        self.ax.set_ylabel("Y (km)", fontsize=10)
        self.ax.grid(True, alpha=0.3, linestyle='--')
        self.ax.legend(loc='upper right', fontsize=9, framealpha=0.9)
        self.ax.set_xlim(-5, 105)
        self.ax.set_ylim(-5, 105)
        
        # Update info labels
        self.iteration_label.config(text=f"{iteration + 1}/{self.iterations_var.get()}")
        self.distance_label.config(text=f"{best_distance:.2f}")
        self.progress.config(value=(iteration + 1) / self.iterations_var.get() * 100)
        
        # Draw
        self.fig.tight_layout()
        self.canvas.draw()

    def edges_from_path(self, path):
        """Return a set of undirected edges (frozenset pairs) from a path list."""
        edges = set()
        if not path:
            return edges
        for i in range(len(path) - 1):
            a, b = path[i], path[i + 1]
            edges.add(frozenset((int(a), int(b))))
        return edges

    def animate_path_change(self, added_edges, removed_edges, final_path, final_distance):
        """Animate transition between previous and current best path.

        - added_edges: set of frozenset edge pairs newly added
        - removed_edges: set of frozenset edge pairs removed
        - final_path: ordered list of node indices for final path
        """
        # Parameters for animation
        duration_ms = max(120, min(800, self.animation_speed_var.get()))
        frame_interval = 60  # ms per frame
        frames = max(3, int(duration_ms / frame_interval))

        # Convert edges to ordered pairs for drawing
        def edge_coords(edge):
            a, b = tuple(edge)
            return (self.cities[a, 0], self.cities[a, 1]), (self.cities[b, 0], self.cities[b, 1])

        added_list = list(added_edges)
        removed_list = list(removed_edges)

        # Animation state
        step = {'frame': 0}

        def draw_frame():
            # Redraw base (cities + pheromones already drawn by caller), so overlay only
            # We'll redraw the whole axes base to ensure consistent visuals
            self.ax.collections.clear()
            self.ax.lines = []
            self.ax.patches = []

            # Redraw cities
            self.ax.scatter(
                self.cities[:, 0], 
                self.cities[:, 1], 
                c='red', 
                s=150, 
                marker='o',
                zorder=4,
                edgecolors='darkred',
                linewidth=1.5,
            )
            for i, city in enumerate(self.cities):
                self.ax.annotate(str(i), (city[0], city[1]), xytext=(3, 3), textcoords='offset points', fontsize=7, fontweight='bold')

            # Draw pheromones lightly from the last snapshot (if available)
            # We won't redraw full pheromone network here to keep animation responsive.

            f = step['frame'] / float(frames)

            # Draw removed edges fading out (red -> vanish)
            for edge in removed_list:
                (x1, y1), (x2, y2) = edge_coords(edge)
                alpha = max(0.0, 0.8 * (1.0 - f))
                width = 2.0 * (1.0 - f) + 0.5
                self.ax.plot([x1, x2], [y1, y2], color='red', linewidth=width, alpha=alpha, zorder=2)

            # Draw added edges pulsing in (transparent -> green)
            for edge in added_list:
                (x1, y1), (x2, y2) = edge_coords(edge)
                alpha = min(0.95, 0.2 + 0.9 * f)
                width = 0.5 + 2.5 * f
                self.ax.plot([x1, x2], [y1, y2], color='green', linewidth=width, alpha=alpha, zorder=3)

            # Draw final path semi-transparent (overlay gradually becomes fully opaque)
            if final_path is not None:
                px = [self.cities[int(c), 0] for c in final_path]
                py = [self.cities[int(c), 1] for c in final_path]
                self.ax.plot(px, py, color='green', linewidth=2.5, alpha=0.25 + 0.65 * f, zorder=5)

            # Title and labels refresh
            self.ax.set_title(
                f'Iteration {self.current_iteration + 1}/{self.iterations_var.get()} | '
                f'Kiến: {self.num_ants_var.get()}, Thành phố: {self.num_cities_var.get()}',
                fontsize=12,
                fontweight='bold'
            )
            self.ax.set_xlim(-5, 105)
            self.ax.set_ylim(-5, 105)

            self.fig.tight_layout()
            self.canvas.draw()

            step['frame'] += 1
            if step['frame'] <= frames:
                self.master.after(frame_interval, draw_frame)
            else:
                # Finalize: draw the final path clearly
                if final_path is not None:
                    px = [self.cities[int(c), 0] for c in final_path]
                    py = [self.cities[int(c), 1] for c in final_path]
                    self.ax.plot(px, py, color='green', linewidth=3.0, alpha=0.95, zorder=6)
                    # draw arrows for direction
                    for i in range(len(final_path) - 1):
                        a = int(final_path[i])
                        b = int(final_path[i + 1])
                        start = self.cities[a]
                        end = self.cities[b]
                        dx = end[0] - start[0]
                        dy = end[1] - start[1]
                        self.ax.arrow(start[0] + dx * 0.7, start[1] + dy * 0.7, dx * 0.2, dy * 0.2,
                                      head_width=2, head_length=1.5, fc='green', ec='green', alpha=0.9, zorder=7)
                # Update labels one last time
                self.iteration_label.config(text=f"{self.current_iteration + 1}/{self.iterations_var.get()}")
                self.distance_label.config(text=f"{final_distance:.2f}")
                self.progress.config(value=(self.current_iteration + 1) / self.iterations_var.get() * 100)

        
    @staticmethod
    def calculate_distances(cities):
        """Tính khoảng cách Euclidean giữa các thành phố"""
        n = len(cities)
        distances = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if i != j:
                    distances[i, j] = np.sqrt(
                        (cities[i, 0] - cities[j, 0])**2 + 
                        (cities[i, 1] - cities[j, 1])**2
                    )
        return distances


if __name__ == "__main__":
    root = tk.Tk()
    app = ACOVisualizer(root)
    root.mainloop()
