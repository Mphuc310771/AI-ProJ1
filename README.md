# AI-ProJ1 — Swarm Intelligence & Traditional Algorithms

**Môn học:** Nhập môn Trí tuệ nhân tạo (AI)
**Chủ đề:** Triển khai, đánh giá và so sánh các thuật toán Swarm Intelligence (ACO, PSO, ABC, FA, CS) cùng các thuật toán tìm kiếm truyền thống.

---

## Mục lục
1. Thông tin nhóm
2. Giới thiệu dự án
3. Thuật toán đã triển khai
4. Cấu trúc thư mục
5. Yêu cầu & Cài đặt
6. Hướng dẫn chạy (Usage)
7. Thực nghiệm & Tái tạo kết quả
8. Kết quả & Báo cáo
9. Đóng góp
10. License & Liên hệ

---

## 1. Thông tin nhóm

| STT | MSSV | Họ và Tên | Vai trò |
|:---:|:---:|:---|:---|
| 1 | 23120098 | Hoàng Kim Trí | Thành viên |
| 2 | 23120156 | Đỗ Trần Minh Phúc | Thành viên |
| 3 | 23120161 | Hồ Chí Quốc | Thành viên |
| 4 | 23120167 | Nguyễn Gia Thịnh | Thành viên |

---

## 2. Giới thiệu dự án

Dự án này tập trung giải quyết hai lớp bài toán tối ưu hóa phổ biến:

1. **Continuous Optimization (Tối ưu liên tục):** Tìm cực trị hàm số — benchmark chính sử dụng hàm **Rastrigin**.
2. **Discrete Optimization (Tối ưu rời rạc):** Bài toán Người du lịch (Traveling Salesman Problem - TSP).

Mục tiêu: triển khai, tinh chỉnh và so sánh hiệu năng giữa các thuật toán Swarm Intelligence và các phương pháp tìm kiếm truyền thống trên hai bài toán mẫu.

---

## 3. Thuật toán đã triển khai

**Swarm Intelligence (Trí tuệ bầy đàn):**
- Ant Colony Optimization (ACO)
- Particle Swarm Optimization (PSO)
- Artificial Bee Colony (ABC)
- Firefly Algorithm (FA)
- Cuckoo Search (CS)

**Traditional Algorithms (Thuật toán truyền thống):**
- Breadth-First Search (BFS)
- Depth-First Search (DFS)
- Hill Climbing

---

## 4. Cấu trúc thư mục

Mã nguồn chính nằm trong thư mục `Project_Swarm` với cấu trúc chính như sau:

```text
AI-ProJ1/
└── Project_Swarm/
    ├── algorithms/                 # Mã nguồn các thuật toán
    │   ├── ABC/                    # Artificial Bee Colony (+ 3D demo)
    │   ├── ACO/                    # Ant Colony (Rastrigin & TSP)
    │   ├── BFS/                    # Breadth-First Search
    │   ├── CS/                     # Cuckoo Search
    │   ├── DFS/                    # Depth-First Search
    │   ├── FA/                     # Firefly Algorithm
    │   ├── HillClimbing/           # Hill Climbing
    │   └── PSO/                    # Particle Swarm Optimization
    ├── data/                       # Dữ liệu đầu vào (Benchmark TSP)
    │   ├── continuous/
    │   └── discrete/               # File .txt (vd: dantzig42_d.txt)
    ├── experiments/                # Các kịch bản chạy so sánh (Benchmarks)
    │   ├── compare_swarm_continuous.py
    │   ├── compare_swarm_discrete.py
    │   └── compare_with_traditional.py
    ├── utils/                      # Thư viện hỗ trợ (đồ thị, visualization)
    ├── requirements.txt            # Danh sách thư viện cần cài đặt
    └── Makefile                    # (Tùy chọn)
```

> Các kết quả chạy (hình ảnh, logs) sẽ được lưu trong `results/` hoặc `visualizations/` bên trong folder của từng thuật toán.

---

## 5. Yêu cầu & Cài đặt

**Yêu cầu hệ thống**: Python 3.8+

**Bước 1 — Di chuyển vào thư mục mã nguồn chính**

```bash
cd Project_Swarm
```

**Bước 2 — Tạo môi trường ảo (khuyến nghị)**

```bash
# Tạo môi trường
python -m venv .venv

# Kích hoạt môi trường:
# - Trên Windows (PowerShell):
.venv\Scripts\Activate.ps1
# - Trên Linux/macOS:
source .venv/bin/activate
```

**Bước 3 — Cài đặt phụ thuộc**

```bash
pip install -r requirements.txt
```

---

## 6. Hướng dẫn chạy (Usage)

**Lưu ý:** Tất cả các lệnh dưới đây đều chạy từ thư mục gốc `Project_Swarm/`.

### A. Chạy demo từng thuật toán


1. **Ant Colony Optimization (ACO)**

```bash
# Bài toán TSP
python algorithms/ACO/TSP/main.py
# Bài toán hàm Rastrigin
python algorithms/ACO/Rastrigin/main.py
```

2. **Particle Swarm Optimization (PSO)**

```bash
python algorithms/PSO/PSO.py            
python algorithms/PSO/PSO_3D.py         
```

3. **Artificial Bee Colony (ABC)**

```bash
python algorithms/ABC/ABC.py
python algorithms/ABC/ABC_3D.py
```

4. **Firefly Algorithm (FA)**

```bash
python algorithms/FA/main.py
python algorithms/FA/run_3d.py
```

5. **Cuckoo Search (CS)**

```bash
python algorithms/CS/cuckoo_search.py
python algorithms/CS/cuckoo_visualization.py
```



### B. Chạy thực nghiệm & so sánh (Experiments)

Để tái tạo các kết quả so sánh trong báo cáo, chạy các script sau:

```bash
# So sánh các thuật toán Swarm (Continuous - Rastrigin)
python experiments/compare_swarm_continuous.py

# So sánh các thuật toán Swarm (Discrete - TSP)
python experiments/compare_swarm_discrete.py

# So sánh Swarm vs Traditional (BFS, DFS, HillClimbing)
python experiments/compare_with_traditional.py
```

> Mỗi script thực nghiệm có thể nhận các tham số đầu vào (số lần chạy, kích thước quần thể, số thế hệ, v.v.). Kiểm tra phần header của từng file `.py` để biết cách cấu hình chi tiết.

---

## 7. Thực nghiệm & Tái tạo kết quả

- Kết quả (đường cong hội tụ, bảng thống kê, file log) sẽ được ghi ra thư mục `results/` hoặc `visualizations/` trong từng module thuật toán.
- Để tái tạo đầy đủ các kết quả trong báo cáo, đảm bảo sử dụng cùng tham số thử nghiệm như đã mô tả trong file `experiments/*`.


---

## 8. Đóng góp

Các đóng góp (fix bugs, cải thiện visualization, thêm thuật toán hoặc benchmark mới) đều hoan nghênh.

Quy trình đóng góp:
1. Fork repository
2. Tạo branch mới cho tính năng / sửa lỗi: `git checkout -b feature/<tên>`
3. Commit và push
4. Tạo Pull Request mô tả thay đổi

---

## 9. License & Liên hệ

- **Repository:** https://github.com/Mphuc310771/AI-ProJ1
- License: (nếu cần, thêm file `LICENSE` — mặc định chưa kèm license)


