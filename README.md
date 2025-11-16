# README — AI-ProJ1 

**Tiêu đề dự án:** AI-ProJ1 — Các thuật toán Swarm Intelligence (ACO, PSO, ABC, FA, CS)

## 1. Nội dung nộp
- **Báo cáo PDF:** `AI 23CNTN - Project 1.pdf` (file báo cáo chính).  
- **Mã nguồn:** thư mục `AI-ProJ1/` chứa toàn bộ code triển khai các thuật toán.  
- **Dependencies:** `requirements.txt`.  
- **README nộp:** file này (tóm tắt hướng dẫn chạy và nội dung nộp).

## 2. Thông tin nhóm
- Thành viên (mã SV — Họ tên)  
  1. 23120098 — Hoàng Kim Trí  
  2. 23120156 — Đỗ Trần Minh Phúc   
  3. 23120161 — Hồ Chí Quốc 
  4. 23120167 — Nguyễn Gia Thịnh 

## 3. Mục tiêu ngắn gọn
Triển khai, đánh giá và so sánh các thuật toán Swarm Intelligence (ACO, PSO, ABC, FA, CS) trên các bài toán chuẩn (ví dụ: Rastrigin, TSP). Báo cáo mô tả phương pháp, thí nghiệm, kết quả và kết luận.

## 4. Hướng dẫn nhanh để chạy và tái tạo kết quả
1. Tạo môi trường ảo và cài phụ thuộc:
```bash
python -m venv .venv
# Linux / macOS
source .venv/bin/activate
# Windows (PowerShell)
.venv\Scripts\Activate.ps1

pip install -r requirements.txt
```
2. Các lệnh chạy mẫu (từ thư mục gốc `AI-ProJ1/`):
```bash
python 02_PSO/PSO.py            
python 02_PSO/PSO_3D.py         
python 03_ABC/ABC.py           
python 03_ABC/ABC_3D.py        
python 04_FA/firefly_algorithm.py
python 05_CS/cuckoo_search.py
```
3. Kết quả, hình và logs thường được lưu trong các thư mục `results/` hoặc `visualizations/` bên trong từng module.

**Lưu ý tái tạo:** để tái tạo kết quả, đặt seed trong code (nếu có) hoặc chạy với cùng tham số; ghi rõ tham số dùng trong báo cáo.

## 5. Nội dung file chính
- `01_ACO/` — triển khai ACO (TSP, hàm Rastrigin).  
- `02_PSO/` — Particle Swarm Optimization, bao gồm demo 3D.  
- `03_ABC/` — Artificial Bee Colony, demo 3D.  
- `04_FA/` — Firefly Algorithm.  
- `05_CS/` — Cuckoo Search, cùng scripts hỗ trợ và benchmarks.  
- `requirements.txt` — danh sách thư viện cần cài.  


## 6. Ghi chú 
- Mọi thuật toán lõi được triển khai bằng Python (sử dụng NumPy cho toán tử ma trận).  
- Kết quả, biểu đồ và phân tích chi tiết nằm trong file báo cáo PDF.  
