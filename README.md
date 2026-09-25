# 🎯 Smart Face Attendance Pro v2.1

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-4.12-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)
![CustomTkinter](https://img.shields.io/badge/CustomTkinter-6.0-blue?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)
![Status](https://img.shields.io/badge/Status-Active-brightgreen?style=for-the-badge)

**Hệ Thống Điểm Danh Khuôn Mặt Thông Minh**  
*Nhận diện khuôn mặt theo thời gian thực · Giao diện Dashboard hiện đại · Chống trùng lặp thông minh*

</div>

---

## 📸 Giao diện

| Màn hình Điểm Danh | Màn hình Huấn Luyện AI |
|---|---|
| ![Dashboard](https://img.shields.io/badge/Dashboard-Live_Attendance-1F3864?style=flat-square) | ![Training](https://img.shields.io/badge/Training-LBPH_Trainer-2E75B6?style=flat-square) |
| Camera giám sát trực tiếp · HUD nhận diện · KPI cards | Huấn luyện AI · Log console · Quét trùng lặp |

---

## ✨ Tính năng nổi bật

- 🎥 **Điểm danh trực tiếp** – Nhận diện khuôn mặt qua webcam theo thời gian thực, hiển thị HUD với tên tiếng Việt có dấu
- 👤 **Đăng ký nhân sự** – Wizard đăng ký người mới, tự động chụp 50 ảnh mẫu với thanh tiến trình
- 🤖 **Huấn luyện AI** – Huấn luyện mô hình LBPH trên Background Thread, không làm đơ giao diện
- 📊 **Lịch sử & Báo cáo** – Xem, lọc, tìm kiếm và xuất báo cáo ra Excel / CSV
- 👥 **Quản lý nhân sự** – CRUD đầy đủ: thêm, sửa tên, xóa nhân sự
- 🔍 **Chống trùng lặp thông minh** – Tự động phát hiện và gộp ID khi cùng một người được đăng ký nhiều lần
- 🌙 **Dark / Light Mode** – Chuyển đổi giao diện tối/sáng, lưu cài đặt tự động
- 🔔 **Âm thanh thông báo** – Beep chào mừng khi điểm danh thành công
- ⚙️ **Cài đặt linh hoạt** – Ngưỡng nhận diện, cooldown, camera index đều tùy chỉnh được

---

## 🏗️ Kiến trúc hệ thống

```
┌─────────────────────────────────────────────────┐
│          TẦNG GIAO DIỆN (CustomTkinter)         │
│  Dashboard · Forms · Treeview · Camera Feed     │
├─────────────────────────────────────────────────┤
│       TẦNG LOGIC NGHIỆP VỤ (Python)             │
│  Haar Cascade · LBPH Recognizer · Duplicate     │
│  Detector · Attendance Recorder · Camera Loop   │
├─────────────────────────────────────────────────┤
│          TẦNG DỮ LIỆU (File-based)              │
│  config.py · attendance.csv · trainer.yml       │
│  dataset/ · settings.json                       │
└─────────────────────────────────────────────────┘
```

---

## 📁 Cấu trúc thư mục

```
Face-Attendance-Project/
│
├── 📄 main_app.py              # Ứng dụng chính (v2.1 Pro - 1900+ dòng)
├── 📄 config.py                # Danh sách nhân sự {ID: "Tên"}
├── 📄 settings.json            # Cấu hình camera, ngưỡng, âm thanh
├── 📄 requirements.txt         # Danh sách thư viện cần cài
│
├── 🚀 run.bat                  # Khởi động nhanh (1 click)
├── 🔧 install.bat              # Cài đặt thư viện tự động
│
├── 📄 1_collect_data.py        # Script thu thập ảnh mẫu (standalone)
├── 📄 2_train_model.py         # Script huấn luyện mô hình (standalone)
├── 📄 3_attendance_app.py      # App điểm danh đơn giản (standalone)
│
├── 📂 dataset/                 # Ảnh mẫu khuôn mặt (User.<ID>.<N>.jpg)
│   ├── User.1.1.jpg
│   ├── User.1.2.jpg
│   └── ...
│
└── 📂 trainer/
    └── trainer.yml             # Mô hình LBPH đã huấn luyện
```

---

## ⚙️ Cài đặt

### Yêu cầu hệ thống
- Python **3.10+**
- Webcam (USB hoặc built-in)
- Windows 10/11 (khuyến nghị)

### Cách 1 – Cài tự động (khuyến nghị)

```bash
# Double-click file install.bat
# Hoặc chạy trong terminal:
install.bat
```

### Cách 2 – Cài thủ công

```bash
pip install -r requirements.txt
```

### Các thư viện cần thiết

| Thư viện | Phiên bản | Mục đích |
|----------|-----------|----------|
| `opencv-contrib-python` | 4.12.0 | Phát hiện & nhận diện khuôn mặt (LBPH) |
| `customtkinter` | 6.0.0 | Giao diện Dashboard hiện đại |
| `Pillow` | 12.3.0 | Vẽ HUD + tiếng Việt trên camera |
| `pandas` | 2.3.2 | Quản lý dữ liệu điểm danh |
| `numpy` | 2.2.6 | Xử lý ma trận ảnh |
| `openpyxl` | 3.1.5 | Xuất báo cáo Excel |

---

## 🚀 Sử dụng

### Khởi động nhanh

```bash
# Double-click run.bat
# Hoặc:
python main_app.py
```

### Quy trình sử dụng lần đầu

```
1. Mở app → vào tab "Đăng Ký Người Mới"
   └── Nhập ID và Họ Tên → Bật camera → Bắt đầu chụp (50 ảnh)

2. Vào tab "Huấn Luyện AI"
   └── Nhấn "Bắt Đầu Huấn Luyện" → Chờ hoàn tất

3. Vào tab "Điểm Danh Trực Tiếp"
   └── Bật camera → Hệ thống tự nhận diện và ghi nhận
```

---

## 🧠 Thuật toán

### LBPH (Local Binary Patterns Histograms)

```
Ảnh khuôn mặt
    ↓
Chia thành các ô nhỏ
    ↓
Mỗi pixel so sánh với 8 láng giềng → Mã LBP 8-bit
    ↓
Tính Histogram từng ô → Nối thành Vector đặc trưng
    ↓
So sánh khoảng cách với mô hình đã lưu
    ↓
Confidence < 70 → Nhận diện thành công ✅
```

### Chống trùng lặp khuôn mặt

```
Khi đăng ký:
  Khuôn mặt mới → Template Matching với dataset
  Tương đồng ≥ 70% → Cảnh báo, từ chối tạo ID mới

Khi điểm danh:
  Bảng ánh xạ {ID_trùng → ID_gốc}
  Nhận diện ra ID trùng → Tự động quy về ID gốc
```

---

## 📊 Hiệu năng

| Tiêu chí | Kết quả |
|----------|---------|
| Tốc độ phát hiện khuôn mặt | ~18–25 ms/frame |
| Tốc độ nhận diện LBPH | ~5–15 ms/lần |
| Độ chính xác (điều kiện tốt) | > 90% |
| Thời gian huấn luyện (50 ảnh) | < 10 giây |
| Thời gian khởi động app | < 3 giây |

---

## 🛠️ Cấu hình

Chỉnh sửa trong tab **Cài Đặt Hệ Thống** hoặc trực tiếp trong `settings.json`:

```json
{
  "camera_index": 0,
  "confidence_threshold": 70,
  "cooldown_seconds": 60,
  "sound_enabled": true,
  "appearance_mode": "Light"
}
```

| Tham số | Mặc định | Mô tả |
|---------|----------|-------|
| `camera_index` | `0` | Cổng webcam (0, 1, 2...) |
| `confidence_threshold` | `70` | Ngưỡng nhận diện (thấp hơn = chặt hơn) |
| `cooldown_seconds` | `60` | Thời gian chờ giữa 2 lần điểm danh |
| `sound_enabled` | `true` | Âm thanh thông báo điểm danh |

---

## ⚠️ Lưu ý

- **Ánh sáng:** Đảm bảo khuôn mặt được chiếu sáng đều, tránh ngược sáng
- **Góc chụp:** Khi đăng ký, nhẹ nhàng xoay đầu để thu nhiều góc độ khác nhau
- **Số ảnh mẫu:** Tối thiểu 30 ảnh/người, khuyến nghị 50 ảnh để đạt độ chính xác tốt
- **Huấn luyện lại:** Sau khi thêm/xóa nhân sự, nhớ huấn luyện lại mô hình AI

---

## 🔮 Hướng phát triển

- [ ] Nâng cấp lên FaceNet / ArcFace (Deep Learning) để tăng độ chính xác
- [ ] Tích hợp cơ sở dữ liệu SQL thay thế file CSV
- [ ] Phát triển phiên bản Web (Flask/FastAPI)
- [ ] Phát triển Mobile App (Flutter)
- [ ] Thêm tính năng nhận dạng khuôn mặt đeo khẩu trang
- [ ] Gửi báo cáo tự động qua Email/Zalo

---

## 👥 Nhóm phát triển

| Thành viên | Phụ trách |
|------------|-----------|
| Thành viên 1 (Nhóm trưởng) | Kiến trúc hệ thống · Camera · Điểm danh · HUD |
| Thành viên 2 | Thuật toán LBPH · Huấn luyện AI · Chống trùng lặp |
| Thành viên 3 | Giao diện Dashboard · Lịch sử · Báo cáo · Testing |

---

## 📄 License

Distributed under the MIT License. See `LICENSE` for more information.

---

<div align="center">

Made with ❤️ using Python · OpenCV · CustomTkinter

⭐ **Nếu project hữu ích, hãy cho một Star nhé!** ⭐

</div>
