# -*- coding: utf-8 -*-
"""
=============================================================================
HỆ THỐNG ĐIỂM DANH KHUÔN MẶT THÔNG MINH - SMART FACE ATTENDANCE SYSTEM PRO
Phiên bản: 2.1 Pro (Bổ sung tính năng Kiểm soát & Chống trùng lặp khuôn mặt)
Được phát triển với CustomTkinter & OpenCV
=============================================================================
"""

import os
import sys
import json
import time
import datetime
import threading
from typing import Dict, List, Tuple, Optional, Set

import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageTk, ImageDraw, ImageFont

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import customtkinter as ctk

# Cấu hình encoding chuẩn UTF-8 cho console Windows
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
if hasattr(sys.stderr, "reconfigure"):
    try:
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Import winsound cho Windows để phát âm thanh thông báo
try:
    import winsound
    HAS_WINSOUND = True
except ImportError:
    HAS_WINSOUND = False


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_FILE = os.path.join(BASE_DIR, "config.py")
ATTENDANCE_FILE = os.path.join(BASE_DIR, "attendance.csv")
TRAINER_FILE = os.path.join(BASE_DIR, "trainer.yml")
TRAINER_DIR_FILE = os.path.join(BASE_DIR, "trainer", "trainer.yml")
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
SETTINGS_FILE = os.path.join(BASE_DIR, "settings.json")

# Đảm bảo các thư mục cần thiết tồn tại
os.makedirs(DATASET_DIR, exist_ok=True)
os.makedirs(os.path.join(BASE_DIR, "trainer"), exist_ok=True)

# Cấu hình mặc định
DEFAULT_SETTINGS = {
    "camera_id": 0,
    "confidence_threshold": 70,     # Khoảng cách LBPH (càng thấp càng khớp, < 70 là nhận)
    "cooldown_seconds": 60,         # Giãn cách giữa 2 lần điểm danh cùng 1 người (giây)
    "sound_enabled": True,          # Bật âm báo thành công
    "appearance_mode": "dark",      # dark / light / system
    "color_theme": "blue"           # blue / green / dark-blue
}

# =============================================================================
# QUẢN LÝ DỮ LIỆU & FILE HỆ THỐNG
# =============================================================================
def load_settings() -> dict:
    settings = DEFAULT_SETTINGS.copy()
    if os.path.exists(SETTINGS_FILE):
        try:
            with open(SETTINGS_FILE, "r", encoding="utf-8") as f:
                saved = json.load(f)
                settings.update(saved)
        except Exception as e:
            print(f"[ERROR] Error loading settings.json: {e}")
    return settings

def save_settings(settings: dict):
    try:
        with open(SETTINGS_FILE, "w", encoding="utf-8") as f:
            json.dump(settings, f, indent=4, ensure_ascii=False)
    except Exception as e:
        print(f"[ERROR] Error saving settings.json: {e}")

def load_users() -> Dict[int, str]:
    """Đọc danh sách users từ config.py (duy trì tương thích 100%)"""
    users = {}
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, "r", encoding="utf-8") as f:
                content = f.read()
                local_vars = {}
                exec(content, {}, local_vars)
                users = local_vars.get("users", {})
        except Exception as e:
            print(f"[ERROR] Error reading config.py: {e}")
    return users

def save_users(users: Dict[int, str]):
    """Ghi danh sách users vào config.py"""
    try:
        with open(CONFIG_FILE, "w", encoding="utf-8") as f:
            f.write("# -*- coding: utf-8 -*-\n")
            f.write("users = {\n")
            for uid in sorted(users.keys()):
                name = users[uid].replace('"', '\\"')
                f.write(f'    {uid}: "{name}",\n')
            f.write("}\n")
    except Exception as e:
        print(f"[ERROR] Error saving config.py: {e}")

def load_attendance_df() -> pd.DataFrame:
    """Tải file điểm danh attendance.csv"""
    if os.path.exists(ATTENDANCE_FILE):
        try:
            df = pd.read_csv(ATTENDANCE_FILE, encoding="utf-8")
            if not {"ID", "Name", "Time"}.issubset(df.columns):
                df = pd.DataFrame(columns=["ID", "Name", "Time"])
            return df
        except Exception:
            return pd.DataFrame(columns=["ID", "Name", "Time"])
    return pd.DataFrame(columns=["ID", "Name", "Time"])

def append_attendance_record(user_id: int, name: str, timestamp_str: str):
    """Ghi trực tiếp bản ghi điểm danh mới vào attendance.csv"""
    try:
        new_row = pd.DataFrame([[user_id, name, timestamp_str]], columns=["ID", "Name", "Time"])
        if not os.path.exists(ATTENDANCE_FILE) or os.path.getsize(ATTENDANCE_FILE) == 0:
            new_row.to_csv(ATTENDANCE_FILE, index=False, encoding="utf-8")
        else:
            new_row.to_csv(ATTENDANCE_FILE, mode="a", header=False, index=False, encoding="utf-8")
    except Exception as e:
        print(f"[ERROR] Error writing attendance.csv: {e}")

# =============================================================================
# THUẬT TOÁN SO KHỚP & PHÁT HIỆN TRÙNG LẶP KHUÔN MẶT
# =============================================================================
def match_face_to_existing_dataset(face_gray: np.ndarray, current_users: Dict[int, str],
                                   threshold: float = 0.70) -> Tuple[Optional[int], float]:
    """
    So sánh khuôn mặt hiện tại với các ảnh mẫu trong dataset/ để phát hiện trùng lặp.
    Trả về (matched_uid, similarity_score). Nếu không trùng trả về (None, 0.0).
    """
    if not os.path.exists(DATASET_DIR) or not current_users:
        return None, 0.0

    face_norm = cv2.resize(face_gray, (100, 100))
    best_uid = None
    best_score = 0.0

    for uid in current_users.keys():
        corrs = []
        # Lấy 5 ảnh mẫu đại diện của mỗi người
        for i in range(1, 10, 2):
            fpath = os.path.join(DATASET_DIR, f"User.{uid}.{i}.jpg")
            if os.path.exists(fpath):
                sample = cv2.imread(fpath, cv2.IMREAD_GRAYSCALE)
                if sample is not None:
                    sample_norm = cv2.resize(sample, (100, 100))
                    try:
                        c = cv2.matchTemplate(face_norm, sample_norm, cv2.TM_CCOEFF_NORMED)[0][0]
                        corrs.append(float(c))
                    except Exception:
                        pass
        if corrs:
            max_c = max(corrs)
            avg_c = sum(corrs) / len(corrs)
            combined_score = 0.6 * max_c + 0.4 * avg_c

            if combined_score > best_score and (max_c >= threshold or avg_c >= (threshold - 0.05)):
                best_score = combined_score
                best_uid = uid

    return best_uid, best_score

def detect_all_dataset_duplicates(current_users: Dict[int, str], threshold: float = 0.72) -> List[Tuple[int, int, float]]:
    """
    Quét toàn bộ dataset để tìm các cặp ID có khuôn mặt giống nhau.
    Trả về danh sách: [(id_primary, id_duplicate, similarity_pct), ...]
    """
    if not os.path.exists(DATASET_DIR):
        return []

    files = os.listdir(DATASET_DIR)
    user_samples: Dict[int, List[np.ndarray]] = {}

    for f in files:
        parts = f.split(".")
        if len(parts) >= 3 and parts[0].lower() == "user":
            try:
                uid = int(parts[1])
                if uid in current_users and len(user_samples.get(uid, [])) < 6:
                    if uid not in user_samples:
                        user_samples[uid] = []
                    img = cv2.imread(os.path.join(DATASET_DIR, f), cv2.IMREAD_GRAYSCALE)
                    if img is not None:
                        user_samples[uid].append(cv2.resize(img, (100, 100)))
            except ValueError:
                pass

    duplicates = []
    uids = sorted(list(user_samples.keys()))

    for i in range(len(uids)):
        for j in range(i + 1, len(uids)):
            u1, u2 = uids[i], uids[j]
            imgs1 = user_samples[u1]
            imgs2 = user_samples[u2]
            corrs = []
            for im1 in imgs1:
                for im2 in imgs2:
                    try:
                        c = cv2.matchTemplate(im1, im2, cv2.TM_CCOEFF_NORMED)[0][0]
                        corrs.append(float(c))
                    except Exception:
                        pass
            if corrs:
                max_c = max(corrs)
                avg_c = sum(corrs) / len(corrs)
                score = 0.6 * max_c + 0.4 * avg_c
                if score >= threshold or max_c >= 0.78:
                    duplicates.append((u1, u2, score))

    return duplicates

# =============================================================================
# HELPER: FONT & VẼ HUD THÔNG MINH TRÊN ẢNH
# =============================================================================
def get_system_font(size: int = 14, bold: bool = False) -> ImageFont.FreeTypeFont:
    """Tải font hệ thống có hỗ trợ tiếng Việt Unicode"""
    font_names = ["segoeui.ttf", "arial.ttf", "tahoma.ttf"]
    if bold:
        font_names = ["segouib.ttf", "arialbd.ttf", "tahomabd.ttf"] + font_names

    win_dir = os.environ.get("WINDIR", "C:\\Windows")
    for fname in font_names:
        fpath = os.path.join(win_dir, "Fonts", fname)
        if os.path.exists(fpath):
            try:
                return ImageFont.truetype(fpath, size)
            except Exception:
                pass
    return ImageFont.load_default()

FONT_HEADER = get_system_font(18, bold=True)
FONT_BOLD = get_system_font(15, bold=True)
FONT_NORMAL = get_system_font(13, bold=False)
FONT_SM = get_system_font(11, bold=False)

def play_chime_sound():
    """Phát âm thanh báo thành công trên nền (không lag giao diện)"""
    if HAS_WINSOUND:
        def _beep():
            try:
                winsound.Beep(1400, 100)
                winsound.Beep(1800, 120)
            except Exception:
                pass
        threading.Thread(target=_beep, daemon=True).start()

def draw_hud_box(draw: ImageDraw.Draw, x: int, y: int, w: int, h: int,
                 title: str, subtitle: str, color_rgb: Tuple[int, int, int]):
    """Vẽ khung nhận diện công nghệ cao (Corner brackets + Badge thông tin)"""
    line_len = max(18, int(min(w, h) * 0.22))
    th = 3

    # 4 Góc viền công nghệ
    # Top-Left
    draw.line([(x, y), (x + line_len, y)], fill=color_rgb, width=th)
    draw.line([(x, y), (x, y + line_len)], fill=color_rgb, width=th)
    # Top-Right
    draw.line([(x + w, y), (x + w - line_len, y)], fill=color_rgb, width=th)
    draw.line([(x + w, y), (x + w, y + line_len)], fill=color_rgb, width=th)
    # Bottom-Left
    draw.line([(x, y + h), (x + line_len, y + h)], fill=color_rgb, width=th)
    draw.line([(x, y + h), (x, y + h - line_len)], fill=color_rgb, width=th)
    # Bottom-Right
    draw.line([(x + w, y + h), (x + w - line_len, y + h)], fill=color_rgb, width=th)
    draw.line([(x + w, y + h), (x + w, y + h - line_len)], fill=color_rgb, width=th)

    # Thẻ Badge phía trên khuôn mặt
    badge_h = 38
    badge_y = max(4, y - badge_h - 6)
    badge_w = max(w, 180)
    badge_x = x

    # Nền mờ cho badge
    draw.rectangle([(badge_x, badge_y), (badge_x + badge_w, badge_y + badge_h)],
                   fill=(15, 23, 42, 230), outline=color_rgb, width=1)
    # Dải màu điểm nhấn bên trái
    draw.rectangle([(badge_x, badge_y), (badge_x + 4, badge_y + badge_h)], fill=color_rgb)

    # Chữ hiển thị
    draw.text((badge_x + 10, badge_y + 3), title, font=FONT_BOLD, fill=(255, 255, 255))
    draw.text((badge_x + 10, badge_y + 20), subtitle, font=FONT_SM, fill=color_rgb)


# =============================================================================
# LỚP CHÍNH: ỨNG DỤNG ĐIỂM DANH KHUÔN MẶT PRO
# =============================================================================
class FaceAttendanceApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        # Cài đặt giao diện CustomTkinter
        self.settings = load_settings()
        ctk.set_appearance_mode(self.settings.get("appearance_mode", "dark"))
        ctk.set_default_color_theme(self.settings.get("color_theme", "blue"))

        # Cửa sổ chính
        self.title("SMART ATTENDANCE AI - HỆ THỐNG ĐIỂM DANH KHUÔN MẶT PRO v2.1")
        self.geometry("1260x780")
        self.minsize(1080, 680)

        # Quản lý dữ liệu bộ nhớ
        self.users = load_users()
        self.attendance_df = load_attendance_df()
        self.last_checkin_times: Dict[int, float] = {}  # {uid: time.time()}

        # Bảng ánh xạ gộp các ID trùng lặp khuôn mặt {duplicate_id: primary_id}
        self.duplicate_map: Dict[int, int] = {}
        self._refresh_duplicate_map()

        # Trạng thái mô hình AI
        self.cascade_detector = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        self.recognizer = cv2.face.LBPHFaceRecognizer_create()
        self.model_loaded = False
        self._init_face_recognizer()

        # Quản lý Camera
        self.cap: Optional[cv2.VideoCapture] = None
        self.is_camera_running = False
        self.current_tab = "attendance"
        self.camera_lock = threading.Lock()
        self.last_raw_frame: Optional[np.ndarray] = None

        # Quản lý biến chụp ảnh đăng ký
        self.reg_capturing = False
        self.reg_user_id = 0
        self.reg_user_name = ""
        self.reg_count = 0
        self.reg_max = 50
        self.reg_auto_train = tk.BooleanVar(value=True)

        # Quản lý biến huấn luyện
        self.is_training = False

        # Khởi tạo giao diện
        self._setup_layout()
        self._init_realtime_clock()

        # Bật camera mặc định cho tab điểm danh
        self.after(500, self.start_camera)

        # Xử lý đóng ứng dụng an toàn
        self.protocol("WM_DELETE_WINDOW", self.on_closing)

    def _init_face_recognizer(self):
        """Tải mô hình nhận diện LBPH nếu file tồn tại"""
        target_file = None
        if os.path.exists(TRAINER_FILE):
            target_file = TRAINER_FILE
        elif os.path.exists(TRAINER_DIR_FILE):
            target_file = TRAINER_DIR_FILE

        if target_file:
            try:
                self.recognizer.read(target_file)
                self.model_loaded = True
            except Exception as e:
                print(f"[ERROR] Error loading trainer.yml: {e}")
                self.model_loaded = False
        else:
            self.model_loaded = False

    def _refresh_duplicate_map(self):
        """Quét và xây dựng bảng ánh xạ gộp các ID trùng lặp khuôn mặt"""
        self.duplicate_map.clear()
        dups = detect_all_dataset_duplicates(self.users, threshold=0.72)
        for u1, u2, score in dups:
            # Quy ước ID nhỏ hơn là ID gốc (Primary ID)
            primary = min(u1, u2)
            dup = max(u1, u2)
            self.duplicate_map[dup] = primary
            print(f"[DUPLICATE DETECTION] Mapped duplicate ID {dup} -> Primary ID {primary} (Similarity: {int(score*100)}%)")

    # =========================================================================
    # GIAO DIỆN TỔNG QUAN (SIDEBAR & CONTAINER)
    # =========================================================================
    def _setup_layout(self):
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # -----------------------------
        # 1. SIDEBAR (THANH ĐIỀU HƯỚNG TRÁI)
        # -----------------------------
        self.sidebar_frame = ctk.CTkFrame(self, width=240, corner_radius=0)
        self.sidebar_frame.grid(row=0, column=0, sticky="nsew")
        self.sidebar_frame.grid_rowconfigure(8, weight=1)

        # App Brand & Logo
        brand_frame = ctk.CTkFrame(self.sidebar_frame, fg_color="transparent")
        brand_frame.pack(fill="x", padx=18, pady=(20, 15))

        lbl_app_logo = ctk.CTkLabel(
            brand_frame, text="⚡ SMART FACE AI",
            font=ctk.CTkFont(family="Segoe UI", size=20, weight="bold")
        )
        lbl_app_logo.pack(anchor="w")

        lbl_app_sub = ctk.CTkLabel(
            brand_frame, text="Hệ thống chấm công v2.1 Pro",
            font=ctk.CTkFont(family="Segoe UI", size=12), text_color="gray"
        )
        lbl_app_sub.pack(anchor="w")

        # Phân cách
        sep = ctk.CTkFrame(self.sidebar_frame, height=2, fg_color=("gray75", "gray30"))
        sep.pack(fill="x", padx=16, pady=(0, 15))

        # Menu Buttons
        self.nav_buttons = {}
        menu_items = [
            ("attendance", "📷  Điểm Danh Trực Tiếp", self.switch_to_attendance),
            ("register",   "➕  Đăng Ký Người Mới", self.switch_to_register),
            ("training",   "⚡  Huấn Luyện AI",     self.switch_to_training),
            ("history",    "📊  Lịch Sử & Báo Cáo",  self.switch_to_history),
            ("users",      "👥  Quản Lý Nhân Sự",   self.switch_to_users),
            ("settings",   "⚙️  Cài Đặt Hệ Thống",   self.switch_to_settings)
        ]

        for key, text, cmd in menu_items:
            btn = ctk.CTkButton(
                self.sidebar_frame, text=text, height=42, corner_radius=8,
                anchor="w", font=ctk.CTkFont(family="Segoe UI", size=14, weight="normal"),
                fg_color="transparent", text_color=("gray10", "gray90"),
                hover_color=("gray80", "#2c3b52"), command=cmd
            )
            btn.pack(fill="x", padx=14, pady=4)
            self.nav_buttons[key] = btn

        # Footer Sidebar (Đổi theme + Trạng thái)
        footer_frame = ctk.CTkFrame(self.sidebar_frame, fg_color="transparent")
        footer_frame.pack(side="bottom", fill="x", padx=16, pady=16)

        lbl_theme = ctk.CTkLabel(footer_frame, text="Giao diện hiển thị:", font=ctk.CTkFont(size=12))
        lbl_theme.pack(anchor="w", pady=(0, 4))

        self.theme_menu = ctk.CTkOptionMenu(
            footer_frame, values=["Tối (Dark)", "Sáng (Light)", "Hệ thống (System)"],
            command=self._on_theme_changed, height=32, corner_radius=6
        )
        self.theme_menu.pack(fill="x")
        curr_mode = self.settings.get("appearance_mode", "dark")
        if curr_mode == "light":
            self.theme_menu.set("Sáng (Light)")
        elif curr_mode == "system":
            self.theme_menu.set("Hệ thống (System)")
        else:
            self.theme_menu.set("Tối (Dark)")

        # -----------------------------
        # 2. KHUNG NỘI DUNG CHÍNH (CONTENT CONTAINER)
        # -----------------------------
        self.content_container = ctk.CTkFrame(self, corner_radius=0, fg_color="transparent")
        self.content_container.grid(row=0, column=1, sticky="nsew", padx=16, pady=16)
        self.content_container.grid_columnconfigure(0, weight=1)
        self.content_container.grid_rowconfigure(0, weight=1)

        # Các View (Trang màn hình)
        self.view_attendance = self._create_attendance_view()
        self.view_register = self._create_register_view()
        self.view_training = self._create_training_view()
        self.view_history = self._create_history_view()
        self.view_users = self._create_users_view()
        self.view_settings = self._create_settings_view()

        # Hiển thị trang mặc định
        self._highlight_nav_button("attendance")
        self.view_attendance.grid(row=0, column=0, sticky="nsew")

    def _highlight_nav_button(self, active_key: str):
        for key, btn in self.nav_buttons.items():
            if key == active_key:
                btn.configure(
                    fg_color=("#3b82f6", "#1d4ed8"),
                    text_color="#ffffff",
                    font=ctk.CTkFont(family="Segoe UI", size=14, weight="bold")
                )
            else:
                btn.configure(
                    fg_color="transparent",
                    text_color=("gray10", "gray90"),
                    font=ctk.CTkFont(family="Segoe UI", size=14, weight="normal")
                )

    def _show_view(self, target_view, key: str):
        # Ẩn tất cả view
        for view in [self.view_attendance, self.view_register, self.view_training,
                     self.view_history, self.view_users, self.view_settings]:
            view.grid_forget()

        self.current_tab = key
        self._highlight_nav_button(key)
        target_view.grid(row=0, column=0, sticky="nsew")

    # =========================================================================
    # VIEW 1: ĐIỂM DANH TRỰC TIẾP (ATTENDANCE DASHBOARD)
    # =========================================================================
    def _create_attendance_view(self) -> ctk.CTkFrame:
        view = ctk.CTkFrame(self.content_container, fg_color="transparent")
        view.grid_columnconfigure(0, weight=7)
        view.grid_columnconfigure(1, weight=4)
        view.grid_rowconfigure(1, weight=1)

        # --- A. HÀNG THẺ THỐNG KÊ (KPI METRICS) ---
        stats_frame = ctk.CTkFrame(view, fg_color="transparent")
        stats_frame.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 12))
        for c in range(4):
            stats_frame.grid_columnconfigure(c, weight=1)

        # Card 1: Đồng hồ
        self.card_clock_frame = ctk.CTkFrame(stats_frame, corner_radius=10)
        self.card_clock_frame.grid(row=0, column=0, sticky="ew", padx=(0, 6), pady=2)
        ctk.CTkLabel(self.card_clock_frame, text="🕒 THỜI GIAN HIỆN TẠI", font=ctk.CTkFont(size=11, weight="bold"), text_color="gray").pack(anchor="w", padx=12, pady=(8, 0))
        self.lbl_clock_time = ctk.CTkLabel(self.card_clock_frame, text="00:00:00", font=ctk.CTkFont(size=20, weight="bold"))
        self.lbl_clock_time.pack(anchor="w", padx=12, pady=(0, 0))
        self.lbl_clock_date = ctk.CTkLabel(self.card_clock_frame, text="Đang tải...", font=ctk.CTkFont(size=11), text_color="gray")
        self.lbl_clock_date.pack(anchor="w", padx=12, pady=(0, 8))

        # Card 2: Tổng nhân sự
        self.card_users_frame = ctk.CTkFrame(stats_frame, corner_radius=10)
        self.card_users_frame.grid(row=0, column=1, sticky="ew", padx=6, pady=2)
        ctk.CTkLabel(self.card_users_frame, text="👥 TỔNG NHÂN SỰ", font=ctk.CTkFont(size=11, weight="bold"), text_color="gray").pack(anchor="w", padx=12, pady=(8, 0))
        self.lbl_kpi_total_users = ctk.CTkLabel(self.card_users_frame, text=str(len(self.users)), font=ctk.CTkFont(size=22, weight="bold"), text_color=("#2563eb", "#60a5fa"))
        self.lbl_kpi_total_users.pack(anchor="w", padx=12, pady=(0, 0))
        ctk.CTkLabel(self.card_users_frame, text="Đã đăng ký trong hệ thống", font=ctk.CTkFont(size=11), text_color="gray").pack(anchor="w", padx=12, pady=(0, 8))

        # Card 3: Đã điểm danh hôm nay
        self.card_checkin_frame = ctk.CTkFrame(stats_frame, corner_radius=10)
        self.card_checkin_frame.grid(row=0, column=2, sticky="ew", padx=6, pady=2)
        ctk.CTkLabel(self.card_checkin_frame, text="✅ ĐÃ ĐIỂM DANH HÔM NAY", font=ctk.CTkFont(size=11, weight="bold"), text_color="gray").pack(anchor="w", padx=12, pady=(8, 0))
        self.lbl_kpi_today_count = ctk.CTkLabel(self.card_checkin_frame, text=str(self._get_today_count()), font=ctk.CTkFont(size=22, weight="bold"), text_color=("#16a34a", "#4ade80"))
        self.lbl_kpi_today_count.pack(anchor="w", padx=12, pady=(0, 0))
        ctk.CTkLabel(self.card_checkin_frame, text="Lượt ghi nhận thành công", font=ctk.CTkFont(size=11), text_color="gray").pack(anchor="w", padx=12, pady=(0, 8))

        # Card 4: Trạng thái AI Model
        self.card_model_frame = ctk.CTkFrame(stats_frame, corner_radius=10)
        self.card_model_frame.grid(row=0, column=3, sticky="ew", padx=(6, 0), pady=2)
        ctk.CTkLabel(self.card_model_frame, text="🧠 MÔ HÌNH NHẬN DIỆN", font=ctk.CTkFont(size=11, weight="bold"), text_color="gray").pack(anchor="w", padx=12, pady=(8, 0))
        model_text = "SẴN SÀNG" if self.model_loaded else "CHƯA HUẤN LUYỆN"
        model_color = ("#16a34a", "#4ade80") if self.model_loaded else ("#d97706", "#f59e0b")
        self.lbl_kpi_model_status = ctk.CTkLabel(self.card_model_frame, text=model_text, font=ctk.CTkFont(size=18, weight="bold"), text_color=model_color)
        self.lbl_kpi_model_status.pack(anchor="w", padx=12, pady=(0, 0))
        ctk.CTkLabel(self.card_model_frame, text="Thuật toán LBPH v2.1", font=ctk.CTkFont(size=11), text_color="gray").pack(anchor="w", padx=12, pady=(0, 8))

        # --- B. KHUNG CAMERA TRỰC TIẾP (BÊN TRÁI) ---
        cam_panel = ctk.CTkFrame(view, corner_radius=12)
        cam_panel.grid(row=1, column=0, sticky="nsew", padx=(0, 10), pady=(0, 0))
        cam_panel.grid_rowconfigure(1, weight=1)
        cam_panel.grid_columnconfigure(0, weight=1)

        # Header camera
        cam_header = ctk.CTkFrame(cam_panel, fg_color="transparent")
        cam_header.grid(row=0, column=0, sticky="ew", padx=14, pady=10)
        
        lbl_cam_title = ctk.CTkLabel(cam_header, text="🎥 CAMERA GIÁM SÁT ĐIỂM DANH", font=ctk.CTkFont(size=15, weight="bold"))
        lbl_cam_title.pack(side="left")

        self.lbl_cam_status_badge = ctk.CTkLabel(
            cam_header, text="● TRỰC TIẾP", text_color="#10b981",
            font=ctk.CTkFont(size=12, weight="bold")
        )
        self.lbl_cam_status_badge.pack(side="right")

        # Vùng hiển thị video Camera
        self.video_container = ctk.CTkFrame(cam_panel, corner_radius=8, fg_color=("#1e293b", "#090d16"))
        self.video_container.grid(row=1, column=0, sticky="nsew", padx=12, pady=(0, 8))
        self.video_container.grid_rowconfigure(0, weight=1)
        self.video_container.grid_columnconfigure(0, weight=1)

        self.video_label = ctk.CTkLabel(
            self.video_container, text="Đang khởi tạo camera...",
            font=ctk.CTkFont(size=14), text_color="gray"
        )
        self.video_label.grid(row=0, column=0, sticky="nsew")

        # Thanh điều khiển dưới camera
        cam_controls = ctk.CTkFrame(cam_panel, fg_color="transparent")
        cam_controls.grid(row=2, column=0, sticky="ew", padx=12, pady=(0, 12))

        self.btn_toggle_cam = ctk.CTkButton(
            cam_controls, text="⏸️ Tạm Dừng Camera", width=160, height=36,
            command=self.toggle_camera, corner_radius=8
        )
        self.btn_toggle_cam.pack(side="left")

        # Thông báo nếu có phát hiện gộp ID trùng
        self.lbl_dup_status = ctk.CTkLabel(
            cam_controls, text="", font=ctk.CTkFont(size=12, weight="bold"), text_color="#f59e0b"
        )
        self.lbl_dup_status.pack(side="left", padx=10)
        self._update_dup_status_label()

        self.lbl_cam_info = ctk.CTkLabel(
            cam_controls, text="Độ phân giải: 640x480 | Ngưỡng nhận diện: 70",
            font=ctk.CTkFont(size=12), text_color="gray"
        )
        self.lbl_cam_info.pack(side="right")

        # --- C. BẢNG THÔNG TIN VỪA GHI NHẬN & FEED (BÊN PHẢI) ---
        feed_panel = ctk.CTkFrame(view, corner_radius=12)
        feed_panel.grid(row=1, column=1, sticky="nsew", padx=(6, 0), pady=(0, 0))
        feed_panel.grid_rowconfigure(2, weight=1)
        feed_panel.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(
            feed_panel, text="👤 VỪA GHI NHẬN MỚI NHẤT",
            font=ctk.CTkFont(size=14, weight="bold")
        ).grid(row=0, column=0, sticky="w", padx=14, pady=(12, 6))

        # Thẻ thông tin người vừa được nhận diện
        self.card_recent_user = ctk.CTkFrame(feed_panel, corner_radius=10, fg_color=("#e2e8f0", "#1e293b"))
        self.card_recent_user.grid(row=1, column=0, sticky="ew", padx=12, pady=(0, 12))

        self.lbl_recent_name = ctk.CTkLabel(
            self.card_recent_user, text="Chưa có ai",
            font=ctk.CTkFont(size=18, weight="bold"), text_color=("#1d4ed8", "#93c5fd")
        )
        self.lbl_recent_name.pack(anchor="w", padx=14, pady=(10, 2))

        self.lbl_recent_details = ctk.CTkLabel(
            self.card_recent_user, text="Đưa khuôn mặt vào camera để điểm danh",
            font=ctk.CTkFont(size=12), text_color="gray"
        )
        self.lbl_recent_details.pack(anchor="w", padx=14, pady=(0, 6))

        self.badge_recent_status = ctk.CTkLabel(
            self.card_recent_user, text="● SẴN SÀNG QUÉT",
            font=ctk.CTkFont(size=12, weight="bold"), text_color="#10b981"
        )
        self.badge_recent_status.pack(anchor="w", padx=14, pady=(0, 10))

        # Danh sách các lượt điểm danh gần nhất hôm nay
        ctk.CTkLabel(
            feed_panel, text="📋 LỊCH SỬ HÔM NAY",
            font=ctk.CTkFont(size=14, weight="bold")
        ).grid(row=2, column=0, sticky="nw", padx=14, pady=(4, 6))

        self.feed_scroll_frame = ctk.CTkScrollableFrame(feed_panel, corner_radius=8)
        self.feed_scroll_frame.grid(row=3, column=0, sticky="nsew", padx=12, pady=(0, 10))
        feed_panel.grid_rowconfigure(3, weight=1)

        # Nút chuyển nhanh đến báo cáo
        btn_view_all_history = ctk.CTkButton(
            feed_panel, text="Xem Toàn Bộ Lịch Sử ➡️", height=34, corner_radius=8,
            command=self.switch_to_history, fg_color="transparent",
            border_width=1, text_color=("gray10", "gray90")
        )
        btn_view_all_history.grid(row=4, column=0, sticky="ew", padx=12, pady=(0, 12))

        self._refresh_live_feed()
        return view

    def _update_dup_status_label(self):
        if hasattr(self, "lbl_dup_status"):
            if self.duplicate_map:
                items = [f"ID {dup}→{orig}" for dup, orig in self.duplicate_map.items()]
                self.lbl_dup_status.configure(text=f"⚡ Đã gộp {len(self.duplicate_map)} ID trùng: {', '.join(items)}")
            else:
                self.lbl_dup_status.configure(text="")

    # =========================================================================
    # VIEW 2: ĐĂNG KÝ NGƯỜI MỚI (REGISTRATION WIZARD)
    # =========================================================================
    def _create_register_view(self) -> ctk.CTkFrame:
        view = ctk.CTkFrame(self.content_container, fg_color="transparent")
        view.grid_columnconfigure(0, weight=4)
        view.grid_columnconfigure(1, weight=5)
        view.grid_rowconfigure(0, weight=1)

        # Cột 1: Form nhập thông tin
        form_panel = ctk.CTkFrame(view, corner_radius=12)
        form_panel.grid(row=0, column=0, sticky="nsew", padx=(0, 10), pady=0)

        ctk.CTkLabel(
            form_panel, text="📝 ĐĂNG KÝ NHÂN SỰ MỚI",
            font=ctk.CTkFont(size=18, weight="bold")
        ).pack(anchor="w", padx=18, pady=(18, 12))

        # Mã ID
        ctk.CTkLabel(form_panel, text="Mã ID nhân viên:", font=ctk.CTkFont(size=13, weight="bold")).pack(anchor="w", padx=18, pady=(4, 2))
        id_row = ctk.CTkFrame(form_panel, fg_color="transparent")
        id_row.pack(fill="x", padx=18, pady=(0, 10))

        self.entry_reg_id = ctk.CTkEntry(id_row, placeholder_text="Mã ID (VD: 1, 2, 3...)", height=38)
        self.entry_reg_id.pack(side="left", fill="x", expand=True, padx=(0, 6))

        btn_auto_id = ctk.CTkButton(id_row, text="Tạo ID Tiếp Theo", width=120, height=38, command=self._suggest_next_id)
        btn_auto_id.pack(side="right")

        # Họ và tên
        ctk.CTkLabel(form_panel, text="Họ và tên nhân sự:", font=ctk.CTkFont(size=13, weight="bold")).pack(anchor="w", padx=18, pady=(4, 2))
        self.entry_reg_name = ctk.CTkEntry(form_panel, placeholder_text="Nhập đầy đủ họ và tên tiếng Việt...", height=38)
        self.entry_reg_name.pack(fill="x", padx=18, pady=(0, 14))

        # Hướng dẫn chụp ảnh
        guide_box = ctk.CTkFrame(form_panel, corner_radius=8, fg_color=("#e2e8f0", "#1e293b"))
        guide_box.pack(fill="x", padx=18, pady=(0, 14))
        ctk.CTkLabel(guide_box, text="💡 TÍNH NĂNG CHỐNG TRÙNG LẶP KHUÔN MẶT:", font=ctk.CTkFont(size=12, weight="bold"), text_color="#10b981").pack(anchor="w", padx=12, pady=(10, 4))
        guide_text = (
            "• Hệ thống tự động kiểm tra xem khuôn mặt đã tồn tại chưa.\n"
            "• Nếu khuôn mặt trùng với một ID đã có, hệ thống sẽ cảnh báo.\n"
            "• Giữ thẳng khuôn mặt trong khung ngắm để chụp đủ 50 ảnh mẫu."
        )
        ctk.CTkLabel(guide_box, text=guide_text, justify="left", font=ctk.CTkFont(size=11), text_color="gray").pack(anchor="w", padx=12, pady=(0, 10))

        # Checkbox tự động huấn luyện
        chk_auto = ctk.CTkCheckBox(
            form_panel, text="Tự động huấn luyện AI ngay sau khi chụp xong",
            variable=self.reg_auto_train, font=ctk.CTkFont(size=13)
        )
        chk_auto.pack(anchor="w", padx=18, pady=(0, 16))

        # Tiến trình chụp
        self.lbl_reg_progress = ctk.CTkLabel(
            form_panel, text="Tiến độ: 0 / 50 ảnh (0%)",
            font=ctk.CTkFont(size=13, weight="bold")
        )
        self.lbl_reg_progress.pack(anchor="w", padx=18, pady=(0, 4))

        self.bar_reg_progress = ctk.CTkProgressBar(form_panel, height=14, corner_radius=7)
        self.bar_reg_progress.pack(fill="x", padx=18, pady=(0, 16))
        self.bar_reg_progress.set(0)

        # Nút bắt đầu chụp
        self.btn_start_capture = ctk.CTkButton(
            form_panel, text="📸 BẮT ĐẦU CHỤP 50 ẢNH MẪU", height=44, corner_radius=8,
            font=ctk.CTkFont(size=14, weight="bold"),
            fg_color="#10b981", hover_color="#059669", command=self.start_registration_capture
        )
        self.btn_start_capture.pack(fill="x", padx=18, pady=(0, 8))

        self.btn_cancel_capture = ctk.CTkButton(
            form_panel, text="⏹️ Hủy Bỏ", height=36, corner_radius=8,
            fg_color="transparent", border_width=1, text_color="gray",
            command=self.cancel_registration_capture
        )
        self.btn_cancel_capture.pack(fill="x", padx=18, pady=(0, 10))

        # Cột 2: Camera xem trước khi đăng ký
        cam_reg_panel = ctk.CTkFrame(view, corner_radius=12)
        cam_reg_panel.grid(row=0, column=1, sticky="nsew", padx=(10, 0), pady=0)
        cam_reg_panel.grid_rowconfigure(1, weight=1)
        cam_reg_panel.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(
            cam_reg_panel, text="📷 XEM TRƯỚC CAMERA ĐĂNG KÝ",
            font=ctk.CTkFont(size=15, weight="bold")
        ).grid(row=0, column=0, sticky="w", padx=14, pady=12)

        reg_video_box = ctk.CTkFrame(cam_reg_panel, corner_radius=8, fg_color=("#1e293b", "#090d16"))
        reg_video_box.grid(row=1, column=0, sticky="nsew", padx=12, pady=(0, 12))
        reg_video_box.grid_rowconfigure(0, weight=1)
        reg_video_box.grid_columnconfigure(0, weight=1)

        self.reg_video_label = ctk.CTkLabel(reg_video_box, text="Camera đăng ký sẵn sàng...", font=ctk.CTkFont(size=14), text_color="gray")
        self.reg_video_label.grid(row=0, column=0, sticky="nsew")

        self._suggest_next_id()
        return view

    def _suggest_next_id(self):
        next_id = max(self.users.keys(), default=0) + 1
        self.entry_reg_id.delete(0, "end")
        self.entry_reg_id.insert(0, str(next_id))

    # =========================================================================
    # VIEW 3: HUẤN LUYỆN MÔ HÌNH AI (TRAINING PANEL)
    # =========================================================================
    def _create_training_view(self) -> ctk.CTkFrame:
        view = ctk.CTkFrame(self.content_container, fg_color="transparent")
        view.grid_columnconfigure(0, weight=1)
        view.grid_rowconfigure(2, weight=1)

        # Header
        head = ctk.CTkFrame(view, corner_radius=12)
        head.grid(row=0, column=0, sticky="ew", pady=(0, 12))

        ctk.CTkLabel(
            head, text="⚡ HUẤN LUYỆN MÔ HÌNH NHẬN DIỆN (LBPH TRAINER)",
            font=ctk.CTkFont(size=18, weight="bold")
        ).pack(anchor="w", padx=18, pady=(16, 6))

        desc = (
            "Hệ thống sẽ quét toàn bộ ảnh mẫu trong thư mục 'dataset/', trích xuất đặc trưng khuôn mặt "
            "bằng thuật toán Local Binary Patterns Histograms (LBPH) và lưu vào file 'trainer.yml'."
        )
        ctk.CTkLabel(head, text=desc, font=ctk.CTkFont(size=13), text_color="gray", justify="left").pack(anchor="w", padx=18, pady=(0, 16))

        # Thống kê mẫu dữ liệu
        data_stats = ctk.CTkFrame(view, corner_radius=12)
        data_stats.grid(row=1, column=0, sticky="ew", pady=(0, 12))
        for c in range(3):
            data_stats.grid_columnconfigure(c, weight=1)

        # Box 1: Tổng ảnh mẫu
        b1 = ctk.CTkFrame(data_stats, fg_color="transparent")
        b1.grid(row=0, column=0, padx=14, pady=12, sticky="ew")
        ctk.CTkLabel(b1, text="TỔNG ẢNH MẪU", font=ctk.CTkFont(size=11, weight="bold"), text_color="gray").pack(anchor="w")
        self.lbl_train_img_count = ctk.CTkLabel(b1, text=str(self._count_dataset_images()), font=ctk.CTkFont(size=22, weight="bold"), text_color="#3b82f6")
        self.lbl_train_img_count.pack(anchor="w")

        # Box 2: Tổng nhân sự
        b2 = ctk.CTkFrame(data_stats, fg_color="transparent")
        b2.grid(row=0, column=1, padx=14, pady=12, sticky="ew")
        ctk.CTkLabel(b2, text="SỐ NGƯỜI ĐÃ CÓ DỮ LIỆU", font=ctk.CTkFont(size=11, weight="bold"), text_color="gray").pack(anchor="w")
        self.lbl_train_user_count = ctk.CTkLabel(b2, text=str(len(self.users)), font=ctk.CTkFont(size=22, weight="bold"), text_color="#10b981")
        self.lbl_train_user_count.pack(anchor="w")

        # Box 3: Thời gian cập nhật mô hình
        b3 = ctk.CTkFrame(data_stats, fg_color="transparent")
        b3.grid(row=0, column=2, padx=14, pady=12, sticky="ew")
        ctk.CTkLabel(b3, text="TRẠNG THÁI FILE TRAINER.YML", font=ctk.CTkFont(size=11, weight="bold"), text_color="gray").pack(anchor="w")
        self.lbl_train_model_date = ctk.CTkLabel(b3, text=self._get_model_modified_time(), font=ctk.CTkFont(size=16, weight="bold"), text_color="#f59e0b")
        self.lbl_train_model_date.pack(anchor="w")

        # Khung tiến trình & Log
        body = ctk.CTkFrame(view, corner_radius=12)
        body.grid(row=2, column=0, sticky="nsew", pady=0)
        body.grid_rowconfigure(2, weight=1)
        body.grid_columnconfigure(0, weight=1)

        # Nút bấm bắt đầu huấn luyện & Nút kiểm tra trùng
        action_row = ctk.CTkFrame(body, fg_color="transparent")
        action_row.grid(row=0, column=0, sticky="ew", padx=16, pady=14)

        self.btn_run_train = ctk.CTkButton(
            action_row, text="🚀 BẮT ĐẦU HUẤN LUYỆN MÔ HÌNH", height=44, corner_radius=8,
            font=ctk.CTkFont(size=14, weight="bold"),
            fg_color="#3b82f6", hover_color="#2563eb", command=self.start_training_thread
        )
        self.btn_run_train.pack(side="left", padx=(0, 10))

        btn_scan_dups = ctk.CTkButton(
            action_row, text="🔍 Quét & Xử Lý Trùng Lặp", height=44, corner_radius=8,
            fg_color="#f59e0b", hover_color="#d97706", font=ctk.CTkFont(size=13, weight="bold"),
            command=self.check_and_resolve_duplicates
        )
        btn_scan_dups.pack(side="left", padx=(0, 12))

        self.lbl_train_status = ctk.CTkLabel(action_row, text="Sẵn sàng thực hiện", font=ctk.CTkFont(size=13), text_color="gray")
        self.lbl_train_status.pack(side="left")

        # Progress bar
        self.bar_train = ctk.CTkProgressBar(body, height=12, corner_radius=6)
        self.bar_train.grid(row=1, column=0, sticky="ew", padx=16, pady=(0, 10))
        self.bar_train.set(0)

        # Log Textbox
        self.txt_train_log = ctk.CTkTextbox(body, corner_radius=8, font=ctk.CTkFont(family="Consolas", size=12))
        self.txt_train_log.grid(row=2, column=0, sticky="nsew", padx=16, pady=(0, 14))
        self.txt_train_log.insert("end", "[HỆ THỐNG] Bảng điều khiển Huấn luyện AI sẵn sàng.\n")

        return view

    def _count_dataset_images(self) -> int:
        if not os.path.exists(DATASET_DIR):
            return 0
        return len([f for f in os.listdir(DATASET_DIR) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])

    def _get_model_modified_time(self) -> str:
        for p in [TRAINER_FILE, TRAINER_DIR_FILE]:
            if os.path.exists(p):
                mtime = os.path.getmtime(p)
                return datetime.datetime.fromtimestamp(mtime).strftime("%d/%m/%Y %H:%M:%S")
        return "Chưa tạo mô hình"

    # =========================================================================
    # VIEW 4: LỊCH SỬ & BÁO CÁO (HISTORY & REPORTS)
    # =========================================================================
    def _create_history_view(self) -> ctk.CTkFrame:
        view = ctk.CTkFrame(self.content_container, fg_color="transparent")
        view.grid_columnconfigure(0, weight=1)
        view.grid_rowconfigure(1, weight=1)

        # Thanh công cụ trên (Filter, Search, Export)
        toolbar = ctk.CTkFrame(view, corner_radius=12)
        toolbar.grid(row=0, column=0, sticky="ew", pady=(0, 12))

        # Tìm kiếm
        self.entry_hist_search = ctk.CTkEntry(toolbar, placeholder_text="🔍 Tìm theo Tên hoặc Mã ID...", width=230, height=36)
        self.entry_hist_search.pack(side="left", padx=(14, 8), pady=12)
        self.entry_hist_search.bind("<KeyRelease>", lambda e: self.filter_history_table())

        # Lọc ngày
        self.combo_hist_filter = ctk.CTkOptionMenu(
            toolbar, values=["Tất cả ngày", "Hôm nay", "7 ngày gần nhất", "Tháng này"],
            height=36, corner_radius=6, command=lambda v: self.filter_history_table()
        )
        self.combo_hist_filter.pack(side="left", padx=6, pady=12)

        # Nút Làm mới
        btn_refresh = ctk.CTkButton(toolbar, text="🔄 Làm Mới", width=90, height=36, corner_radius=6, command=self.reload_history_table)
        btn_refresh.pack(side="left", padx=6, pady=12)

        # Nút Xuất Excel
        btn_export_excel = ctk.CTkButton(
            toolbar, text="📥 Xuất File Excel", width=130, height=36, corner_radius=6,
            fg_color="#10b981", hover_color="#059669", command=self.export_history_excel
        )
        btn_export_excel.pack(side="right", padx=(6, 14), pady=12)

        # Nút Xuất CSV
        btn_export_csv = ctk.CTkButton(
            toolbar, text="📄 Xuất File CSV", width=120, height=36, corner_radius=6,
            fg_color="#3b82f6", hover_color="#2563eb", command=self.export_history_csv
        )
        btn_export_csv.pack(side="right", padx=6, pady=12)

        # Nút Xóa lịch sử
        btn_clear_data = ctk.CTkButton(
            toolbar, text="🗑️ Xóa Bản Ghi", width=110, height=36, corner_radius=6,
            fg_color="#ef4444", hover_color="#dc2626", command=self.clear_attendance_history
        )
        btn_clear_data.pack(side="right", padx=6, pady=12)

        # Bảng hiển thị dữ liệu (Treeview bọc trong Frame)
        table_container = ctk.CTkFrame(view, corner_radius=12)
        table_container.grid(row=1, column=0, sticky="nsew", pady=(0, 6))
        table_container.grid_rowconfigure(0, weight=1)
        table_container.grid_columnconfigure(0, weight=1)

        # Cấu hình Treeview
        cols = ("stt", "id", "name", "date", "time", "status")
        self.tree_history = ttk.Treeview(table_container, columns=cols, show="headings", selectmode="extended")

        self.tree_history.heading("stt", text="STT")
        self.tree_history.heading("id", text="Mã ID")
        self.tree_history.heading("name", text="Họ và Tên")
        self.tree_history.heading("date", text="Ngày Điểm Danh")
        self.tree_history.heading("time", text="Giờ Điểm Danh")
        self.tree_history.heading("status", text="Trạng Thái")

        self.tree_history.column("stt", width=60, anchor="center")
        self.tree_history.column("id", width=90, anchor="center")
        self.tree_history.column("name", width=240, anchor="w")
        self.tree_history.column("date", width=140, anchor="center")
        self.tree_history.column("time", width=140, anchor="center")
        self.tree_history.column("status", width=160, anchor="center")

        # Scrollbar
        scroll_y = ttk.Scrollbar(table_container, orient="vertical", command=self.tree_history.yview)
        self.tree_history.configure(yscrollcommand=scroll_y.set)

        self.tree_history.grid(row=0, column=0, sticky="nsew", padx=(10, 0), pady=10)
        scroll_y.grid(row=0, column=1, sticky="ns", padx=(0, 10), pady=10)

        # Footer thống kê bảng
        footer = ctk.CTkFrame(view, fg_color="transparent")
        footer.grid(row=2, column=0, sticky="ew", pady=(2, 0))

        self.lbl_hist_count = ctk.CTkLabel(footer, text="Tổng số bản ghi: 0", font=ctk.CTkFont(size=12, weight="bold"), text_color="gray")
        self.lbl_hist_count.pack(side="left", padx=4)

        self._style_treeview()
        self.reload_history_table()
        return view

    def _style_treeview(self):
        style = ttk.Style()
        style.theme_use("clam")

        is_dark = ctk.get_appearance_mode() == "Dark"
        bg_color = "#1e293b" if is_dark else "#ffffff"
        fg_color = "#f8fafc" if is_dark else "#0f172a"
        head_bg = "#0f172a" if is_dark else "#e2e8f0"
        head_fg = "#ffffff" if is_dark else "#0f172a"
        select_bg = "#3b82f6"

        style.configure(
            "Treeview",
            background=bg_color,
            foreground=fg_color,
            fieldbackground=bg_color,
            rowheight=32,
            font=("Segoe UI", 11)
        )
        style.configure(
            "Treeview.Heading",
            background=head_bg,
            foreground=head_fg,
            relief="flat",
            font=("Segoe UI", 11, "bold")
        )
        style.map("Treeview", background=[("selected", select_bg)])

    # =========================================================================
    # VIEW 5: QUẢN LÝ NHÂN SỰ (USER MANAGEMENT)
    # =========================================================================
    def _create_users_view(self) -> ctk.CTkFrame:
        view = ctk.CTkFrame(self.content_container, fg_color="transparent")
        view.grid_columnconfigure(0, weight=1)
        view.grid_rowconfigure(1, weight=1)

        # Toolbar
        toolbar = ctk.CTkFrame(view, corner_radius=12)
        toolbar.grid(row=0, column=0, sticky="ew", pady=(0, 12))

        ctk.CTkLabel(toolbar, text="👥 DANH SÁCH NHÂN SỰ ĐÃ ĐĂNG KÝ", font=ctk.CTkFont(size=16, weight="bold")).pack(side="left", padx=16, pady=12)

        btn_add_user = ctk.CTkButton(
            toolbar, text="➕ Thêm Người Mới", width=140, height=36, corner_radius=6,
            command=self.switch_to_register, fg_color="#10b981", hover_color="#059669"
        )
        btn_add_user.pack(side="right", padx=(6, 14), pady=12)

        btn_scan_dups = ctk.CTkButton(
            toolbar, text="🔍 Quét & Xử Lý Trùng Lặp", width=170, height=36, corner_radius=6,
            command=self.check_and_resolve_duplicates, fg_color="#f59e0b", hover_color="#d97706"
        )
        btn_scan_dups.pack(side="right", padx=6, pady=12)

        btn_del_user = ctk.CTkButton(
            toolbar, text="🗑️ Xóa Người Chọn", width=130, height=36, corner_radius=6,
            command=self.delete_selected_user, fg_color="#ef4444", hover_color="#dc2626"
        )
        btn_del_user.pack(side="right", padx=6, pady=12)

        btn_rename_user = ctk.CTkButton(
            toolbar, text="✏️ Đổi Tên", width=100, height=36, corner_radius=6,
            command=self.rename_selected_user
        )
        btn_rename_user.pack(side="right", padx=6, pady=12)

        # Bảng Users
        table_container = ctk.CTkFrame(view, corner_radius=12)
        table_container.grid(row=1, column=0, sticky="nsew", pady=0)
        table_container.grid_rowconfigure(0, weight=1)
        table_container.grid_columnconfigure(0, weight=1)

        cols = ("id", "name", "photos", "status")
        self.tree_users = ttk.Treeview(table_container, columns=cols, show="headings", selectmode="browse")

        self.tree_users.heading("id", text="Mã ID")
        self.tree_users.heading("name", text="Họ và Tên")
        self.tree_users.heading("photos", text="Số Ảnh Mẫu Trong Dataset")
        self.tree_users.heading("status", text="Trạng Thái")

        self.tree_users.column("id", width=100, anchor="center")
        self.tree_users.column("name", width=300, anchor="w")
        self.tree_users.column("photos", width=200, anchor="center")
        self.tree_users.column("status", width=220, anchor="center")

        scroll_y = ttk.Scrollbar(table_container, orient="vertical", command=self.tree_users.yview)
        self.tree_users.configure(yscrollcommand=scroll_y.set)

        self.tree_users.grid(row=0, column=0, sticky="nsew", padx=(10, 0), pady=10)
        scroll_y.grid(row=0, column=1, sticky="ns", padx=(0, 10), pady=10)

        self.reload_users_table()
        return view

    # =========================================================================
    # VIEW 6: CÀI ĐẶT HỆ THỐNG (SETTINGS VIEW)
    # =========================================================================
    def _create_settings_view(self) -> ctk.CTkFrame:
        view = ctk.CTkFrame(self.content_container, fg_color="transparent")
        view.grid_columnconfigure(0, weight=1)

        panel = ctk.CTkFrame(view, corner_radius=12)
        panel.pack(fill="both", expand=True)

        ctk.CTkLabel(panel, text="⚙️ CÀI ĐẶT HỆ THỐNG & THAM SỐ NHẬN DIỆN", font=ctk.CTkFont(size=18, weight="bold")).pack(anchor="w", padx=20, pady=(20, 16))

        # Cổng Camera
        row_cam = ctk.CTkFrame(panel, fg_color="transparent")
        row_cam.pack(fill="x", padx=20, pady=10)
        ctk.CTkLabel(row_cam, text="Cổng kết nối Camera (Camera Index):", width=280, anchor="w", font=ctk.CTkFont(size=13, weight="bold")).pack(side="left")
        self.spin_cam_id = ctk.CTkComboBox(row_cam, values=["0", "1", "2", "3"], width=100)
        self.spin_cam_id.set(str(self.settings.get("camera_id", 0)))
        self.spin_cam_id.pack(side="left", padx=10)

        # Ngưỡng nhận diện (Confidence Distance)
        row_conf = ctk.CTkFrame(panel, fg_color="transparent")
        row_conf.pack(fill="x", padx=20, pady=12)
        ctk.CTkLabel(row_conf, text="Ngưỡng tin cậy nhận diện (Threshold):", width=280, anchor="w", font=ctk.CTkFont(size=13, weight="bold")).pack(side="left")
        self.slider_conf = ctk.CTkSlider(row_conf, from_=40, to=90, number_of_steps=50, width=220)
        self.slider_conf.set(self.settings.get("confidence_threshold", 70))
        self.slider_conf.pack(side="left", padx=10)
        self.lbl_conf_val = ctk.CTkLabel(row_conf, text=str(int(self.slider_conf.get())), width=50, font=ctk.CTkFont(size=13, weight="bold"))
        self.lbl_conf_val.pack(side="left")
        self.slider_conf.configure(command=lambda v: self.lbl_conf_val.configure(text=str(int(v))))

        # Thời gian Cooldown
        row_cool = ctk.CTkFrame(panel, fg_color="transparent")
        row_cool.pack(fill="x", padx=20, pady=12)
        ctk.CTkLabel(row_cool, text="Giãn cách điểm danh cùng 1 người:", width=280, anchor="w", font=ctk.CTkFont(size=13, weight="bold")).pack(side="left")
        self.slider_cooldown = ctk.CTkSlider(row_cool, from_=10, to=300, number_of_steps=29, width=220)
        self.slider_cooldown.set(self.settings.get("cooldown_seconds", 60))
        self.slider_cooldown.pack(side="left", padx=10)
        self.lbl_cool_val = ctk.CTkLabel(row_cool, text=f"{int(self.slider_cooldown.get())}s", width=50, font=ctk.CTkFont(size=13, weight="bold"))
        self.lbl_cool_val.pack(side="left")
        self.slider_cooldown.configure(command=lambda v: self.lbl_cool_val.configure(text=f"{int(v)}s"))

        # Bật/tắt âm thanh
        row_sound = ctk.CTkFrame(panel, fg_color="transparent")
        row_sound.pack(fill="x", padx=20, pady=12)
        ctk.CTkLabel(row_sound, text="Âm thanh khi điểm danh thành công:", width=280, anchor="w", font=ctk.CTkFont(size=13, weight="bold")).pack(side="left")
        self.switch_sound = ctk.CTkSwitch(row_sound, text="Bật âm thanh (Chime Beep)")
        if self.settings.get("sound_enabled", True):
            self.switch_sound.select()
        else:
            self.switch_sound.deselect()
        self.switch_sound.pack(side="left", padx=10)

        # Nút Lưu Cài Đặt
        btn_save_settings = ctk.CTkButton(
            panel, text="💾 LƯU CẤU HÌNH HỆ THỐNG", width=220, height=42, corner_radius=8,
            font=ctk.CTkFont(size=14, weight="bold"),
            fg_color="#10b981", hover_color="#059669", command=self.save_system_settings
        )
        btn_save_settings.pack(anchor="w", padx=20, pady=(25, 20))

        return view

    # =========================================================================
    # ĐIỀU HƯỚNG CHUYỂN TAB
    # =========================================================================
    def switch_to_attendance(self):
        self._show_view(self.view_attendance, "attendance")
        self.start_camera()

    def switch_to_register(self):
        self._show_view(self.view_register, "register")
        self._suggest_next_id()
        self.start_camera()

    def switch_to_training(self):
        self._show_view(self.view_training, "training")
        self.stop_camera()  # Tắt camera khi chuyển sang tab huấn luyện để giải phóng CPU
        self.lbl_train_img_count.configure(text=str(self._count_dataset_images()))
        self.lbl_train_user_count.configure(text=str(len(self.users)))
        self.lbl_train_model_date.configure(text=self._get_model_modified_time())

    def switch_to_history(self):
        self._show_view(self.view_history, "history")
        self.stop_camera()
        self.reload_history_table()

    def switch_to_users(self):
        self._show_view(self.view_users, "users")
        self.stop_camera()
        self.reload_users_table()

    def switch_to_settings(self):
        self._show_view(self.view_settings, "settings")
        self.stop_camera()

    # =========================================================================
    # VÒNG LẶP XỬ LÝ CAMERA & NHẬN DIỆN THỜI GIAN THỰC
    # =========================================================================
    def start_camera(self):
        with self.camera_lock:
            if self.is_camera_running and self.cap is not None and self.cap.isOpened():
                return
            cam_idx = int(self.settings.get("camera_id", 0))
            self.cap = cv2.VideoCapture(cam_idx, cv2.CAP_DSHOW if os.name == 'nt' else cv2.CAP_ANY)
            if not self.cap.isOpened():
                self.cap = cv2.VideoCapture(cam_idx)
            self.is_camera_running = self.cap.isOpened()

        if self.is_camera_running:
            self.btn_toggle_cam.configure(text="⏸️ Tạm Dừng Camera")
            self.lbl_cam_status_badge.configure(text="● TRỰC TIẾP", text_color="#10b981")
            self._camera_loop()
        else:
            self.btn_toggle_cam.configure(text="▶️ Khởi Động Camera")
            self.lbl_cam_status_badge.configure(text="● LỖI KẾT NỐI", text_color="#ef4444")
            self.video_label.configure(text="Không thể mở Camera. Vui lòng kiểm tra kết nối!", image="")

    def stop_camera(self):
        with self.camera_lock:
            self.is_camera_running = False
            if self.cap is not None:
                self.cap.release()
                self.cap = None

        if hasattr(self, "btn_toggle_cam"):
            self.btn_toggle_cam.configure(text="▶️ Khởi Động Camera")
            self.lbl_cam_status_badge.configure(text="● TẠM DỪNG", text_color="gray")

    def toggle_camera(self):
        if self.is_camera_running:
            self.stop_camera()
            self.video_label.configure(text="Camera đang tạm dừng.", image="")
        else:
            self.start_camera()

    def _camera_loop(self):
        if not self.is_camera_running or self.cap is None:
            return

        ret, frame = self.cap.read()
        if not ret or frame is None:
            self.after(30, self._camera_loop)
            return

        # Lật ảnh gương cho tự nhiên
        frame = cv2.flip(frame, 1)
        self.last_raw_frame = frame.copy()
        h_frame, w_frame = frame.shape[:2]

        # Xử lý theo tab đang mở
        if self.current_tab == "attendance":
            self._process_attendance_frame(frame, w_frame, h_frame)
        elif self.current_tab == "register":
            self._process_register_frame(frame, w_frame, h_frame)

        # Lặp lại tiếp tục
        if self.is_camera_running:
            self.after(16, self._camera_loop)

    def _process_attendance_frame(self, frame: np.ndarray, w_frame: int, h_frame: int):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        small_gray = cv2.resize(gray, (0, 0), fx=0.5, fy=0.5)
        faces_small = self.cascade_detector.detectMultiScale(small_gray, 1.2, 5, minSize=(30, 30))

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb_frame)
        draw = ImageDraw.Draw(pil_img)

        now = time.time()
        conf_thresh = float(self.settings.get("confidence_threshold", 70))
        cooldown = float(self.settings.get("cooldown_seconds", 60))

        for (sx, sy, sw, sh) in faces_small:
            x, y, w, h = sx * 2, sy * 2, sw * 2, sh * 2
            face_crop = gray[y:y+h, x:x+w]
            if face_crop.size == 0:
                continue

            if self.model_loaded:
                try:
                    pred_id, confidence = self.recognizer.predict(face_crop)
                except Exception:
                    pred_id, confidence = -1, 999
            else:
                pred_id, confidence = -1, 999

            # XỬ LÝ KHỬ TRÙNG LẶP ID:
            # Nếu pred_id là một ID bị trùng lặp với ID gốc, tự động quy về ID gốc!
            canonical_id = self.duplicate_map.get(pred_id, pred_id)

            if self.model_loaded and confidence < conf_thresh:
                user_name = self.users.get(canonical_id, self.users.get(pred_id, f"ID: {canonical_id}"))
                match_percent = max(10, min(99, int(100 - (confidence / conf_thresh) * 35)))

                # Cooldown áp dụng cho cả nhóm ID trùng lặp (canonical_id)
                last_time = self.last_checkin_times.get(canonical_id, 0)
                is_cooldown = (now - last_time) < cooldown

                if not is_cooldown:
                    # GHI NHẬN ĐIỂM DANH MỚI
                    self.last_checkin_times[canonical_id] = now
                    # Đánh dấu cooldown cho cả ID trùng nếu có
                    if pred_id != canonical_id:
                        self.last_checkin_times[pred_id] = now

                    dt_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    append_attendance_record(canonical_id, user_name, dt_str)

                    if self.settings.get("sound_enabled", True):
                        play_chime_sound()

                    self._on_attendance_success(canonical_id, user_name, match_percent, dt_str)

                    status_tag = "VỪA GHI NHẬN!"
                    color = (16, 185, 129)  # Emerald green
                else:
                    status_tag = "ĐÃ ĐIỂM DANH"
                    color = (59, 130, 246)  # Blue

                # Thêm nhãn nếu đang tự động gộp trùng
                if pred_id != canonical_id:
                    title_text = f"{user_name} ({match_percent}%)"
                    subtitle_text = f"ID: #{canonical_id} (Đã gộp trùng ID #{pred_id}) • {status_tag}"
                else:
                    title_text = f"{user_name} ({match_percent}%)"
                    subtitle_text = f"ID: #{canonical_id} • {status_tag}"
            else:
                title_text = "Chưa nhận diện"
                subtitle_text = "Vui lòng nhìn thẳng"
                color = (239, 68, 68)  # Red

            draw_hud_box(draw, x, y, w, h, title_text, subtitle_text, color)

        display_w, display_h = 640, 480
        pil_resized = pil_img.resize((display_w, display_h), Image.Resampling.BILINEAR)
        ctk_img = ctk.CTkImage(light_image=pil_resized, dark_image=pil_resized, size=(display_w, display_h))
        self.video_label.configure(image=ctk_img, text="")
        self.video_label.image = ctk_img

    def _on_attendance_success(self, uid: int, name: str, match_pct: int, dt_str: str):
        """Cập nhật các thẻ thông tin và danh sách live feed"""
        self.lbl_recent_name.configure(text=f"{name}")
        self.lbl_recent_details.configure(text=f"Mã ID: #{uid} • Thời gian: {dt_str.split()[-1]} • Độ khớp: {match_pct}%")
        self.badge_recent_status.configure(text="✅ ĐIỂM DANH THÀNH CÔNG!", text_color="#10b981")

        self.lbl_kpi_today_count.configure(text=str(self._get_today_count()))
        self._refresh_live_feed()

    def _process_register_frame(self, frame: np.ndarray, w_frame: int, h_frame: int):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.cascade_detector.detectMultiScale(gray, 1.3, 5, minSize=(60, 60))

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb_frame)
        draw = ImageDraw.Draw(pil_img)

        # Vùng căn chỉnh khuôn mặt
        center_w, center_h = int(w_frame * 0.45), int(h_frame * 0.6)
        cx1 = (w_frame - center_w) // 2
        cy1 = (h_frame - center_h) // 2
        cx2 = cx1 + center_w
        cy2 = cy1 + center_h

        draw.rectangle([(cx1, cy1), (cx2, cy2)], outline=(100, 116, 139), width=2)
        draw.text((cx1 + 10, cy1 + 10), "VÙNG CĂN CHỈNH KHUÔN MẶT", font=FONT_SM, fill=(148, 163, 184))

        if self.reg_capturing and len(faces) > 0:
            for (x, y, w, h) in faces:
                self.reg_count += 1
                img_path = os.path.join(DATASET_DIR, f"User.{self.reg_user_id}.{self.reg_count}.jpg")
                cv2.imwrite(img_path, gray[y:y+h, x:x+w])

                draw_hud_box(draw, x, y, w, h, f"Đang chụp: {self.reg_count}/50", "Giữ nguyên tư thế...", (16, 185, 129))

                pct = self.reg_count / self.reg_max
                self.bar_reg_progress.set(pct)
                self.lbl_reg_progress.configure(text=f"Tiến độ: {self.reg_count} / {self.reg_max} ảnh ({int(pct*100)}%)")
                break

            if self.reg_count >= self.reg_max:
                self.reg_capturing = False
                self._on_registration_finished()
        elif len(faces) > 0:
            for (x, y, w, h) in faces:
                draw_hud_box(draw, x, y, w, h, "Phát hiện khuôn mặt", "Sẵn sàng chụp", (59, 130, 246))

        display_w, display_h = 560, 420
        pil_resized = pil_img.resize((display_w, display_h), Image.Resampling.BILINEAR)
        ctk_img = ctk.CTkImage(light_image=pil_resized, dark_image=pil_resized, size=(display_w, display_h))
        self.reg_video_label.configure(image=ctk_img, text="")
        self.reg_video_label.image = ctk_img

    # =========================================================================
    # QUY TRÌNH ĐĂNG KÝ NGƯỜI MỚI (CÓ KIỂM SOÁT TRÙNG LẶP KHUÔN MẶT)
    # =========================================================================
    def start_registration_capture(self):
        id_str = self.entry_reg_id.get().strip()
        name_str = self.entry_reg_name.get().strip()

        if not id_str.isdigit():
            messagebox.showerror("Lỗi", "Mã ID người dùng phải là số nguyên dương!")
            return
        if not name_str:
            messagebox.showerror("Lỗi", "Vui lòng nhập họ và tên người mới!")
            return

        new_uid = int(id_str)

        # -------------------------------------------------------------
        # BƯỚC KIỂM TRA TRÙNG LẶP KHUÔN MẶT (CHỐNG TRÙNG ID)
        # -------------------------------------------------------------
        if self.last_raw_frame is not None:
            gray = cv2.cvtColor(self.last_raw_frame, cv2.COLOR_BGR2GRAY)
            faces = self.cascade_detector.detectMultiScale(gray, 1.2, 5, minSize=(60, 60))

            if len(faces) == 0:
                messagebox.showwarning("Chưa Thấy Mặt", "Không phát hiện khuôn mặt nào trước camera!\nVui lòng đứng trước camera và nhìn thẳng vào khung ngắm.")
                return

            # Lấy khuôn mặt đầu tiên để kiểm tra trùng lặp
            fx, fy, fw, fh = faces[0]
            face_crop = gray[fy:fy+fh, fx:fx+fw]

            # 1. So sánh với các ảnh đã có trong thư mục dataset
            matched_uid, sim_score = match_face_to_existing_dataset(face_crop, self.users, threshold=0.70)

            # 2. Hoặc kiểm tra với mô hình đã huấn luyện nếu có
            if matched_uid is None and self.model_loaded:
                try:
                    pred_id, conf = self.recognizer.predict(face_crop)
                    thresh = float(self.settings.get("confidence_threshold", 70))
                    if conf < (thresh - 5) and pred_id in self.users:
                        matched_uid = pred_id
                        sim_score = max(0.75, 1.0 - (conf / thresh) * 0.4)
                except Exception:
                    pass

            # NẾU PHÁT HIỆN KHUÔN MẶT NÀY ĐÃ THUỘC VỀ MỘT ID ĐÃ CÓ VÀ KHÁC NEW_UID:
            if matched_uid is not None and matched_uid != new_uid:
                existing_name = self.users.get(matched_uid, f"ID #{matched_uid}")
                sim_pct = int(sim_score * 100)

                msg = (
                    f"⚠️ PHÁT HIỆN TRÙNG LẶP KHUÔN MẶT!\n\n"
                    f"Khuôn mặt này ĐÃ ĐƯỢC ĐĂNG KÝ trong hệ thống cho:\n"
                    f"• Họ và tên: {existing_name} (Mã ID: #{matched_uid})\n"
                    f"• Độ khớp nhận diện: ~{sim_pct}%\n\n"
                    f"Hệ thống không cho phép đăng ký cùng 1 khuôn mặt thành 2 mã ID khác nhau (sẽ gây xung đột khi nhận diện).\n\n"
                    f"Bạn có muốn:\n"
                    f"- Chọn 'YES' để chuyển sang CẬP NHẬT ẢNH cho {existing_name} (ID: #{matched_uid})\n"
                    f"- Chọn 'NO' để HỦY BỎ thao tác đăng ký ID mới này"
                )
                choice = messagebox.askyesno("Cảnh Báo Trùng Lặp", msg)
                if choice:
                    # Chuyển sang cập nhật ID cũ
                    self.entry_reg_id.delete(0, "end")
                    self.entry_reg_id.insert(0, str(matched_uid))
                    self.entry_reg_name.delete(0, "end")
                    self.entry_reg_name.insert(0, existing_name)
                    new_uid = matched_uid
                    name_str = existing_name
                else:
                    return

        # Lưu thông tin nhân sự
        self.users[new_uid] = name_str
        save_users(self.users)

        self.reg_user_id = new_uid
        self.reg_user_name = name_str
        self.reg_count = 0
        self.reg_capturing = True

        self.btn_start_capture.configure(state="disabled", text="⏳ ĐANG CHỤP ẢNH...")
        self.bar_reg_progress.set(0)
        self.lbl_reg_progress.configure(text="Tiến độ: 0 / 50 ảnh (0%)")

    def cancel_registration_capture(self):
        self.reg_capturing = False
        self.btn_start_capture.configure(state="normal", text="📸 BẮT ĐẦU CHỤP 50 ẢNH MẪU")
        self.lbl_reg_progress.configure(text="Đã dừng chụp.")

    def _on_registration_finished(self):
        self.btn_start_capture.configure(state="normal", text="📸 BẮT ĐẦU CHỤP 50 ẢNH MẪU")
        self.lbl_reg_progress.configure(text=f"Hoàn thành! Đã thu thập 50 ảnh cho {self.reg_user_name}.")
        self.lbl_kpi_total_users.configure(text=str(len(self.users)))
        self._refresh_duplicate_map()
        self._update_dup_status_label()

        if self.reg_auto_train.get():
            messagebox.showinfo("Thành Công", f"Đã thu thập 50 ảnh mẫu cho '{self.reg_user_name}' (ID: {self.reg_user_id})!\nHệ thống sẽ tự động chuyển sang huấn luyện AI.")
            self.switch_to_training()
            self.start_training_thread()
        else:
            messagebox.showinfo("Thành Công", f"Đã đăng ký thành công {self.reg_user_name} với ID {self.reg_user_id}!")
            self._suggest_next_id()
            self.entry_reg_name.delete(0, "end")

    # =========================================================================
    # QUY TRÌNH HUẤN LUYỆN MÔ HÌNH AI (TRAINING)
    # =========================================================================
    def start_training_thread(self):
        if self.is_training:
            return
        self.is_training = True
        self.btn_run_train.configure(state="disabled", text="⏳ Đang huấn luyện...")
        self.bar_train.set(0)
        self.lbl_train_status.configure(text="Đang xử lý dữ liệu...", text_color="#3b82f6")

        threading.Thread(target=self._train_worker, daemon=True).start()

    def _train_worker(self):
        def log(msg: str):
            t_str = datetime.datetime.now().strftime("%H:%M:%S")
            self.after(0, lambda: self.txt_train_log.insert("end", f"[{t_str}] {msg}\n"))
            self.after(0, lambda: self.txt_train_log.see("end"))

        log("Bắt đầu quá trình huấn luyện mô hình nhận diện khuôn mặt...")
        if not os.path.exists(DATASET_DIR):
            log("LỖI: Thư mục 'dataset' không tồn tại!")
            self._finish_training(False)
            return

        image_files = [f for f in os.listdir(DATASET_DIR) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        total_files = len(image_files)

        if total_files == 0:
            log("LỖI: Chưa có ảnh mẫu nào trong thư mục 'dataset'!")
            self._finish_training(False)
            return

        # Cảnh báo nếu có ID trùng lặp trong dataset trước khi huấn luyện
        dups = detect_all_dataset_duplicates(self.users, threshold=0.72)
        if dups:
            for u1, u2, sc in dups:
                n1 = self.users.get(u1, f"ID {u1}")
                n2 = self.users.get(u2, f"ID {u2}")
                log(f"[CẢNH BÁO TRÙNG] ID {u1} ({n1}) và ID {u2} ({n2}) có khuôn mặt giống nhau {int(sc*100)}%! Hệ thống đã kích hoạt cơ chế gộp ID tự động.")

        log(f"Tìm thấy {total_files} file ảnh mẫu. Bắt đầu trích xuất đặc trưng khuôn mặt...")
        face_samples = []
        ids = []

        for idx, fname in enumerate(image_files):
            fpath = os.path.join(DATASET_DIR, fname)
            try:
                parts = fname.split(".")
                if len(parts) >= 3 and parts[0].lower() == "user":
                    uid = int(parts[1])
                else:
                    continue

                pil_img = Image.open(fpath).convert("L")
                img_numpy = np.array(pil_img, "uint8")
                faces = self.cascade_detector.detectMultiScale(img_numpy)

                for (x, y, w, h) in faces:
                    face_samples.append(img_numpy[y:y+h, x:x+w])
                    ids.append(uid)

                pct = (idx + 1) / total_files * 0.8
                self.after(0, lambda p=pct: self.bar_train.set(p))
            except Exception as e:
                log(f"Lỗi đọc file {fname}: {e}")

        if len(face_samples) == 0:
            log("LỖI: Không nhận diện được khuôn mặt nào trong các ảnh mẫu!")
            self._finish_training(False)
            return

        log(f"Trích xuất thành công {len(face_samples)} mẫu khuôn mặt từ {len(set(ids))} nhân sự.")
        log("Đang huấn luyện bộ phân loại LBPH Face Recognizer...")

        try:
            self.recognizer.train(face_samples, np.array(ids))
            self.recognizer.write(TRAINER_FILE)
            self.recognizer.write(TRAINER_DIR_FILE)

            self.after(0, lambda: self.bar_train.set(1.0))
            log("Huấn luyện hoàn tất! Đã lưu mô hình thành công vào 'trainer.yml'.")
            self._finish_training(True)
        except Exception as e:
            log(f"Lỗi trong quá trình huấn luyện: {e}")
            self._finish_training(False)

    def _finish_training(self, success: bool):
        def _update():
            self.is_training = False
            self.btn_run_train.configure(state="normal", text="🚀 BẮT ĐẦU HUẤN LUYỆN MÔ HÌNH")
            if success:
                self.model_loaded = True
                self.lbl_train_status.configure(text="Huấn luyện thành công!", text_color="#10b981")
                self.lbl_kpi_model_status.configure(text="SẴN SÀNG", text_color=("#16a34a", "#4ade80"))
                self.lbl_train_model_date.configure(text=self._get_model_modified_time())
                self._refresh_duplicate_map()
                self._update_dup_status_label()
                messagebox.showinfo("Thành Công", "Mô hình nhận diện AI đã được huấn luyện thành công!")
            else:
                self.lbl_train_status.configure(text="Huấn luyện thất bại!", text_color="#ef4444")
                messagebox.showerror("Thất Bại", "Quá trình huấn luyện gặp sự cố. Vui lòng kiểm tra lại log!")
        self.after(0, _update)

    # =========================================================================
    # CÔNG CỤ QUÉT & XỬ LÝ TRÙNG LẶP ID (DUPLICATE RESOLVER)
    # =========================================================================
    def check_and_resolve_duplicates(self):
        """Phát hiện và cho phép người dùng 1-click gộp hoặc xóa các ID trùng khuôn mặt"""
        dups = detect_all_dataset_duplicates(self.users, threshold=0.70)
        if not dups:
            messagebox.showinfo("Kiểm Tra Dữ Liệu", "✅ Dữ liệu hoàn hảo!\nKhông phát hiện nhân sự nào bị trùng lặp khuôn mặt giữa các mã ID.")
            return

        for u1, u2, sc in dups:
            name1 = self.users.get(u1, f"ID #{u1}")
            name2 = self.users.get(u2, f"ID #{u2}")
            pct = int(sc * 100)

            msg = (
                f"⚠️ PHÁT HIỆN 2 MÃ ID CÓ CÙNG KHUÔN MẶT!\n\n"
                f"• ID #{u1}: {name1}\n"
                f"• ID #{u2}: {name2}\n"
                f"• Độ tương đồng ảnh mẫu: {pct}%\n\n"
                f"Bạn có muốn GỘP ID #{u2} vào ID #{u1} không?\n"
                f"(Thao tác này sẽ chuyển toàn bộ lịch sử điểm danh của ID #{u2} sang ID #{u1}, "
                f"xóa các ảnh thừa của ID #{u2} và tự động huấn luyện lại AI)."
            )
            ans = messagebox.askyesno("Xử Lý Trùng Lặp Nhân Sự", msg)
            if ans:
                self._merge_duplicate_users(u1, u2)
                break

    def _merge_duplicate_users(self, primary_id: int, duplicate_id: int):
        """Thực hiện gộp duplicate_id vào primary_id"""
        primary_name = self.users.get(primary_id, f"ID #{primary_id}")
        duplicate_name = self.users.get(duplicate_id, f"ID #{duplicate_id}")

        # 1. Xóa ảnh của duplicate_id trong dataset
        if os.path.exists(DATASET_DIR):
            prefix = f"user.{duplicate_id}."
            for fname in os.listdir(DATASET_DIR):
                if fname.lower().startswith(prefix):
                    try:
                        os.remove(os.path.join(DATASET_DIR, fname))
                    except Exception:
                        pass

        # 2. Xóa khỏi users config.py
        if duplicate_id in self.users:
            del self.users[duplicate_id]
            save_users(self.users)

        # 3. Chuẩn hóa lịch sử điểm danh: chuyển duplicate_id thành primary_id
        if not self.attendance_df.empty:
            mask = self.attendance_df["ID"] == duplicate_id
            self.attendance_df.loc[mask, "ID"] = primary_id
            self.attendance_df.loc[mask, "Name"] = primary_name
            self.attendance_df.to_csv(ATTENDANCE_FILE, index=False, encoding="utf-8")

        # 4. Cập nhật giao diện & Huấn luyện lại AI
        self.reload_users_table()
        self.reload_history_table()
        self._refresh_duplicate_map()
        self._update_dup_status_label()
        self.lbl_kpi_total_users.configure(text=str(len(self.users)))

        messagebox.showinfo(
            "Đã Gộp Thành Công",
            f"Đã gộp thành công ID #{duplicate_id} ({duplicate_name}) vào ID #{primary_id} ({primary_name})!\n"
            f"Hệ thống sẽ tiến hành huấn luyện lại mô hình AI ngay bây giờ."
        )
        self.switch_to_training()
        self.start_training_thread()

    # =========================================================================
    # LỊCH SỬ ĐIỂM DANH & XUẤT BÁO CÁO (HISTORY & EXPORT)
    # =========================================================================
    def reload_history_table(self):
        self.attendance_df = load_attendance_df()
        self.filter_history_table()

    def filter_history_table(self):
        for item in self.tree_history.get_children():
            self.tree_history.delete(item)

        df = self.attendance_df.copy()
        if df.empty:
            self.lbl_hist_count.configure(text="Tổng số bản ghi: 0")
            return

        query = self.entry_hist_search.get().strip().lower()
        if query:
            df = df[
                df["Name"].astype(str).str.lower().str.contains(query) |
                df["ID"].astype(str).str.contains(query)
            ]

        filter_mode = self.combo_hist_filter.get()
        today_str = datetime.date.today().strftime("%Y-%m-%d")

        if filter_mode == "Hôm nay":
            df = df[df["Time"].astype(str).str.startswith(today_str)]
        elif filter_mode == "7 ngày gần nhất":
            seven_days_ago = (datetime.date.today() - datetime.timedelta(days=7)).strftime("%Y-%m-%d")
            df = df[df["Time"].astype(str) >= seven_days_ago]
        elif filter_mode == "Tháng này":
            this_month = datetime.date.today().strftime("%Y-%m")
            df = df[df["Time"].astype(str).str.startswith(this_month)]

        df = df.iloc[::-1]

        for i, (_, row) in enumerate(df.iterrows(), start=1):
            time_full = str(row["Time"])
            parts = time_full.split()
            date_part = parts[0] if len(parts) > 0 else ""
            time_part = parts[1] if len(parts) > 1 else ""

            self.tree_history.insert(
                "", "end",
                values=(i, row["ID"], row["Name"], date_part, time_part, "Thành công")
            )

        self.lbl_hist_count.configure(text=f"Hiển thị {len(df)} bản ghi (Tổng dữ liệu: {len(self.attendance_df)})")

    def export_history_excel(self):
        if self.attendance_df.empty:
            messagebox.showwarning("Thông Báo", "Chưa có dữ liệu điểm danh để xuất file!")
            return

        filepath = filedialog.asksaveasfilename(
            defaultextension=".xlsx",
            filetypes=[("Excel Files", "*.xlsx"), ("All Files", "*.*")],
            initialfile=f"BaoCao_DiemDanh_{datetime.date.today().strftime('%Y%m%d')}.xlsx"
        )
        if not filepath:
            return

        try:
            df = self.attendance_df.copy()
            df["Ngày"] = df["Time"].apply(lambda t: str(t).split()[0] if len(str(t).split()) > 0 else "")
            df["Giờ"] = df["Time"].apply(lambda t: str(t).split()[1] if len(str(t).split()) > 1 else "")
            df["Trạng Thái"] = "Hợp Lệ"
            df.rename(columns={"ID": "Mã Nhân Viên", "Name": "Họ và Tên", "Time": "Thời Gian Đầy Đủ"}, inplace=True)

            with pd.ExcelWriter(filepath, engine="openpyxl") as writer:
                df.to_excel(writer, index=False, sheet_name="Điểm Danh")

            messagebox.showinfo("Thành Công", f"Đã xuất file báo cáo Excel thành công tại:\n{filepath}")
        except Exception as e:
            messagebox.showerror("Lỗi", f"Không thể xuất file Excel: {e}")

    def export_history_csv(self):
        if self.attendance_df.empty:
            messagebox.showwarning("Thông Báo", "Chưa có dữ liệu điểm danh để xuất file!")
            return

        filepath = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV Files", "*.csv"), ("All Files", "*.*")],
            initialfile=f"BaoCao_DiemDanh_{datetime.date.today().strftime('%Y%m%d')}.csv"
        )
        if not filepath:
            return

        try:
            self.attendance_df.to_csv(filepath, index=False, encoding="utf-8-sig")
            messagebox.showinfo("Thành Công", f"Đã xuất file CSV thành công tại:\n{filepath}")
        except Exception as e:
            messagebox.showerror("Lỗi", f"Không thể xuất file CSV: {e}")

    def clear_attendance_history(self):
        selected = self.tree_history.selection()
        if not selected:
            ans = messagebox.askyesno("Xác Nhận", "Bạn có chắc chắn muốn XÓA TOÀN BỘ lịch sử điểm danh không?\nHành động này không thể hoàn tác!")
            if ans:
                self.attendance_df = pd.DataFrame(columns=["ID", "Name", "Time"])
                self.attendance_df.to_csv(ATTENDANCE_FILE, index=False, encoding="utf-8")
                self.reload_history_table()
                self._refresh_live_feed()
                self.lbl_kpi_today_count.configure(text="0")
                messagebox.showinfo("Thông Báo", "Đã xóa toàn bộ lịch sử điểm danh.")
        else:
            ans = messagebox.askyesno("Xác Nhận", f"Bạn có chắc muốn xóa {len(selected)} bản ghi đã chọn?")
            if ans:
                for item in selected:
                    vals = self.tree_history.item(item, "values")
                    uid = int(vals[1])
                    d_str = vals[3]
                    t_str = vals[4]
                    full_time = f"{d_str} {t_str}"
                    self.attendance_df = self.attendance_df[
                        ~((self.attendance_df["ID"] == uid) & (self.attendance_df["Time"] == full_time))
                    ]
                self.attendance_df.to_csv(ATTENDANCE_FILE, index=False, encoding="utf-8")
                self.reload_history_table()
                self._refresh_live_feed()

    # =========================================================================
    # QUẢN LÝ NHÂN SỰ (USER MANAGEMENT)
    # =========================================================================
    def reload_users_table(self):
        for item in self.tree_users.get_children():
            self.tree_users.delete(item)

        for uid, name in sorted(self.users.items()):
            photo_count = 0
            if os.path.exists(DATASET_DIR):
                prefix = f"user.{uid}."
                photo_count = sum(1 for f in os.listdir(DATASET_DIR) if f.lower().startswith(prefix))

            if uid in self.duplicate_map:
                orig_id = self.duplicate_map[uid]
                status = f"⚠️ Trùng khuôn mặt với ID #{orig_id}"
            elif photo_count >= 50:
                status = "Đã có dữ liệu (Hợp lệ)"
            else:
                status = f"Chưa đủ ảnh ({photo_count}/50)"

            self.tree_users.insert("", "end", values=(uid, name, f"{photo_count} ảnh", status))

    def rename_selected_user(self):
        selected = self.tree_users.selection()
        if not selected:
            messagebox.showwarning("Thông Báo", "Vui lòng chọn nhân sự cần đổi tên!")
            return

        item = selected[0]
        vals = self.tree_users.item(item, "values")
        uid = int(vals[0])

        dialog = ctk.CTkInputDialog(text=f"Nhập họ và tên mới cho ID #{uid}:", title="Đổi Tên Nhân Sự")
        new_name = dialog.get_input()

        if new_name and new_name.strip():
            self.users[uid] = new_name.strip()
            save_users(self.users)
            self.reload_users_table()
            self.attendance_df.loc[self.attendance_df["ID"] == uid, "Name"] = new_name.strip()
            self.attendance_df.to_csv(ATTENDANCE_FILE, index=False, encoding="utf-8")
            self._refresh_live_feed()
            messagebox.showinfo("Thành Công", f"Đã cập nhật tên thành '{new_name.strip()}'!")

    def delete_selected_user(self):
        selected = self.tree_users.selection()
        if not selected:
            messagebox.showwarning("Thông Báo", "Vui lòng chọn nhân sự cần xóa!")
            return

        item = selected[0]
        vals = self.tree_users.item(item, "values")
        uid = int(vals[0])
        name = vals[1]

        ans = messagebox.askyesno(
            "Xác Nhận Xóa",
            f"Bạn có chắc muốn xóa nhân sự '{name}' (ID: {uid}) khỏi hệ thống?\n"
            "Các ảnh mẫu trong 'dataset/' cũng sẽ được xóa."
        )
        if not ans:
            return

        if uid in self.users:
            del self.users[uid]
            save_users(self.users)

        if os.path.exists(DATASET_DIR):
            prefix = f"user.{uid}."
            for fname in os.listdir(DATASET_DIR):
                if fname.lower().startswith(prefix):
                    try:
                        os.remove(os.path.join(DATASET_DIR, fname))
                    except Exception:
                        pass

        self.reload_users_table()
        self._refresh_duplicate_map()
        self._update_dup_status_label()
        self.lbl_kpi_total_users.configure(text=str(len(self.users)))
        messagebox.showinfo("Đã Xóa", f"Đã xóa nhân sự '{name}' thành công!\nHãy huấn luyện lại AI để áp dụng thay đổi.")

    # =========================================================================
    # LƯU CÀI ĐẶT & TIỆN ÍCH KHÁC
    # =========================================================================
    def save_system_settings(self):
        try:
            cam_id = int(self.spin_cam_id.get())
            conf_thresh = int(self.slider_conf.get())
            cooldown = int(self.slider_cooldown.get())
            sound = self.switch_sound.get() == 1

            self.settings["camera_id"] = cam_id
            self.settings["confidence_threshold"] = conf_thresh
            self.settings["cooldown_seconds"] = cooldown
            self.settings["sound_enabled"] = sound

            save_settings(self.settings)
            self.lbl_cam_info.configure(text=f"Độ phân giải: 640x480 | Ngưỡng nhận diện: {conf_thresh}")
            messagebox.showinfo("Thành Công", "Đã lưu cài đặt hệ thống thành công!")
        except Exception as e:
            messagebox.showerror("Lỗi", f"Không thể lưu cấu hình: {e}")

    def _on_theme_changed(self, choice: str):
        mode_map = {"Tối (Dark)": "dark", "Sáng (Light)": "light", "Hệ thống (System)": "system"}
        mode = mode_map.get(choice, "dark")
        ctk.set_appearance_mode(mode)
        self.settings["appearance_mode"] = mode
        save_settings(self.settings)
        self._style_treeview()

    def _get_today_count(self) -> int:
        if self.attendance_df.empty:
            return 0
        today_str = datetime.date.today().strftime("%Y-%m-%d")
        sub = self.attendance_df[self.attendance_df["Time"].astype(str).str.startswith(today_str)]
        return sub["ID"].nunique()

    def _refresh_live_feed(self):
        """Cập nhật danh sách điểm danh gần nhất hôm nay trên Dashboard"""
        for w in self.feed_scroll_frame.winfo_children():
            w.destroy()

        if self.attendance_df.empty:
            ctk.CTkLabel(self.feed_scroll_frame, text="Chưa có ai điểm danh hôm nay", text_color="gray").pack(pady=20)
            return

        today_str = datetime.date.today().strftime("%Y-%m-%d")
        today_df = self.attendance_df[self.attendance_df["Time"].astype(str).str.startswith(today_str)]

        if today_df.empty:
            ctk.CTkLabel(self.feed_scroll_frame, text="Chưa có ai điểm danh hôm nay", text_color="gray").pack(pady=20)
            return

        recent_items = today_df.tail(8).iloc[::-1]

        for _, row in recent_items.iterrows():
            item_frame = ctk.CTkFrame(self.feed_scroll_frame, corner_radius=8, fg_color=("#f1f5f9", "#1e293b"))
            item_frame.pack(fill="x", pady=3)

            t_val = str(row["Time"]).split()[-1]

            left = ctk.CTkFrame(item_frame, fg_color="transparent")
            left.pack(side="left", padx=10, pady=6)
            ctk.CTkLabel(left, text=str(row["Name"]), font=ctk.CTkFont(size=13, weight="bold")).pack(anchor="w")
            ctk.CTkLabel(left, text=f"ID: #{row['ID']}", font=ctk.CTkFont(size=11), text_color="gray").pack(anchor="w")

            right = ctk.CTkFrame(item_frame, fg_color="transparent")
            right.pack(side="right", padx=10, pady=6)
            ctk.CTkLabel(right, text=t_val, font=ctk.CTkFont(size=12, weight="bold"), text_color="#10b981").pack(anchor="e")
            ctk.CTkLabel(right, text="Đã ghi nhận", font=ctk.CTkFont(size=10), text_color="gray").pack(anchor="e")

    def _init_realtime_clock(self):
        """Đồng hồ kỹ thuật số thời gian thực"""
        def update_clock():
            now = datetime.datetime.now()
            time_str = now.strftime("%H:%M:%S")
            days_vn = ["Thứ Hai", "Thứ Ba", "Thứ Tư", "Thứ Năm", "Thứ Sáu", "Thứ Bảy", "Chủ Nhật"]
            day_str = f"{days_vn[now.weekday()]}, {now.strftime('%d/%m/%Y')}"

            if hasattr(self, "lbl_clock_time"):
                self.lbl_clock_time.configure(text=time_str)
                self.lbl_clock_date.configure(text=day_str)

            self.after(1000, update_clock)
        update_clock()

    def on_closing(self):
        """Đóng ứng dụng và giải phóng tài nguyên an toàn"""
        self.stop_camera()
        self.destroy()


# =============================================================================
# ĐIỂM BẮT ĐẦU CHƯƠNG TRÌNH (ENTRY POINT)
# =============================================================================
if __name__ == "__main__":
    app = FaceAttendanceApp()
    app.mainloop()
