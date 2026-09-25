@echo off
chcp 65001 >nul
title HE THONG DIEM DANH KHUON MAT - SMART ATTENDANCE PRO v2.1
color 0b

:: Chuyển về đúng thư mục chứa file bat
cd /d "%~dp0"

echo =====================================================================
echo    SMART FACE ATTENDANCE SYSTEM PRO v2.0
echo    He Thong Diem Danh Khuon Mat Thong Minh
echo =====================================================================
echo.
echo [*] Dang kiem tra moi truong Python...

where python >nul 2>nul
if %errorlevel% neq 0 (
    echo [LOI] Khong tim thay Python tren he thong!
    echo Vui long cai dat Python va tick chon "Add Python to PATH".
    echo.
    pause
    exit /b 1
)

echo [*] Dang khoi dong ung dung...
echo.

python main_app.py

if %errorlevel% neq 0 (
    echo.
    echo =====================================================================
    echo [THONG BAO] Chuong trinh da dung lai hoac gap su co.
    echo =====================================================================
    pause
)
