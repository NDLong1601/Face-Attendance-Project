@echo off
title CAI DAT THU VIEN - SMART ATTENDANCE PRO
color 0a

cd /d "%~dp0"

echo =====================================================================
echo    CAI DAT TU DONG THU VIEN CHO HE THONG DIEM DANH KHUON MAT
echo =====================================================================
echo.

echo [*] BUOC 1: Kiem tra Python tren he thong...
where python >nul 2>nul
if %errorlevel% neq 0 (
    color 0c
    echo [LOI] Khong tim thay Python!
    echo Vui long cai dat Python tu: https://www.python.org/downloads/
    echo Luu y: Hay tich chon "Add Python to PATH" khi cai dat.
    echo.
    pause
    exit /b 1
)

python --version
echo [OK] Python da san sang.
echo.

echo [*] BUOC 2: Kiem tra cong cu quan ly goi pip...
python -m pip --version >nul 2>nul
if %errorlevel% neq 0 (
    color 0c
    echo [LOI] Khong tim thay pip!
    pause
    exit /b 1
)
echo [OK] pip da san sang.
echo.

echo [*] BUOC 3: Dang cai dat cac thu vien tu requirements.txt...
echo Cac goi: opencv-contrib-python, customtkinter, pillow, pandas, numpy, openpyxl
echo Qua trinh nay co the mat 1 - 3 phut tuy toc do mang...
echo.

python -m pip install -r requirements.txt

if %errorlevel% neq 0 (
    echo.
    color 0e
    echo [CANH BAO] Thu cai dat truc tiep cac goi...
    python -m pip install opencv-contrib-python customtkinter pillow pandas numpy openpyxl
)

echo.
echo [*] BUOC 4: Kiem tra tinh san sang cua cac thu vien...
python -c "import cv2, cv2.face, customtkinter, PIL, pandas, numpy, openpyxl; print('[OK] Kiem tra tat ca thu vien thanh cong!')"

if %errorlevel% neq 0 (
    echo.
    color 0c
    echo =====================================================================
    echo [LOI] Chua cai dat day du cac thu vien. Vui long kiem tra ket noi mang!
    echo =====================================================================
    echo.
    pause
    exit /b 1
)

echo.
color 0b
echo =====================================================================
echo [THANH CONG] TAT CA CAC THU VIEN DA DUOC CAI DAT HOAN TAT!
echo Ban co the nhap dup chuot vao file 'run.bat' de khoi dong chuong trinh.
echo =====================================================================
echo.
pause
