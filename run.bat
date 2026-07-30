@echo off
title Auto Video Translation & Dubbing
echo ===================================================
echo   Dang kich hoat moi truong ao va khoi chay AI...
echo ===================================================

:: 1. Kiểm tra và kích hoạt môi trường ảo (tự động thử venv hoặc .venv)
if exist "venv\Scripts\activate.bat" (
    call venv\Scripts\activate.bat
) else if exist ".venv\Scripts\activate.bat" (
    call .venv\Scripts\activate.bat
) else (
    echo [LOI] Khong tim thay thu muc venv hoac .venv!
    echo Vui long kiem tra lai ten thu muc moi truong ao cua ban.
    pause
    exit /b
)

:: 2. Khởi chạy chương trình main.py
python main.py

:: 3. Giữ cửa sổ cmd không bị đóng ngay khi chương trình kết thúc
pause