@echo off
chcp 65001 >nul
cd /d "%~dp0"
echo ============================================================
echo   人脸考勤系统 - 启动中...
echo ============================================================
call .\venv\Scripts\activate.bat
python app.py
pause
