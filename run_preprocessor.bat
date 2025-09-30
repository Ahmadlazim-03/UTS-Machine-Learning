@echo off
echo ===============================================
echo   Cervical Cancer Data Preprocessing Toolkit
echo ===============================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo Error: Python tidak ditemukan. Pastikan Python sudah terinstall.
    pause
    exit /b 1
)

REM Check if CSV file exists
if not exist "risk_factors_cervical_cancer.csv" (
    echo Error: File 'risk_factors_cervical_cancer.csv' tidak ditemukan.
    echo Pastikan file CSV ada di direktori yang sama dengan script ini.
    pause
    exit /b 1
)

REM Install dependencies if needed
echo Checking dependencies...
pip install -r requirements.txt >nul 2>&1
if errorlevel 1 (
    echo Warning: Gagal install dependencies. Silakan install manual dengan:
    echo pip install -r requirements.txt
    echo.
)

echo.
echo Menjalankan Cervical Cancer Data Preprocessing Toolkit...
echo.

REM Run the main program
python main_preprocessor.py

echo.
echo Program selesai. Tekan tombol apapun untuk keluar...
pause >nul