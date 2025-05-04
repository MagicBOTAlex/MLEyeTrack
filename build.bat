@echo off
call conda activate et || exit /b

REM Make sure PyInstaller is installed
pip install pyinstaller || exit /b

REM Build using your .spec file
pyinstaller MLEyetrack.spec -y || exit /b

REM Copy models folder
xcopy /E /I /Y models dist\MLEyetrack\models

REM Copy Settings.json
copy /Y Settings.json dist\MLEyetrack\

echo.
echo Build complete. Files copied to dist\MLEyetrack
pause
