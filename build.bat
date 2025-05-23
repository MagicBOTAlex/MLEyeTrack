@echo off
call conda activate et || exit /b

REM Make sure PyInstaller is installed
pip install pyinstaller || exit /b

@REM del /s /q .\build\*
@REM rmdir /s /q .\build\

REM Build using your .spec file
pyinstaller MLEyetrack.spec -y || exit /b

REM Copy models folder
xcopy /E /I /Y models dist\MLEyetrack\models

REM Copy Settings.json
copy /Y Settings.json dist\MLEyetrack\

REM Delete existing ZIP if it exists
if exist dist\MLEyetrack.zip del /f /q dist\MLEyetrack.zip

REM Try to use 7-Zip if available
where 7z >nul 2>&1
if %errorlevel%==0 (
    echo 7-Zip found. Compressing with 7-Zip...
    7z a -tzip dist\MLEyetrack.zip .\dist\MLEyetrack\* -mx=1
) else (
    echo 7-Zip not found. Falling back to PowerShell compression...
    powershell -Command "Compress-Archive -Path 'dist\MLEyetrack\*' -DestinationPath 'dist\MLEyetrack.zip' -CompressionLevel Optimal"
)

echo.
echo Build complete. Files copied to dist\MLEyetrack and compressed to dist\MLEyetrack.zip
pause
