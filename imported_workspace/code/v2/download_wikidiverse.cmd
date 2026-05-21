@echo off
setlocal

REM Download WikiDiverse into code\v2\data\wikidiverse using the existing Python downloader.

set "SCRIPT_DIR=%~dp0"
for %%I in ("%SCRIPT_DIR%..\..") do set "REPO_ROOT=%%~fI"
set "OUTPUT_DIR=%SCRIPT_DIR%data\wikidiverse"

if not exist "%OUTPUT_DIR%" mkdir "%OUTPUT_DIR%"

echo Downloading WikiDiverse to:
echo   %OUTPUT_DIR%
echo.

python "%REPO_ROOT%\code\v1\download_wikidiverse.py" --output-dir "%OUTPUT_DIR%" %*

endlocal
