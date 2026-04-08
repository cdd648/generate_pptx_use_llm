@echo off
chcp 65001 >nul
title 信息图转PPTX生成器

echo ==========================================
echo    信息图转 PPTX 生成器
echo ==========================================
echo.

REM 检查 Python 是否安装
python --version >nul 2>&1
if errorlevel 1 (
    echo [错误] 未检测到 Python，请先安装 Python 3.8+
    echo 下载地址: https://www.python.org/downloads/
    pause
    exit /b 1
)

echo [1/3] Python 版本:
python --version
echo.

REM 检查虚拟环境
if exist venv\Scripts\activate.bat (
    echo [2/3] 激活虚拟环境...
    call venv\Scripts\activate.bat
) else (
    echo [2/3] 未检测到虚拟环境，使用系统 Python
)
echo.

echo [3/3] 启动 Streamlit 服务...
echo.
echo 启动成功后，浏览器会自动打开应用
echo 如未自动打开，请手动访问: http://localhost:8501
echo.
echo 按 Ctrl+C 可停止服务
echo ==========================================
echo.

streamlit run app.py

if errorlevel 1 (
    echo.
    echo [错误] 启动失败，请检查依赖是否已安装
    echo 运行以下命令安装依赖:
    echo   pip install -r requirements.txt
    pause
)
