@echo off
setlocal

set "ROOT_DIR=%~dp0"
cd /d "%ROOT_DIR%"

if not exist ".venv\Scripts\python.exe" (
  echo [ERROR] 未找到虚拟环境: .venv\Scripts\python.exe
  echo 请先在项目根目录创建并安装依赖，然后再运行此脚本。
  pause
  exit /b 1
)

start "SDM Console" "http://127.0.0.1:8000/console"
echo 正在启动 SDM 服务...
echo 启动后请访问 http://127.0.0.1:8000/console
".venv\Scripts\python.exe" -m uvicorn apps.langserve_chat:app --host 0.0.0.0 --port 8000 --reload

echo.
echo 服务已退出。
pause