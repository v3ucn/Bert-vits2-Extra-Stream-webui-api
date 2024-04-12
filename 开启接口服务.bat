@echo off
chcp 65001

call venv\python.exe api.py

@echo 启动完毕，请按任意键关闭
call pause