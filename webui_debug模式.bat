@echo off
chcp 65001

call venv\Scripts\gradio.exe webui.py

@echo 启动完毕，请按任意键关闭
call pause