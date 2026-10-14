@echo off
cd /d C:\Users\felip\Agent-employment
if not exist output mkdir output
echo ==== %date% %time% >> output\daily.log
.venv\Scripts\python.exe -m job_assistant daily >> output\daily.log 2>&1
echo exit=%errorlevel% >> output\daily.log
