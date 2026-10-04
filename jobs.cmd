@echo off
rem Run the assistant for one person:  jobs.cmd <profile> <command> [options]
rem Examples:  jobs.cmd novia setup        jobs.cmd novia postular        jobs.cmd novia index
if "%~1"=="" (
  echo Usage: jobs.cmd ^<profile^> ^<command^> [options]
  exit /b 2
)
set "JOB_ASSISTANT_PROFILE=%~1"
shift
"%~dp0.venv\Scripts\python.exe" -m job_assistant %1 %2 %3 %4 %5 %6 %7 %8 %9
