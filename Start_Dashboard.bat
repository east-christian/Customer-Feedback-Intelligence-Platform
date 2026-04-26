@echo off
TITLE Sentiment Analysis Dashboard Launcher
echo Starting Feedback Intelligence Platform...

:: Navigate to the script's directory
cd /d "%~dp0"

:: Activate the virtual environment
echo Activating virtual environment...
call .venv\Scripts\activate.bat

:: Run the Streamlit application
echo Launching Dashboard in your browser...
streamlit run src\scripts\main.py

pause
