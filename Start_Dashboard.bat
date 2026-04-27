@echo off
TITLE Sentiment Analysis Dashboard Launcher
echo Starting Feedback Intelligence Platform...

:: finds the correct directory
cd /d "%~dp0"

:: activates virtual environment
echo Activating virtual environment...
call .venv\Scripts\activate.bat

:: starts streamlit
echo Launching Dashboard in your browser...
streamlit run src\scripts\main.py

pause
