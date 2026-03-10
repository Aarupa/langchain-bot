@echo off
REM Setup script for Voice Chatbot with LangChain RAG
REM Run this file to set up the project

echo ========================================
echo Voice Chatbot Setup
echo ========================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python from https://www.python.org/downloads/
    pause
    exit /b 1
)

echo [1/4] Installing dependencies...
pip install -r requirements.txt
if errorlevel 1 (
    echo ERROR: Failed to install dependencies
    pause
    exit /b 1
)

echo.
echo [2/4] Creating necessary directories...
if not exist "documents" mkdir documents
if not exist "vector_store" mkdir vector_store
echo ✓ Directories created

echo.
echo [3/4] Setting up environment...
if not exist ".env" (
    echo Copying .env.example to .env
    copy .env.example .env
    echo ✓ .env file created - PLEASE EDIT with your OpenAI API Key
) else (
    echo ✓ .env file already exists
)

echo.
echo [4/4] Running application...
echo.
echo ========================================
echo Setup Complete!
echo ========================================
echo.
echo Next steps:
echo 1. Edit .env file and add your OpenAI API Key
echo 2. Add your documents to the 'documents' folder
echo 3. Run: streamlit run ui.py
echo.
pause
