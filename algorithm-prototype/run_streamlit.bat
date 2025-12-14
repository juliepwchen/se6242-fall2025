@echo off
REM Run Streamlit app for Yelp Algorithm Prototype
title Yelp Algorithm Prototype - Streamlit UI

echo 🚀 Starting Streamlit UI for Yelp Algorithm Prototype...

REM Check or create virtual environment
if not exist .venv (
    echo 🔧 Creating virtual environment (.venv)...
    python -m venv .venv
)

REM Activate environment
call .venv\Scripts\activate

REM Install dependencies if needed
echo 📦 Installing dependencies...
pip install -r requirements.txt --quiet

REM Run Streamlit
echo 🌐 Launching app at http://localhost:8501 ...
streamlit run app_streamlit.py

pause
