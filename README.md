# Sentiment Analysis & Feedback Intelligence Platform

CSCI 491 Sentiment Analysis Project with the goal of predicting customer sentiments and extraction of important themes using review data


## Table of Contents

- Installation
- Ollama Setup
- Usage
- Project Structure
- Dashboard Modules
- Dependencies
- Configuration

Installation

Prerequisites

- Python 3.10 or higher
- Git
- Ollama

Step 1: Clone the Repository


git clone https://github.com/YOUR_USERNAME/Sentiment-Analysis-Project.git
cd Sentiment-Analysis-Project


Step 2: Create Virtual Environment


Windows

python -m venv .venv
.venv\Scripts\activate

macOS/Linux

python3 -m venv .venv
source .venv/bin/activate


Step 3: Install Dependencies


pip install -r requirements.txt


Ollama Setup

The project uses Ollama to run a local LLM for theme extraction from customer reviews.

Install Ollama

1. Download Ollama:
   - Visit https://ollama.com/download
   - Download the installer for your operating system (Windows, macOS, or Linux)
   - Run the installer and follow the setup instructions

2. Verify Installation:
   ollama --version

Download Required Model

This project uses the Gemma2:9b model for theme extraction:
ollama pull gemma2:9b

(keep in mind, an rtx 3080 ti was used for this project, and you will likely need a similarly powerful gpu to run this project effectively.)

Start Ollama Service

Ollama runs as a background service. Ensure it's running before launching the dashboard:

Check if Ollama is running:
ollama list

If not running, start it (usually starts automatically on install)
ollama serve

Usage

Double-click `Start_Dashboard.bat` in the project root directory. This will:
1. Activate the virtual environment
2. Launch the Streamlit dashboard
3. Open your browser automatically

The dashboard will open at `http://localhost:8501`
