# Cyber Threat Detection System

This repository contains a Streamlit app for training and testing ensemble machine-learning models to detect network threats.

Quick start

1. Create and activate a Python virtual environment (Windows PowerShell):

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

2. Run the app:

```powershell
.venv\Scripts\streamlit.exe run app.py
```

Notes
- Large files (models, datasets) are excluded via `.gitignore`. Use external storage or Git LFS for large assets.
- If you deploy to Streamlit Cloud, include `requirements.txt`. Consider removing `tensorflow` to speed up deploys unless required.
