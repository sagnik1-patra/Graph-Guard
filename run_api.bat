@echo off
cd /d "C:\Users\sagni\Downloads\GraphGuard"
python -m pip install --upgrade pip
pip install -r requirements_api.txt
uvicorn app:app --host 0.0.0.0 --port 8000
