@echo off

echo Creating virtual environment if it does not exist...
IF NOT EXIST venv (
    python -m venv venv
)

echo Activating virtual environment...
call venv\Scripts\activate

echo Installing requirements...
pip install --upgrade pip
pip install -r requirements.txt

echo Running main.py...
python main.py

pause
