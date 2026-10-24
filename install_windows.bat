@echo off
echo Installing Facial Recognition Attendance System for Windows...

:: Create a Python virtual environment
echo Creating Python virtual environment...
python -m venv venv
call venv\Scripts\activate.bat

:: Install numpy first to ensure correct version
echo Installing numpy...
pip install numpy==1.23.5

:: Install pre-built dlib for Windows (supporting common Python versions)
echo Installing pre-built dlib...
pip install https://github.com/jloh02/dlib/releases/download/v19.22/dlib-19.22.99-cp39-cp39-win_amd64.whl

:: If the above fails, try a different Python version (3.8, 3.7)
if %ERRORLEVEL% NEQ 0 (
    echo Trying dlib for Python 3.8...
    pip install https://github.com/jloh02/dlib/releases/download/v19.22/dlib-19.22.99-cp38-cp38-win_amd64.whl
)

if %ERRORLEVEL% NEQ 0 (
    echo Trying dlib for Python 3.7...
    pip install https://github.com/jloh02/dlib/releases/download/v19.22/dlib-19.22.99-cp37-cp37m-win_amd64.whl
)

:: Install face-recognition after dlib
echo Installing face-recognition...
pip install face-recognition==1.3.0

:: Install other requirements
echo Installing other requirements...
pip install opencv-python==4.8.0.74
pip install pandas==2.0.3
pip install scikit-learn==1.3.0
pip install tensorflow==2.12.0
pip install matplotlib==3.7.2
pip install pillow==10.0.0
pip install tqdm==4.65.0

echo.
echo Installation completed! You can now run the following commands:
echo.
echo To download required models:    python download_models.py
echo To capture face images:         python capture_faces.py
echo To run the attendance system:   python main.py
echo To test accuracy:               python test_accuracy.py
echo.
echo Press any key to exit...
pause > nul 