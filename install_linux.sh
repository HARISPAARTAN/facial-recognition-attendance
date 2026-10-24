#!/bin/bash

echo "Installing Facial Recognition Attendance System for Linux..."

# Install required system packages for dlib
echo "Installing system dependencies..."
sudo apt-get update
sudo apt-get install -y cmake build-essential libopenblas-dev liblapack-dev libx11-dev libgtk-3-dev python3-dev python3-venv

# Create a Python virtual environment
echo "Creating Python virtual environment..."
python3 -m venv venv
source venv/bin/activate

# Install dependencies in the correct order to avoid conflicts
echo "Installing numpy first to ensure correct version..."
pip install --upgrade pip
pip install wheel
pip install numpy==1.23.5

echo "Installing dlib..."
pip install dlib==19.24.1

echo "Installing face-recognition..."
pip install face-recognition==1.3.0

echo "Installing other requirements..."
pip install opencv-python==4.8.0.74
pip install pandas==2.0.3
pip install scikit-learn==1.3.0
pip install tensorflow==2.12.0
pip install matplotlib==3.7.2
pip install pillow==10.0.0
pip install tqdm==4.65.0

echo ""
echo "Installation completed! You can now run the following commands:"
echo ""
echo "First activate the virtual environment:"
echo "source venv/bin/activate"
echo ""
echo "Then run any of these commands:"
echo "To download required models:    python download_models.py"
echo "To capture face images:         python capture_faces.py"
echo "To run the attendance system:   python main.py"
echo "To test accuracy:               python test_accuracy.py" 