import sys
import platform
import os
import subprocess

def check_python_version():
    """Check if the Python version is compatible with the project."""
    python_version = sys.version_info
    print(f"Current Python version: {python_version.major}.{python_version.minor}.{python_version.micro}")
    
    if (python_version.major == 3 and 7 <= python_version.minor <= 9):
        print("✓ Python version is compatible.")
        return True
    else:
        print("✗ Python version is not compatible. Please use Python 3.7-3.9.")
        return False

def check_cmake():
    """Check if CMake is installed."""
    try:
        result = subprocess.run(['cmake', '--version'], 
                               stdout=subprocess.PIPE, 
                               stderr=subprocess.PIPE, 
                               text=True)
        if result.returncode == 0:
            cmake_version = result.stdout.split('\n')[0]
            print(f"✓ CMake is installed: {cmake_version}")
            return True
        else:
            print("✗ CMake is not installed or not in PATH.")
            return False
    except FileNotFoundError:
        print("✗ CMake is not installed or not in PATH.")
        return False

def check_system():
    """Check system information and compatibility."""
    system = platform.system()
    print(f"Operating System: {system} {platform.version()}")
    
    if system == "Windows":
        print("\nRecommended installation method for Windows:")
        print("1. Run 'install_windows.bat' script")
        print("\nAlternatively, you can use pre-built dlib wheels:")
        print("pip install https://github.com/jloh02/dlib/releases/download/v19.22/dlib-19.22.99-cp39-cp39-win_amd64.whl")
        print("(Replace cp39 with your Python version, e.g., cp37 for Python 3.7)")
    
    elif system == "Linux":
        print("\nRecommended installation method for Linux:")
        print("1. Install system dependencies: sudo apt-get install -y cmake build-essential libopenblas-dev liblapack-dev libx11-dev libgtk-3-dev")
        print("2. Run './install_linux.sh' script")
    
    elif system == "Darwin":  # macOS
        print("\nRecommended installation method for macOS:")
        print("1. Install cmake: brew install cmake")
        print("2. Create a virtual environment and install dependencies:")
        print("   python3 -m venv venv")
        print("   source venv/bin/activate")
        print("   pip install numpy==1.23.5")
        print("   pip install -r requirements.txt")
    
    else:
        print(f"Unsupported operating system: {system}")

def check_directories():
    """Check if required directories exist, create them if not."""
    directories = [
        "data",
        "data/images",
        "data/attendance",
        "models"
    ]
    
    print("\nChecking required directories:")
    for directory in directories:
        if os.path.exists(directory):
            print(f"✓ {directory} exists")
        else:
            print(f"✗ {directory} does not exist - creating...")
            os.makedirs(directory, exist_ok=True)

def main():
    """Main function to check the system setup."""
    print("=" * 60)
    print("Facial Recognition Attendance System - Setup Check")
    print("=" * 60)
    
    # Check Python version
    python_ok = check_python_version()
    
    # Check system
    print("\n" + "=" * 60)
    check_system()
    
    # Check if cmake is installed
    print("\n" + "=" * 60)
    print("Checking CMake (required for dlib):")
    cmake_ok = check_cmake()
    
    # Check directories
    print("\n" + "=" * 60)
    check_directories()
    
    # Summary
    print("\n" + "=" * 60)
    print("Summary:")
    if python_ok:
        print("✓ Python version is compatible")
    else:
        print("✗ Python version is not compatible - please use Python 3.7-3.9")
    
    if cmake_ok:
        print("✓ CMake is installed")
    else:
        print("✗ CMake is not installed - required for building dlib from source")
        print("  If using Windows, you can use pre-built wheels instead (see above)")
    
    print("\nNext steps:")
    print("1. Install dependencies (use the appropriate installation script for your OS)")
    print("2. Download required models: python download_models.py")
    print("3. Capture face images: python capture_faces.py")
    print("4. Run the system: python main.py")

if __name__ == "__main__":
    main() 