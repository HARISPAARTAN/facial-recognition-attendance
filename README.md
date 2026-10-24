# Facial Recognition Attendance System

A webcam-based student attendance tracker written in Python. OpenCV reads the camera and draws on the video, the `face_recognition` library (built on dlib) finds faces and matches them against a folder of reference photos, and a Tkinter window starts and stops recognition and shows the day's attendance. The first time a known face is matched in a run, the person is saved with a time stamp to a CSV file.

`simple_version/` is a fallback that does not need dlib. It matches faces with basic image processing, and `simple_version/generate_data.py` creates sample data for it.

## Status

Scripts compile under Python 3.13 (2026-10-24). Recognition was not re-run in this check-in; no accuracy figure is claimed.

The check was `python -I -m py_compile <file>` on every `.py` file, `simple_version/` included. It is a syntax check only: it does not cover installing dlib, downloading the models or sample data, the webcam, or the Tkinter windows, and the pinned dependencies were not installed under Python 3.13.

Known gaps, found by reading the code rather than running it:

- `requirements.txt` pins `tensorflow==2.12.0`, but no script imports TensorFlow.
- `process_images.py` and `download_sample_data.py` import `requests`, and `test_accuracy.py` imports `seaborn`. Neither is listed in `requirements.txt`; run `pip install requests seaborn` if one of those imports fails.

## Features

- Live webcam face detection and recognition against the reference photos in `data/images/`
- Attendance marked automatically, with a time stamp, the first time a known face is matched in a run
- Tkinter interface: start and stop recognition, reload the known faces, view and export the day's attendance
- Attendance saved as CSV files in `data/attendance/`
- Helper tools: reference-photo capture, recognition over still images, and an accuracy test script (see [Scripts](#scripts))

## Requirements

- Python 3.7 to 3.9 (the pre-built Windows dlib wheels used below are for Python 3.7, 3.8 and 3.9)
- A webcam for live recognition and photo capture
- Dependencies pinned in `requirements.txt`

## Installation

### Easy Installation (Recommended)

#### Windows:
1. Clone this repository:
```
git clone https://github.com/HARISPAARTAN/facial-recognition-attendance.git
cd facial-recognition-attendance
```

2. Run the Windows installation script:
```
install_windows.bat
```

#### Linux:
1. Clone this repository:
```
git clone https://github.com/HARISPAARTAN/facial-recognition-attendance.git
cd facial-recognition-attendance
```

2. Make the installation script executable and run it:
```
chmod +x install_linux.sh
./install_linux.sh
```

### Manual Installation

1. Clone this repository:
```
git clone https://github.com/HARISPAARTAN/facial-recognition-attendance.git
cd facial-recognition-attendance
```

2. Create and activate a virtual environment (recommended):
```
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

3. Install dependencies:

For Windows users with installation issues:
```
# Install numpy first
pip install numpy==1.23.5

# Install pre-built dlib (choose the one matching your Python version)
# For Python 3.9:
pip install https://github.com/jloh02/dlib/releases/download/v19.22/dlib-19.22.99-cp39-cp39-win_amd64.whl
# For Python 3.8:
pip install https://github.com/jloh02/dlib/releases/download/v19.22/dlib-19.22.99-cp38-cp38-win_amd64.whl
# For Python 3.7:
pip install https://github.com/jloh02/dlib/releases/download/v19.22/dlib-19.22.99-cp37-cp37m-win_amd64.whl

# Then install other dependencies
pip install face-recognition==1.3.0
pip install opencv-python==4.8.0.74
pip install pandas==2.0.3
pip install scikit-learn==1.3.0
pip install tensorflow==2.12.0
pip install matplotlib==3.7.2
pip install pillow==10.0.0
pip install tqdm==4.65.0
```

For Linux/Mac users:
```
# Install system requirements for dlib
sudo apt-get update
sudo apt-get install -y cmake build-essential libopenblas-dev liblapack-dev libx11-dev libgtk-3-dev

# Install dependencies
pip install -r requirements.txt
```

4. Download the dlib model files (details under [Models and sample data](#models-and-sample-data)):
```
python download_models.py
```

## Models and sample data

**Models.** The dlib model files are large, so they are not stored in this repository (`models/` holds only a `.gitkeep`). `python download_models.py` downloads the shape-predictor and face-recognition ResNet files from [davisking/dlib-models](https://github.com/davisking/dlib-models) into `models/`. On Windows it extracts the `.bz2` archives with 7-Zip if `7z` is on your PATH; on Linux and macOS it uses `bzip2`. `main.py` creates the `models/` folder but does not load files from it. The models are third-party files with their own licence terms, so check those before any commercial use.

**Sample data.** `python download_sample_data.py` fills `data/images/` (known faces) and `data/test_images/`. It first tries to download a sample set from GitHub and, if that fails, generates placeholder pictures locally: a coloured square with a drawn smiley and a name. `python download_sample_data.py --generate` skips the download. The generated pictures are drawings, not photographs, and this check-in did not test whether the face detector finds a face in them. The download links in `download_sample_data.py` and `process_images.py` were not tested either.

## Usage

1. Add student face images to the `data/images` folder:
   - Take clear photos of student faces
   - Name the images as `FirstName_LastName.jpg`
   - One face per image is recommended
   - You can use the provided capture tool: `python capture_faces.py` (it saves `Name_1.jpg`, `Name_2.jpg`, and so on)
   - The name shown on screen and written to the CSV is the image's file name without its extension

2. Run the application:
```
python main.py
```

3. Using the application:
   - Click "Start Recognition" to begin face detection
   - The system will automatically mark attendance when a known face is detected
   - Click "View Attendance" to see today's attendance records; its "Export" button saves a copy as `data/attendance/export_<date>.csv`
   - Attendance records are stored in the `data/attendance` folder as CSV files

Reference photos, test photos and attendance records are personal data. `data/images/`, `data/test_images/` and `data/attendance/` are git-ignored so that they are not committed by accident.

## Scripts

| File | What it does |
|---|---|
| `main.py` | The application: a Tkinter window with live recognition, today's attendance table, and view and export |
| `capture_faces.py` | Tkinter webcam tool that saves reference photos to `data/images/` |
| `process_images.py` | Runs recognition over still images (`--dir`, `--known`, `--gui`); annotated copies go to `data/results/` and attendance to `data/attendance/` |
| `test_accuracy.py` | Compares recognition of `data/test_images/` against `data/images/`, taking the true names from the file names, and writes tables and plots to `test_results/`. Run it on your own photos to get a figure; none is claimed here |
| `check_setup.py` | Prints Python, CMake and operating-system install hints and creates the folders the project needs |
| `download_models.py` | Downloads the dlib model files, see above |
| `download_sample_data.py` | Downloads or generates sample images, see above |
| `install_windows.bat`, `install_linux.sh` | Installation scripts, see Installation |
| `fix_dependencies.sh` | Linux helper that reinstalls dlib (built from source) and face_recognition; it also rewrites the files in `simple_version/` |
| `simple_version/` | The fallback without dlib; see its own README |

## Troubleshooting

### Common Installation Issues

1. **dlib installation fails**: 
   - For Windows: Use the pre-built wheels as shown in the Windows installation steps
   - For Linux: Make sure you have cmake and build-essential installed
   - For Mac: `brew install cmake` before installing dlib

2. **Dependency conflicts**:
   - Use the provided installation scripts
   - Make sure to install numpy==1.23.5 before other packages
   - Install dlib before face-recognition

3. **Other issues**:
   - Make sure you have the correct Python version (3.7-3.9)
   - Use a virtual environment to avoid conflicts with other projects

## Directory Structure

```
facial-recognition-attendance/
├── data/
│   ├── images/              # Reference photos (git-ignored, .gitkeep only)
│   ├── test_images/         # Test photos (git-ignored, .gitkeep only)
│   └── attendance/          # Attendance CSVs, created on first run (git-ignored)
├── models/                  # Model files from download_models.py (git-ignored, .gitkeep only)
├── simple_version/          # Fallback without dlib
│   ├── generate_data.py
│   ├── simple_recognition.py
│   └── README.md
├── main.py                  # Main application
├── capture_faces.py         # Tool to capture face images
├── process_images.py        # Recognition over still images
├── test_accuracy.py         # Script to test recognition accuracy
├── check_setup.py           # Environment check
├── download_models.py       # Script to download the model files
├── download_sample_data.py  # Script to download or generate sample images
├── requirements.txt         # Dependencies list
├── install_windows.bat      # Windows installation script
├── install_linux.sh         # Linux installation script
├── fix_dependencies.sh      # Linux dlib and face_recognition reinstall helper
└── LICENSE                  # MIT licence
```

## How It Works

1. The system loads the known faces from `data/images`: one encoding per image, named after the file
2. OpenCV reads frames from the webcam and shrinks each one before detection
3. The `face_recognition` library (dlib) finds the faces in the frame and computes an encoding for each
4. Each encoding is compared with the known encodings: the closest known face supplies the name if it is within a distance tolerance set in `main.py`, otherwise the face is labelled "Unknown"
5. The first time a name is matched in a run, the time is recorded
6. The records are written to `data/attendance/<YYYY-MM-DD>.csv` (columns `Name` and `Time`); the day's file is rewritten from the current run's records each time a new name is added

## Tips for Reference Photos

- Use clear, well-lit photos in which the face is fully visible and not obscured
- Use one face per image; only the first face found in an image is used

## License

[MIT License](LICENSE)

---

This project was developed using:
- OpenCV for the camera, resizing and drawing
- face_recognition (based on dlib) for face detection and matching
- pandas for the CSV files
- Pillow for showing frames in Tkinter
- Tkinter for the GUI
