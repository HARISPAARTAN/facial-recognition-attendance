import os
import urllib.request
import platform
import zipfile
import tarfile
import subprocess
import sys

def download_progress(count, block_size, total_size):
    percent = int(count * block_size * 100 / total_size)
    sys.stdout.write(f"\r...{percent}% downloaded")
    sys.stdout.flush()

def main():
    print("Downloading face recognition models and dependencies...")
    
    # Create models directory if it doesn't exist
    os.makedirs("models", exist_ok=True)
    
    # Download face recognition model (dlib shape predictor)
    print("\nDownloading dlib face shape predictor model (this may take a while)...")
    shape_predictor_url = "https://github.com/davisking/dlib-models/raw/master/shape_predictor_68_face_landmarks_GTX.dat.bz2"
    shape_predictor_path = "models/shape_predictor_68_face_landmarks.dat.bz2"
    
    try:
        if not os.path.exists(shape_predictor_path):
            urllib.request.urlretrieve(shape_predictor_url, shape_predictor_path, download_progress)
            print("\nShape predictor model downloaded successfully!")
            
            # Extract the model
            print("Extracting model...")
            # Use appropriate extraction command based on platform
            if platform.system() == "Windows":
                # On Windows, we can use 7-Zip if available
                try:
                    subprocess.run(['7z', 'e', shape_predictor_path, '-omodels', '-y'], check=True)
                    print("Extraction successful using 7-Zip!")
                except (subprocess.SubprocessError, FileNotFoundError):
                    print("7-Zip not found. Please manually extract the downloaded file.")
                    print(f"Extract {shape_predictor_path} to the models directory.")
            else:
                # On Linux/Mac, use bzip2
                subprocess.run(['bzip2', '-dk', shape_predictor_path], check=True)
                print("Extraction successful!")
        else:
            print("Shape predictor model already exists. Skipping download.")
    except Exception as e:
        print(f"\nError downloading shape predictor model: {str(e)}")
        print("You can manually download it from:")
        print(shape_predictor_url)
        print(f"And save it to: {shape_predictor_path}")
        print("Then extract it using an appropriate tool.")
    
    # Download face recognition model (dlib face recognition resnet model)
    print("\nDownloading dlib face recognition model (this may take a while)...")
    face_rec_url = "https://github.com/davisking/dlib-models/raw/master/dlib_face_recognition_resnet_model_v1.dat.bz2"
    face_rec_path = "models/dlib_face_recognition_resnet_model_v1.dat.bz2"
    
    try:
        if not os.path.exists(face_rec_path):
            urllib.request.urlretrieve(face_rec_url, face_rec_path, download_progress)
            print("\nFace recognition model downloaded successfully!")
            
            # Extract the model
            print("Extracting model...")
            # Use appropriate extraction command based on platform
            if platform.system() == "Windows":
                # On Windows, we can use 7-Zip if available
                try:
                    subprocess.run(['7z', 'e', face_rec_path, '-omodels', '-y'], check=True)
                    print("Extraction successful using 7-Zip!")
                except (subprocess.SubprocessError, FileNotFoundError):
                    print("7-Zip not found. Please manually extract the downloaded file.")
                    print(f"Extract {face_rec_path} to the models directory.")
            else:
                # On Linux/Mac, use bzip2
                subprocess.run(['bzip2', '-dk', face_rec_path], check=True)
                print("Extraction successful!")
        else:
            print("Face recognition model already exists. Skipping download.")
    except Exception as e:
        print(f"\nError downloading face recognition model: {str(e)}")
        print("You can manually download it from:")
        print(face_rec_url)
        print(f"And save it to: {face_rec_path}")
        print("Then extract it using an appropriate tool.")
    
    print("\nAlternative method to get these models:")
    print("1. Install dlib and face_recognition packages according to their documentation")
    print("2. The models will be downloaded automatically when first using the libraries")
    
    print("\nNeeded dependencies and installation instructions:")
    print("1. OpenCV: pip install opencv-python")
    print("2. NumPy: pip install numpy")
    print("3. dlib: pip install dlib (may require C++ build tools)")
    print("4. face_recognition: pip install face-recognition")
    print("5. pandas: pip install pandas")
    print("6. Pillow: pip install pillow")
    
    print("\nAll done! Now run 'pip install -r requirements.txt' to install required Python packages.")
    print("Then run 'python main.py' to start the facial recognition system.")

if __name__ == "__main__":
    main() 