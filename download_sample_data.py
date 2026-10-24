import os
import requests
import argparse
from zipfile import ZipFile
from tqdm import tqdm
import shutil
import sys
import tempfile
import random
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import cv2
import time

def download_file(url, output_path):
    """Download a file with progress bar"""
    print(f"Downloading from {url}...")
    
    # Stream the download to show progress
    try:
        # For Dropbox URLs, ensure we're getting the direct download
        if 'dropbox.com' in url and 'dl=1' not in url:
            url = url + ('&' if '?' in url else '?') + 'dl=1'
        
        response = requests.get(url, stream=True, headers={'User-Agent': 'Mozilla/5.0'})
        response.raise_for_status()  # Raise an exception for 4XX/5XX responses
        
        # Check if we got an HTML response instead of a zip file (common with Dropbox)
        content_type = response.headers.get('Content-Type', '')
        if 'text/html' in content_type:
            print(f"Error: Received HTML response instead of a file. URL may be invalid or requires authentication.")
            return False
        
        total_size_in_bytes = int(response.headers.get('content-length', 0))
        block_size = 1024
        progress_bar = tqdm(total=total_size_in_bytes, unit='iB', unit_scale=True)
        
        with open(output_path, 'wb') as file:
            for data in response.iter_content(block_size):
                progress_bar.update(len(data))
                file.write(data)
        
        progress_bar.close()
        
        if total_size_in_bytes != 0 and progress_bar.n != total_size_in_bytes:
            print("ERROR: Download incomplete")
            return False
        
        # Verify the file is a valid zip file
        if not is_valid_zip(output_path):
            print("ERROR: Downloaded file is not a valid zip file")
            return False
            
        return True
        
    except requests.exceptions.RequestException as e:
        print(f"ERROR: Download failed: {str(e)}")
        return False

def is_valid_zip(file_path):
    """Check if a file is a valid zip file"""
    try:
        with ZipFile(file_path, 'r') as zipf:
            return True
    except:
        return False

def extract_zip(zip_path, extract_to):
    """Extract a zip file"""
    print(f"Extracting {zip_path} to {extract_to}...")
    
    try:
        with ZipFile(zip_path, 'r') as zip_ref:
            # Get total number of files for progress bar
            total_files = len(zip_ref.infolist())
            
            # Extract all files with progress bar
            for i, file in enumerate(zip_ref.infolist()):
                zip_ref.extract(file, extract_to)
                # Update progress
                sys.stdout.write(f"\rExtracting: {i+1}/{total_files} files")
                sys.stdout.flush()
                
        print("\nExtraction completed!")
        return True
    except Exception as e:
        print(f"Error extracting file: {str(e)}")
        return False

def download_known_faces():
    """Download sample known faces for training"""
    known_faces_dir = "data/images"
    os.makedirs(known_faces_dir, exist_ok=True)
    
    # URLs for sample known faces (properly named for training)
    # Using GitHub as it tends to be more reliable for direct downloads
    urls = [
        "https://github.com/murtazahassan/Face-Recognition/raw/main/training_dataset.zip",
        "https://github.com/opencv/opencv/raw/master/data/haarcascades/haarcascade_frontalface_default.xml"
    ]
    
    # Try each URL until one works
    for url in urls:
        # Download to a temporary file first
        with tempfile.NamedTemporaryFile(delete=False, suffix='.zip') as tmp_file:
            temp_path = tmp_file.name
        
        if download_file(url, temp_path):
            # Move to final location if download succeeds
            zip_path = os.path.join(known_faces_dir, "sample_known_faces.zip")
            shutil.move(temp_path, zip_path)
            
            # Extract files
            if extract_zip(zip_path, known_faces_dir):
                # Remove zip file
                os.remove(zip_path)
                
                # Count downloaded files
                face_files = [f for f in os.listdir(known_faces_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
                print(f"Downloaded {len(face_files)} sample known faces to {known_faces_dir}")
                print("These faces are named for immediate use with the facial recognition system.")
                
                return True
            else:
                # Try next URL if extraction fails
                continue
        else:
            # Remove the temp file and try the next URL
            if os.path.exists(temp_path):
                os.remove(temp_path)
    
    # If we reach here, all URLs failed
    print("WARNING: Failed to download known faces from online sources.")
    print("Generating local sample faces instead...")
    
    # Generate local sample faces as fallback
    return generate_sample_faces()

def download_test_images(dataset_type="mixed"):
    """Download test images for recognition"""
    test_images_dir = "data/test_images"
    os.makedirs(test_images_dir, exist_ok=True)
    
    # URLs for different datasets (GitHub, more reliable)
    urls = {
        "lfw": "https://github.com/murtazahassan/Face-Recognition/raw/main/training_dataset.zip",
        "celeba": "https://github.com/murtazahassan/Face-Recognition/raw/main/training_dataset.zip",
        "mixed": "https://github.com/murtazahassan/Face-Recognition/raw/main/training_dataset.zip"
    }
    
    # Try the primary URL
    with tempfile.NamedTemporaryFile(delete=False, suffix='.zip') as tmp_file:
        temp_path = tmp_file.name
    
    # Try primary URL
    url = urls[dataset_type]
    if not download_file(url, temp_path):
        # If download fails, generate local test images
        if os.path.exists(temp_path):
            os.remove(temp_path)
        
        print("Download failed, generating local test images...")
        return generate_sample_test_images()
    
    # Move to final location
    zip_path = os.path.join(test_images_dir, f"{dataset_type}-sample.zip")
    shutil.move(temp_path, zip_path)
    
    # Extract files
    if not extract_zip(zip_path, test_images_dir):
        return generate_sample_test_images()
    
    # Remove zip file
    os.remove(zip_path)
    
    # Count downloaded files
    test_files = [f for f in os.listdir(test_images_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
    print(f"Downloaded {len(test_files)} test images to {test_images_dir}")
    
    return True

def generate_sample_faces():
    """Generate sample known faces locally"""
    print("Generating sample known faces locally...")
    
    known_faces_dir = "data/images"
    os.makedirs(known_faces_dir, exist_ok=True)
    
    # List of sample names
    names = [
        "John_Smith",
        "Emma_Johnson",
        "Michael_Williams",
        "Sarah_Brown",
        "David_Jones",
        "Emily_Davis",
        "James_Miller",
        "Olivia_Wilson",
        "Robert_Taylor",
        "Sophia_Anderson"
    ]
    
    face_count = 0
    
    # Create a simple colored rectangle for each name with the name written on it
    for name in names:
        # Create a 400x400 image with a random color
        img = Image.new('RGB', (400, 400), color=(
            random.randint(150, 250),
            random.randint(150, 250),
            random.randint(150, 250)
        ))
        
        # Draw the name on the image
        draw = ImageDraw.Draw(img)
        # Try to use a system font, fallback to default
        try:
            # Try a common font
            font = ImageFont.truetype("arial.ttf", 40)
        except IOError:
            try:
                # Try a different common font
                font = ImageFont.truetype("DejaVuSans.ttf", 40)
            except IOError:
                # Use default font
                font = ImageFont.load_default()
        
        # Draw text in the center
        draw.text((50, 150), name.replace("_", " "), fill="black", font=font)
        draw.text((50, 200), "Sample Face", fill="black", font=font)
        
        # Draw a simple face outline
        draw.ellipse((150, 50, 250, 150), outline="black", width=2)  # Face
        draw.ellipse((175, 80, 185, 90), outline="black", width=2)  # Left eye
        draw.ellipse((215, 80, 225, 90), outline="black", width=2)  # Right eye
        draw.arc((175, 90, 225, 120), 0, 180, fill="black", width=2)  # Smile
        
        # Save the image
        output_path = os.path.join(known_faces_dir, f"{name}.jpg")
        img.save(output_path)
        face_count += 1
        print(f"Generated {name}.jpg")
    
    print(f"Successfully generated {face_count} sample faces")
    return True

def generate_sample_test_images():
    """Generate sample test images locally"""
    print("Generating sample test images locally...")
    
    test_images_dir = "data/test_images"
    os.makedirs(test_images_dir, exist_ok=True)
    
    # Get the list of names from the known faces directory
    known_faces_dir = "data/images"
    known_names = []
    
    # If known faces exist, use their names
    if os.path.exists(known_faces_dir):
        for filename in os.listdir(known_faces_dir):
            if filename.endswith(('.jpg', '.jpeg', '.png')):
                name = os.path.splitext(filename)[0]
                known_names.append(name)
    
    # If no known faces, use default names
    if not known_names:
        known_names = [
            "John_Smith",
            "Emma_Johnson",
            "Michael_Williams",
            "Sarah_Brown",
            "David_Jones"
        ]
    
    test_count = 0
    
    # Create test images - some with known names, some as "Unknown"
    for i in range(20):  # Generate 20 test images
        # Select a name, 70% chance of using a known name
        if random.random() < 0.7 and known_names:
            name = random.choice(known_names)
        else:
            name = "Unknown_Person"
        
        # Create a 500x500 image with a different random color
        img = Image.new('RGB', (500, 500), color=(
            random.randint(100, 200),
            random.randint(100, 200),
            random.randint(100, 200)
        ))
        
        # Draw the name on the image
        draw = ImageDraw.Draw(img)
        # Try to use a system font, fallback to default
        try:
            font = ImageFont.truetype("arial.ttf", 30)
        except IOError:
            try:
                font = ImageFont.truetype("DejaVuSans.ttf", 30)
            except IOError:
                font = ImageFont.load_default()
        
        # Add random noise to make each image different
        draw.text((50 + random.randint(-20, 20), 200 + random.randint(-20, 20)), 
                  name.replace("_", " "), 
                  fill="black", 
                  font=font)
        
        # Draw a simple face with slight variations
        center_x, center_y = 250 + random.randint(-20, 20), 150 + random.randint(-20, 20)
        size = 120 + random.randint(-20, 20)
        
        # Face
        draw.ellipse((center_x - size//2, center_y - size//2, 
                      center_x + size//2, center_y + size//2), 
                     outline="black", width=2)
        
        # Eyes
        eye_size = size // 8
        left_eye_x = center_x - size//4
        right_eye_x = center_x + size//4
        eye_y = center_y - size//8
        
        draw.ellipse((left_eye_x - eye_size, eye_y - eye_size, 
                      left_eye_x + eye_size, eye_y + eye_size), 
                     outline="black", width=2)
        
        draw.ellipse((right_eye_x - eye_size, eye_y - eye_size, 
                      right_eye_x + eye_size, eye_y + eye_size), 
                     outline="black", width=2)
        
        # Mouth - happy or neutral
        mouth_y = center_y + size//4
        if random.random() < 0.7:  # 70% happy
            draw.arc((center_x - size//3, mouth_y - size//6, 
                      center_x + size//3, mouth_y + size//6), 
                     0, 180, fill="black", width=2)
        else:  # 30% neutral
            draw.line((center_x - size//3, mouth_y, 
                       center_x + size//3, mouth_y), 
                      fill="black", width=2)
        
        # Save the image
        output_path = os.path.join(test_images_dir, f"test_{i+1}.jpg")
        img.save(output_path)
        test_count += 1
        
        if i % 5 == 0:
            print(f"Generated {i+1} test images...")
    
    print(f"Successfully generated {test_count} test images")
    return True

def generate_complete_dataset():
    """Generate a complete dataset locally (both known faces and test images)"""
    print("=" * 60)
    print("Generating complete dataset locally")
    print("=" * 60)
    
    # Generate known faces
    print("\nStep 1: Generating known faces (training data)...")
    if not generate_sample_faces():
        print("Failed to generate known faces.")
        return False
    
    # Generate test images
    print("\nStep 2: Generating test images...")
    if not generate_sample_test_images():
        print("Failed to generate test images.")
        return False
    
    print("\n" + "=" * 60)
    print("Dataset generation complete!")
    print("Known faces: data/images/")
    print("Test images: data/test_images/")
    print("=" * 60)
    print("\nYou can now run the facial recognition system with the generated data:")
    print("1. Run the main application: python main.py")
    print("2. Or process test images: python process_images.py --gui")
    print("=" * 60)
    
    return True

def setup_complete_dataset():
    """Try to download dataset first, fall back to generating locally"""
    print("=" * 60)
    print("Setting up dataset for facial recognition attendance system")
    print("=" * 60)
    
    # First try zip download for known faces
    print("\nStep 1: Downloading known faces (training data)...")
    
    try:
        # Try downloading
        if not download_known_faces():
            print("Failed to download or generate known faces.")
            return False
    except Exception as e:
        print(f"Error during known faces download: {str(e)}")
        print("Falling back to local generation...")
        if not generate_sample_faces():
            print("Failed to generate known faces.")
            return False
    
    # Next try zip download for test images
    print("\nStep 2: Setting up test images...")
    
    try:
        # Try downloading
        if not download_test_images("mixed"):
            print("Test images download failed, falling back to generation...")
            if not generate_sample_test_images():
                print("Failed to generate test images.")
                return False
    except Exception as e:
        print(f"Error during test images download: {str(e)}")
        print("Falling back to local generation...")
        if not generate_sample_test_images():
            print("Failed to generate test images.")
            return False
    
    print("\n" + "=" * 60)
    print("Dataset setup complete!")
    print("Known faces: data/images/")
    print("Test images: data/test_images/")
    print("=" * 60)
    print("\nYou can now run the facial recognition system with sample data:")
    print("1. Run the main application: python main.py")
    print("2. Or process test images: python process_images.py --gui")
    print("=" * 60)
    
    return True

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Download or generate sample data for facial recognition attendance system')
    parser.add_argument('--known-faces', action='store_true', help='Download sample known faces (training data)')
    parser.add_argument('--test-images', choices=['lfw', 'celeba', 'mixed'], 
                        help='Download test images (specify dataset: lfw, celeba, or mixed)')
    parser.add_argument('--complete', action='store_true', help='Download both known faces and test images (complete dataset)')
    parser.add_argument('--generate', action='store_true', help='Generate sample data locally (no download required)')
    parser.add_argument('--direct', action='store_true', help='Use direct file download instead of zip files')
    
    args = parser.parse_args()
    
    # Generate local dataset if requested
    if args.generate:
        generate_complete_dataset()
        return
    
    # Direct file download if requested (now using local generation)
    if args.direct:
        print("Using local data generation instead of direct downloads...")
        generate_complete_dataset()
        return
    
    # Default action if no arguments provided
    if not (args.known_faces or args.test_images or args.complete):
        setup_complete_dataset()
        return
    
    # Download/generate known faces if requested
    if args.known_faces:
        download_known_faces()
    
    # Download/generate test images if requested
    if args.test_images:
        download_test_images(args.test_images)
    
    # Download/generate complete dataset if requested
    if args.complete:
        setup_complete_dataset()

if __name__ == "__main__":
    main() 