"""
Generate sample data for the simple facial recognition attendance system.
"""
import os
import random
from PIL import Image, ImageDraw, ImageFont
import numpy as np

def generate_sample_faces():
    """Generate sample known faces"""
    print("Generating sample known faces...")
    
    # Create directory
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
        # Create an image with a random color
        color = (
            random.randint(150, 250),
            random.randint(150, 250),
            random.randint(150, 250)
        )
        img = Image.new('RGB', (400, 400), color=color)
        
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

def generate_test_images():
    """Generate test images"""
    print("Generating test images...")
    
    # Create directory
    test_dir = "data/test_images"
    os.makedirs(test_dir, exist_ok=True)
    
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
    
    # Create test images
    for i in range(20):
        # Select a name, 70% chance of using a known name
        if random.random() < 0.7 and known_names:
            name = random.choice(known_names)
        else:
            name = "Unknown_Person"
        
        # Create an image with a random color
        color = (
            random.randint(100, 200),
            random.randint(100, 200),
            random.randint(100, 200)
        )
        img = Image.new('RGB', (500, 500), color=color)
        
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
        
        # Add some random noise to make each image different
        draw.text(
            (50 + random.randint(-20, 20), 200 + random.randint(-20, 20)),
            name.replace("_", " "),
            fill="black",
            font=font
        )
        
        # Draw a simple face
        center_x, center_y = 250 + random.randint(-20, 20), 150 + random.randint(-20, 20)
        size = 120 + random.randint(-20, 20)
        
        # Face
        draw.ellipse(
            (center_x - size//2, center_y - size//2, center_x + size//2, center_y + size//2),
            outline="black",
            width=2
        )
        
        # Eyes
        eye_size = size // 8
        left_eye_x = center_x - size//4
        right_eye_x = center_x + size//4
        eye_y = center_y - size//8
        
        draw.ellipse(
            (left_eye_x - eye_size, eye_y - eye_size, left_eye_x + eye_size, eye_y + eye_size),
            outline="black",
            width=2
        )
        
        draw.ellipse(
            (right_eye_x - eye_size, eye_y - eye_size, right_eye_x + eye_size, eye_y + eye_size),
            outline="black",
            width=2
        )
        
        # Mouth
        mouth_y = center_y + size//4
        if random.random() < 0.7:  # 70% happy
            draw.arc(
                (center_x - size//3, mouth_y - size//6, center_x + size//3, mouth_y + size//6),
                0, 180,
                fill="black",
                width=2
            )
        else:  # 30% neutral
            draw.line(
                (center_x - size//3, mouth_y, center_x + size//3, mouth_y),
                fill="black",
                width=2
            )
        
        # Save the image
        output_path = os.path.join(test_dir, f"test_{i+1}.jpg")
        img.save(output_path)
        test_count += 1
        
        if i % 5 == 0:
            print(f"Generated {i+1} test images...")
    
    print(f"Successfully generated {test_count} test images")

def main():
    print("=" * 60)
    print("Generating sample data for facial recognition")
    print("=" * 60)
    
    # Generate known faces
    generate_sample_faces()
    
    # Generate test images
    generate_test_images()
    
    print("\nData generation complete!")
    print("Known faces: data/images/")
    print("Test images: data/test_images/")
    print("\nYou can now run the facial recognition system with:")
    print("python simple_recognition.py")

if __name__ == "__main__":
    main()
