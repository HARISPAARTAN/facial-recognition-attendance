import os
import cv2
import numpy as np
import face_recognition
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import argparse
from tqdm import tqdm

class FaceRecognitionTester:
    def __init__(self, train_dir, test_dir, tolerance=0.6):
        self.train_dir = train_dir
        self.test_dir = test_dir
        self.tolerance = tolerance
        self.known_face_encodings = []
        self.known_face_names = []
        
        # Create output directory for results
        os.makedirs("test_results", exist_ok=True)
        
        # Load training faces
        self.load_known_faces()
    
    def load_known_faces(self):
        """Load known faces from training directory"""
        print("Loading training faces...")
        
        # Clear existing data
        self.known_face_encodings = []
        self.known_face_names = []
        
        # Ensure directory exists
        if not os.path.exists(self.train_dir):
            print(f"Error: Training directory {self.train_dir} not found")
            sys.exit(1)
        
        # List all images in the directory
        for filename in os.listdir(self.train_dir):
            if filename.endswith(('.jpg', '.jpeg', '.png')):
                # Get person name from filename (without extension)
                name = os.path.splitext(filename)[0]
                # If format is name_number, remove the number
                if '_' in name and name.split('_')[-1].isdigit():
                    name = '_'.join(name.split('_')[:-1])
                
                # Load image
                image_path = os.path.join(self.train_dir, filename)
                image = face_recognition.load_image_file(image_path)
                
                # Find face encodings
                try:
                    encoding = face_recognition.face_encodings(image)[0]
                    self.known_face_encodings.append(encoding)
                    self.known_face_names.append(name)
                    print(f"Loaded training face for: {name}")
                except IndexError:
                    print(f"No face found in image: {filename}")
                    continue
        
        print(f"Loaded {len(self.known_face_names)} training faces")
    
    def test_recognition(self):
        """Test face recognition on test images"""
        print("\nTesting face recognition...")
        
        # Check if test directory exists
        if not os.path.exists(self.test_dir):
            print(f"Error: Test directory {self.test_dir} not found")
            sys.exit(1)
        
        # Prepare results
        results = []
        y_true = []
        y_pred = []
        
        # Process each test image
        test_files = [f for f in os.listdir(self.test_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
        
        for filename in tqdm(test_files, desc="Testing images"):
            # Get actual name from filename
            actual_name = os.path.splitext(filename)[0]
            # If format is name_number, remove the number
            if '_' in actual_name and actual_name.split('_')[-1].isdigit():
                actual_name = '_'.join(actual_name.split('_')[:-1])
            
            # Load image
            image_path = os.path.join(self.test_dir, filename)
            image = face_recognition.load_image_file(image_path)
            
            # Find faces in test image
            face_locations = face_recognition.face_locations(image)
            
            if not face_locations:
                print(f"No face found in test image: {filename}")
                # Count as incorrect prediction
                y_true.append(actual_name)
                y_pred.append("Unknown")
                results.append({
                    'image': filename,
                    'actual': actual_name,
                    'predicted': "Unknown",
                    'correct': False,
                    'face_found': False,
                    'distance': 1.0  # Maximum distance
                })
                continue
            
            # Get face encodings
            face_encodings = face_recognition.face_encodings(image, face_locations)
            
            # Use the first face found
            face_encoding = face_encodings[0]
            
            # Compare with known faces
            matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding, tolerance=self.tolerance)
            name = "Unknown"
            face_distance = 1.0  # Default to maximum distance
            
            # Use the known face with the smallest distance
            if len(self.known_face_encodings) > 0:
                face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
                best_match_index = np.argmin(face_distances)
                face_distance = face_distances[best_match_index]
                
                if matches[best_match_index]:
                    name = self.known_face_names[best_match_index]
            
            # Record result
            is_correct = (name == actual_name)
            
            results.append({
                'image': filename,
                'actual': actual_name,
                'predicted': name,
                'correct': is_correct,
                'face_found': True,
                'distance': face_distance
            })
            
            y_true.append(actual_name)
            y_pred.append(name)
        
        # Convert results to DataFrame
        df_results = pd.DataFrame(results)
        
        # Calculate metrics
        accuracy = df_results['correct'].mean() * 100
        face_detection_rate = df_results['face_found'].mean() * 100
        
        # Print results
        print(f"\nTest results:")
        print(f"Total test images: {len(df_results)}")
        print(f"Faces detected: {df_results['face_found'].sum()} ({face_detection_rate:.2f}%)")
        print(f"Correct predictions: {df_results['correct'].sum()} of {df_results['face_found'].sum()} detected faces")
        print(f"Recognition accuracy: {accuracy:.2f}%")
        
        # Save results
        df_results.to_csv("test_results/recognition_results.csv", index=False)
        
        # Create confusion matrix
        labels = sorted(list(set(y_true + y_pred)))
        cm = confusion_matrix(y_true, y_pred, labels=labels)
        
        # Plot confusion matrix
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
        plt.title("Confusion Matrix")
        plt.ylabel("Actual")
        plt.xlabel("Predicted")
        plt.tight_layout()
        plt.savefig("test_results/confusion_matrix.png")
        
        # Generate classification report
        report = classification_report(y_true, y_pred, labels=labels, output_dict=True)
        df_report = pd.DataFrame(report).transpose()
        df_report.to_csv("test_results/classification_report.csv")
        
        # Print classification report
        print("\nClassification Report:")
        print(classification_report(y_true, y_pred, labels=labels))
        
        return accuracy
    
    def test_different_tolerances(self, tolerance_range=None):
        """Test with different tolerance values"""
        if tolerance_range is None:
            tolerance_range = [0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7]
        
        print("\nTesting different tolerance values...")
        
        results = []
        
        for tolerance in tolerance_range:
            self.tolerance = tolerance
            accuracy = self.test_recognition()
            results.append({'tolerance': tolerance, 'accuracy': accuracy})
        
        # Convert to DataFrame
        df_tolerance = pd.DataFrame(results)
        df_tolerance.to_csv("test_results/tolerance_results.csv", index=False)
        
        # Plot results
        plt.figure(figsize=(10, 6))
        plt.plot(df_tolerance['tolerance'], df_tolerance['accuracy'], marker='o')
        plt.title("Recognition Accuracy vs Tolerance")
        plt.xlabel("Tolerance Value")
        plt.ylabel("Accuracy (%)")
        plt.grid(True)
        plt.savefig("test_results/tolerance_accuracy.png")
        
        # Find best tolerance
        best_idx = df_tolerance['accuracy'].idxmax()
        best_tolerance = df_tolerance.iloc[best_idx]['tolerance']
        best_accuracy = df_tolerance.iloc[best_idx]['accuracy']
        
        print(f"\nBest tolerance value: {best_tolerance} with accuracy: {best_accuracy:.2f}%")

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Test face recognition accuracy')
    parser.add_argument('--train', default='data/images', help='Directory with training images')
    parser.add_argument('--test', default='data/test_images', help='Directory with test images')
    parser.add_argument('--tolerance', type=float, default=0.6, help='Recognition tolerance (default: 0.6)')
    parser.add_argument('--test-tolerances', action='store_true', help='Test different tolerance values')
    
    args = parser.parse_args()
    
    # Ensure test directory exists
    if not os.path.exists(args.test):
        print(f"Test directory {args.test} does not exist. Creating...")
        os.makedirs(args.test, exist_ok=True)
        print(f"Please copy test images to {args.test} directory")
        sys.exit(0)
    
    # Create tester
    tester = FaceRecognitionTester(args.train, args.test, args.tolerance)
    
    if args.test_tolerances:
        tester.test_different_tolerances()
    else:
        tester.test_recognition()

if __name__ == "__main__":
    main() 