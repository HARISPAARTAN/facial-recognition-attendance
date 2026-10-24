import os
import cv2
import numpy as np
import face_recognition
import pandas as pd
from datetime import datetime
import argparse
import shutil
import requests
from tqdm import tqdm
from zipfile import ZipFile
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

class BatchImageProcessor:
    def __init__(self, images_dir="data/test_images", known_faces_dir="data/images"):
        self.images_dir = images_dir
        self.known_faces_dir = known_faces_dir
        self.known_face_encodings = []
        self.known_face_names = []
        self.results_dir = "data/results"
        self.attendance_data = {}
        self.current_date = datetime.now().strftime("%Y-%m-%d")
        
        # Create necessary directories
        os.makedirs(self.images_dir, exist_ok=True)
        os.makedirs(self.known_faces_dir, exist_ok=True)
        os.makedirs(self.results_dir, exist_ok=True)
        
        # Load known faces
        self.load_known_faces()
    
    def load_known_faces(self):
        """Load known faces from images directory"""
        print("Loading known faces...")
        
        # Clear existing data
        self.known_face_encodings = []
        self.known_face_names = []
        
        # List all images in the directory
        for filename in os.listdir(self.known_faces_dir):
            if filename.endswith(('.jpg', '.jpeg', '.png')):
                # Get person name from filename (without extension)
                name = os.path.splitext(filename)[0]
                
                # Load image
                image_path = os.path.join(self.known_faces_dir, filename)
                image = face_recognition.load_image_file(image_path)
                
                # Find face encodings
                try:
                    encoding = face_recognition.face_encodings(image)[0]
                    self.known_face_encodings.append(encoding)
                    self.known_face_names.append(name)
                    print(f"Loaded face for: {name}")
                except IndexError:
                    print(f"No face found in image: {filename}")
                    continue
        
        print(f"Loaded {len(self.known_face_names)} faces")
    
    def process_image(self, image_path, save_output=True):
        """Process a single image for face recognition"""
        print(f"Processing image: {image_path}")
        
        # Load the image
        image = face_recognition.load_image_file(image_path)
        
        # Convert to BGR format for OpenCV
        image_cv = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        # Find all faces in the image
        face_locations = face_recognition.face_locations(image)
        face_encodings = face_recognition.face_encodings(image, face_locations)
        
        face_names = []
        
        # Process each face
        for face_encoding in face_encodings:
            # Compare with known faces
            matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding, tolerance=0.6)
            name = "Unknown"
            
            # Use the known face with the smallest distance
            if len(self.known_face_encodings) > 0:
                face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
                best_match_index = np.argmin(face_distances)
                
                if matches[best_match_index]:
                    name = self.known_face_names[best_match_index]
                    # Mark attendance
                    self.mark_attendance(name)
            
            face_names.append(name)
        
        # Display results on the image
        for (top, right, bottom, left), name in zip(face_locations, face_names):
            # Draw rectangle around face
            cv2.rectangle(image_cv, (left, top), (right, bottom), (0, 255, 0), 2)
            
            # Draw name label
            cv2.rectangle(image_cv, (left, bottom - 35), (right, bottom), (0, 255, 0), cv2.FILLED)
            cv2.putText(image_cv, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 1)
        
        # Save the output image if requested
        if save_output:
            output_path = os.path.join(self.results_dir, os.path.basename(image_path))
            cv2.imwrite(output_path, image_cv)
            print(f"Saved output image to: {output_path}")
        
        return image_cv, face_locations, face_names
    
    def process_directory(self):
        """Process all images in the specified directory"""
        print(f"Processing all images in {self.images_dir}...")
        
        image_files = [f for f in os.listdir(self.images_dir) 
                      if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        if not image_files:
            print(f"No images found in {self.images_dir}")
            return []
        
        results = []
        
        for filename in tqdm(image_files):
            image_path = os.path.join(self.images_dir, filename)
            processed_img, face_locations, face_names = self.process_image(image_path)
            
            results.append({
                'filename': filename,
                'faces_found': len(face_locations),
                'face_names': face_names
            })
        
        # Save attendance
        self.save_attendance()
        
        # Generate report
        self.generate_report(results)
        
        return results
    
    def mark_attendance(self, name):
        """Mark attendance for recognized person"""
        now = datetime.now()
        time_str = now.strftime("%H:%M:%S")
        
        # Check if person is already marked
        if name in self.attendance_data:
            return
        
        # Add to the attendance data
        self.attendance_data[name] = time_str
        print(f"Marked attendance for {name}")
    
    def save_attendance(self):
        """Save attendance data to CSV file"""
        if not self.attendance_data:
            print("No attendance data to save")
            return
        
        # Create DataFrame
        df = pd.DataFrame(list(self.attendance_data.items()), columns=['Name', 'Time'])
        
        # Save to CSV
        attendance_file = f"data/attendance/batch_process_{self.current_date}.csv"
        df.to_csv(attendance_file, index=False)
        
        print(f"Saved attendance data to {attendance_file}")
        
        return attendance_file
    
    def generate_report(self, results):
        """Generate a summary report of the processing"""
        if not results:
            return
        
        total_images = len(results)
        total_faces = sum(r['faces_found'] for r in results)
        recognized_faces = sum(1 for r in results for name in r['face_names'] if name != "Unknown")
        
        # Create DataFrame
        df = pd.DataFrame(results)
        
        # Save to CSV
        report_file = os.path.join(self.results_dir, f"report_{self.current_date}.csv")
        df.to_csv(report_file, index=False)
        
        print("\n" + "=" * 50)
        print("Processing Report:")
        print(f"Total images processed: {total_images}")
        print(f"Total faces detected: {total_faces}")
        print(f"Recognized faces: {recognized_faces}")
        print(f"Unknown faces: {total_faces - recognized_faces}")
        print(f"Recognition rate: {recognized_faces/total_faces*100:.2f}% (if faces were detected)")
        print(f"Report saved to: {report_file}")
        print("=" * 50)


def download_test_dataset(dataset_url, output_dir):
    """Download a dataset for testing"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Download the dataset
    print(f"Downloading dataset from {dataset_url}...")
    local_filename = os.path.join(output_dir, "downloaded_dataset.zip")
    
    # Stream the download to show progress
    response = requests.get(dataset_url, stream=True)
    total_size_in_bytes = int(response.headers.get('content-length', 0))
    block_size = 1024
    progress_bar = tqdm(total=total_size_in_bytes, unit='iB', unit_scale=True)
    
    with open(local_filename, 'wb') as file:
        for data in response.iter_content(block_size):
            progress_bar.update(len(data))
            file.write(data)
    
    progress_bar.close()
    
    # Extract the dataset
    print("Extracting dataset...")
    with ZipFile(local_filename, 'r') as zip_ref:
        zip_ref.extractall(output_dir)
    
    # Remove the zip file
    os.remove(local_filename)
    
    print(f"Dataset downloaded and extracted to {output_dir}")


class ImageProcessorGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Batch Image Processor")
        self.root.geometry("900x700")
        
        # Initialize the processor
        self.processor = BatchImageProcessor()
        
        # Create UI elements
        self.create_widgets()
    
    def create_widgets(self):
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Configuration frame
        config_frame = ttk.LabelFrame(main_frame, text="Configuration", padding="10")
        config_frame.pack(fill=tk.X, pady=10)
        
        # Input directory
        ttk.Label(config_frame, text="Input Images Directory:").grid(row=0, column=0, sticky=tk.W, pady=5)
        self.input_dir_var = tk.StringVar(value=self.processor.images_dir)
        ttk.Entry(config_frame, textvariable=self.input_dir_var, width=50).grid(row=0, column=1, padx=5, pady=5)
        ttk.Button(config_frame, text="Browse", command=self.browse_input_dir).grid(row=0, column=2, padx=5, pady=5)
        
        # Known faces directory
        ttk.Label(config_frame, text="Known Faces Directory:").grid(row=1, column=0, sticky=tk.W, pady=5)
        self.known_dir_var = tk.StringVar(value=self.processor.known_faces_dir)
        ttk.Entry(config_frame, textvariable=self.known_dir_var, width=50).grid(row=1, column=1, padx=5, pady=5)
        ttk.Button(config_frame, text="Browse", command=self.browse_known_dir).grid(row=1, column=2, padx=5, pady=5)
        
        # Dataset download frame
        dataset_frame = ttk.LabelFrame(main_frame, text="Download Test Dataset", padding="10")
        dataset_frame.pack(fill=tk.X, pady=10)
        
        # Dataset options
        ttk.Label(dataset_frame, text="Sample Dataset:").grid(row=0, column=0, sticky=tk.W, pady=5)
        self.dataset_var = tk.StringVar(value="LFW (Labeled Faces in the Wild) Sample")
        ttk.Combobox(dataset_frame, textvariable=self.dataset_var, width=50, 
                    values=["LFW (Labeled Faces in the Wild) Sample", 
                           "CelebA Sample"]).grid(row=0, column=1, padx=5, pady=5)
        ttk.Button(dataset_frame, text="Download", command=self.download_dataset).grid(row=0, column=2, padx=5, pady=5)
        
        # Actions frame
        actions_frame = ttk.Frame(main_frame)
        actions_frame.pack(fill=tk.X, pady=10)
        
        ttk.Button(actions_frame, text="Process Images", command=self.process_images).pack(side=tk.LEFT, padx=5)
        ttk.Button(actions_frame, text="View Attendance", command=self.view_attendance).pack(side=tk.LEFT, padx=5)
        ttk.Button(actions_frame, text="Reload Known Faces", command=self.reload_faces).pack(side=tk.LEFT, padx=5)
        
        # Status frame
        status_frame = ttk.LabelFrame(main_frame, text="Status", padding="10")
        status_frame.pack(fill=tk.X, pady=10)
        
        self.status_var = tk.StringVar(value="Ready")
        ttk.Label(status_frame, textvariable=self.status_var).pack(anchor=tk.W)
        
        self.progress_var = tk.DoubleVar(value=0)
        self.progress_bar = ttk.Progressbar(status_frame, variable=self.progress_var, maximum=100)
        self.progress_bar.pack(fill=tk.X, pady=5)
        
        # Results frame
        results_frame = ttk.LabelFrame(main_frame, text="Results", padding="10")
        results_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        # Treeview for results
        self.results_tree = ttk.Treeview(results_frame, columns=("file", "faces", "names"), show="headings")
        self.results_tree.heading("file", text="Image File")
        self.results_tree.heading("faces", text="Faces Found")
        self.results_tree.heading("names", text="Names")
        self.results_tree.column("file", width=300)
        self.results_tree.column("faces", width=100)
        self.results_tree.column("names", width=300)
        
        # Add scrollbar to treeview
        scrollbar = ttk.Scrollbar(results_frame, orient=tk.VERTICAL, command=self.results_tree.yview)
        self.results_tree.configure(yscroll=scrollbar.set)
        
        # Pack treeview and scrollbar
        self.results_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
    
    def browse_input_dir(self):
        """Browse for input directory"""
        directory = filedialog.askdirectory(initialdir=self.processor.images_dir)
        if directory:
            self.input_dir_var.set(directory)
            self.processor.images_dir = directory
    
    def browse_known_dir(self):
        """Browse for known faces directory"""
        directory = filedialog.askdirectory(initialdir=self.processor.known_faces_dir)
        if directory:
            self.known_dir_var.set(directory)
            self.processor.known_faces_dir = directory
    
    def download_dataset(self):
        """Download a sample dataset"""
        self.status_var.set("Downloading dataset...")
        self.root.update()
        
        try:
            dataset_name = self.dataset_var.get()
            
            if "LFW" in dataset_name:
                # LFW dataset (small sample)
                url = "https://www.dropbox.com/scl/fi/l2fh72qcmaw0vl6rw1bqf/lfw-sample.zip?rlkey=ur7uljk1wi5p6hwh7zcq1v7lv&dl=1"
                download_test_dataset(url, self.processor.images_dir)
            elif "CelebA" in dataset_name:
                # CelebA dataset (small sample)
                url = "https://www.dropbox.com/scl/fi/9zto2rif566y1d97vb6qv/celeba-sample.zip?rlkey=gfp4w4d2mlkf1w1rqtxmgtboh&dl=1"
                download_test_dataset(url, self.processor.images_dir)
            
            self.status_var.set(f"Downloaded {dataset_name} to {self.processor.images_dir}")
            messagebox.showinfo("Download Complete", f"Dataset downloaded to {self.processor.images_dir}")
            
        except Exception as e:
            self.status_var.set(f"Error: {str(e)}")
            messagebox.showerror("Download Error", str(e))
    
    def process_images(self):
        """Process all images in the input directory"""
        # Update processor directories
        self.processor.images_dir = self.input_dir_var.get()
        self.processor.known_faces_dir = self.known_dir_var.get()
        
        # Clear previous results
        for item in self.results_tree.get_children():
            self.results_tree.delete(item)
        
        # Process images in a separate thread
        self.status_var.set("Processing images...")
        self.root.update()
        
        try:
            # Reload faces to ensure latest reference images are used
            self.processor.load_known_faces()
            
            # Process all images
            results = self.processor.process_directory()
            
            # Update results in the treeview
            for result in results:
                self.results_tree.insert("", tk.END, values=(
                    result['filename'],
                    result['faces_found'],
                    ", ".join(result['face_names'])
                ))
            
            self.status_var.set(f"Processed {len(results)} images")
            messagebox.showinfo("Processing Complete", f"Processed {len(results)} images")
            
        except Exception as e:
            self.status_var.set(f"Error: {str(e)}")
            messagebox.showerror("Processing Error", str(e))
    
    def view_attendance(self):
        """View attendance records"""
        try:
            attendance_file = self.processor.save_attendance()
            
            if os.path.exists(attendance_file):
                df = pd.read_csv(attendance_file)
                
                # Create a new window
                attendance_window = tk.Toplevel(self.root)
                attendance_window.title(f"Attendance for {self.processor.current_date}")
                attendance_window.geometry("400x500")
                
                # Create treeview
                tree = ttk.Treeview(attendance_window, columns=list(df.columns), show="headings")
                
                for col in df.columns:
                    tree.heading(col, text=col)
                
                for _, row in df.iterrows():
                    tree.insert("", tk.END, values=list(row))
                
                tree.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
                
                # Add export button
                ttk.Button(attendance_window, text="Export", 
                          command=lambda: self.export_attendance(df)).pack(pady=10)
            else:
                messagebox.showinfo("No Data", "No attendance data available")
        except Exception as e:
            messagebox.showerror("Error", f"Error viewing attendance: {str(e)}")
    
    def export_attendance(self, df):
        """Export attendance data to a CSV file"""
        file_path = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
            initialfile=f"attendance_{self.processor.current_date}.csv"
        )
        
        if file_path:
            df.to_csv(file_path, index=False)
            messagebox.showinfo("Export Complete", f"Attendance exported to {file_path}")
    
    def reload_faces(self):
        """Reload known faces"""
        self.processor.known_faces_dir = self.known_dir_var.get()
        self.processor.load_known_faces()
        messagebox.showinfo("Faces Loaded", f"Successfully loaded {len(self.processor.known_face_names)} faces")


def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Process images for face recognition')
    parser.add_argument('--dir', help='Directory containing images to process')
    parser.add_argument('--known', help='Directory containing known face images')
    parser.add_argument('--download', help='Download a test dataset (lfw or celeba)', choices=['lfw', 'celeba'])
    parser.add_argument('--gui', action='store_true', help='Launch GUI interface')
    
    args = parser.parse_args()
    
    # Launch GUI if requested
    if args.gui:
        root = tk.Tk()
        app = ImageProcessorGUI(root)
        root.mainloop()
        return
    
    # Download dataset if requested
    if args.download:
        output_dir = "data/test_images"
        if args.download == 'lfw':
            url = "https://www.dropbox.com/scl/fi/l2fh72qcmaw0vl6rw1bqf/lfw-sample.zip?rlkey=ur7uljk1wi5p6hwh7zcq1v7lv&dl=1"
            download_test_dataset(url, output_dir)
        elif args.download == 'celeba':
            url = "https://www.dropbox.com/scl/fi/9zto2rif566y1d97vb6qv/celeba-sample.zip?rlkey=gfp4w4d2mlkf1w1rqtxmgtboh&dl=1"
            download_test_dataset(url, output_dir)
    
    # Initialize the processor with specified directories
    images_dir = args.dir if args.dir else "data/test_images"
    known_faces_dir = args.known if args.known else "data/images"
    
    processor = BatchImageProcessor(images_dir, known_faces_dir)
    processor.process_directory()


if __name__ == "__main__":
    main() 