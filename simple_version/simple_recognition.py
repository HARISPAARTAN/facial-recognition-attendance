"""
Simplified facial recognition attendance system that doesn't require dlib.
This is a fallback option for environments where dlib installation is problematic.
"""
import os
import cv2
import numpy as np
import pandas as pd
from datetime import datetime
import tkinter as tk
from tkinter import messagebox, ttk
from PIL import Image, ImageTk
import threading
import json
import random

class SimpleFaceRecognition:
    def __init__(self):
        self.known_faces = {}
        self.current_date = datetime.now().strftime("%Y-%m-%d")
        self.attendance_file = f"data/attendance/simple_{self.current_date}.csv"
        self.attendance_data = {}
        
        # Ensure directories exist
        os.makedirs("data/images", exist_ok=True)
        os.makedirs("data/attendance", exist_ok=True)
        
        # Load known faces
        self.load_known_faces()
    
    def load_known_faces(self):
        """Load known faces from images directory using simple features"""
        print("Loading known faces...")
        
        # Clear existing data
        self.known_faces = {}
        
        # List all images in the directory
        for filename in os.listdir("data/images"):
            if filename.endswith(('.jpg', '.jpeg', '.png')):
                # Get person name from filename (without extension)
                name = os.path.splitext(filename)[0]
                
                # Load image
                image_path = os.path.join("data/images", filename)
                image = cv2.imread(image_path)
                
                if image is not None:
                    # Calculate simple features (average color in different regions)
                    features = self._extract_simple_features(image)
                    self.known_faces[name] = features
                    print(f"Loaded face for: {name}")
                else:
                    print(f"Could not load image: {filename}")
        
        print(f"Loaded {len(self.known_faces)} faces")
    
    def _extract_simple_features(self, image):
        """Extract simple features from an image (color histograms)"""
        # Resize image for consistency
        image = cv2.resize(image, (100, 100))
        
        # Convert to different color spaces
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Calculate histograms
        gray_hist = cv2.calcHist([gray], [0], None, [16], [0, 256]).flatten()
        h_hist = cv2.calcHist([hsv], [0], None, [16], [0, 180]).flatten()
        s_hist = cv2.calcHist([hsv], [1], None, [16], [0, 256]).flatten()
        
        # Normalize histograms
        gray_hist = cv2.normalize(gray_hist, gray_hist).flatten()
        h_hist = cv2.normalize(h_hist, h_hist).flatten()
        s_hist = cv2.normalize(s_hist, s_hist).flatten()
        
        # Combine features
        return np.concatenate((gray_hist, h_hist, s_hist))
    
    def recognize_face(self, image):
        """Recognize a face in the image using simple feature matching"""
        if len(self.known_faces) == 0:
            return "Unknown"
        
        # Extract features from the image
        features = self._extract_simple_features(image)
        
        # Compare with known faces
        min_distance = float('inf')
        best_match = "Unknown"
        
        for name, known_features in self.known_faces.items():
            # Calculate distance between feature vectors
            distance = np.linalg.norm(features - known_features)
            
            if distance < min_distance:
                min_distance = distance
                best_match = name
        
        # Use a threshold to determine if it's a match
        if min_distance < 0.5:  # Adjust threshold as needed
            return best_match
        else:
            return "Unknown"
    
    def mark_attendance(self, name):
        """Mark attendance for recognized person"""
        now = datetime.now()
        time_str = now.strftime("%H:%M:%S")
        
        # Check if person is already marked
        if name in self.attendance_data:
            return
        
        # Add to the attendance data
        self.attendance_data[name] = time_str
        
        # Save to CSV file
        df = pd.DataFrame(list(self.attendance_data.items()), columns=['Name', 'Time'])
        df.to_csv(self.attendance_file, index=False)
        
        print(f"Marked attendance for {name} at {time_str}")
    
    def process_frame(self, frame):
        """Process a video frame for face recognition"""
        # Convert to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Use Haar cascade for face detection
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        
        for (x, y, w, h) in faces:
            # Extract face region
            face_image = frame[y:y+h, x:x+w]
            
            # Recognize the face
            name = self.recognize_face(face_image)
            
            # Mark attendance if the person is recognized
            if name != "Unknown":
                self.mark_attendance(name)
            
            # Draw rectangle around face
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            # Draw name label
            cv2.rectangle(frame, (x, y+h), (x+w, y+h+30), (0, 255, 0), cv2.FILLED)
            cv2.putText(frame, name, (x+6, y+h+20), cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), 1)
        
        return frame

class SimpleAttendanceGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Simple Facial Recognition Attendance")
        self.root.geometry("1000x600")
        
        # Create the recognition system
        self.recognition = SimpleFaceRecognition()
        
        # Initialize variables
        self.is_running = False
        self.cap = None
        
        # Create widgets
        self.create_widgets()
    
    def create_widgets(self):
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Left panel (video feed)
        left_panel = ttk.Frame(main_frame)
        left_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Video label
        self.video_label = ttk.Label(left_panel)
        self.video_label.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)
        
        # Right panel (controls and data)
        right_panel = ttk.Frame(main_frame, width=300)
        right_panel.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Controls
        controls_frame = ttk.LabelFrame(right_panel, text="Controls")
        controls_frame.pack(fill=tk.X, padx=10, pady=10)
        
        self.start_button = ttk.Button(controls_frame, text="Start Recognition", command=self.start_recognition)
        self.start_button.pack(fill=tk.X, padx=10, pady=5)
        
        self.stop_button = ttk.Button(controls_frame, text="Stop Recognition", command=self.stop_recognition, state=tk.DISABLED)
        self.stop_button.pack(fill=tk.X, padx=10, pady=5)
        
        ttk.Button(controls_frame, text="Reload Faces", command=self.reload_faces).pack(fill=tk.X, padx=10, pady=5)
        ttk.Button(controls_frame, text="View Attendance", command=self.view_attendance).pack(fill=tk.X, padx=10, pady=5)
        
        # Status
        status_frame = ttk.LabelFrame(right_panel, text="Status")
        status_frame.pack(fill=tk.X, padx=10, pady=10)
        
        self.status_var = tk.StringVar(value="Ready")
        ttk.Label(status_frame, textvariable=self.status_var).pack(padx=10, pady=5, anchor=tk.W)
        
        # Attendance
        attendance_frame = ttk.LabelFrame(right_panel, text="Today's Attendance")
        attendance_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        self.attendance_tree = ttk.Treeview(attendance_frame, columns=("Name", "Time"), show="headings")
        self.attendance_tree.heading("Name", text="Name")
        self.attendance_tree.heading("Time", text="Time")
        self.attendance_tree.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
    
    def start_recognition(self):
        """Start face recognition"""
        if self.is_running:
            return
        
        self.cap = cv2.VideoCapture(0)
        
        if not self.cap.isOpened():
            messagebox.showerror("Error", "Could not open webcam")
            return
        
        self.is_running = True
        self.start_button.config(state=tk.DISABLED)
        self.stop_button.config(state=tk.NORMAL)
        self.status_var.set("Recognition Running...")
        
        # Start video thread
        self.video_thread = threading.Thread(target=self.update_frame)
        self.video_thread.daemon = True
        self.video_thread.start()
    
    def stop_recognition(self):
        """Stop face recognition"""
        self.is_running = False
        self.start_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)
        self.status_var.set("Recognition Stopped")
        
        if self.cap:
            self.cap.release()
    
    def update_frame(self):
        """Update video frame continuously"""
        while self.is_running:
            ret, frame = self.cap.read()
            
            if not ret:
                self.stop_recognition()
                messagebox.showerror("Error", "Failed to capture frame")
                break
            
            # Process frame
            processed_frame = self.recognition.process_frame(frame)
            
            # Convert for display
            cv_image = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(cv_image)
            
            # Resize to fit label
            label_width = self.video_label.winfo_width()
            label_height = self.video_label.winfo_height()
            
            if label_width > 1 and label_height > 1:
                pil_image = pil_image.resize((label_width, label_height), Image.LANCZOS)
            
            # Convert to Tkinter image
            tk_image = ImageTk.PhotoImage(image=pil_image)
            
            # Update label
            self.video_label.config(image=tk_image)
            self.video_label.image = tk_image
            
            # Update attendance display
            self.update_attendance_display()
    
    def update_attendance_display(self):
        """Update attendance treeview"""
        # Clear treeview
        for item in self.attendance_tree.get_children():
            self.attendance_tree.delete(item)
        
        # Add attendance data
        for name, time in self.recognition.attendance_data.items():
            self.attendance_tree.insert("", tk.END, values=(name, time))
    
    def reload_faces(self):
        """Reload known faces"""
        self.recognition.load_known_faces()
        messagebox.showinfo("Faces Loaded", f"Successfully loaded {len(self.recognition.known_faces)} faces")
    
    def view_attendance(self):
        """View attendance records"""
        try:
            if os.path.exists(self.recognition.attendance_file):
                df = pd.read_csv(self.recognition.attendance_file)
                
                # Create a new window
                attendance_window = tk.Toplevel(self.root)
                attendance_window.title(f"Attendance for {self.recognition.current_date}")
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
                           command=lambda: df.to_csv(f"data/attendance/export_{self.recognition.current_date}.csv", index=False)
                          ).pack(pady=10)
            else:
                messagebox.showinfo("No Data", "No attendance data available for today")
        except Exception as e:
            messagebox.showerror("Error", f"Error viewing attendance: {str(e)}")

def main():
    root = tk.Tk()
    app = SimpleAttendanceGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()
