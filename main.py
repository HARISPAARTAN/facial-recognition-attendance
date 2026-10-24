import os
import cv2
import numpy as np
import face_recognition
import pandas as pd
from datetime import datetime
import tkinter as tk
from tkinter import messagebox, ttk
from PIL import Image, ImageTk
import threading

class FacialRecognitionAttendance:
    def __init__(self):
        self.known_face_encodings = []
        self.known_face_names = []
        self.frame_resizing = 0.25
        self.attendance_data = {}
        self.current_date = datetime.now().strftime("%Y-%m-%d")
        self.attendance_file = f"data/attendance/{self.current_date}.csv"
        
        # Initialize paths
        self.images_path = "data/images"
        self.model_path = "models"
        
        # Ensure directories exist
        os.makedirs(self.images_path, exist_ok=True)
        os.makedirs("data/attendance", exist_ok=True)
        os.makedirs(self.model_path, exist_ok=True)
        
        # Load known faces
        self.load_known_faces()
        
    def load_known_faces(self):
        """Load known faces from images directory"""
        print("Loading known faces...")
        
        # Clear existing data
        self.known_face_encodings = []
        self.known_face_names = []
        
        # List all images in the directory
        for filename in os.listdir(self.images_path):
            if filename.endswith(('.jpg', '.jpeg', '.png')):
                # Get person name from filename (without extension)
                name = os.path.splitext(filename)[0]
                
                # Load image
                image_path = os.path.join(self.images_path, filename)
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
    
    def process_frame(self, frame):
        """Process each frame for face recognition"""
        # Resize frame for faster processing
        small_frame = cv2.resize(frame, (0, 0), fx=self.frame_resizing, fy=self.frame_resizing)
        
        # Convert BGR to RGB
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
        
        # Find all faces in the current frame
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
        
        face_names = []
        
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
        
        # Display results
        for (top, right, bottom, left), name in zip(face_locations, face_names):
            # Scale back face locations
            top = int(top / self.frame_resizing)
            right = int(right / self.frame_resizing)
            bottom = int(bottom / self.frame_resizing)
            left = int(left / self.frame_resizing)
            
            # Draw rectangle around face
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            
            # Draw name label
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 255, 0), cv2.FILLED)
            cv2.putText(frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 1)
        
        return frame
    
    def mark_attendance(self, name):
        """Mark attendance for recognized person"""
        now = datetime.now()
        time_str = now.strftime("%H:%M:%S")
        
        # Check if person is already marked for today
        if name in self.attendance_data:
            # Already marked, do nothing
            return
        
        # Add to the attendance data
        self.attendance_data[name] = time_str
        
        # Save to CSV file
        df = pd.DataFrame(list(self.attendance_data.items()), columns=['Name', 'Time'])
        df.to_csv(self.attendance_file, index=False)
        
        print(f"Marked attendance for {name} at {time_str}")

    def run_recognition(self):
        """Run the main recognition loop without GUI"""
        # Initialize webcam
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("Error: Could not open webcam")
            return
        
        print("Starting facial recognition...")
        
        while True:
            # Read frame
            ret, frame = cap.read()
            
            if not ret:
                print("Error: Failed to capture image")
                break
            
            # Process frame
            processed_frame = self.process_frame(frame)
            
            # Display the resulting frame
            cv2.imshow('Facial Recognition Attendance', processed_frame)
            
            # Break loop with 'q' key
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        # Release resources
        cap.release()
        cv2.destroyAllWindows()

class AttendanceSystemGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Facial Recognition Attendance System")
        self.root.geometry("1000x600")
        self.root.resizable(True, True)
        
        # Create the facial recognition system
        self.attendance_system = FacialRecognitionAttendance()
        
        # Create GUI elements
        self.create_widgets()
        
        # Video capture variables
        self.cap = None
        self.is_running = False
        self.video_thread = None
    
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
        
        # Control buttons
        controls_frame = ttk.LabelFrame(right_panel, text="Controls", padding="10")
        controls_frame.pack(padx=10, pady=10, fill=tk.X)
        
        self.start_button = ttk.Button(controls_frame, text="Start Recognition", command=self.start_recognition)
        self.start_button.pack(pady=5, fill=tk.X)
        
        self.stop_button = ttk.Button(controls_frame, text="Stop Recognition", command=self.stop_recognition, state=tk.DISABLED)
        self.stop_button.pack(pady=5, fill=tk.X)
        
        ttk.Button(controls_frame, text="Load Known Faces", command=self.reload_faces).pack(pady=5, fill=tk.X)
        ttk.Button(controls_frame, text="View Attendance", command=self.view_attendance).pack(pady=5, fill=tk.X)
        
        # Status frame
        status_frame = ttk.LabelFrame(right_panel, text="Status", padding="10")
        status_frame.pack(padx=10, pady=10, fill=tk.X)
        
        self.status_label = ttk.Label(status_frame, text="System Ready")
        self.status_label.pack(pady=5)
        
        self.loaded_faces_label = ttk.Label(status_frame, text=f"Loaded Faces: {len(self.attendance_system.known_face_names)}")
        self.loaded_faces_label.pack(pady=5)
        
        # Attendance frame
        attendance_frame = ttk.LabelFrame(right_panel, text="Today's Attendance", padding="10")
        attendance_frame.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)
        
        # Attendance treeview
        self.attendance_tree = ttk.Treeview(attendance_frame, columns=("Name", "Time"), show="headings")
        self.attendance_tree.heading("Name", text="Name")
        self.attendance_tree.heading("Time", text="Time")
        self.attendance_tree.pack(fill=tk.BOTH, expand=True)
    
    def start_recognition(self):
        """Start the recognition process"""
        if self.is_running:
            return
        
        self.cap = cv2.VideoCapture(0)
        
        if not self.cap.isOpened():
            messagebox.showerror("Error", "Could not open webcam")
            return
        
        self.is_running = True
        self.start_button.config(state=tk.DISABLED)
        self.stop_button.config(state=tk.NORMAL)
        self.status_label.config(text="Recognition Running...")
        
        # Start video thread
        self.video_thread = threading.Thread(target=self.update_frame)
        self.video_thread.daemon = True
        self.video_thread.start()
    
    def stop_recognition(self):
        """Stop the recognition process"""
        self.is_running = False
        self.start_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)
        self.status_label.config(text="Recognition Stopped")
        
        if self.cap is not None:
            self.cap.release()
    
    def update_frame(self):
        """Update video frame (runs in separate thread)"""
        while self.is_running:
            ret, frame = self.cap.read()
            
            if not ret:
                self.stop_recognition()
                messagebox.showerror("Error", "Failed to capture image")
                break
            
            # Process frame
            processed_frame = self.attendance_system.process_frame(frame)
            
            # Convert to ImageTk format
            cv_image = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(cv_image)
            
            # Resize to fit the label
            label_width = self.video_label.winfo_width()
            label_height = self.video_label.winfo_height()
            
            if label_width > 1 and label_height > 1:  # Ensure valid dimensions
                pil_image = pil_image.resize((label_width, label_height), Image.LANCZOS)
            
            tk_image = ImageTk.PhotoImage(image=pil_image)
            
            # Update label
            self.video_label.config(image=tk_image)
            self.video_label.image = tk_image  # Keep a reference
            
            # Update attendance display
            self.update_attendance_display()
    
    def update_attendance_display(self):
        """Update the attendance treeview"""
        # Clear treeview
        for item in self.attendance_tree.get_children():
            self.attendance_tree.delete(item)
        
        # Add attendance data
        for name, time in self.attendance_system.attendance_data.items():
            self.attendance_tree.insert("", tk.END, values=(name, time))
    
    def reload_faces(self):
        """Reload known faces"""
        self.attendance_system.load_known_faces()
        self.loaded_faces_label.config(text=f"Loaded Faces: {len(self.attendance_system.known_face_names)}")
        messagebox.showinfo("Faces Loaded", f"Successfully loaded {len(self.attendance_system.known_face_names)} faces")
    
    def view_attendance(self):
        """View attendance records"""
        try:
            if os.path.exists(self.attendance_system.attendance_file):
                df = pd.read_csv(self.attendance_system.attendance_file)
                
                # Create a new window
                attendance_window = tk.Toplevel(self.root)
                attendance_window.title(f"Attendance for {self.attendance_system.current_date}")
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
                           command=lambda: df.to_csv(f"data/attendance/export_{self.attendance_system.current_date}.csv", index=False)
                          ).pack(pady=10)
            else:
                messagebox.showinfo("No Data", "No attendance data available for today")
        except Exception as e:
            messagebox.showerror("Error", f"Error viewing attendance: {str(e)}")

def add_sample_faces():
    """Add some sample faces to the dataset (for testing)"""
    import shutil
    import requests
    from io import BytesIO
    
    try:
        # Create sample directory if it doesn't exist
        sample_dir = "data/images"
        os.makedirs(sample_dir, exist_ok=True)
        
        print("This is just a placeholder for adding sample faces.")
        print("In a real implementation, you would:")
        print("1. Take photos of students")
        print("2. Save them with their names as filenames")
        print("3. The system will automatically load them")
        
        print("\nTo add a new student:")
        print("1. Take a clear photo of their face")
        print("2. Save it as 'FirstName_LastName.jpg' in the 'data/images' folder")
        print("3. Reload the faces in the application")
        
    except Exception as e:
        print(f"Error adding sample faces: {str(e)}")

if __name__ == "__main__":
    # Create sample faces for testing (optional)
    # add_sample_faces()
    
    # Start the GUI application
    root = tk.Tk()
    app = AttendanceSystemGUI(root)
    root.mainloop()
    
    # If you want to run without GUI, uncomment this line:
    # FacialRecognitionAttendance().run_recognition() 