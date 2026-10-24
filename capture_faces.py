import os
import cv2
import tkinter as tk
from tkinter import simpledialog, messagebox
from PIL import Image, ImageTk
import threading

class FaceCaptureApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Face Capture Tool")
        self.root.geometry("800x600")
        
        # Initialize variables
        self.cap = None
        self.is_running = False
        self.student_name = ""
        self.images_folder = "data/images"
        self.current_frame = None
        
        # Ensure images directory exists
        os.makedirs(self.images_folder, exist_ok=True)
        
        # Create GUI
        self.create_widgets()
        
    def create_widgets(self):
        # Main frame
        self.main_frame = tk.Frame(self.root)
        self.main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Video frame
        self.video_frame = tk.Frame(self.main_frame)
        self.video_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        self.video_label = tk.Label(self.video_frame)
        self.video_label.pack(fill=tk.BOTH, expand=True)
        
        # Control frame
        self.control_frame = tk.Frame(self.main_frame, width=200)
        self.control_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=10)
        
        # Student name entry
        tk.Label(self.control_frame, text="Student Name:").pack(anchor=tk.W, pady=(0, 5))
        self.name_var = tk.StringVar()
        self.name_entry = tk.Entry(self.control_frame, textvariable=self.name_var, width=20)
        self.name_entry.pack(fill=tk.X, pady=(0, 10))
        
        # Buttons
        self.start_button = tk.Button(self.control_frame, text="Start Camera", command=self.start_camera)
        self.start_button.pack(fill=tk.X, pady=5)
        
        self.capture_button = tk.Button(self.control_frame, text="Capture Face", command=self.capture_face, state=tk.DISABLED)
        self.capture_button.pack(fill=tk.X, pady=5)
        
        self.stop_button = tk.Button(self.control_frame, text="Stop Camera", command=self.stop_camera, state=tk.DISABLED)
        self.stop_button.pack(fill=tk.X, pady=5)
        
        # Status
        tk.Label(self.control_frame, text="Status:").pack(anchor=tk.W, pady=(20, 5))
        self.status_var = tk.StringVar(value="Camera off")
        self.status_label = tk.Label(self.control_frame, textvariable=self.status_var)
        self.status_label.pack(anchor=tk.W)
        
        # Captured images list
        tk.Label(self.control_frame, text="Captured Images:").pack(anchor=tk.W, pady=(20, 5))
        self.captured_listbox = tk.Listbox(self.control_frame, height=10)
        self.captured_listbox.pack(fill=tk.X, pady=5)
        
        # Load existing images
        self.load_existing_images()
    
    def load_existing_images(self):
        """Load existing images in the data folder"""
        for filename in os.listdir(self.images_folder):
            if filename.endswith(('.jpg', '.jpeg', '.png')):
                self.captured_listbox.insert(tk.END, filename)
    
    def start_camera(self):
        """Start the camera feed"""
        if self.is_running:
            return
        
        self.cap = cv2.VideoCapture(0)
        
        if not self.cap.isOpened():
            messagebox.showerror("Error", "Could not open webcam")
            return
        
        self.is_running = True
        self.start_button.config(state=tk.DISABLED)
        self.stop_button.config(state=tk.NORMAL)
        self.capture_button.config(state=tk.NORMAL)
        self.status_var.set("Camera running")
        
        # Start video thread
        self.video_thread = threading.Thread(target=self.update_frame)
        self.video_thread.daemon = True
        self.video_thread.start()
    
    def stop_camera(self):
        """Stop the camera feed"""
        self.is_running = False
        self.start_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)
        self.capture_button.config(state=tk.DISABLED)
        self.status_var.set("Camera off")
        
        if self.cap is not None:
            self.cap.release()
    
    def update_frame(self):
        """Update video frame (runs in separate thread)"""
        while self.is_running:
            ret, frame = self.cap.read()
            
            if not ret:
                self.stop_camera()
                messagebox.showerror("Error", "Failed to capture image")
                break
            
            # Detect faces
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)
            
            # Draw rectangles around faces
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            # Store current frame for capture
            self.current_frame = frame.copy()
            
            # Convert to ImageTk format
            cv_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
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
    
    def capture_face(self):
        """Capture and save the current face"""
        if not self.is_running or self.current_frame is None:
            messagebox.showerror("Error", "Camera not running")
            return
        
        # Get student name
        student_name = self.name_var.get().strip()
        
        if not student_name:
            messagebox.showerror("Error", "Please enter a student name")
            return
        
        # Format filename
        # Replace spaces with underscores and remove special characters
        filename = ''.join(c if c.isalnum() or c in ['_', '-'] else '_' for c in student_name)
        filename = filename.replace(' ', '_')
        
        # Get all face images for this student
        existing_files = [f for f in os.listdir(self.images_folder) 
                         if f.startswith(filename) and f.endswith(('.jpg', '.jpeg', '.png'))]
        
        # Create a new filename with index
        new_filename = f"{filename}_{len(existing_files) + 1}.jpg"
        file_path = os.path.join(self.images_folder, new_filename)
        
        # Save the image
        cv2.imwrite(file_path, self.current_frame)
        
        # Update status and list
        self.status_var.set(f"Captured image for {student_name}")
        self.captured_listbox.insert(tk.END, new_filename)
        
        messagebox.showinfo("Success", f"Face captured and saved as {new_filename}")

if __name__ == "__main__":
    # Create the root window
    root = tk.Tk()
    app = FaceCaptureApp(root)
    root.mainloop() 