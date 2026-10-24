# Simple Version (No Dlib)

If you're still having issues with dlib, you can use our simplified version that doesn't require dlib.

Run both commands from the repository root: the scripts read and write `data/` relative to the current folder.

1. Run the generate_data.py script to create sample data:
   ```
   python simple_version/generate_data.py
   ```

2. Run the simplified facial recognition system:
   ```
   python simple_version/simple_recognition.py
   ```

This version uses basic image processing instead of deep learning: OpenCV's Haar-cascade face detector and colour-histogram matching. It is meant for trying out the workflow without dlib; its recognition quality was not measured.
