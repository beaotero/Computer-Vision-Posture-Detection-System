# Lab Project: Gesture-Based Security & Posture Detection System

This project implements a computer vision system capable of camera calibration, object tracking using Kalman filters, and a security interface based on a Finite State Machine (FSM) that unlocks access through specific facial and body gestures.

--- 

### 📂 Project Structure
The project is organized within the src/ folder, keeping external assets (images, data logs) separated for better maintainability.

```bash
Lab_Project/
├── assets/                   # External resources
└── src/                      # Source code
    ├── calibration.py        # Camera calibration pipeline (Chessboard method)
    ├── camera.py             # Main script 
    ├── image_calibration.py  # Image-specific calibration utility
    ├── kalman_tracking.py    # Kalman Filter + MeanShift implementation
    ├── landmark_detection.py # Gesture and posture logic 
    ├── maquina_estados.py    # FSM logic for gesture password
    ├── UI.py                 # Visualization tools
    ├── utils.py              # Helper functions (Conversions, HSV tools, I/O)
    └── variables.py          # Constants and MediaPipe landmark indices
```
---


### 🛠️ Module Descriptions

1. **camera.py (Main Entry Point)**

This is the core of the application. It manages the real-time video stream, integrates MediaPipe Pose, and coordinates data flow between landmark detection, Kalman tracking, and the state machine. It handles the UI overlay, FPS calculation, and keyboard controls.

2. **maquina_estados.py (Finite State Machine)**

Defines the system logic for the security layer. It implements the PasswordFSM class, which requires the user to perform a specific sequence of movements to reach the UNLOCKED state:

Start $\rightarrow$ Center Face $\rightarrow$ Look Left $\rightarrow$ Look Right $\rightarrow$ Raise Right Hand $\rightarrow$ Raise Left Hand $\rightarrow$ Access Granted

3. **landmark_detection.py**

Contains the specialized algorithms to interpret MediaPipe coordinates:

- Head Tracking: Calculates ear-to-nose ratios to determine head rotation.
- Posture Analysis: Detects curved backs, shoulder asymmetry, or if the face is too close to the sensor.
- Shape Detection: Uses HSV color segmentation and contour approximation to identify geometric shapes (like triangles).

4. **kalman_tracking.py**

Provides robust tracking of the user's face. It uses a Kalman Filter to predict movement and MeanShift (based on HSV histograms) to correct the prediction. This ensures the tracking window remains stable even if the primary detector loses confidence for a few frames.

5. **calibration.py**

Computes the camera's intrinsic parameters and distortion coefficients using chessboard patterns. This is essential for correcting lens distortion and mapping 2D coordinates to 3D space accurately.

6. **utils.py, variables.py & UI.py**

utils.py: Tools for coordinate mapping, CSV data handling, and an interactive HSV trackbar utility for color thresholding.

variables.py: Centralizes all MediaPipe landmark indices and global configuration constants.

UI.py: Visualization Tools used for the display.

--- 

### 🚀 Installation & Usage
**Prerequisites:**
Ensure you have Python 3.8+ and the following libraries installed:

```bash
pip install opencv-python mediapipe numpy pandas imageio matplotlib pygame pillow
```
**Running the System:**
To start the real-time detection and security interface:

```bash
python src/camera.py
```

**Keyboard Controls:**
| Key | Action |
| :--- | :--- |
| **`q`** o **`ESC`** | Exit program |
| **`Space`** | Save landmarks to buffer |
| **`g`** | Save pose data to a csv file|
| **`j`** | Save image from video (`image.png`) |

---

## 🙍🏼‍♀️ <span id="contributors">Contributors</span>

- <a href="https://github.com/LuciaHC" target="_blank">Lucía Herraiz</a>
- <a href="https://github.com/beaotero" target="_blank">Beatriz Otero</a>

---