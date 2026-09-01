# 🚗 Driver Drowsiness Detection

A real-time driver drowsiness detection system built with **Python, OpenCV, TensorFlow/Keras, and Pygame**.

The system uses Haar Cascade classifiers to detect the driver's face and eyes, a CNN model to classify eye states as **open or closed**, and a time-based rule to detect prolonged eye closure and trigger an audible warning.

---

## 📌 Overview

Driver drowsiness is an important safety concern, particularly during long periods of driving.

The goal of this project is to build a simple real-time computer vision system capable of detecting prolonged eye closure from a webcam feed and warning the driver when the eyes remain closed for a predefined period.

The project combines traditional computer vision techniques with deep learning:

```text
Webcam
   ↓
OpenCV Frame Capture
   ↓
Haar Cascade Face / Eye Detection
   ↓
Eye Region Extraction
   ↓
Image Preprocessing
   ↓
CNN Eye-State Classification
   ↓
Both Eyes Closed?
   ↓
5-Second Temporal Threshold
   ↓
Audio Alert
```

> **Project status:** Computer vision prototype / portfolio project
> This system is intended for educational and experimental purposes and is not a certified automotive safety system.

---

## ✨ Features

* 🎥 Real-time webcam processing
* 👤 Face detection using Haar Cascade
* 👁️ Left and right eye detection
* 🧠 CNN-based eye-state classification
* 🔄 Frame-by-frame inference
* ⏱️ Temporal monitoring of eye closure
* 🚨 Audio alarm after prolonged eye closure
* 🔴 Visual warning with a red frame
* 📸 Automatic frame capture when drowsiness is detected
* ⚙️ Simple threshold-based decision logic

---

## 🏗️ System Architecture

The complete processing pipeline is:

```text
┌─────────────────────┐
│       Webcam        │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│   Capture Frame     │
│      OpenCV         │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│    Face Detection   │
│    Haar Cascade     │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│     Eye Detection   │
│    Haar Cascades    │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│    Eye ROI Crop     │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│   Preprocessing     │
│ Gray → 24×24        │
│ Normalize /255      │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│   CNN Classifier    │
│    TensorFlow       │
└──────────┬──────────┘
           │
           ▼
    ┌──────┴───────┐
    │              │
  Open           Closed
    │              │
    │         Start Timer
    │              │
    │              ▼
    │       Closed ≥ 5 sec?
    │              │
    │              ▼
    │        Audio Warning
    │              │
    └──────────────┴───────┐
                           ▼
                    Visual Feedback
```

---

## 🔍 Computer Vision Pipeline

### 1. Video Capture

The system captures live frames using OpenCV:

```python
video_cap = cv.VideoCapture(0)
```

The default webcam is used as the input source.

Each frame is processed inside a continuous loop.

---

### 2. Face Detection

The project uses the Haar Cascade classifier provided by OpenCV:

```python
face_cascade = cv.CascadeClassifier(
    os.path.join(
        cv.data.haarcascades,
        "haarcascade_frontalface_default.xml"
    )
)
```

Face detection is performed on a grayscale version of the frame.

```python
gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
```

The detector is configured with:

* `scaleFactor = 1.1`
* `minNeighbors = 5`
* `minSize = (25, 25)`

The detected face is displayed with a bounding box.

---

## 👁️ Eye Detection

Two Haar Cascade classifiers are used independently:

```text
haarcascade_lefteye_2splits.xml
haarcascade_righteye_2splits.xml
```

This allows the system to process the left and right eye separately.

The detected eye regions are cropped directly from the original frame.

---

## 🧹 Image Preprocessing

Before classification, each detected eye undergoes the following preprocessing steps:

### 1. Convert to grayscale

```python
eye = cv.cvtColor(eye, cv.COLOR_BGR2GRAY)
```

### 2. Resize

```python
eye = cv.resize(eye, (24, 24))
```

### 3. Normalize pixel values

```python
eye = eye / 255
```

### 4. Reshape

The image is reshaped into the format expected by the CNN:

```text
(24, 24)
      ↓
(24, 24, 1)
      ↓
(1, 24, 24, 1)
```

The final tensor represents a batch containing one grayscale eye image.

---

## 🧠 CNN Eye-State Classification

A trained TensorFlow/Keras model is loaded from:

```text
models/cnnCat2.keras
```

The model predicts the state of each eye.

The output is converted into a class index using:

```python
np.argmax(model.predict(eye), axis=1)
```

The project uses the following class mapping:

```text
0 → Closed
1 → Open
```

Therefore:

```python
if prediction[0] == 1:
    eye_state = "open"

if prediction[0] == 0:
    eye_state = "closed"
```

The same model is applied independently to the left and right eye.

---

## ⏱️ Temporal Drowsiness Detection

The system does not trigger an alarm simply because one frame is classified as closed.

Instead, it checks whether **both eyes are simultaneously classified as closed**:

```python
if rpred[0] == 0 and lpred[0] == 0:
```

When both eyes become closed, a timer starts:

```python
if closed_start_time is None:
    closed_start_time = time.time()
```

The duration of continuous eye closure is then calculated:

```python
closed_duration = time.time() - closed_start_time
```

If either eye becomes open again, the timer is reset:

```python
closed_start_time = None
```

This provides a simple temporal mechanism for distinguishing prolonged eye closure from an isolated prediction.

---

## 🚨 Alert Mechanism

The current drowsiness threshold is:

```text
5 seconds
```

When:

```text
closed_duration >= 5
```

the system triggers several actions.

### 🔊 Audio Alert

An alarm sound is played using Pygame:

```python
sound.play()
```

The sound file is:

```text
alarm.wav
```

### 🔴 Visual Alert

A red border is drawn around the entire frame.

The border thickness is dynamically increased and decreased to create a flashing effect.

### 📸 Frame Capture

The current frame is also saved:

```text
image.jpg
```

This provides a snapshot of the frame in which the drowsiness condition was triggered.

---

## 🖥️ Real-Time Visualization

The application displays:

* Face bounding box
* Eye bounding boxes
* Current eye state
* Closed-eye duration
* Visual drowsiness warning

A simplified example of the runtime states is:

```text
Normal:

┌─────────────────────────────┐
│                             │
│        Driver Face          │
│       ┌──────────┐          │
│       │ 👁️  👁️ │          │
│       └──────────┘          │
│                             │
│                    OPEN     │
└─────────────────────────────┘
```

When prolonged eye closure is detected:

```text
┌─────────────────────────────┐
│  ⚠️ DROWSINESS DETECTED ⚠️  │
│                             │
│        Driver Face          │
│       ┌──────────┐          │
│       │  -    -  │          │
│       └──────────┘          │
│                             │
│              CLOSED: 5+ sec │
└─────────────────────────────┘
```

---

## 🧩 Technologies

| Technology         | Purpose                                  |
| ------------------ | ---------------------------------------- |
| Python             | Core programming language                |
| OpenCV             | Video processing, face and eye detection |
| Haar Cascade       | Face and eye detection                   |
| TensorFlow / Keras | CNN model inference                      |
| NumPy              | Numerical operations and preprocessing   |
| Pygame             | Audio alarm                              |

---

## 📁 Project Structure

The project can be organized as follows:

```text
driver-drowsiness-detection/
│
├── models/
│   └── cnnCat2.keras
│
├── alarm.wav
│
├── drowsiness.py
│
├── image.jpg
│
├── requirements.txt
│
└── README.md
```

`image.jpg` is generated automatically when the drowsiness threshold is reached.

---

## 🚀 Installation

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/driver-drowsiness-detection.git
cd driver-drowsiness-detection
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

Activate the environment.

**Windows:**

```bash
venv\Scripts\activate
```

**Linux / macOS:**

```bash
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## 📦 Requirements

The project requires the following main packages:

```text
opencv-python
numpy
tensorflow
pygame
```

The exact versions should be specified in `requirements.txt` based on the environment in which the project was tested.

---

## ▶️ Running the Project

Make sure:

* A webcam is connected
* `alarm.wav` exists in the expected location
* The trained model exists at:

```text
models/cnnCat2.keras
```

Then run:

```bash
python drowsiness.py
```

The webcam window will open and real-time detection will begin.

### Exit

Press:

```text
ESC
```

to stop the application.

---

## 🔬 Decision Logic

The core decision logic can be summarized as:

```text
                 ┌──────────────┐
                 │ Left Eye     │
                 │ Prediction   │
                 └──────┬───────┘
                        │
                        ▼
                 ┌──────────────┐
                 │ Right Eye    │
                 │ Prediction   │
                 └──────┬───────┘
                        │
                        ▼
              Both Eyes Closed?
                  /           \
                No             Yes
                │               │
                ▼               ▼
          Reset Timer       Start Timer
                                │
                                ▼
                       Closure ≥ 5 sec?
                          /          \
                        No            Yes
                        │              │
                        ▼              ▼
                     Monitor      Trigger Alert
```

This rule-based temporal layer is intentionally simple and easy to interpret.

---

## ⚠️ Limitations

The current implementation has several limitations.

### Haar Cascade Limitations

Haar Cascade detectors can be sensitive to:

* Lighting conditions
* Head rotation
* Face orientation
* Occlusion
* Camera position

### Eye Detection

The eye detectors operate independently across the frame. Therefore, false or incorrect eye detections may occur in some situations.

### Drowsiness Definition

The current system primarily uses **continuous eye closure** as the indicator of drowsiness.

However:

> Eye closure alone is not a complete representation of driver fatigue.

For example, the system does not currently model:

* Head nodding
* Yawning
* Gaze direction
* Blink frequency
* Driving duration
* Steering behavior
* Long-term fatigue patterns

### Threshold-Based Decision

The current threshold is fixed at:

```text
5 seconds
```

A fixed threshold may not be optimal for every person or driving condition.

---

## 📈 Possible Improvements

Several improvements could make the system more robust.

### 1. Better Face Detection

Replace Haar Cascade with a more modern detector such as:

* MediaPipe Face Detection
* YOLO
* RetinaFace

### 2. Better Facial Landmark Detection

Facial landmarks could provide more precise eye localization and enable additional features such as:

* Eye Aspect Ratio (EAR)
* Head pose
* Gaze estimation

### 3. Temporal Deep Learning

Instead of relying only on a fixed time threshold, a temporal model could learn patterns across consecutive frames.

Possible approaches include:

```text
CNN + LSTM
CNN + GRU
3D CNN
Temporal Transformer
```

### 4. PERCLOS

A future version could implement **PERCLOS (Percentage of Eye Closure)** to measure the proportion of time that the eyes remain closed during a predefined time window.

This would provide a more meaningful fatigue-related measure than a single continuous-closure threshold.

### 5. Robustness Evaluation

The system could be evaluated under different conditions:

```text
Normal Lighting
Low Light
Glasses
Different Head Poses
Different Subjects
Different Camera Positions
```

### 6. Performance Optimization

The inference pipeline could be optimized through:

* Frame skipping
* Reduced input resolution
* More efficient face detection
* Model optimization
* TensorFlow Lite / ONNX deployment

---

## 🧪 Evaluation

The CNN classifier and the complete real-time system should be evaluated separately.

### Classification Evaluation

Recommended metrics:

| Metric           | Purpose                              |
| ---------------- | ------------------------------------ |
| Accuracy         | Overall classification performance   |
| Precision        | Reliability of predictions           |
| Recall           | Ability to detect the target class   |
| F1-Score         | Balance between precision and recall |
| Confusion Matrix | Analysis of classification errors    |

### Real-Time Evaluation

The complete system should additionally be evaluated based on:

* Detection latency
* FPS
* False alarm rate
* Missed drowsiness events
* Stability under different lighting conditions

> Quantitative results should only be reported after running the experiments on a clearly defined test set and hardware configuration.

---

## 💡 Key Learning Outcomes

This project provided practical experience with several core computer vision concepts:

* Real-time video capture using OpenCV
* Grayscale image processing
* Haar Cascade object detection
* Region of Interest (ROI) extraction
* Image resizing and normalization
* TensorFlow/Keras model inference
* CNN image classification
* Processing predictions from multiple regions
* Temporal state tracking
* Rule-based event detection
* Real-time visual feedback
* Audio alert integration

---

## 🔍 Technical Takeaways

One of the main ideas explored in this project is the difference between **frame-level classification** and **event-level detection**.

The CNN answers:

```text
"What is the state of this eye in this frame?"
```

while the temporal logic answers:

```text
"Has the driver remained in a potentially dangerous state for long enough?"
```

Combining these two levels makes the application more suitable for a real-time monitoring scenario than using a single-frame prediction alone.

---

## 🛣️ Future Roadmap

### Computer Vision

* [ ] Replace Haar Cascade with a modern face detector
* [ ] Add facial landmark detection
* [ ] Implement Eye Aspect Ratio (EAR)
* [ ] Add head-pose estimation
* [ ] Add yawning detection

### Deep Learning

* [ ] Improve the eye-state classifier
* [ ] Evaluate the model on unseen subjects
* [ ] Add temporal modeling
* [ ] Experiment with CNN + LSTM
* [ ] Compare multiple architectures

### System

* [ ] Implement PERCLOS
* [ ] Improve alert logic
* [ ] Add configurable thresholds
* [ ] Add performance benchmarking
* [ ] Optimize inference for edge devices

---

## ⚖️ Disclaimer

This project is intended for **educational and research purposes**.

It is a computer vision prototype and should **not** be considered a certified driver-monitoring or automotive safety system.

The system may produce false positives or false negatives under real-world conditions.

---

## 📄 License

This project is licensed under the MIT License.

---

## 👤 Author

**Ziba Hatamian**

Machine Learning & Computer Vision

---

## ⭐ Acknowledgment

This project was developed as a practical exploration of real-time computer vision, CNN inference, and OpenCV-based video processing.
