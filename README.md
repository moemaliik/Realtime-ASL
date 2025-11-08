<<<<<<< HEAD
# Realtime-ASL
 A Real-Time Sign Language Recognition System
=======
# Realtime-ASL  
*A Real-Time Sign Language Recognition System*  
**Author:** Mwafag Malik Omer Ahmed  

---

## Table of Contents  

1. [Overview](#overview)  
2. [Key Features](#key-features)  
3. [System Architecture](#system-architecture)  
   - [Core Components](#core-components)  
   - [Recognition Pipeline](#recognition-pipeline)  
4. [Requirements](#requirements)  
   - [Hardware Requirements](#hardware-requirements)  
   - [Software Requirements](#software-requirements)  
   - [Python Dependencies](#python-dependencies)  
5. [Quick Start](#quick-start)  
6. [How to Use](#how-to-use)  
   - [Basic Gestures](#basic-gestures)  
   - [Control Gestures](#control-gestures)  
   - [Using the Interface](#using-the-interface)  
7. [Technical Details](#technical-details)  
   - [CNN Model Architecture](#cnn-model-architecture)  
   - [Hand Landmark Detection](#hand-landmark-detection)  
   - [Gesture Smoothing](#gesture-smoothing)  
8. [Project Structure](#project-structure)  
9. [Development](#development)  
   - [Data Collection](#data-collection)  
   - [Model Training](#model-training)  
   - [Customization](#customization)  
10. [Troubleshooting](#troubleshooting)  
    - [Common Issues](#common-issues)  
    - [Performance Optimization](#performance-optimization)  
11. [Performance Metrics](#performance-metrics)  
12. [Contributing](#contributing)  
13. [License](#license)  
14. [Acknowledgments](#acknowledgments)  
15. [Support](#support)  
16. [Future Enhancements](#future-enhancements)  

---

## Overview  

**Realtime-ASL** is a computer vision system designed to translate **American Sign Language (ASL)** gestures into **text and speech** in real time. It leverages **deep learning**, **hand landmark detection**, and **CNN-based classification** to facilitate seamless communication between deaf/hard-of-hearing individuals and the hearing community.

The system processes hand gestures via webcam, identifies ASL letters, assembles them into words, and produces text and speech outputs through an intuitive and modern interface.

---

## Key Features  

- **Real-Time Recognition:** Instant ASL-to-text conversion  
- **High Accuracy:** Achieves up to 97–99% accuracy under optimal conditions  
- **Modern Web Interface:** Responsive glassmorphism UI for clean visualization  
- **Text-to-Speech Integration:** Converts recognized text into natural speech  
- **Word Suggestions:** Intelligent spell-checking and word completion  
- **Gesture Smoothing:** Stability ensured through temporal filtering  
- **Cross-Platform:** Compatible with Windows, macOS, and Linux  
- **Multiple Interfaces:** Accessible via both web and desktop applications  

---

## System Architecture  

### Core Components  

1. **Hand Detection:** Uses *MediaPipe* for real-time hand landmark extraction  
2. **Skeleton Generation:** Converts landmarks to standardized skeleton images  
3. **CNN Classification:** Performs 8-group gesture classification  
4. **Post-Processing:** Refines predictions through mathematical rules  
5. **Text Processing:** Implements spell-check and word suggestion  
6. **Speech Synthesis:** Utilizes Google Text-to-Speech (gTTS) with offline fallback  

### Recognition Pipeline  

```
Camera Input → Hand Detection → Skeleton Drawing → CNN Classification → 
Group Classification → Letter Refinement → Text Assembly → Speech Output
```

---

## Requirements  

### Hardware Requirements  

| Component | Minimum | Recommended |
|------------|----------|-------------|
| **Webcam** | Built-in or USB camera | HD camera |
| **RAM** | 4 GB | 8 GB |
| **CPU** | Dual-core processor | Quad-core or higher |
| **Storage** | 500 MB | 1 GB free |

### Software Requirements  

- **OS:** Windows 8+, macOS 10.14+, or Linux  
- **Python:** Version 3.8 or higher  
- **Browser:** Chrome, Firefox, Safari, or Edge  

### Python Dependencies  

```bash
pip install opencv-python numpy tensorflow keras mediapipe cvzone
pip install flask flask-socketio pyttsx3 gtts pygame enchant
pip install pillow base64 math time
```

---

## Quick Start  

1. **Install dependencies:**  
   ```bash
   pip install -r requirements.txt
   ```  
2. **Run the application:**  
   ```bash
   python app.py
   ```  
3. **Access in browser:**  
   Visit `http://localhost:5000`  
4. **Allow camera access** when prompted  
5. **Start signing** to see real-time recognition  

---

## How to Use  

### Basic Gestures  

Supports all 26 ASL finger-spelling gestures (A–Z).  
Examples:  
- **A:** Closed fist with thumb on side  
- **B:** Fingers extended, palm forward  
- **C:** Curved hand shape forming “C”  
- **D:** Index extended, others closed  

*(Full list provided in source documentation)*

### Control Gestures  

| Function | Gesture Description |
|-----------|--------------------|
| **Next (Add Character)** | Make E, Y, or B gesture, then point thumb down |
| **Backspace** | All fingers up, palm facing camera |
| **Space** | Make E, S, X, Y, or B gesture with thumb down |

### Using the Interface  

- **Camera View:** Real-time hand feed  
- **Skeleton Display:** Shows hand landmarks  
- **Current Character:** Displays recognized letter  
- **Sentence Builder:** Accumulates text output  
- **Word Suggestions:** Clickable spell-check options  
- **Speak Button:** Converts text to speech  
- **Clear Button:** Resets text buffer  

---

## Technical Details  

### CNN Model Architecture  

#### Stage 1: Group Classification  
CNN classifies gestures into 8 primary groups for similarity reduction.  

#### Stage 2: Letter Refinement  
Mathematical post-processing maps groups to specific ASL letters using hand landmark analysis.

### Hand Landmark Detection  

Uses **MediaPipe’s 21-point model**, identifying each finger and palm landmark for accurate geometric analysis.

### Gesture Smoothing  

- **Temporal Filtering:** Requires 3 stable frames before confirming recognition  
- **Confidence Thresholding:** Filters low-confidence predictions  
- **Gesture History:** Maintains rolling history for stability  

---

## Project Structure  

```
RealtimeASL/
├── app.py
├── templates/
│   └── index.html
├── AtoZ/
│   ├── A/
│   ├── B/
│   └── ...
├── cnn8grps_rad1_model.h5
├── data_collection_final.py
├── data_collection_binary.py
├── white.jpg
├── sign-language-frontend/
│   ├── src/
│   ├── src-tauri/
│   └── package.json
└── README.md
```

---

## Development  

### Data Collection  

1. Run:  
   ```bash
   python data_collection_final.py
   ```  
2. **Controls:**  
   - `s` → Save image  
   - `n` → Next letter  
   - `q` → Quit  

Collect ~180 images per letter for optimal training.  

### Model Training  

- **Framework:** TensorFlow/Keras  
- **Dataset:** 4,680+ images (A–Z)  
- **Optimization:** Adam optimizer, categorical crossentropy loss  
- **Split:** 80/20 training-validation  

### Customization  

- **New Gestures:** Add images, retrain model, and update `predict_gesture()`  
- **UI Changes:** Modify `index.html` or Svelte files under `sign-language-frontend/`  

---

## Troubleshooting  

### Common Issues  

| Issue | Solution |
|--------|-----------|
| Camera not detected | Check browser permissions or USB connection |
| Low accuracy | Improve lighting and background clarity |
| App won’t start | Verify Python 3.8+ and re-install dependencies |
| Model load error | Ensure `cnn8grps_rad1_model.h5` exists and TensorFlow installed |

### Performance Optimization  

- **Performance:** Close background applications, lower camera resolution  
- **Accuracy:** Maintain steady hand and consistent lighting  

---

## Performance Metrics  

| Metric | Value |
|---------|-------|
| **Recognition Accuracy (Optimal)** | 99% |
| **Recognition Accuracy (Average)** | 95–97% |
| **Frame Rate** | 10–15 FPS |
| **CPU Usage** | 30–50% |
| **RAM Usage** | 2–4 GB |

---

## Contributing  

Contributions are welcome via pull requests and issue reports.  

**How to Contribute:**  
1. Fork the repository  
2. Create a feature branch  
3. Implement and test changes  
4. Submit a pull request  

---

## License  

This project is open source. 

---

## Acknowledgments  

- **MediaPipe** — Hand tracking framework  
- **TensorFlow/Keras** — Deep learning architecture  
- **OpenCV** — Computer vision library  
- **Flask** — Web backend  
- **SvelteKit** — Frontend framework  

---

## Support  

For inquiries or issues:  
- Open a GitHub issue  
- Review project documentation
- Contact me if the aforementioned did not help

---

## Future Enhancements  

**Planned Features**  
- Multi-language sign support  
- Full sentence recognition  
- Mobile application support  
- Cloud-based accuracy optimization  

**Research Directions**  
- 3D hand tracking and depth estimation  
- Emotion and facial expression recognition  
- Context-aware sign interpretation  
- Advanced accessibility integration  

