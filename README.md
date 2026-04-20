# 🤖 Sign2Text

> **Real-time ASL Sign Recognition**  
> Convert hand gestures into text using computer vision and deep learning.

---

## ✨ Features

- 🎥 **Real-time Detection** — Recognize ASL signs through your webcam instantly
- 🧠 **Custom CNN Model** — Lightweight custom convolutional neural network (240×240 input)
- 🤲 **Gesture Support** — Three intuitive gestures:
  - **Open Palm** = Add Space
  - **Two Open Palms** = Delete Last Character  
  - **ASL Signs** = Recognize A-Z letters + blank
- 📊 **Confidence Display** — Visual feedback with stability bars and confidence percentages
- ⚡ **Optimized** — Fast inference with preprocessing pipeline and stability thresholds

---

## 🎬 Demo

![ASL Sign2Language Demo](./demo/demo.gif)

---
## 🚀 Quick Start

### Prerequisites
- Python 3.7+
- Webcam
- ~100MB disk space (for dependencies)

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/ThaGeekiestOne/sign2text.git
cd sign2text
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Run the application**
```bash
python src/inference.py
```

---

## 🎮 Usage

Once the app is running:

### Controls
| Input | Action |
|-------|--------|
| **Open Palm** | Add space to word |
| **Both Palms Open** | Delete last character |
| **ASL Sign** | Recognize and add letter (A-Z) |
| **`q`** | Quit application |

### Display
- **Top Bar** — Current word being typed
- **Hand Box** — Green (confident) or Orange (uncertain)
- **Confidence Bar** — Visual representation of prediction confidence
- **Stability Meter** — Shows how many frames the current sign has been held

---

## 🧪 Configuration

Edit `src/inference.py` to adjust these parameters:

```python
# Detection thresholds
STABLE_THRESHOLD = 6        # Frames to hold before capturing
MIN_CONFIDENCE = 60.0       # Confidence threshold (%)
CAPTURE_COOLDOWN = 3.0      # Cooldown between captures (seconds)

# Gesture thresholds
PALM_HOLD = 20              # Frames to hold palm for space
GESTURE_COOLDOWN = 2.0      # Cooldown between gestures (seconds)

# Model input
IMG_SIZE = 240              # Input image size (240×240)
```

---

## 📊 Model Architecture

**Input:** 240×240 RGB image  
**Output:** 27 classes (A-Z + blank/space)

### Layers
```
Conv2D(32) → BatchNorm → MaxPool
    ↓
Conv2D(64) → BatchNorm → MaxPool
    ↓
Conv2D(128) → BatchNorm → MaxPool
    ↓
Flatten
    ↓
Dense(256) → BatchNorm → Dropout(0.5)
    ↓
Dense(128) → Dropout(0.3)
    ↓
Dense(27) → Softmax
```

**Total Parameters:** 8.5M  
**Training Data:** 12,845 images (27 classes)  
**Test Data:** 4,268 images

---

## 📚 Dataset

Training data sourced from:  
🔗 [ASL Signs Dataset - Kaggle](https://www.kaggle.com/datasets/amakii/asl-signs-preprocessed)

- 12,845 training images
- 4,268 test images
- 27 classes (A-Z + blank)
- Pre-preprocessed and balanced

---

## 🔍 How It Works

### 1. **Hand Detection**
   - MediaPipe detects hand landmarks in real-time
   - Bounding box extracted with padding

### 2. **Preprocessing**
   - Resize to 240×240
   - Convert to grayscale
   - Adaptive thresholding
   - Edge detection & dilation
   - Normalize to [0, 1]

### 3. **Prediction**
   - Forward pass through CNN
   - Argmax to get predicted class
   - Stability check (must be stable for N frames)
   - Confidence threshold filtering

### 4. **Gesture Recognition**
   - Open palm detection (4 fingers extended)
   - Both palms detection for delete

---

## 📈 Performance

| Metric | Value |
|--------|-------|
| **Validation Accuracy** | ~97% |
| **Model Size** | ~70 MB |
| **Inference Time** | ~50-100ms |
| **FPS** | 20-30 |

---

## 🛠️ Troubleshooting

### **App won't start**
- Ensure webcam is connected
- Check Python version (3.7+)
- Verify `best_asl_eff.h5` is in the root directory

### **Low detection accuracy**
- Ensure proper lighting and white background
- Keep hand in frame and centered
- Hold gesture steady for required frames
- Adjust `STABLE_THRESHOLD` or `MIN_CONFIDENCE`

### **Slow performance**
- Close background apps
- Reduce `PREDICT_EVERY` value
- Check GPU availability (TensorFlow)

---

## 🤝 Contributing

Contributions welcome! Feel free to:
- Report bugs
- Suggest improvements
- Add new gestures
- Improve preprocessing

---

## 📜 License

MIT License — See LICENSE file for details

---

## 🙏 Acknowledgments

- **MediaPipe** — Hand detection framework
- **TensorFlow/Keras** — Deep learning
- **OpenCV** — Image processing
- **Kaggle** — Dataset source

---

## 📧 Contact

Questions or feedback? Feel free to reach out!

---

**Made with ❤️ for ASL Recognition**
