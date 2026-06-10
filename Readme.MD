# 🎭 AI Emotion Detector

Detects **facial emotions in real time** from a webcam feed or uploaded photo using a pre-trained DeepFace model. Classifies 7 emotions: happy, sad, angry, surprise, fear, disgust, neutral.

## 📁 Project Structure
```
emotion-detector/
├── src/
│   ├── detector.py    ← Core: DeepFace analysis + OpenCV drawing
│   └── webcam.py      ← Live webcam mode (OpenCV window)
├── app.py             ← Streamlit image-upload interface
└── requirements.txt
```

## 🚀 Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```
> ⚠️ DeepFace will automatically download the pre-trained model (~100MB) on first run.

### 2a. Run the web app (recommended)
```bash
streamlit run app.py
```
Open http://localhost:8501 — upload any photo to detect emotions.

### 2b. Run live webcam mode
```bash
python src/webcam.py
```
A window opens showing your webcam with emotion labels overlaid. Press **Q** to quit.

## 🧠 How It Works

| Step | What happens |
|------|-------------|
| **Face detection** | DeepFace uses OpenCV's Haar cascade to find faces |
| **Emotion model** | Pre-trained mini-Xception CNN predicts 7 emotion classes |
| **Overlay** | OpenCV draws bounding boxes + labels onto each frame |
| **Optimisation** | Webcam mode only runs DeepFace every 5 frames for smoothness |

## 📊 Emotions Detected
😄 Happy · 😢 Sad · 😠 Angry · 😲 Surprise · 😨 Fear · 🤢 Disgust · 😐 Neutral

## 💡 Ideas to Extend This
- Log emotion history over time and plot a mood graph
- Add age and gender estimation (DeepFace supports both!)
- Build a "mood journal" that saves timestamped snapshots
- Deploy to Hugging Face Spaces for a live public demo
- Add audio: play a song that matches the detected mood
