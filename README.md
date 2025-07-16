# 🎭 Multimodal Emotion Recognition System (Facial + Audio)

This project implements a real-time emotion recognition system using both **facial expressions** and **audio signals**. It leverages deep learning models trained on standard datasets to detect emotions from webcam video and microphone input.

---

## 📦 Datasets Used

### 🧠 Facial Emotion Dataset
- **Source**: [FER2013](https://www.kaggle.com/datasets/msambare/fer2013)
- **Classes**: Angry, Disgust, Fear, Happy, Neutral, Sad, Surprise
- **Format**: Grayscale images (48x48), organized by emotion folders

### 🎙️ Audio Emotion Dataset
- **Source**: [RAVDESS](https://www.kaggle.com/datasets/uwrfkaggler/ravdess-emotional-speech-audio)
- **Classes**: Neutral, Calm, Happy, Sad, Angry, Fearful, Disgust, Surprised
- **Format**: `.wav` files with emotion encoded in filename

---

## 🧠 Algorithms Used

- **Facial Emotion Recognition**: Convolutional Neural Network (CNN)
- **Audio Emotion Recognition**: CNN trained on MFCC features
- **Feature Extraction**:
  - Facial: Resized grayscale image → normalized → CNN input
  - Audio: MFCC + delta features → mean pooled → CNN input
- **Smoothing**: Rolling window (deque) for stable predictions
- **Confidence Filtering**: Only predictions above threshold are accepted

---

## 🔄 Flow of Code Execution

1. **Load Models**: Facial and audio models are loaded from `.json` and `.h5` files
2. **Start Webcam**: Captures frames and detects faces using Haar cascade
3. **Face Prediction**:
   - Every few frames, face is cropped and passed to CNN
   - Emotion is predicted and smoothed over time
4. **Manual Audio Trigger**:
   - Press `'a'` to record 2 seconds of audio
   - MFCC features are extracted and passed to audio model
   - Emotion is predicted and smoothed
5. **Display Output**:
   - Webcam frame shows bounding box, facial emotion, and audio emotion
   - Press `'q'` to quit

---

## 🖼️ Flow of Output

| Component        | Behavior                                                                |
|------------------|-------------------------------------------------------------------------|
| Webcam Feed      | Real-time video with face detection and emotion overlay                 |
| Facial Emotion   | Updated every 10 frames, smoothed for stability                         |
| Audio Emotion    | Triggered manually, displayed below webcam feed                         |
| Rectangle Flicker| Eliminated using face hold mechanism                                    |
| Prediction Delay | Minimized with optimized intervals and threading                        |

---

## 🎮 Manual Controls

- Press `'a'` → Record audio and predict emotion  
- Press `'q'` → Quit the application cleanly

---

## 📊 Accuracy

| Model        | Accuracy (on full dataset) |
|--------------|----------------------------|
| Facial CNN   | ~96.09%–97.89% (FER2013)   |
| Audio CNN    | ~92–95.3% (RAVDESS)        |

> Accuracy can be improved with transfer learning, data augmentation, and balanced datasets.

---

## ✅ Results

- Real-time multimodal emotion detection with smooth UI
- Stable predictions using temporal smoothing
- Manual audio trigger for controlled evaluation
- Modular codebase for easy extension and retraining


---

## 🚀 Future Improvements

- Fuse face and audio predictions into a unified emotion output  
- Add real-time emotion dashboard or logging  
- Deploy as a desktop or web app  
- Use MobileNetV2 or EfficientNet for higher accuracy

---





