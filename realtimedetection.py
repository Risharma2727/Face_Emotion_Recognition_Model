import cv2
import numpy as np
import threading
import sounddevice as sd
import librosa
from collections import deque, Counter
from tensorflow.keras.models import model_from_json, Sequential
import tensorflow as tf

# ✅ Load and compile face model
with open("facialemotionmodel.json") as f:
    face_model = model_from_json(f.read(), custom_objects={"Sequential": Sequential})
face_model.load_weights("facialemotionmodel.h5")
face_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# ✅ Load and compile audio model
with open("audio_emotion_model.json") as f:
    audio_model = model_from_json(f.read(), custom_objects={"Sequential": Sequential})
audio_model.load_weights("audio_emotion_model.weights.h5")
audio_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# 🎭 Emotion labels
face_labels = {0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy', 4: 'neutral', 5: 'sad', 6: 'surprise'}
audio_labels = {0: 'neutral', 1: 'calm', 2: 'happy', 3: 'sad', 4: 'angry', 5: 'fearful', 6: 'disgust', 7: 'surprised'}

# 🧠 Haar cascade for face detection
haar_file = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(haar_file)

# 🎙️ Audio settings
SAMPLE_RATE = 22050
DURATION = 2
audio_emotion = "None"

def extract_face_features(image):
    image = image.reshape(1, 48, 48, 1) / 255.0
    return image

def record_audio():
    global audio_emotion
    try:
        audio = sd.rec(int(SAMPLE_RATE * DURATION), samplerate=SAMPLE_RATE, channels=1)
        sd.wait()
        audio = audio.flatten()
        mfccs = librosa.feature.mfcc(y=audio, sr=SAMPLE_RATE, n_mfcc=40)
        mfccs_processed = np.mean(mfccs.T, axis=0)
        audio_features = np.expand_dims(mfccs_processed, axis=0)
        pred = audio_model.predict(audio_features)
        audio_emotion = audio_labels[pred.argmax()]
        print(f"[AUDIO EMOTION] {audio_emotion}")
    except Exception as e:
        print(f"[ERROR] Audio processing failed: {e}")

def start_audio_thread():
    audio_thread = threading.Thread(target=record_audio, daemon=True)
    audio_thread.start()

# 🎥 Webcam setup
webcam = cv2.VideoCapture(0)
webcam.set(cv2.CAP_PROP_BUFFERSIZE, 1)

print("[INFO] Press 'a' to record audio | 'q' to quit")

frame_count = 0
detect_interval = 50  # Detect faces every 5 frames
predict_interval = 100  # Predict emotion every 10 frames
last_prediction_frame = 0
face_history = deque(maxlen=10)
stable_emotion = "neutral"

while True:
    ret, frame = webcam.read()
    if not ret:
        continue

    frame = cv2.resize(frame, (640, 480))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    frame_count += 1
    if frame_count % detect_interval == 0:
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)
    else:
        faces = []

    for (x, y, w, h) in faces:
        face_img = gray[y:y + h, x:x + w]
        face_img = cv2.resize(face_img, (48, 48))
        img = extract_face_features(face_img)

        # 🔄 Predict only every few frames
        if frame_count - last_prediction_frame >= predict_interval:
            pred = face_model.predict(img)
            face_emotion = face_labels[pred.argmax()]
            face_history.append(face_emotion)
            stable_emotion = Counter(face_history).most_common(1)[0][0]
            last_prediction_frame = frame_count

        # 🖼️ Display stable emotion
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.putText(frame, stable_emotion, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

    cv2.putText(frame, f"Audio Emotion: {audio_emotion}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    cv2.imshow("Multimodal Emotion Detection", frame)

    key = cv2.waitKey(10) & 0xFF
    if key == ord('a'):
        print("[INFO] Recording audio...")
        start_audio_thread()
    elif key == ord('q'):
        break

webcam.release()
cv2.destroyAllWindows()
