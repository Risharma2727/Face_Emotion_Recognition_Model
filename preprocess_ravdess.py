import cv2
import os
import pandas as pd

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
output_dir = "dataset/images"
os.makedirs(output_dir, exist_ok=True)

emotion_map = {
    '01': 'neutral', '02': 'calm', '03': 'happy', '04': 'sad',
    '05': 'angry', '06': 'fearful', '07': 'disgust', '08': 'surprised'
}

labels = []

def get_emotion_label(filename):
    return emotion_map[filename.split('-')[2]]

def process_video(video_path, actor_id):
    cap = cv2.VideoCapture(video_path)
    count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        for (x, y, w, h) in faces:
            face = gray[y:y+h, x:x+w]
            face = cv2.resize(face, (48, 48))
            filename = f"{actor_id}_{count}.jpg"
            cv2.imwrite(os.path.join(output_dir, filename), face)
            labels.append((filename, get_emotion_label(os.path.basename(video_path))))
            count += 1
    cap.release()

# Loop through all actors
base_dir = "RAVDESS"
for actor in os.listdir(base_dir):
    actor_dir = os.path.join(base_dir, actor)
    for file in os.listdir(actor_dir):
        if file.endswith(".mp4"):
            process_video(os.path.join(actor_dir, file), actor)

# Save labels
df = pd.DataFrame(labels, columns=["filename", "emotion"])
df.to_csv("dataset/labels.csv", index=False)
print("Preprocessing complete.")
