import os
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import model_from_json
from tensorflow.keras.utils import to_categorical

# ✅ Load CNN model
with open("facialemotionmodel.json", "r") as f:
    face_model = model_from_json(f.read(), custom_objects={"Sequential": tf.keras.Sequential})
face_model.load_weights("facialemotionmodel.h5")
face_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# ✅ Load dataset
df = pd.read_csv("images/train_labels.csv")
X, y = [], []

for idx, row in df.iterrows():
    img_path = os.path.join("images/images/train", row["filename"])
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is not None:
        img = cv2.resize(img, (48, 48))
        img = img.reshape(48, 48, 1) / 255.0
        X.append(img)
        y.append(row["emotion"])

X = np.array(X)
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# 🔍 Predict and evaluate on full dataset
y_pred = face_model.predict(X)
y_pred_labels = np.argmax(y_pred, axis=1)

accuracy = accuracy_score(y_encoded, y_pred_labels)

# 💾 Save accuracy to file
with open("cnn_model_accuracy.txt", "w") as f:
    f.write(f"CNN Facial Emotion Model Accuracy: {accuracy * 100:.2f}%\n")

print(f"[RESULT] CNN Facial Emotion Model Accuracy: {accuracy * 100:.2f}%")
