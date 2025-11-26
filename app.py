import streamlit as st
from tensorflow.keras.models import load_model
import cv2
import numpy as np
from PIL import Image
import os
import requests

# -------------------------
# Project Title & Overview
# -------------------------
st.set_page_config(page_title="InsideOut: Real-Time Emotion Detection", layout="centered")
st.title("InsideOut: An Emotion Recognition System")
st.markdown("""
Welcome to **InsideOut**, a real-time emotion recognition system.  
Upload an image or take a live photo, and let the AI detect facial emotions instantly!  
Supported emotions: Angry, Disgust, Fear, Happy, Neutral, Sad, Surprise.
""")

# -------------------------
# Constants & Configuration
# -------------------------
MODEL_URL = "https://huggingface.co/AhsanFarabi/inside_out/resolve/main/inside_out.h5"
MODEL_LOCAL_PATH = "inside_out.h5"
CLASS_LABELS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

# -------------------------
# Load Model & Cascade
# -------------------------
@st.cache_resource
def download_and_load_model():
    if not os.path.exists(MODEL_LOCAL_PATH):
        with st.spinner("⏳ Downloading model..."):
            response = requests.get(MODEL_URL)
            with open(MODEL_LOCAL_PATH, "wb") as f:
                f.write(response.content)
    model = load_model(MODEL_LOCAL_PATH)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    return model, face_cascade

model, face_cascade = download_and_load_model()

# -------------------------
# Input Options
# -------------------------
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
camera_input = st.camera_input("📷 Or take a picture")

image_data = uploaded_file if uploaded_file else camera_input

# -------------------------
# Processing Image
# -------------------------
if image_data is not None:
    image = Image.open(image_data).convert("RGB")
    open_cv_image = np.array(image)
    frame = cv2.cvtColor(open_cv_image, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        face_roi = gray[y:y + h, x:x + w]
        face_resized = cv2.resize(face_roi, (48, 48))
        face_resized = np.expand_dims(face_resized, axis=-1)
        face_resized = np.expand_dims(face_resized, axis=0)
        face_resized = face_resized / 255.0

        prediction = model.predict(face_resized)
        emotion = CLASS_LABELS[np.argmax(prediction)]
        confidence = np.max(prediction)

        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        label = f"{emotion} ({confidence*100:.1f}%)"
        cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

    # Display the result
    st.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), caption="Emotion Detection Result", use_container_width=True)
else:
    st.info("Please upload an image or take a picture to begin.")
