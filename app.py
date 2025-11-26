import streamlit as st
from tensorflow.keras.models import load_model
import cv2
import numpy as np
from PIL import Image
import os
import requests

# ---------------------------------------------------
# Modern UI Styling (CSS)
# ---------------------------------------------------
st.set_page_config(page_title="InsideOut: Real-Time Emotion Detection", layout="wide")

st.markdown("""
<style>

    /* Page background */
    .stApp {
        background: linear-gradient(135deg, #2b5876, #4e4376);
        color: #ffffff;
    }

    /* Title styling */
    .title-container {
        text-align: center;
        padding: 10px 0 30px 0;
    }
    .title-container h1 {
        font-size: 3rem;
        font-weight: 700;
        color: #ffffff;
    }

    /* Description box */
    .description-box {
        background: rgba(255, 255, 255, 0.12);
        padding: 20px;
        border-radius: 12px;
        text-align: center;
        margin-bottom: 35px;
        font-size: 1.1rem;
    }

    /* Uploader section */
    .upload-section {
        background: rgba(255, 255, 255, 0.10);
        padding: 25px;
        border-radius: 12px;
        backdrop-filter: blur(6px);
    }

    /* Result Styling */
    .result-container {
        margin-top: 25px;
    }

</style>
""", unsafe_allow_html=True)

# -------------------------
# Page Title
# -------------------------
st.markdown('<div class="title-container"><h1>InsideOut: Emotion Recognition</h1></div>', unsafe_allow_html=True)

# -------------------------
# Description
# -------------------------
st.markdown("""
<div class="description-box">
Welcome to <strong>InsideOut</strong> — a real-time emotion recognition system.<br>
Upload an image or take a live picture, and let the AI detect facial emotions instantly.<br>
Supported emotions: Angry, Disgust, Fear, Happy, Neutral, Sad, Surprise.
</div>
""", unsafe_allow_html=True)

# -------------------------
# Constants
# -------------------------
MODEL_URL = "https://huggingface.co/AhsanFarabi/inside_out/resolve/main/inside_out.h5"
MODEL_LOCAL_PATH = "inside_out.h5"
CLASS_LABELS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']


# -------------------------
# Load Model
# -------------------------
@st.cache_resource
def download_and_load_model():
    if not os.path.exists(MODEL_LOCAL_PATH):
        with st.spinner("Downloading model..."):
            response = requests.get(MODEL_URL)
            with open(MODEL_LOCAL_PATH, "wb") as f:
                f.write(response.content)
    model = load_model(MODEL_LOCAL_PATH)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    return model, face_cascade

model, face_cascade = download_and_load_model()

# -------------------------
# Input Area
# -------------------------
st.subheader("Upload or Capture Image")

col1, col2 = st.columns([1,1])

with col1:
    st.markdown('<div class="upload-section">', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="upload-section">', unsafe_allow_html=True)
    camera_input = st.camera_input("Or take a picture")
    st.markdown('</div>', unsafe_allow_html=True)

image_data = uploaded_file if uploaded_file else camera_input

# -------------------------
# Image Processing
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

        cv2.rectangle(frame, (x, y), (x + w, y + h), (64, 255, 128), 2)
        label = f"{emotion} ({confidence*100:.1f}%)"
        cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.8, (255, 255, 0), 2)

    st.markdown('<div class="result-container">', unsafe_allow_html=True)
    st.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), caption="Emotion Detection Result", use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

else:
    st.info("Please upload an image or take a picture to begin.")
