import streamlit as st
from tensorflow.keras.models import load_model
import cv2
import numpy as np
from PIL import Image
import os
import requests
from pathlib import Path
from streamlit_webrtc import webrtc_streamer
import av
import threading

# Global lock for thread-safe TensorFlow inference
inference_lock = threading.Lock()

# --- Configuration & Styling ---
st.set_page_config(
    page_title="InsideOut: Real-Time Emotion Detection",
    layout="centered",
    initial_sidebar_state="expanded"
)

# -------------------------
# Project Title & Overview
# -------------------------
st.title("InsideOut: An Emotion Recognition System")
st.markdown("""
Welcome to **InsideOut**, a real-time emotion recognition system.
Use your webcam for live emotion detection, or upload a static image!
Supported emotions: **Angry, Disgust, Fear, Happy, Neutral, Sad, Surprise.**
""")

st.markdown("---")

# -------------------------
# Constants & Configuration
# -------------------------
MODEL_URL = "https://huggingface.co/AhsanFarabi/inside_out/resolve/main/inside_out.h5"
MODEL_LOCAL_PATH = Path("inside_out.h5")
CLASS_LABELS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

# -------------------------
# Load Model & Cascade
# -------------------------
@st.cache_resource
def download_and_load_model():
    """Downloads model from URL and loads it along with the face cascade."""
    if not MODEL_LOCAL_PATH.exists():
        with st.spinner("⏳ Downloading AI model (25MB)... This may take a moment."):
            try:
                response = requests.get(MODEL_URL, stream=True)
                response.raise_for_status() 
                with open(MODEL_LOCAL_PATH, "wb") as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                st.success("Model downloaded successfully!")
            except requests.exceptions.RequestException as e:
                st.error(f"Error downloading model: {e}. Please check your connection.")
                return None, None
                
    model = load_model(str(MODEL_LOCAL_PATH))
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    return model, face_cascade

model, face_cascade = download_and_load_model()

if model is None or face_cascade is None:
    st.stop()

# -------------------------
# Core Processing Function
# -------------------------
def process_frame(frame_bgr):
    """Detects faces, predicts emotions, and draws bounding boxes."""
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
    )
    
    face_details = []
    
    for (x, y, w, h) in faces:
        face_roi = gray[y:y + h, x:x + w]
        face_resized = cv2.resize(face_roi, (48, 48))
        face_resized = np.expand_dims(face_resized, axis=-1)  
        face_resized = np.expand_dims(face_resized, axis=0)   
        face_resized = face_resized / 255.0                  

        with inference_lock:
            prediction = model(face_resized, training=False).numpy()
            
        emotion_index = np.argmax(prediction)
        emotion = CLASS_LABELS[emotion_index]
        confidence = np.max(prediction)
        
        top_indices = np.argsort(prediction[0])[-len(CLASS_LABELS):][::-1]
        top_emotions = [(CLASS_LABELS[i], prediction[0][i]) for i in top_indices]
        
        face_details.append({
            'coords': (x, y, w, h),
            'emotion': emotion,
            'confidence': confidence,
            'top_emotions': top_emotions
        })

        color_map = {
            'Happy': (40, 200, 255), 'Angry': (0, 0, 255), 'Sad': (255, 0, 0),
            'Surprise': (0, 255, 255), 'Neutral': (128, 128, 128),
            'Fear': (0, 69, 255), 'Disgust': (0, 255, 0)
        }
        color = color_map.get(emotion, (0, 255, 0))

        cv2.rectangle(frame_bgr, (x, y), (x + w, y + h), color, 2)
        label = f"{emotion} ({confidence*100:.1f}%)"
        (text_w, text_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        cv2.rectangle(frame_bgr, (x, y - text_h - baseline - 10), (x + text_w + 10, y), color, -1)
        cv2.putText(frame_bgr, label, (x + 5, y - baseline - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        
    return frame_bgr, face_details

# -------------------------
# UI Layout (Tabs)
# -------------------------
tab1, tab2 = st.tabs(["🎥 Live Video (WebRTC)", "🖼️ Static Image"])

with tab1:
    st.info("Click 'Start' to allow webcam access and begin live emotion detection.")
    
    def video_frame_callback(frame):
        img = frame.to_ndarray(format="bgr24")
        processed_img, _ = process_frame(img)
        return av.VideoFrame.from_ndarray(processed_img, format="bgr24")

    webrtc_streamer(
        key="emotion-detection",
        video_frame_callback=video_frame_callback,
        rtc_configuration={
            "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
        },
        media_stream_constraints={"video": True, "audio": False}
    )

with tab2:
    col1, col2 = st.columns(2)
    with col1:
        uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])
    with col2:
        camera_input = st.camera_input("Take a Static Picture")
        
    image_data = uploaded_file if uploaded_file else camera_input
    
    if image_data is not None:
        try:
            image = Image.open(image_data).convert("RGB")
        except Exception as e:
            st.error(f"Error loading image: {e}")
            st.stop()
            
        open_cv_image = np.array(image)
        frame = cv2.cvtColor(open_cv_image, cv2.COLOR_RGB2BGR)
        
        processed_frame, details = process_frame(frame)
        
        if len(details) == 0:
            st.warning("No faces detected in the image. Please try another one.")
        else:
            st.success(f"Detected {len(details)} face(s). Processing emotions...")
            
            for face in details:
                x, y, w, h = face['coords']
                st.subheader(f"Face at ({x}, {y})")
                st.write(f"**Primary Emotion:** {face['emotion']} ({face['confidence']*100:.1f}%)")
                with st.expander("Top 3 Probabilities"):
                    for e in face['top_emotions'][:3]:
                        st.write(f"- {e[0]}: {e[1]*100:.1f}%")
                        
        st.markdown("---")
        st.subheader("Processed Image Result") 
        st.image(cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB), caption="Emotion Detection Result", width="stretch")

# -------------------------
# Footer / Credits
# -------------------------
st.markdown("---")
st.caption("Powered by Keras, OpenCV, WebRTC, and Streamlit. Model: InsideOut by Ahsan Farabi.")
