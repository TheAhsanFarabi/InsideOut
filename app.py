import streamlit as st
from tensorflow.keras.models import load_model
import cv2
import numpy as np
from PIL import Image
import os
import requests
from pathlib import Path

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
Upload an image or take a live photo, and let the AI detect facial emotions instantly!
Supported emotions: **Angry, Disgust, Fear, Happy, Neutral, Sad, Surprise.**
""")

# --- Separator ---
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
                response.raise_for_status() # Check for request errors
                with open(MODEL_LOCAL_PATH, "wb") as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                st.success("Model downloaded successfully!")
            except requests.exceptions.RequestException as e:
                st.error(f"Error downloading model: {e}. Please check your connection.")
                return None, None
                
    model = load_model(str(MODEL_LOCAL_PATH))
    # Load OpenCV's default frontal face detection cascade
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    return model, face_cascade

# Load resources
model, face_cascade = download_and_load_model()

# Check if model loading was successful
if model is None or face_cascade is None:
    st.stop()

# -------------------------
# Input Options (Using columns for better layout)
# -------------------------
col1, col2 = st.columns(2)

with col1:
    uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])

with col2:
    camera_input = st.camera_input("Take a Live Picture")

# Determine the source of the image data
image_data = uploaded_file if uploaded_file else camera_input

# --- Separator ---
st.markdown("---")

# -------------------------
# Processing Image
# -------------------------
if image_data is not None:
    # 1. Image Loading and Conversion
    try:
        image = Image.open(image_data).convert("RGB")
    except Exception as e:
        st.error(f"Error loading image: {e}")
        st.stop()
        
    open_cv_image = np.array(image)
    # Convert RGB image (from PIL/Streamlit) to BGR (for OpenCV)
    frame = cv2.cvtColor(open_cv_image, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 2. Face Detection
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,  
        minNeighbors=5,
        minSize=(30, 30)
    )

    detected_faces_count = len(faces)
    
    if detected_faces_count == 0:
        st.warning("No faces detected in the image. Please try another one.")
    else:
        st.success(f"Detected {detected_faces_count} face(s). Processing emotions...")
        
        # 3. Emotion Prediction and Drawing
        for (x, y, w, h) in faces:
            # Extract face ROI (Region of Interest)
            face_roi = gray[y:y + h, x:x + w]
            
            # Preprocessing for the Keras model
            face_resized = cv2.resize(face_roi, (48, 48))
            face_resized = np.expand_dims(face_resized, axis=-1)  
            face_resized = np.expand_dims(face_resized, axis=0)   
            face_resized = face_resized / 255.0                  

            # Make prediction
            prediction = model.predict(face_resized, verbose=0)
            emotion_index = np.argmax(prediction)
            emotion = CLASS_LABELS[emotion_index]
            confidence = np.max(prediction)

            # Draw rectangle and label
            # Use a dynamic color based on the emotion
            color_map = {
                'Happy': (40, 200, 255),    # Light Blue/Yellow BGR
                'Angry': (0, 0, 255),       # Red BGR
                'Sad': (255, 0, 0),         # Blue BGR
                'Surprise': (0, 255, 255),  # Yellow BGR
                'Neutral': (128, 128, 128), # Gray BGR
                'Fear': (0, 69, 255),       # Orange BGR 
                'Disgust': (0, 255, 0)      # Green BGR
            }
            color = color_map.get(emotion, (0, 255, 0)) # Default Green

            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            label = f"{emotion} ({confidence*100:.1f}%)"
            
            # Calculate text size for background box
            (text_w, text_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            
            # Draw a filled rectangle as a background for the text
            cv2.rectangle(frame, (x, y - text_h - baseline - 10), (x + text_w + 10, y), color, -1)
            
            # Draw the text label
            cv2.putText(frame, label, (x + 5, y - baseline - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
            
            # Optional: Display top predictions for each face
            top_indices = np.argsort(prediction[0])[-len(CLASS_LABELS):][::-1]
            top_emotions = [(CLASS_LABELS[i], prediction[0][i]) for i in top_indices]
            
            st.subheader(f"Face at ({x}, {y})")
            st.write(f"**Primary Emotion:** {emotion} ({confidence*100:.1f}%)")
            
            with st.expander("Top 3 Probabilities"):
                for e in top_emotions[:3]:
                    st.write(f"- {e[0]}: {e[1]*100:.1f}%")


        # 4. Display the Result
        st.markdown("---")
        st.subheader("Processed Image Result") 
        st.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), caption="Emotion Detection Result", width="stretch")

else:
    # Initial state message
    st.info("Please upload an image or take a picture to begin the emotion detection.")
    
# -------------------------
# Footer / Credits
# -------------------------
st.markdown("---")
st.caption("Powered by Keras, OpenCV, and Streamlit. Model: InsideOut by Ahsan Farabi.")
