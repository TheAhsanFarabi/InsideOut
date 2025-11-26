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

# Custom CSS for a modern look
st.markdown("""
<style>
    /* Main Streamlit container styling */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        padding-left: 1rem;
        padding-right: 1rem;
    }

    /* Title and Header styling */
    h1 {
        font-size: 3rem;
        color: #1a73e8; /* Google Blue */
        text-align: center;
        margin-bottom: 0.5rem;
    }
    
    .stMarkdown p {
        font-size: 1.1rem;
        color: #5f6368; /* Grey text */
        text-align: center;
    }

    /* Info box styling */
    div[data-testid="stAlert"] {
        border-radius: 0.5rem;
        border-left: 8px solid #fbbc05; /* Google Yellow */
    }

    /* File uploader and camera input container */
    div[data-testid="stFileUploader"] > div:first-child, div[data-testid="stCameraInput"] > div:first-child {
        border: 2px dashed #dadce0;
        border-radius: 0.75rem;
        padding: 1.5rem;
        text-align: center;
    }

    /* Result image border */
    div.stImage img {
        border-radius: 1rem;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
    }
    
    /* Emotion label color coding (Optional, can be expanded) */
    .emotion-Angry { color: #dc3545; }
    .emotion-Happy { color: #28a745; }
    .emotion-Surprise { color: #ffc107; }
    .emotion-Neutral { color: #6c757d; }
    /* ... add more classes for other emotions */

</style>
""", unsafe_allow_html=True)

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
# Use Pathlib for better path handling
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
    uploaded_file = st.file_uploader("🖼️ Upload an Image", type=["jpg", "jpeg", "png"])

with col2:
    camera_input = st.camera_input("📸 Take a Live Picture")

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
        scaleFactor=1.1,  # Reduced for slightly better detection
        minNeighbors=5,
        minSize=(30, 30)
    )

    detected_faces_count = len(faces)
    
    if detected_faces_count == 0:
        st.warning("⚠️ No faces detected in the image. Please try another one.")
    else:
        st.success(f"✅ Detected {detected_faces_count} face(s). Processing emotions...")
        
        # 3. Emotion Prediction and Drawing
        for (x, y, w, h) in faces:
            # Extract face ROI (Region of Interest)
            face_roi = gray[y:y + h, x:x + w]
            
            # Preprocessing for the Keras model
            face_resized = cv2.resize(face_roi, (48, 48))
            face_resized = np.expand_dims(face_resized, axis=-1)  # Add channel dimension
            face_resized = np.expand_dims(face_resized, axis=0)   # Add batch dimension
            face_resized = face_resized / 255.0                  # Normalize

            # Make prediction
            prediction = model.predict(face_resized, verbose=0)
            emotion_index = np.argmax(prediction)
            emotion = CLASS_LABELS[emotion_index]
            confidence = np.max(prediction)

            # Draw rectangle and label
            # Use a dynamic color based on the emotion (optional, but adds flair)
            color = (0, 255, 0) # Default Green
            if emotion == 'Happy': color = (40, 200, 255) # Light Blue/Yellow
            elif emotion == 'Angry': color = (0, 0, 255) # Red
            elif emotion == 'Sad': color = (255, 0, 0) # Blue
            elif emotion == 'Surprise': color = (0, 255, 255) # Yellow

            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            label = f"{emotion} ({confidence*100:.1f}%)"
            
            # Calculate text size for background box
            (text_w, text_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            
            # Draw a filled rectangle as a background for the text
            cv2.rectangle(frame, (x, y - text_h - baseline - 10), (x + text_w + 10, y), color, -1)
            
            # Draw the text label
            cv2.putText(frame, label, (x + 5, y - baseline - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
            
            # Optional: Display top 3 predictions in a sidebar/expander
            top_indices = np.argsort(prediction[0])[-3:][::-1]
            top_emotions = [(CLASS_LABELS[i], prediction[0][i]) for i in top_indices]
            
            st.markdown(
                f"""
                <div style="border: 1px solid #dadce0; border-radius: 8px; padding: 10px; margin-top: 10px;">
                    <h3 style="margin-top: 0; font-size: 1.25rem;">Face at ({x}, {y})</h3>
                    <p style="text-align: left; margin-bottom: 5px;">
                        <strong>Primary Emotion:</strong> <span class="emotion-{emotion}">{emotion} ({confidence*100:.1f}%)</span>
                    </p>
                    <details>
                        <summary>Top 3 Probabilities</summary>
                        <ul>
                            {''.join([f'<li>{e}: {p*100:.1f}%</li>' for e, p in top_emotions])}
                        </ul>
                    </details>
                </div>
                """, unsafe_allow_html=True
            )


        # 4. Display the Result
        st.markdown("---")
        st.subheader("🖼️ Processed Image Result")
        # Convert BGR (from OpenCV) back to RGB (for Streamlit display)
        st.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), caption="Emotion Detection Result", use_container_width=True)

else:
    # Initial state message
    st.info("Please upload an image or take a picture to begin the emotion detection.")
    
# -------------------------
# Footer / Credits
# -------------------------
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #9aa0a6; font-size: 0.85rem;'>
        Powered by Keras, OpenCV, and Streamlit. Model: InsideOut by Ahsan Farabi.
    </div>
    """, unsafe_allow_html=True
)
