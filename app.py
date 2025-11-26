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

# Custom CSS for a lively, streamlined look (Vibrant Accents, No Emojis)
st.markdown("""
<style>
    /* Main Streamlit container styling */
    .main .block-container {
        padding-top: 3rem;
        padding-bottom: 3rem;
        max-width: 800px;
    }

    /* Title and Header styling */
    h1 {
        font-size: 3.5rem;
        color: #FF4B4B; /* Streamlit Red/Vibrant Accent */
        text-align: center;
        margin-bottom: 0.5rem;
        text-shadow: 1px 1px 2px rgba(0, 0, 0, 0.1);
    }
    
    .stMarkdown p {
        font-size: 1.15rem;
        color: #333333; /* Darker text for readability */
        text-align: center;
        line-height: 1.6;
    }

    /* Subheader for sections */
    h2, h3 {
        color: #4CAF50; /* Green Accent */
        border-bottom: 2px solid #EEEEEE;
        padding-bottom: 5px;
        margin-top: 2rem;
    }

    /* Info box styling */
    div[data-testid="stAlert"] {
        border-radius: 0.75rem;
        border-left: 6px solid #FF4B4B; /* Match Accent */
        background-color: #FFF0F0;
    }
    
    /* Input containers */
    div[data-testid="stFileUploader"] > div:first-child, div[data-testid="stCameraInput"] > div:first-child {
        border: 2px dashed #FF4B4B; /* Vibrant dashed border */
        background-color: #F8F8F8;
        border-radius: 1rem;
        padding: 2rem;
        text-align: center;
        transition: all 0.3s ease;
    }

    /* Hover effect for input containers */
    div[data-testid="stFileUploader"] > div:first-child:hover, div[data-testid="stCameraInput"] > div:first-child:hover {
        background-color: #EDEDED;
        box-shadow: 0 4px 12px rgba(255, 75, 75, 0.2);
    }

    /* Result image border and shadow */
    div.stImage img {
        border-radius: 1.5rem;
        box-shadow: 0 8px 16px rgba(0, 0, 0, 0.2);
        transition: transform 0.3s ease;
    }
    
    div.stImage img:hover {
        transform: scale(1.01);
    }
    
    /* Custom Result Box Styling (Lively) */
    .emotion-result-box {
        background-color: #E8F5E9; /* Light Green Background */
        border: 1px solid #C8E6C9;
        border-radius: 12px;
        padding: 15px;
        margin-top: 15px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
    }

    /* Emotion label color coding */
    .emotion-Angry { color: #D32F2F; font-weight: bold; }
    .emotion-Disgust { color: #7B1FA2; font-weight: bold; }
    .emotion-Fear { color: #303F9F; font-weight: bold; }
    .emotion-Happy { color: #FFD600; font-weight: bold; }
    .emotion-Neutral { color: #616161; font-weight: bold; }
    .emotion-Sad { color: #1976D2; font-weight: bold; }
    .emotion-Surprise { color: #FF9800; font-weight: bold; }

</style>
""", unsafe_allow_html=True)

# -------------------------
# Project Title & Overview
# -------------------------
st.title("InsideOut: Real-Time Emotion Analysis")
st.markdown("""
Welcome to **InsideOut**, a dynamic system designed to instantly analyze facial expressions. 
Simply provide an image—either by upload or a live photo—and let our AI identify the primary emotion! 
We detect: **Angry, Disgust, Fear, Happy, Neutral, Sad, and Surprise.**
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
        with st.spinner("Downloading AI model (25MB)... This may take a moment to establish the connection."):
            try:
                response = requests.get(MODEL_URL, stream=True)
                response.raise_for_status() 
                with open(MODEL_LOCAL_PATH, "wb") as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                st.success("Analysis model loaded successfully!")
            except requests.exceptions.RequestException as e:
                st.error(f"Error downloading model: {e}. Please check your connection and refresh.")
                return None, None
                
    model = load_model(str(MODEL_LOCAL_PATH))
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    return model, face_cascade

model, face_cascade = download_and_load_model()

if model is None or face_cascade is None:
    st.stop()

# -------------------------
# Input Options (Using columns for better layout)
# -------------------------
st.header("1. Input Image")
col1, col2 = st.columns(2)

with col1:
    uploaded_file = st.file_uploader("Upload an Image File", type=["jpg", "jpeg", "png"])

with col2:
    camera_input = st.camera_input("Use Live Camera Snapshot")

# Determine the source of the image data
image_data = uploaded_file if uploaded_file else camera_input

# --- Separator ---
st.markdown("---")

# -------------------------
# Processing Image
# -------------------------
if image_data is not None:
    st.header("2. Analysis in Progress")
    
    # 1. Image Loading and Conversion
    try:
        image = Image.open(image_data).convert("RGB")
    except Exception as e:
        st.error(f"Error loading image: {e}")
        st.stop()
        
    open_cv_image = np.array(image)
    frame = cv2.cvtColor(open_cv_image, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 2. Face Detection
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(40, 40)
    )

    detected_faces_count = len(faces)
    
    if detected_faces_count == 0:
        st.warning("No facial expressions were clearly detected in the image. Please try adjusting the lighting or composition.")
    else:
        st.info(f"Successfully detected {detected_faces_count} face(s). Generating results...")
        
        # 3. Emotion Prediction and Drawing
        
        # Container for results to appear before the image
        results_container = st.container()

        for (x, y, w, h) in faces:
            face_roi = gray[y:y + h, x:x + w]
            
            # Preprocessing
            face_resized = cv2.resize(face_roi, (48, 48))
            face_resized = np.expand_dims(face_resized, axis=-1)
            face_resized = np.expand_dims(face_resized, axis=0)
            face_resized = face_resized / 255.0

            # Prediction
            prediction = model.predict(face_resized, verbose=0)
            emotion_index = np.argmax(prediction)
            emotion = CLASS_LABELS[emotion_index]
            confidence = np.max(prediction)

            # Define color based on emotion (Professional/Lively Palette)
            color_map = {
                'Happy': (40, 255, 100),   # Light Green/Yellow BGR
                'Surprise': (0, 255, 255), # Yellow BGR
                'Sad': (255, 100, 50),     # Light Blue BGR
                'Angry': (0, 0, 255),      # Red BGR
                'Neutral': (150, 150, 150),# Gray BGR
                'Fear': (200, 50, 0),      # Dark Blue BGR
                'Disgust': (50, 200, 50)   # Medium Green BGR
            }
            color = color_map.get(emotion, (255, 255, 255)) # Default White

            # Draw rectangle
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 3) # Thicker border

            # Draw label with background
            label = f"{emotion} ({confidence*100:.1f}%)"
            (text_w, text_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            
            # Draw a filled background box
            cv2.rectangle(frame, (x, y - text_h - baseline - 10), (x + text_w + 10, y), color, -1)
            
            # Draw the text label in black for contrast
            cv2.putText(frame, label, (x + 5, y - baseline - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
            
            
            # Display detailed result box
            with results_container:
                st.markdown(
                    f"""
                    <div class="emotion-result-box">
                        <h3 style="margin-top: 0; color: #FF4B4B;">Face Analysis Report</h3>
                        <p style="text-align: left; margin-bottom: 5px;">
                            <strong>Primary Emotion Detected:</strong> <span class="emotion-{emotion}">{emotion}</span>
                            (Confidence: {confidence*100:.1f}%)
                        </p>
                        <details>
                            <summary style="color: #4CAF50; cursor: pointer;">Show Detailed Probabilities</summary>
                            <ul>
                                {''.join([f'<li><span class="emotion-{CLASS_LABELS[i]}">{CLASS_LABELS[i]}</span>: {prediction[0][i]*100:.1f}%</li>' 
                                          for i in np.argsort(prediction[0])[-len(CLASS_LABELS):][::-1]])}
                            </ul>
                        </details>
                    </div>
                    """, unsafe_allow_html=True
                )


        # 4. Display the Result
        st.subheader("Final Output Image")
        # Convert BGR (from OpenCV) back to RGB (for Streamlit display)
        st.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), caption="Facial Emotion Detection Output", use_container_width=True)

else:
    # Initial state message
    st.info("The system is ready. Please use the controls above to upload or capture an image to begin the analysis.")
    
# -------------------------
# Footer / Credits
# -------------------------
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #9aa0a6; font-size: 0.85rem;'>
        This application is powered by Keras, OpenCV, and Streamlit. The underlying model is InsideOut.
    </div>
    """, unsafe_allow_html=True
)
