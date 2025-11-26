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

# Custom CSS for a modern look with a blurry background effect
st.markdown("""
<style>
    /* Blurred background effect - Using a placeholder URL */
    .stApp {
        background: url("https://images.unsplash.com/photo-1518066000714-cdcd825c673e?q=80&w=2070&auto=format&fit=crop&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D");
        background-size: cover;
        background-attachment: fixed;
        background-position: center;
    }

    /* Apply blur effect to the background, and darken it */
    .stApp::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: inherit;
        filter: blur(15px) brightness(60%); /* Increased blur strength, added darkening */
        z-index: -1; 
    }

    /* Main Streamlit container styling - Semi-transparent white content block */
    .main .block-container {
        background-color: rgba(255, 255, 255, 0.92); /* Slightly less transparent white for better contrast */
        border-radius: 1rem;
        padding-top: 2rem;
        padding-bottom: 2rem;
        padding-left: 1.5rem;
        padding-right: 1.5rem;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.3); /* Stronger shadow for more pop */
        margin-top: 2rem;
        margin-bottom: 2rem;
    }

    /* Title and Header styling */
    h1 {
        font-size: 3rem;
        color: #1a73e8; /* Google Blue */
        text-align: center;
        margin-bottom: 0.5rem;
        text-shadow: 2px 2px 5px rgba(0, 0, 0, 0.2); /* Slightly stronger text shadow */
    }
    
    .stMarkdown p {
        font-size: 1.1rem;
        color: #212529; /* Even darker grey text for maximum readability */
        text-align: center;
    }

    /* Info box styling */
    div[data-testid="stAlert"] {
        border-radius: 0.5rem;
        border-left: 8px solid #fbbc05; /* Google Yellow */
        background-color: rgba(255, 250, 220, 0.95); /* Less transparent background */
    }

    /* File uploader and camera input container */
    div[data-testid="stFileUploader"] > div:first-child, div[data-testid="stCameraInput"] > div:first-child {
        border: 2px dashed #aab0b6; 
        border-radius: 0.75rem;
        padding: 1.5rem;
        text-align: center;
        background-color: rgba(248, 249, 250, 0.95); /* Less transparent light background */
        transition: all 0.3s ease;
    }

    div[data-testid="stFileUploader"] > div:first-child:hover, div[data-testid="stCameraInput"] > div:first-child:hover {
        background-color: rgba(241, 243, 244, 1); /* Fully opaque on hover */
        border-color: #1a73e8; 
    }

    /* Result image border */
    div.stImage img {
        border-radius: 1rem;
        box-shadow: 0 8px 20px rgba(0, 0, 0, 0.25); /* Stronger shadow */
    }
    
    /* Emotion label color coding */
    .emotion-Angry { color: #dc3545; font-weight: bold; }
    .emotion-Happy { color: #28a745; font-weight: bold; }
    .emotion-Surprise { color: #ffc107; font-weight: bold; }
    .emotion-Neutral { color: #6c757d; font-weight: bold; }
    .emotion-Sad { color: #007bff; font-weight: bold; }
    .emotion-Fear { color: #6f42c1; font-weight: bold; }
    .emotion-Disgust { color: #20c997; font-weight: bold; }

    /* Custom detailed result box styling */
    div.stMarkdown > div > div > div[data-testid^="stVerticalBlock"] > div:has(h3) { 
        background-color: rgba(241, 243, 244, 0.95); /* Less transparent light gray background */
        border: 1px solid #e8eaed;
        border-radius: 0.75rem;
        padding: 15px;
        margin-top: 15px;
        box-shadow: 0 2px 10px rgba(0, 0, 0, 0.15); /* Stronger shadow */
    }
    
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
            
            st.markdown(
                f"""
                <div style="border: 1px solid #dadce0; border-radius: 8px; padding: 10px; margin-top: 10px; background-color: rgba(255, 255, 255, 0.7);">
                    <h3 style="margin-top: 0; font-size: 1.25rem; color: #1a73e8;">Face at ({x}, {y})</h3>
                    <p style="text-align: left; margin-bottom: 5px; color: #3c4043;">
                        <strong>Primary Emotion:</strong> <span class="emotion-{emotion}">{emotion} ({confidence*100:.1f}%)</span>
                    </p>
                    <details>
                        <summary style="color: #5f6368;">Top 3 Probabilities</summary>
                        <ul style="color: #3c4043;">
                            {''.join([f'<li><span class="emotion-{e[0]}">{e[0]}</span>: {e[1]*100:.1f}%</li>' for e in top_emotions[:3]])}
                        </ul>
                    </details>
                </div>
                """, unsafe_allow_html=True
            )


        # 4. Display the Result
        st.markdown("---")
        st.subheader("Processed Image Result") 
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
    <div style='text-align: center; color: #7f878a; font-size: 0.85rem; background-color: rgba(255, 255, 255, 0.7); padding: 10px; border-radius: 0.5rem;'>
        Powered by Keras, OpenCV, and Streamlit. Model: InsideOut by Ahsan Farabi.
    </div>
    """, unsafe_allow_html=True
)
