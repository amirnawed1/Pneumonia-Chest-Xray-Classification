import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os
import gdown

# -----------------------------
# Page Configuration
# -----------------------------

st.set_page_config(
    page_title="AI Powered Pneumonia Detection",
    page_icon="🩺",
    layout="wide"
)

# -----------------------------
# Custom CSS
# -----------------------------

st.markdown("""
<style>

.main {
    background-color: #0E1117;
}

.big-title{
    font-size:50px;
    font-weight:bold;
    text-align:center;
    color:#00D4FF;
}

.sub-title{
    text-align:center;
    color:white;
    font-size:20px;
}

.footer{
    text-align:center;
    color:gray;
}

</style>
""", unsafe_allow_html=True)

# -----------------------------
# Sidebar
# -----------------------------

st.sidebar.title("🩺 About Project")

st.sidebar.info(
"""
### Pneumonia Detection System

#### Model Used
Custom CNN

#### Test Accuracy
86.7%

#### Technologies

- TensorFlow
- Keras
- Streamlit
- NumPy
- Pillow

Made by Amir Nawed 🚀
"""
)

# -----------------------------
# Main Title
# -----------------------------

st.markdown(
'<p class="big-title">🩺 AI Powered Pneumonia Detection System</p>',
unsafe_allow_html=True
)

st.markdown(
'<p class="sub-title">Upload a Chest X-ray image and let AI predict whether it is Normal or Pneumonia.</p>',
unsafe_allow_html=True
)

# -----------------------------
# Download Model
# -----------------------------

MODEL_PATH = "final_model.keras"

FILE_ID = "1lTKEAF0tDKFLiFsAEO56dSWbt6735nBS"

URL = f"https://drive.google.com/uc?id={FILE_ID}"

if not os.path.exists(MODEL_PATH):

    with st.spinner("Downloading AI model... Please wait ⏳"):

        gdown.download(
            URL,
            MODEL_PATH,
            quiet=False
        )

# -----------------------------
# Load Model
# -----------------------------

@st.cache_resource
def load_model():

    model = tf.keras.models.load_model(
        MODEL_PATH
    )

    return model

model = load_model()
