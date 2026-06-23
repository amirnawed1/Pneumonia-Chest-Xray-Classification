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

# -----------------------------
# Upload Image
# -----------------------------

uploaded_file = st.file_uploader(
    "📂 Upload Chest X-ray Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    col1, col2 = st.columns(2)

    with col1:

        st.image(
            image,
            caption="Uploaded Image",
            use_container_width=True
        )

    # -----------------------------
    # Image Preprocessing
    # -----------------------------

    img = image.resize((224,224))

    img = np.array(img)

    img = img / 255.0

    img = np.expand_dims(
        img,
        axis=0
    )

    # -----------------------------
    # Prediction
    # -----------------------------

    prediction = model.predict(img)

    st.success("Prediction completed")

    st.write(prediction)

    predicted_class = np.argmax(
        prediction
    )

    confidence = np.max(
        prediction
    ) * 100

    classes = [

        "Normal",

        "Pneumonia"

    ]

    result = classes[predicted_class]

    with col2:

        st.subheader("🤖 AI Prediction")

        if result == "Normal":

            st.success(
                f"✅ Prediction: {result}"
            )

        else:

            st.error(
                f"⚠️ Prediction: {result}"
            )

        st.metric(

            label="Confidence Score",

            value=f"{confidence:.2f}%"

        )

if uploaded_file is not None:

    normal_prob = prediction[0][0] * 100

    pneumonia_prob = prediction[0][1] * 100

    st.subheader("📊 Prediction Probability")

    chart_data = {

        "Normal": normal_prob,

        "Pneumonia": pneumonia_prob

    }

    st.bar_chart(chart_data)



# -----------------------------
# About Prediction
# -----------------------------

st.markdown("---")

st.subheader("🧠 About the AI Model")

st.info(
"""
This system uses a Custom CNN model trained on Chest X-ray images.

Model Test Accuracy: **86.7%**

The model predicts whether the X-ray image belongs to:

- Normal
- Pneumonia

The image is resized to 224 × 224 before prediction.
"""
)

# -----------------------------
# Medical Disclaimer
# -----------------------------

st.markdown("---")

st.warning(
"""
⚠️ Medical Disclaimer

This AI system is intended for educational and research purposes only.

It should not be used as a substitute for professional medical diagnosis.

Always consult a qualified healthcare professional.
"""
)

# -----------------------------
# Footer
# -----------------------------

st.markdown("---")

st.markdown(
"""
<div class='footer'>

Made with ❤️ using TensorFlow, Keras and Streamlit

### 🚀 Developed by Amir Nawed

</div>
""",
unsafe_allow_html=True
)
