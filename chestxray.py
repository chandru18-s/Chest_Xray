import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image
import io

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(
    page_title="Chest X-ray Predictor",
    page_icon="🩺",
    layout="centered"
)

st.title("🩺 Chest X-ray Pneumonia Prediction")

# -----------------------------
# Load Model
# -----------------------------
@st.cache_resource
def load_cnn_model():
    return load_model(
        "ChestXray.h5"
    )

model = load_cnn_model()

class_names = ["NORMAL", "PNEUMONIA"]

# -----------------------------
# Upload
# -----------------------------
uploaded_file = st.file_uploader(
    "Upload Chest X-ray Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    try:
        # Read image
        bytes_data = uploaded_file.read()
        image = Image.open(io.BytesIO(bytes_data)).convert("RGB")

        st.image(image, caption="Uploaded X-ray")

        # Predict button
        if st.button("🔍 Predict"):

            img = image.resize((150,150))
            img_array = np.array(img)

            # Ensure 3 channels
            if img_array.shape[-1] == 1:
                img_array = np.stack((img_array,)*3, axis=-1)

            img_array = np.expand_dims(img_array, axis=0)
            img_array = img_array / 255.0

            # Prediction
            pred = model.predict(img_array)

            predicted_class = class_names[np.argmax(pred)]
            confidence = np.max(pred) * 100

            st.success(f"Prediction: {predicted_class}")
            st.info(f"Confidence: {confidence:.2f}%")

    except Exception as e:
        st.error("❌ File is not a valid X-ray image.")
        st.error(e)