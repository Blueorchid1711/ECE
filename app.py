import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image

# -------------------------------
# Page Config
# -------------------------------
st.set_page_config(page_title="COVID-19 Detection", page_icon="🦠", layout="centered")
st.title("🦠 COVID-19 X-Ray Classifier")
st.markdown("Upload a chest X-ray image to detect COVID-19 infection using our trained model.")

# -------------------------------
# Load your trained model
# -------------------------------
# Ensure 'final_model.keras' is in the same repo
model = load_model("final_model.keras")

CLASS_NAMES = ['Normal', 'COVID-19', 'Pneumonia']  # update if different

# -------------------------------
# Upload and Predict
# -------------------------------
uploaded_file = st.file_uploader("📤 Upload Chest X-Ray Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB").resize((224, 224))
    st.image(image, caption="Uploaded Image", use_column_width=True)

    img_array = np.expand_dims(np.array(image) / 255.0, axis=0)
    preds = model.predict(img_array)

    label = CLASS_NAMES[np.argmax(preds)]
    confidence = np.max(preds)

    st.success(f"Prediction: **{label}**")
    st.metric("Confidence", f"{confidence*100:.2f}%")
    st.progress(float(confidence))
