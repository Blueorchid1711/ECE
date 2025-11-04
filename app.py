import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image

# --------------------------------
# PAGE CONFIG
# --------------------------------
st.set_page_config(page_title="Object Classifier", page_icon="🎯", layout="centered")
st.title("🎯 ML-Based Image Classification App")
st.markdown("Upload an image to classify using your trained model.")

# --------------------------------
# LOAD MODEL
# --------------------------------
model = load_model("Covid_19.ipynb")

# Replace with your actual class labels
CLASS_NAMES = ['class1', 'class2', 'class3']

# --------------------------------
# IMAGE UPLOAD & PREDICTION
# --------------------------------
uploaded_file = st.file_uploader("📤 Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB").resize((224, 224))
    st.image(image, caption="Uploaded Image", use_column_width=True)
    img_array = np.expand_dims(np.array(image) / 255.0, axis=0)

    preds = model.predict(img_array)
    pred_label = CLASS_NAMES[np.argmax(preds)]
    confidence = np.max(preds)

    st.success(f"Prediction: **{pred_label}**")
    st.metric("Confidence", f"{confidence*100:.2f}%")
    st.progress(float(confidence))
