# ============================================================
# 🦠 COVID-19 X-Ray Classification Web App
# Built with Streamlit | Deployed on Hugging Face / Streamlit Cloud
# ============================================================

import streamlit as st
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from PIL import Image

# ============================================================
# 🎨 PAGE CONFIGURATION
# ============================================================
st.set_page_config(
    page_title="COVID-19 X-Ray Classifier",
    page_icon="",
    layout="centered"
)

# ============================================================
# 🏥 MAIN HEADING
# ============================================================
st.title("COVID-19 X-Ray Classification App")
st.markdown(
    """
    ### 🔍 Detect COVID-19 or Normal Chest X-Rays  
    Upload a **chest X-ray image**, and the trained deep learning model will classify it as either:
    -  **Normal**
    -  **COVID-19**
    ---
    """
)

# ============================================================
# ⚙️ LOAD TRAINED MODEL
# ============================================================
st.sidebar.header("Model Information")

try:
    model = load_model("final_model.keras")
    st.sidebar.success("Model loaded successfully!")
except Exception as e:
    st.sidebar.error(f"Failed to load model.\n\n**Error:** {e}")
    st.stop()

# Define original class labels (from training)
CLASS_NAMES = ['Normal', 'COVID-19']
st.sidebar.write("**Model classes:**", ", ".join(CLASS_NAMES))
st.sidebar.markdown("---")

# ============================================================
# 📤 IMAGE UPLOAD SECTION
# ============================================================
uploaded_file = st.file_uploader(
    "Upload a Chest X-Ray Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:
    # ========================================================
    # 🖼️ PREPROCESS IMAGE (RGB)
    # ========================================================
    image = Image.open(uploaded_file).convert("RGB").resize((128, 128))
    st.image(image, caption="Uploaded X-Ray", use_column_width=True)

    # Model expects RGB input: (1, 128, 128, 3)
    img_array = np.array(image).astype("float32") / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    with st.expander("Model Input Details"):
        st.write("Model expects:", model.input_shape)
        st.write("Input provided:", img_array.shape)

    # ========================================================
    # 🧠 PREDICTION
    # ========================================================
    preds = model.predict(img_array)
    pred_probs = preds[0]
    original_label = CLASS_NAMES[np.argmax(pred_probs)]
    confidence = np.max(pred_probs)

    # ========================================================
    # 🔁 CUSTOM LOGIC: SWAP LABELS & REMOVE PNEUMONIA
    # ========================================================
    if original_label == "Normal":
        pred_label = "COVID-19"
    elif original_label == "COVID-19":
        pred_label = "Normal"
    else:
        pred_label = "Unknown"  # For Pneumonia (ignored)

    # ========================================================
    # 📊 DISPLAY RESULTS
    # ========================================================
    st.markdown("---")
    st.subheader("Prediction Result")
    if pred_label == "Unknown":
        st.warning("")
    else:
        st.success(f"**Predicted Label:** {pred_label}")
        st.metric("Confidence", f"{confidence*100:.2f}%")
        st.progress(float(confidence))

# ============================================================
# 📘 SIDEBAR: APP INFORMATION
# ============================================================
st.sidebar.header(" About This App")
st.sidebar.markdown(
    """
    **Developed by:** Team ECE Hackathon 2025  
    **Model:** CNN (TensorFlow / Keras)  
    **Input Size:** 128×128×3 (RGB)  
    **Frameworks:** Streamlit, TensorFlow, PIL  

    ---
     *This application is for demonstration and educational purposes only.*
    """
)
