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
    page_icon="🩺",
    layout="centered"
)

# ============================================================
# 🏥 MAIN HEADING
# ============================================================
st.title("🩺 COVID-19 X-Ray Classification App")
st.markdown(
    """
    ### 🔍 Detect COVID-19, Pneumonia, or Normal Chest X-Rays  
    Upload a **chest X-ray image**, and the trained deep learning model will classify it into one of three categories:
    - 🧍 **Normal**
    - 🦠 **COVID-19**
    - 💨 **Pneumonia**

    ---
    """
)

# ============================================================
# ⚙️ LOAD TRAINED MODEL
# ============================================================
st.sidebar.header("⚙️ Model Information")

try:
    model = load_model("final_model.keras")
    st.sidebar.success("✅ Model loaded successfully!")
except Exception as e:
    st.sidebar.error(f"❌ Failed to load model.\n\n**Error:** {e}")
    st.stop()

# Define class labels (adjust if needed)
CLASS_NAMES = ['Normal', 'COVID-19', 'Pneumonia']
st.sidebar.write("**Classes:**", ", ".join(CLASS_NAMES))
st.sidebar.markdown("---")

# ============================================================
# 📤 IMAGE UPLOAD SECTION
# ============================================================
uploaded_file = st.file_uploader(
    "📤 Upload a Chest X-Ray Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:
    # ========================================================
    # 🖼️ PREPROCESS IMAGE (RGB)
    # ========================================================
    image = Image.open(uploaded_file).convert("RGB").resize((128, 128))
    st.image(image, caption="🩻 Uploaded X-Ray", use_column_width=True)

    # Model expects RGB input: (1, 128, 128, 3)
    img_array = np.array(image).astype("float32") / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    with st.expander("🔍 Model Input Details"):
        st.write("Model expects:", model.input_shape)
        st.write("Input provided:", img_array.shape)

    # ========================================================
    # 🧠 PREDICTION
    # ========================================================
    preds = model.predict(img_array)
    pred_probs = preds[0]
    pred_label = CLASS_NAMES[np.argmax(pred_probs)]
    confidence = np.max(pred_probs)

    # ========================================================
    # 📊 DISPLAY RESULTS
    # ========================================================
    st.markdown("---")
    st.subheader("🧾 Prediction Result")
    st.success(f"**Prediction:** {pred_label}")
    st.metric("Model Confidence", f"{confidence*100:.2f}%")
    st.progress(float(confidence))

    # Show class probabilities
    st.subheader("📈 Class Probabilities")
    prob_df = pd.DataFrame({
        "Class": CLASS_NAMES,
        "Probability (%)": pred_probs * 100
    }).set_index("Class")
    st.bar_chart(prob_df)

    # ========================================================
    # 🩺 EXPLANATION
    # ========================================================
    st.info(
        f"The model predicts **{pred_label}** with a confidence of "
        f"**{confidence*100:.2f}%**.\n\n"
        f"⚠️ *This app is for research and educational purposes only.*"
    )

# ============================================================
# 📘 SIDEBAR: APP INFORMATION
# ============================================================
st.sidebar.header("📘 About This App")
st.sidebar.markdown(
    """
    **Developed by:** Team ECE Hackathon 2025  
    **Model:** CNN (TensorFlow / Keras)  
    **Input Size:** 128×128×3 (RGB)  
    **Frameworks:** Streamlit, TensorFlow, PIL  

    ---
    ⚠️ *This application is for research and educational use only.*
    """
)

