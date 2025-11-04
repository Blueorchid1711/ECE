import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image

uploaded_file = st.file_uploader("📤 Upload Chest X-Ray", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("L").resize((224, 224))
    st.image(image, caption="Uploaded X-ray", use_column_width=True)
    img_array = np.expand_dims(np.expand_dims(np.array(image) / 255.0, axis=-1), axis=0)
    
    st.write("Input shape to model:", img_array.shape)
    preds = model.predict(img_array)
    
    label = CLASS_NAMES[np.argmax(preds)]
    confidence = np.max(preds)
    
    st.success(f"Prediction: **{label}**")
    st.metric("Confidence", f"{confidence*100:.2f}%")
    st.progress(float(confidence))

