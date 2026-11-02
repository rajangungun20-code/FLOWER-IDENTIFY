
import streamlit as st
import tensorflow as tf
import numpy as np
import json
from PIL import Image

st.title("🌸 Flower Classification")

# Load trained model
model = tf.keras.models.load_model("FLOWER.keras")

# Load class names
with open("classes.json", "r") as f:
    class_names = json.load(f)

# Upload image
uploaded_file = st.file_uploader(
    "Upload a Flower Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    img = Image.open(uploaded_file).convert("RGB")

    st.image(img, caption="Uploaded Image", width=300)

    if st.button("Predict"):

        img = img.resize((64, 64))

        x = np.array(img)
        x = np.expand_dims(x, axis=0)

        p = model.predict(x)

        index = np.argmax(p)

        st.success("Predicted Flower: " + class_names[index])

        st.write("Confidence:", round(float(p[0][index]) * 100, 2), "%")
