import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image, ImageOps

st.title("🖍️ Handwritten Digit Recognition")
st.write("Upload a clear image of a handwritten digit (0-9) to predict it using CNN.")

uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("L")
    image = ImageOps.invert(image).resize((28, 28))
    img_array = np.array(image) / 255.0
    img_array = img_array.reshape(1, 28, 28, 1)

    model = tf.keras.models.load_model("model/mnist_cnn.h5")
    prediction = model.predict(img_array)
    st.success(f"Predicted Digit: {np.argmax(prediction)}")
