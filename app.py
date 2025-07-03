import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

# Load the model (update path as needed)
model = load_model("cat_dog_cnn_model.h5")  # Change to your model filename

# Set page title
st.set_page_config(page_title="Cat vs Dog Classifier")

# Title
st.title("🐶 Cat vs Dog Image Classifier 🐱")
st.write("Upload an image of a cat or dog, and the model will predict which it is!")

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image", use_container_width=True)
    
    # Preprocess image
    img = img.resize((150, 150))  # Match with your model input
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0  # Normalize if your model expects it
    
    # Prediction
    prediction = model.predict(img_array)
    class_name = "Dog" if prediction[0][0] > 0.5 else "Cat"
    confidence = prediction[0][0] if prediction[0][0] > 0.5 else 1 - prediction[0][0]

    # Display result
    st.markdown(f"### Prediction: **This is a {class_name}**")
    st.markdown(f"### Confidence: **{confidence * 100:.2f}%**")
