import streamlit as st
from PIL import Image
import numpy as np
import tensorflow.lite as tflite  # Import TFLite

# Function to load and preprocess image
def load_image(image_file):
    img = Image.open(image_file)
    return img

# Streamlit UI
st.title("Cats and Dogs Classification")

# Upload Image
image_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

# Load the TFLite Model
interpreter = tflite.Interpreter(model_path="model.tflite")
interpreter.allocate_tensors()

# Get input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

if image_file is not None:
    # Display Uploaded Image
    st.image(load_image(image_file), width=250)

    # Preprocess Image
    image = Image.open(image_file)
    image = image.resize((224, 224))
    image_arr = np.array(image.convert("RGB"), dtype=np.float32)
    image_arr = np.expand_dims(image_arr, axis=0)  # Add batch dimension

    # Set Input Tensor
    interpreter.set_tensor(input_details[0]['index'], image_arr)

    # Run Inference
    interpreter.invoke()

    # Get Predictions
    result = interpreter.get_tensor(output_details[0]['index'])
    ind = np.argmax(result)

    # Class Labels
    classes = ["Cat", "Dog"]
    st.header(f"Prediction: {classes[ind]}")
