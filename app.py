import streamlit as st
from PIL import Image
import numpy as np
from model import load_my_model, preprocess_image, predict_image

# Load the pre-trained model
model_path = 'krishimitra.h5'
model = load_my_model(model_path)

# List of class names
class_names = ['jute', 'maize', 'rice','sugarcane','wheat']  # Replace with your actual class names

# Set the title of the app
st.title('Crop Classification')

# Add a header
st.header('Upload an image to classify the crop')

# Add a file uploader widget
uploaded_file = st.file_uploader("Upload Photo", type=["jpg", "jpeg", "png"])

# Display the uploaded image and classify it
if uploaded_file is not None:
    # Open the uploaded image
    image = Image.open(uploaded_file)

    # Display the uploaded image
    st.image(image, caption='Uploaded Photo', use_column_width=True)
    
    # Preprocess the image
    preprocessed_image = preprocess_image(image)
    
    # Add a button to trigger prediction
    if st.button('Classify'):
        # Make a prediction
        prediction = predict_image(model, preprocessed_image)
        
        # Convert prediction to class name
        predicted_class = class_names[np.argmax(prediction)]
        
        # Display the prediction
        st.write("Prediction:", predicted_class)
