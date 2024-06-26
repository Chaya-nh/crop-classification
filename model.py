# model.py

import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image

def load_my_model(model_path):
    model = load_model(model_path)
    return model

def preprocess_image(image):
    # Convert the image to a tensor and preprocess it
    image = image.resize((256, 256))  # Resize to match the input size required by your model
    image = np.array(image)
    image = image / 255.0  # Normalize to [0, 1]
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

def predict_image(model, image):
    prediction = model.predict(image)
    return prediction
