# Import the necessary libraries
import numpy as np
import tensorflow as tf
from tensorflow import keras
from PIL import Image
import streamlit as st
import urllib.request

# Load the pre-trained CNN model
model = keras.models.load_model('model_new.h5')

# Create a function to make predictions on uploaded images
@st.cache(allow_output_mutation=True)
def predict_digit(img):
    # Load the image and convert it to grayscale
    img = img.convert('L')
    img = img.resize((28, 28))
    img_array = np.array(img)
    img_array = img_array.reshape((1, 28, 28, 1)) / 255.0
    # Make the prediction
    prediction = model.predict(img_array)
    digit = np.argmax(prediction)
    return digit

# Create the Streamlit app
st.title('Hand-written Digit Recognition')
image_url = st.text_input('Enter the URL of an image')
if image_url != '':
    try:
        with urllib.request.urlopen(image_url) as url:
            image = Image.open(url)
            st.image(image, caption='Uploaded Image', use_column_width=True)
            prediction = predict_digit(image)
            st.write('Prediction:', prediction)
    except:
        st.write('Error: Invalid URL or Image cannot be processed')
