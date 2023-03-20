# Importing the required libraries
import streamlit as st
import tensorflow as tf

from PIL import Image
import numpy as np

# Loading the trained model
model = tf.keras.models.load_model('model11.h5')


# Defining the function to predict the digit
def predict_digit(img):
    # Preprocessing the image
    img = img.convert('L')
    img = img.resize((28, 28))
    img = np.array(img)
    img = img.reshape(1, 28, 28, 1)
    img = img / 255.0

    # Predicting the digit
    res = model.predict([img])[0]
    return np.argmax(res), max(res)


# Creating the web app
st.set_page_config(page_title="Handwritten Digit Recognition", page_icon=":pencil2:")
st.title("Handwritten Digit Recognition using CNN")

# Uploading the image
uploaded_file = st.file_uploader("Choose an image of a digit", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Displaying the image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Predicting the digit
    digit, confidence = predict_digit(image)
    st.write("Prediction:", digit)
    st.write("Confidence:", round(confidence * 100, 2), "%")

