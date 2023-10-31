import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf

# Load the handwritten text classification models
models = {
    'MNIST': tf.keras.models.load_model('handwritten_text_classification_model_mnist.h5'),
    'EMNIST': tf.keras.models.load_model('handwritten_text_classification_model_emnist.h5'),
}

# Define the GUI
st.title('Handwritten Text Classification')

# Select a model
model_name = st.sidebar.selectbox('Model:', models.keys())
model = models[model_name]

# Upload an image
st.write('Upload an image of handwritten text:')
image = st.file_uploader('Image', type=['png', 'jpg', 'jpeg'])

# Adjust the prediction threshold
threshold = st.sidebar.slider('Prediction threshold:', 0.0, 1.0, 0.5)

# Generate a new handwritten text image
if st.sidebar.button('Generate new image'):
    image = np.random.randint(0, 255, size=(28, 28, 1), dtype=np.uint8)

# Classify the image
if image:
    # Preprocess the image
    image = Image.open(image)
    image = image.resize((28, 28))
    image = np.array(image)
    image = image.astype('float32') / 255.0

    # Make a prediction
    prediction = model.predict(np.array([image]))

    # Display the prediction
    st.write('Prediction:')
    if prediction.argmax() >= threshold:
        st.write(prediction.argmax())
    else:
        st.write('None')

    # Provide explanations of the prediction
    st.write('Explanation:')
    st.write(f'The model is {prediction.argmax() * 100:.2f}% confident that the prediction is correct.')
else:
    st.write('Please upload an image to classify.')
