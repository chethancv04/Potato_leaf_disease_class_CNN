import numpy as np
import PIL.Image as Image
import tensorflow as tf
import streamlit as st
from streamlit_extras.add_vertical_space import add_vertical_space
from warnings import filterwarnings

filterwarnings('ignore')


def streamlit_config():
    # Page configuration
    st.set_page_config(page_title='Potato Disease Classification', layout='wide')

    add_vertical_space(2)

# Function to handle prediction
def prediction(image_path, class_names=['Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy']):
    img = Image.open(image_path)
    img_resized = img.resize((256, 256))
    img_array = tf.keras.preprocessing.image.img_to_array(img_resized)
    img_array = np.expand_dims(img_array, axis=0)

    model = tf.keras.models.load_model(r'D:\FRO\model.h5')

    prediction = model.predict(img_array)

    predicted_class = class_names[np.argmax(prediction)]
    confidence = round(np.max(prediction) * 100, 2)

    add_vertical_space(1)
    st.markdown(f'<h4 style="color: orange;">Predicted Class: {predicted_class}<br>Confidence: {confidence}%</h4>',
                unsafe_allow_html=True)

    add_vertical_space(1)
    st.image(img.resize((400, 300)))

# Streamlit Configuration Setup
streamlit_config()

# Sidebar for disease type explanations
st.sidebar.title("Disease Types")
st.sidebar.write("""
- **Potato___Early_blight**: A fungal disease caused by *Alternaria solani*, characterized by brown lesions on the leaves, often with concentric rings.
  
- **Potato___Late_blight**: Caused by the pathogen *Phytophthora infestans*, this disease leads to large, dark, and water-soaked spots on leaves and stems, often with a white mold on the undersides of leaves.

- **Potato___healthy**: Indicates a healthy potato leaf without any visible disease symptoms.
""")

# Main Section for Image Upload
st.markdown('<h3 style="text-align: center;">Upload an Image for Disease Classification</h3>', unsafe_allow_html=True)

# Centering the file uploader with increased spacing
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    add_vertical_space(4)  # Increased space above file uploader
    input_image = st.file_uploader(label='Upload an Image', type=['jpg', 'jpeg', 'png'])
    add_vertical_space(4)  # Increased space below file uploader

if input_image is not None:
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        prediction(input_image)

# About Section
st.markdown('<h3 style="text-align: center;">About Me and This Project</h3>', unsafe_allow_html=True)
st.markdown("""
<p style="text-align: center;">
Hi, I'm Chethan, a CV student at the Maharaja Institute of Technology (MIT), Mysore. This application is part of my project work in using deep learning for agricultural applications. The goal is to classify potato diseases by analyzing images of potato leaves. By identifying diseases early, we can help farmers and agricultural experts take timely action to prevent crop loss.
</p>
""", unsafe_allow_html=True)
