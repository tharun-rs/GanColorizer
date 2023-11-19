import streamlit as st
from GanColorizer import ColorizerModel
from PIL import Image
import numpy as np
model = ColorizerModel()

model.load_weights()


st.title("GAN Colorizer App")
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png", "bmp", "gif"])
if uploaded_file is not None:
    col1, col2 = st.columns(2)
    original_image = Image.open(uploaded_file)
    original_size = original_image.size
    col1.image(original_image, caption="Original Image", use_column_width=True)
    colorized_image_array = model.generate_color(uploaded_file)
    colorized_image_array = np.reshape(colorized_image_array, (256, 256, 3))
    colorized_image = Image.fromarray(colorized_image_array.astype('uint8'))
    colorized_image = colorized_image.resize(original_size, Image.ANTIALIAS)
    col2.image(colorized_image, caption="Colorized Image", use_column_width=True)
