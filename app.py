import streamlit as st
from PIL import Image
import toml
from src.inference import image_inference
from src.model import vit_model

st.title("Image Classification with ViT")

# Image uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    # Display image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    predicted_class =  image_inference(image)
    
    # Show result
    st.markdown(f"### Predicted Class: `{predicted_class}`")
