import os
import streamlit as st
import tensorflow as tf
import numpy as np
import pickle
import rasterio
from PIL import Image
import io
import matplotlib.pyplot as plt
from deployment_utils import load_deployment_model, load_pca_model, predict_image, get_rgb_image
import requests


def download_model():
    import gdown
    url = "https://drive.google.com/uc?id=1VGEo5syNKrgDOJZmOadcuwTIGrkr4oke"
    output = "best_model.h5"
    gdown.download(url, output, quiet=False)

if not os.path.exists("best_model.h5"):
    download_model()

# Set page config
st.set_page_config(
    page_title="Land Type Classification",
    page_icon="🌍",
    layout="wide"
)

# Title and description
st.title("🌍 Land Type Classification")
st.markdown("""
This app classifies satellite images into different land types using a deep learning model.
Upload a Sentinel-2 satellite image (.tif format) to get started.
""")

# Load models and encoders
@st.cache_resource
def load_models():
    model = load_deployment_model('best_model.h5')
    pca_model = load_pca_model('pca_model.pkl')
    with open('label_encoder.pkl', 'rb') as f:
        label_encoder = pickle.load(f)
    return model, pca_model, label_encoder

try:
    model, pca_model, label_encoder = load_models()
except Exception as e:
    st.error(f"Error loading models: {str(e)}")
    st.stop()

# File uploader
uploaded_file = st.file_uploader("Upload a Sentinel-2 satellite image (.tif)", type=['tif'])

if uploaded_file is not None:
    try:
        # Save the uploaded file temporarily
        with open("temp_image.tif", "wb") as f:
            f.write(uploaded_file.getvalue())
        
        # Display the RGB visualization
        with rasterio.open("temp_image.tif") as src:
            data = src.read()
            rgb_img = get_rgb_image(data)
            
            # Create two columns for layout
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Satellite Image (RGB Visualization)")
                fig, ax = plt.subplots(figsize=(10, 10))
                ax.imshow(rgb_img)
                ax.axis('off')
                st.pyplot(fig)
                plt.close()
            
            # Make prediction
            class_name, confidence = predict_image(model, "temp_image.tif", pca_model, label_encoder)
            
            with col2:
                st.subheader("Classification Results")
                st.markdown(f"### Predicted Land Type: **{class_name}**")
                st.markdown(f"### Confidence: **{confidence:.2%}**")
                
                # Display confidence bar
                st.progress(float(confidence))
                
                # Display class descriptions
                st.markdown("""
                ### Land Type Descriptions:
                - **Annual Crop**: Fields with crops that are planted and harvested within a year
                - **Forest**: Dense tree coverage
                - **Herbaceous Vegetation**: Areas covered with grasses and non-woody plants
                - **Highway**: Major roads and transportation routes
                - **Industrial**: Manufacturing and industrial facilities
                - **Pasture**: Land used for grazing animals
                - **Permanent Crop**: Long-term crops like orchards and vineyards
                - **Residential**: Urban areas with housing
                - **River**: Flowing water bodies
                - **Sea/Lake**: Large bodies of water
                """)
        
        # Clean up temporary file
        import os
        os.remove("temp_image.tif")
        
    except Exception as e:
        st.error(f"Error processing image: {str(e)}")
        st.error("Please make sure you've uploaded a valid Sentinel-2 satellite image in .tif format.")

# Add footer
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p>Built with Streamlit • Using TensorFlow • Trained on EuroSAT Dataset</p>
</div>
""", unsafe_allow_html=True)
