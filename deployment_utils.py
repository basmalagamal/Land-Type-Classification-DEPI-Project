import numpy as np
import tensorflow as tf
import rasterio
from tensorflow.keras.models import load_model
import pickle
import os

# Constants
IMG_SIZE = (224, 224)
NUM_CLASSES = 10
BAND_NAMES = [
    'B01 - Coastal Aerosol',
    'B02 - Blue',
    'B03 - Green',
    'B04 - Red',
    'B05 - Red Edge 1',
    'B06 - Red Edge 2',
    'B07 - Red Edge 3',
    'B08 - NIR',
    'B8A - Narrow NIR',
    'B09 - Water Vapor',
    'B10 - SWIR - Cirrus',
    'B11 - SWIR 1',
    'B12 - SWIR 2'
]

def load_deployment_model(model_path):
    """Load the trained model"""
    return load_model(model_path)

def load_pca_model(pca_path):
    """Load the PCA model"""
    with open(pca_path, 'rb') as f:
        return pickle.load(f)

def get_rgb_image(data):
    """Create a natural RGB image using Bands 4, 3, 2 (Red, Green, Blue)"""
    rgb = np.dstack((data[3], data[2], data[1]))  # Bands are 0-indexed
    rgb = rgb.astype(np.float32)
    rgb /= 2750
    rgb = np.clip(rgb, 0, 1)
    return rgb

def preprocess_image(image_path, pca_model):
    """Preprocess a single image for prediction"""
    # Load image
    with rasterio.open(image_path) as src:
        data = src.read().astype(np.float32)
    
    # Compute NDVI and NDBI
    red = data[3]  # B04 - Red
    nir = data[7]  # B08 - NIR
    swir = data[10]  # B11 - SWIR 1
    
    ndvi = (nir - red) / (nir + red + 1e-6)
    ndbi = (swir - nir) / (swir + nir + 1e-6)
    
    # Stack all bands
    img = np.transpose(data, (1, 2, 0))  # (H, W, C)
    img = np.concatenate([img, ndvi[..., np.newaxis], ndbi[..., np.newaxis]], axis=-1)
    
    # Min-Max scaling
    min_val = img.min(axis=(0, 1), keepdims=True)
    max_val = img.max(axis=(0, 1), keepdims=True)
    scaled_data = (img - min_val) / (max_val - min_val + 1e-6)
    scaled_data = np.clip(scaled_data, 0.0, 1.0)
    
    # Apply PCA
    H, W, C = scaled_data.shape
    img_pca = pca_model.transform(scaled_data.reshape(-1, C)).reshape(H, W, -1)
    
    # Resize to model input size
    img_pca = tf.image.resize(img_pca, IMG_SIZE)
    
    return img_pca

def predict_image(model, image_path, pca_model, label_encoder):
    """Make prediction on a single image"""
    # Preprocess image
    processed_img = preprocess_image(image_path, pca_model)
    
    # Add batch dimension
    processed_img = tf.expand_dims(processed_img, axis=0)
    
    # Make prediction
    prediction = model.predict(processed_img)
    predicted_class = np.argmax(prediction)
    confidence = np.max(prediction)
    
    # Get class name
    class_name = label_encoder.inverse_transform([predicted_class])[0]