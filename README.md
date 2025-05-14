# Land Type Classification using Sentinel-2 Satellite Images

This project is a deep learning solution for automatically classifying different types of land (such as farms, water bodies, urban areas, roads, and forests) from Sentinel-2 satellite imagery. The model is trained on the EuroSAT dataset and deployed as an interactive web app using Streamlit.

## Features
- Classifies satellite images into 10 land types
- Uses a ResNet-based deep learning model
- Includes PCA and NDVI/NDBI preprocessing
- User-friendly Streamlit web interface

## Land Types
- Annual Crop
- Forest
- Herbaceous Vegetation
- Highway
- Industrial
- Pasture
- Permanent Crop
- Residential
- River
- Sea/Lake

## Project Structure
```
├── app.py                # Streamlit web app
├── deployment_utils.py   # Preprocessing and prediction utilities
├── best_model.h5         # Trained Keras model
├── pca.pkl               # PCA model (pickled)
├── label_encoder.pkl     # LabelEncoder (pickled)
├── requirements.txt      # Python dependencies
└── README.md             # Project documentation
```
## Notebook
- `Land_Type_Classification_using_Sentinel_2_Satellite_Images.ipynb`:  
  This Jupyter notebook contains all steps for data exploration, preprocessing, model training, and exporting the trained model and encoders. Use it to reproduce or modify the training pipeline.

## Setup Instructions
1. **Clone the repository**
   ```bash
   git clone https://github.com/basmalagamal/Land-Type-Classification-DEPI-Project.git
   cd your-repo-name
   ```
2. **(Recommended) Create a virtual environment**
   ```bash
   conda create -n landclass python=3.10
   conda activate landclass
   conda install pip
   ```
3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```
4. **Run the Streamlit app**
   ```bash
   streamlit run app.py
   ```

## Usage
- Upload a Sentinel-2 `.tif` satellite image via the web interface.
- The app will display an RGB visualization and predict the land type with a confidence score.

## Notes
- Make sure `best_model.h5`, `pca.pkl`, and `label_encoder.pkl` are present in the project directory.
- The app expects images in the same format as the EuroSAT dataset (Sentinel-2 `.tif` files with 13 bands).
- The main branch contains the Jupyter notebook for data exploration and model training.  
- The `Streamlit-Deployment` branch contains the Streamlit app and deployment code.

## Requirements
- Python 3.10+
- See `requirements.txt` for all dependencies

## Credits
- Model trained on the [EuroSAT dataset (GitHub)](https://github.com/phelber/eurosat)
- EuroSAT dataset also available on [Kaggle](https://www.kaggle.com/datasets/apollo2506/eurosat-dataset)
- Built with TensorFlow, scikit-learn, and Streamlit

## Model Download
Due to GitHub file size limits, download `best_model.h5` from [[Google Drive link here](https://drive.google.com/drive/folders/1bbqpKrO2U20FSr0LA82eE8KfqsJOdCr8?usp=drive_link)].
Place it in the project directory before running the app.

## Branches

- **main**: Contains the Jupyter notebook for data exploration, preprocessing, and model training.
- **Streamlit-Deployment**: Contains the Streamlit app, deployment utilities, trained model, and all files needed for deployment.
---

**For questions or issues, please open an issue on GitHub.** 
