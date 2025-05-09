#Land Cover Classification Using Deep Learning

This project focuses on classifying geospatial images into 10 distinct land cover classes using a Convolutional Neural Network (CNN). It uses multi-band .tif satellite images and is designed to handle complex terrain types with high accuracy.

🗂️ Dataset
Source: EuroSAT dataset (or your dataset source)

Format: .tif images with multiple bands (e.g., RGB, NDVI)

Classes:

AnnualCrop

Forest

HerbaceousVegetation

Highway

Industrial

Pasture

PermanentCrop

Residential

River

SeaLake

🖼️ Image Details
Input Size: 64x64 pixels

Bands: Typically 3 (RGB), but adaptable for multi-band input.

Preprocessing: Images are normalized and augmented using the Albumentations library (rotation, flipping, brightness adjustment, etc.).

🛠️ Model Architecture
Backbone: ResNet50 (pre-trained on ImageNet)

Input Layer: Modified to accept multi-band .tif images

Output Layer: 10-class softmax

Key Hyperparameters:

Optimizer: Adam

Learning Rate: 0.0001

Batch Size: 32

Epochs: 50

Loss: Categorical Crossentropy

📈 Performance Metrics
Accuracy: ~95% on the test set

Confusion Matrix Insights:

Highest performance: SeaLake, River, Residential (f1-score ~0.98)

Lower but solid: HerbaceousVegetation & PermanentCrop (f1-score ~0.90)

Balanced precision & recall across all classes

Macro & Weighted Averages:

Precision: 0.95

Recall: 0.95

F1-Score: 0.95

🌀 ROC Curve Analysis
High AUC: SeaLake, River, Forest (~0.99)

Slightly lower AUC: HerbaceousVegetation, PermanentCrop (~0.95–0.96)

Micro-average AUC: ~0.95–0.96

ROC curves show strong model performance with steep rises and high true positive rates.

🖥️ Visualizations
Confusion Matrix

ROC Curves (one-vs-rest for all classes)

Sample Predictions:

True vs. Predicted labels for selected images
