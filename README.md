# Drought-and-Irrigation-Analysis
This project focuses on soil moisture classification using deep learning and computer vision techniques.
The goal is to support irrigation decisions by analyzing soil and plant conditions from images.

## Project Overview
Developed a CNN model with transfer learning (DenseNet121).
Applied data augmentation to improve generalization, with extra emphasis on the minority class.
Used class weights to balance class distribution during training.
Created a custom test set for more reliable evaluation.

## Tech Stack
- Python 3.12
- TensorFlow / Keras
- scikit-learn
- OpenCV
- Matplotlib/Seaborn

## Dataset
- Dataset Source: Soil Moisture Classification Dataset by Chanakya Project
- https://universe.roboflow.com/chanakya-project/soil-moisture/dataset/4
- Images are split into train, validation, and test sets.
- Single-time data augmentation is applied to expand training samples.

## Model Architecture
- Base Model: DenseNet121
- Global Average Pooling
- Dense layers with Dropout for regularization

## Requirements
- Python 3.8+
- TensorFlow / Keras
- scikit-learn
- matplotlib, seaborn
- numpy, requests, zipfile, shutil

## Installation
1. Clone the repository:
   '''git clone https://github.com/alt4ble/Drought-and-Irrigation-Analysis.git
   cd Drought-and-Irrigation-Analysis'''

2. Install dependencies:
   '''pip install -r requirements.txt'''

3. Train the Model:
   '''python train.py'''

4. Evaluate:
   '''python evaluate.py'''

## Usage
Run the training script:
'''python scripts/training.py'''

The script will:
- Download the dataset if not present
- Split the dataset into train/valid/test
- Perform single-time augmentation
- Train a DenseNet121 model
- Save the final model as final_model.keras

## Results
- Accuracy: %79

- Classification Report:
<img width="543" height="251" alt="v10_4" src="https://github.com/user-attachments/assets/47d71ec1-f392-4f0e-bf18-c18bcb02cba5" />

- Confusion Matrix:
<img width="600" height="600" alt="v10_1" src="https://github.com/user-attachments/assets/701c3e68-f8b3-47dc-8f8b-a87168d1987a" />

- Test Accuracy & Loss Graphs:
<img width="640" height="480" alt="v10_2" src="https://github.com/user-attachments/assets/ad97f2ab-774a-4e04-a0e2-16983bc34f9a" />
<img width="640" height="480" alt="v10_3" src="https://github.com/user-attachments/assets/94fe55a6-4518-4913-a650-e2d634eb9e93" />
