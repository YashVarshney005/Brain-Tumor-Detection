# Brain Tumor Classification Using Transfer Learning (MobileNetV2)

---

## Overview

This project implements a deep learning pipeline to classify brain tumor images into two classes (e.g., tumor vs. no tumor) using transfer learning with MobileNetV2. It leverages data augmentation, class balancing, and visualization to ensure robust performance and interpretability.

---

## Dataset

The dataset should be provided as a ZIP archive (`Brain Tumor.zip`) containing subfolders per class with images inside.

The directory structure is compatible with Keras’ `flow_from_directory()` for automatic labeling.

---

## Methodology

1. **Data Preparation:**  
   - Extract dataset from ZIP file and clean up nested folders.  
   - Split dataset into training and validation sets (80/20 split).  
   - Apply data augmentation (rotations, zoom, shifts, flips) to training data for better generalization.

2. **Class Balancing:**  
   - Compute class weights to mitigate the effect of class imbalance during training.

3. **Model Building:**  
   - Use MobileNetV2 pretrained on ImageNet as a frozen base model (feature extractor).  
   - Add custom classification head: global average pooling, dense layer with ReLU, dropout, and sigmoid output.

4. **Training:**  
   - Compile model with Adam optimizer and binary cross-entropy loss.  
   - Use callbacks: EarlyStopping (to prevent overfitting) and ReduceLROnPlateau (to adjust learning rate).  
   - Train model silently for up to 25 epochs with class weights.

5. **Evaluation:**  
   - Predict on validation set and generate a classification report (precision, recall, F1-score).  
   - Visualize confusion matrix.  
   - Display sample predictions with true and predicted labels.

---

## Features

- Automated dataset extraction and cleanup  
- Image augmentation for improved robustness  
- Balanced training using class weights  
- Transfer learning for efficient training  
- Early stopping and learning rate scheduler for better convergence  
- Detailed evaluation with classification report and confusion matrix  
- Visualization of prediction correctness on sample images

---

## Requirements

- Python 3.7+  
- TensorFlow 2.x  
- scikit-learn  
- matplotlib  
- seaborn  
- numpy  

Install dependencies with:

```bash
pip install tensorflow scikit-learn matplotlib seaborn numpy
