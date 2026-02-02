# 🐶🐱 Dogs vs Cats Image Classifier

A deep learning project that classifies images of dogs and cats using Convolutional Neural Networks (CNNs) with Keras and TensorFlow.

##  Project Overview

This project implements a binary image classification model to distinguish between images of dogs and cats. The model achieves **85.9% accuracy** on the test set using a custom CNN architecture with data augmentation techniques.

##  Key Features

- **Custom CNN Architecture** with BatchNormalization layers for improved training stability
- **Data Augmentation** to enhance model generalization and reduce overfitting
- **Comprehensive Visualization** of training metrics and model performance
- **Well-structured Pipeline** from data loading to model evaluation

## Dataset

- **Source**: Kaggle Dogs vs Cats Classification Dataset
- **Total Images**: 25,000 images
- **Classes**: 2 (Dogs and Cats)
- **Image Size**: 150x150 pixels (resized)
- **Split Ratio**:
  - Training: 75%
  - Validation: 10%
  - Test: 15%

##  Model Architecture

The CNN architecture consists of:

- **3 Convolutional Blocks** (Conv2D + BatchNorm + MaxPooling + Dropout)
- **Dense Layers** for classification
- **Batch Normalization** for faster convergence
- **Dropout Layers** to prevent overfitting

### Hyperparameters

```python
Image Size: 150x150x3
Batch Size: 128
Epochs: 20
Optimizer: Adam
Loss Function: Binary Crossentropy
```

## Data Augmentation Techniques

To improve model generalization, the following augmentation techniques are applied:

- Rotation (±10°)
- Horizontal Flip
- Width/Height Shift (10%)
- Shear Transformation (20%)
- Zoom (20%)

## Results

- **Test Accuracy**: 85.9%
- **Training Strategy**: Early stopping with validation monitoring
- **Visualization**: Confusion matrix, accuracy/loss curves, and sample predictions

## Technologies Used

- **Python 3.x**
- **TensorFlow / Keras** - Deep learning framework
- **NumPy** - Numerical computations
- **Pandas** - Data manipulation
- **Matplotlib / Seaborn** - Data visualization
- **Scikit-learn** - Model evaluation metrics

## Getting Started

### Prerequisites

```bash
pip install tensorflow numpy pandas matplotlib seaborn scikit-learn
```

### Running the Notebook

1. Clone this repository
2. Download the dataset from [Kaggle](https://www.kaggle.com/datasets/chetankv/dogs-cats-images)
3. Update the `BASE_PATH` variable to point to your dataset location
4. Run all cells in the notebook

##  Exploratory Data Analysis

The notebook includes:
- Class distribution visualization
- Sample image displays
- Augmentation effect demonstrations
- Training/validation curves
- Confusion matrix analysis

##  Learning Outcomes

This project demonstrates:
- Building CNN architectures from scratch
- Implementing data augmentation pipelines
- Handling image datasets with Keras
- Model evaluation and performance metrics
- Visualization of deep learning results


---

⭐ If you found this project helpful, please consider giving it a star!
