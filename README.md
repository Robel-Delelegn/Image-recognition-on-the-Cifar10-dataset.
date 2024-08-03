# Image Recognition with CIFAR-10 Dataset

## Description
This project demonstrates a simple image classification model using the CIFAR-10 dataset. The dataset consists of 60,000 32x32 color images in 10 different classes. The model is built using TensorFlow and Keras, utilizing convolutional layers for feature extraction and dense layers for classification.

## Requirements
- Python 3.x
- TensorFlow
- Matplotlib
- TensorFlow Datasets
- OpenCV (optional, for additional image processing)

## Installation
To run this project, install the required Python packages using pip.

## How to Run
Load and Visualize CIFAR-10 Data: The script loads the CIFAR-10 dataset, displays an example image, and prints its label.

Build the Model: A Convolutional Neural Network (CNN) is defined and compiled. The model includes:

Two convolutional layers with ReLU activation and max pooling.
A flatten layer to reshape the data.
Two dense layers, with the final dense layer using softmax activation for classification.
Train the Model: The model is trained on the CIFAR-10 training data for 5 epochs.

## Acknowledgements
This project uses the CIFAR-10 dataset, a popular dataset for image classification tasks. The dataset is provided by the TensorFlow Datasets library.

Evaluate the Model: The trained model is evaluated on the CIFAR-10 test data.


Save the Model: The trained model is saved to a file named cifar10.model

## Usage Notes
Data Visualization: The example image from the CIFAR-10 dataset is displayed to give a visual understanding of the data.
Model Training: The model is trained for 5 epochs; adjust this as needed based on your computing resources and desired accuracy.
Model Saving: The model is saved to cifar10.model, which can be loaded later for inference or further training.
