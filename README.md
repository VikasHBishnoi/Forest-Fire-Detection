# Forest Fire Detection using CNN

This repository contains a machine learning project for detecting forest fires using Convolutional Neural Networks (CNN).

## Project Description

The goal of this project is to develop a CNN algorithm that can detect whether an image contains a fire or not. The project utilizes a dataset of forest images, some of which depict fires, and trains a CNN model to classify the images as "Fire" or "No-Fire".

## Data Description

The dataset consists of a collection of forest images, including both fire and non-fire images. The data is organized into the following directories:

- "train and validation": This folder contains the training and validation images.
- "test": This folder contains the images for detection.

The dataset can be accessed [here](https://www.kaggle.com/datasets/brsdincer/wildfire-detection-image-data).

## Project Methodology

The project utilizes the following libraries:

- TensorFlow
- NumPy
- Keras
- OpenCV
- joblib
- Matplotlib

The images are preprocessed by resizing and augmenting them using the ImageDataGenerator from Keras. The preprocessing steps include rescaling, rotation, shifting, shearing, zooming, and horizontal flipping.

The CNN model is constructed and trained using the training dataset. The model architecture includes Conv2D, MaxPooling2D, BatchNormalization, Flatten, and Dense layers. Dropout regularization is also applied to reduce overfitting.

The model is evaluated using the test dataset, and the accuracy and loss values are calculated. Finally, the trained model is saved for future use.

## Model Performance

The trained model achieved an accuracy of 95% on the test dataset. The performance of the model is visualized using accuracy and loss plots.

## Conclusion

In this project, we successfully built a CNN model for detecting forest fires with a high accuracy of 95%. The trained model is saved as "model.pkl" for future use.