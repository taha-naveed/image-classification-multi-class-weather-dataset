# Weather Image Classification Project

Welcome to my Weather Image Classification project repository! In this project, I developed an end-to-end deep learning solution to classify images based on different weather conditions. The project explores various model architectures to capture nuances in weather images, particularly addressing the challenge of distinguishing between similar conditions such as "sunrise" and "shine."

---

## Table of Contents

- [Project Overview](#project-overview)
- [Models Overview](#models-overview)
  - [1. Fully Connected Classifier](#1-fully-connected-classifier)
  - [2. Tuned Multi-Layer Neural Network with Dropout](#2-tuned-multi-layer-neural-network-with-dropout)
  - [3. Convolutional Neural Network (CNN)](#3-convolutional-neural-network-cnn)
  - [4. Transfer Learning with Pretrained MobileNet](#4-transfer-learning-with-pretrained-mobilenet)
  - [5. Fine-Tuned MobileNet with Dynamic Learning Rate (Most Recent)](#5-fine-tuned-mobilenet-with-dynamic-learning-rate-most-recent)
- [Training Details](#training-details)
- [Usage](#usage)
- [Acknowledgements](#acknowledgements)

---

## Project Overview

The goal of this project was to classify images of various weather conditions using deep learning. I experimented with five different models to identify the best approach for handling the multi-class classification task. The project uses categorical cross-entropy loss given the multi-class nature of the problem and incorporates early stopping to avoid overfitting.

---

## Models Overview

### 1. Fully Connected Classifier

- **Description:** A simple neural network built using Keras' Sequential API.
- **Architecture:** Consists of a flatten layer and a Dense layer with softmax activation.
- **Optimizer:** RMSprop.
- **Purpose:** Served as a baseline model to quickly assess performance on weather image classification.

### 2. Tuned Multi-Layer Neural Network with Dropout

- **Description:** A more complex model that includes multiple hidden layers with dropout for regularization.
- **Enhancements:** Utilized Keras Tuner with Bayesian Optimization to fine-tune hyperparameters such as learning rate, number of hidden layers, and dropout rate.
- **Purpose:** Improved performance over the baseline by reducing overfitting and optimizing network complexity.

### 3. Convolutional Neural Network (CNN)

- **Description:** A deep CNN designed to extract spatial features from weather images.
- **Architecture:** Contains three Conv2D layers each followed by MaxPooling2D layers, then a Flatten layer, and a Dense softmax classifier.
- **Optimizer:** Adam.
- **Purpose:** Leverages convolutional layers to capture local spatial patterns in images, enhancing classification accuracy.

### 4. Transfer Learning with Pretrained MobileNet

- **Description:** A model that fine-tunes a MobileNet model pretrained on ImageNet.
- **Approach:** Used MobileNet as a feature extractor and added a Dense softmax classification layer.
- **Purpose:** Improved accuracy by leveraging learned features from a large-scale dataset, providing a solid starting point for weather image classification.

### 5. Fine-Tuned MobileNet with Dynamic Learning Rate (Most Recent)

- **Description:** An enhanced version of the transfer learning model.
- **Enhancements:** Uses a dynamic learning rate schedule instead of a fixed rate, coupled with the SGD optimizer with momentum.
- **Purpose:** Specifically addresses the challenge of distinguishing between similar weather conditions (e.g., sunrise vs. shine) by allowing the model to adapt more effectively to the nuances of the dataset.

---

## Training Details

- **Loss Function:** Categorical Cross-Entropy.
- **Early Stopping:** Implemented to prevent overfitting by monitoring validation loss.
- **Optimization:** Different optimizers were used based on model architecture (RMSprop for the fully connected model, Adam for the CNN and first MobileNet model, and SGD with momentum for the fine-tuned MobileNet).
- **Hyperparameter Tuning:** Keras Tuner (Bayesian Optimization) was employed for one of the models to optimize network architecture and training parameters.

---

## Usage

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/your_username/weather-image-classification.git
   cd weather-image-classification
