# Image Classification using the Multi-class Weather Dataset

**Author:** Taha Naveed Shiblu

## Introduction

This README provides an overview of the image classification project using the Multi-class Weather Dataset (MWD). The primary objective of this project is to develop and evaluate machine learning models capable of accurately classifying images into various weather scenarios. The MWD is a widely-used dataset that provides labeled images for different weather conditions, making it a suitable choice for practicing image classification techniques.

## Dataset Overview

The Multi-class Weather Dataset (MWD) consists of images labeled into different weather categories, such as sunny, cloudy, rainy, snowy, and foggy. The dataset was sourced from [Mendeley Data](https://data.mendeley.com/datasets/4drtyfjtfy/1), and is widely used for educational purposes in computer vision and machine learning.

### Data Distribution

The dataset contains a balanced number of images across each weather category, allowing for an unbiased training and evaluation process. The distribution of images per category is as follows:

- **Sunny:** X images
- **Cloudy:** Y images
- **Rainy:** Z images
- **Snowy:** A images
- **Foggy:** B images

## Data Preprocessing

To prepare the data for model training, several preprocessing steps were performed:

1. **Resizing**: All images were resized to a uniform size of 128x128 pixels to ensure consistency in input dimensions for the model.
2. **Normalization**: Pixel values were normalized to a range of [0, 1] to facilitate faster and more stable training of the neural network.
3. **Data Augmentation**: Techniques such as rotation, flipping, zooming, and shifting were applied to artificially increase the size of the training dataset and enhance the model's ability to generalize to unseen data.

## Model Training

For the classification task, a Convolutional Neural Network (CNN) was selected due to its proven effectiveness in image recognition tasks. The model architecture included:

- **Input Layer**: Accepts input images of size 128x128x3.
- **Convolutional Layers**: Multiple convolutional layers with ReLU activation functions and max-pooling layers to down-sample the feature maps.
- **Fully Connected Layers**: Dense layers to learn complex patterns and relationships between features.
- **Output Layer**: A softmax layer to output the probability distribution over the weather categories.

### Hyperparameter Tuning

Several hyperparameters, such as learning rate, batch size, and the number of epochs, were tuned to optimize the model's performance. A grid search approach was used to find the optimal combination of these parameters.

## Model Evaluation

The model was evaluated using a separate validation dataset to ensure unbiased assessment. The following metrics were used to evaluate model performance:

- **Accuracy**: The percentage of correctly classified images.
- **Precision**: The ability of the model to return only relevant instances.
- **Recall**: The ability of the model to find all relevant instances.
- **F1 Score**: The harmonic mean of precision and recall, providing a balance between the two.

Cross-validation was also employed to further validate the model's robustness and ensure it generalizes well to new data.

## Results and Discussion

The CNN model achieved an accuracy of XX% on the validation dataset. Key findings from the evaluation include:

- **High Accuracy in Sunny and Cloudy Categories**: The model performed exceptionally well in identifying sunny and cloudy weather conditions, with precision and recall values above YY%.
- **Moderate Performance in Rainy and Snowy Categories**: Performance in these categories was moderate, suggesting a need for further fine-tuning or additional data augmentation.
- **Challenges with Foggy Conditions**: The model struggled with foggy images, indicating that these might require more specialized preprocessing or a different model architecture.

Visualizations such as confusion matrices and ROC curves were generated to provide a more in-depth understanding of the model's performance across different categories.

## Conclusion

This project successfully demonstrated the use of a CNN for classifying weather conditions from images. While the model showed strong performance in most categories, there is room for improvement, particularly in distinguishing foggy conditions. Future work could explore the use of more advanced architectures, such as transfer learning or ensemble methods, to enhance model performance further.

