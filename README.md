# Handwritten Digit Recognition App

## Overview

This application is a web-based tool designed to recognize handwritten digits. Utilizing a Convolutional Neural Network (CNN) trained on the MNIST dataset, it can accurately predict digits ranging from 0 to 9. The system is built using Flask for the web framework and TensorFlow for the machine learning model, providing an intuitive interface for digit recognition.

## Features

- **Machine Learning Model:** The core of the application is a CNN trained on the MNIST dataset, a benchmark dataset in the field of machine learning.
- **Web Interface:** Users can upload images of handwritten digits through a simple web interface. The application processes these images and returns the predicted digit.
- **Data Augmentation:** During training, the model employs data augmentation techniques such as random rotations, zooms, and shifts to improve its accuracy and robustness.
- **Model Persistence:** The trained model is saved and can be loaded for predictions without retraining, making it efficient and quick to use.

## Technical Details

- **Model Architecture:**
  - Convolutional Layers: Three convolutional layers with ReLU activation functions to extract features from the input images.
  - Pooling Layers: Max-pooling layers to downsample the feature maps.
  - Dropout Layers: Dropout layers to prevent overfitting.
  - Batch Normalization: Batch normalization to stabilize and accelerate training.
  - Dense Layers: Fully connected layers to perform the final classification.

- **Training Process:**
  - The model is trained on the MNIST dataset, consisting of 60,000 training images and 10,000 testing images.
  - Data augmentation is applied to increase the diversity of the training data.
  - A learning rate scheduler is used to adjust the learning rate during training, improving the model's performance.

- **Prediction:**
  - Uploaded images are preprocessed (resized, normalized, and reshaped) to match the input format expected by the model.
  - The model predicts the digit, and the result is displayed on the web interface.

## Purpose and Applications

The primary purpose of this application is to demonstrate the capabilities of neural networks in image recognition tasks. It serves as an educational tool for understanding how machine learning models can be trained and deployed in real-world applications.

Potential applications include:
- **Educational Tools:** Helping students learn about machine learning and neural networks.
- **Digit Recognition:** Useful in digitizing handwritten documents and forms.
- **Interactive Demos:** Providing interactive demonstrations for workshops and presentations on AI and machine learning.

## Installation

1. **Clone the repository:**
    ```sh
    git clone https://github.com/yourusername/handwritten-digit-recognition.git
    cd handwritten-digit-recognition
    ```

2. **Create a virtual environment:**
    ```sh
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. **Install the required packages:**
    ```sh
    pip install -r requirements.txt
    ```

## Training the Model

1. **Run the training script:**
    ```sh
    python model.py
    ```
   This script will train the CNN on the MNIST dataset and save the model as `handwritten.model.keras`.

## Running the Application

1. **Start the Flask application:**
    ```sh
    python app.py
    ```

2. **Access the application:**
   Open a web browser and navigate to `http://0.0.0.0:8000` or `http://localhost:8000`.

## Using the Application

1. **Upload an Image:**
   - Use the web interface to upload a handwritten digit image. The image should be in grayscale and preferably centered on a white background.

2. **Get Prediction:**
   - The application will process the image and display the predicted digit.

## Conclusion

This Handwritten Digit Recognition App showcases the power of convolutional neural networks in processing and recognizing handwritten digits. It combines a robust machine learning model with an easy-to-use web interface, making it accessible for educational purposes and practical applications alike. The project highlights the seamless integration of AI with web technologies to create functional and interactive tools.
