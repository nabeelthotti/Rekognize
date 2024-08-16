# Rekognize

## Live Site: https://rekognize.fly.dev

## Overview

Rekognize is a web-based tool designed to recognize both handwritten digits and alphabets. By leveraging a Convolutional Neural Network (CNN) trained on the MNIST and EMNIST datasets, the application can accurately predict characters ranging from digits (0-9) to alphabets (A-Z, both uppercase and lowercase). The system is built using Flask for the web framework and TensorFlow for the machine learning model, providing an intuitive interface for character recognition.

## Features

- **Machine Learning Model:** At the heart of the application is a CNN trained on both the MNIST dataset for digits and the EMNIST dataset for alphabets. These datasets are standard benchmarks in the field of machine learning.
- **Comprehensive Character Recognition:** The application can recognize not just digits but also uppercase and lowercase letters, making it versatile for various tasks.
- **Web Interface:** Users can upload images of handwritten characters through a simple and intuitive web interface. The application processes these images and returns the predicted character.
- **Data Augmentation:** During training, the model employs advanced data augmentation techniques such as random rotations, zooms, shifts, and elastic distortions to improve its accuracy and robustness.
- **Model Persistence:** The trained model is saved and can be loaded for predictions without retraining, ensuring efficiency and quick responses during use.

## Technical Details

### Model Architecture:
- **Convolutional Layers:** The model includes multiple convolutional layers that utilize filters to detect features such as edges, corners, and textures in the input images. These layers use ReLU (Rectified Linear Unit) activation functions to introduce non-linearity, which allows the network to model complex patterns in the data.
- **Pooling Layers:** Max-pooling layers follow the convolutional layers to downsample the feature maps, reducing the spatial dimensions and computational load while retaining essential information.
- **Dropout Layers:** To prevent overfitting, dropout layers are strategically placed in the network. These layers randomly deactivate a fraction of neurons during training, forcing the network to learn more robust and generalized features.
- **Batch Normalization:** Batch normalization layers are used to stabilize and accelerate training by normalizing the inputs of each layer. This helps in faster convergence and allows the use of higher learning rates.
- **Dense Layers:** After flattening the pooled feature maps, the model uses fully connected (dense) layers to perform the final classification. These layers consolidate the extracted features and output probabilities for each class (digits and alphabets).

### Training Process:
- **Dataset Utilization:** The model is trained on a combined dataset of MNIST for digits and EMNIST for alphabets, providing a comprehensive training set of handwritten characters.
- **Data Augmentation:** Extensive data augmentation is applied to create a more diverse and balanced training set. Techniques include random rotations, zooms, shifts, and elastic distortions, which help the model generalize better to real-world inputs.
- **Learning Rate Scheduling:** A learning rate scheduler is employed to adjust the learning rate dynamically during training. This approach improves model performance by allowing more significant updates during the initial stages and finer adjustments as training progresses.
- **Model Evaluation:** The model is evaluated on a separate test set to ensure it generalizes well to new, unseen data. Metrics such as accuracy, precision, and recall are used to assess performance.
  
### Prediction:
- **Image Preprocessing:** Uploaded images are preprocessed by resizing, normalizing, and reshaping them to match the input format expected by the model. This ensures consistency and accuracy in predictions.
- **Character Recognition:** The model predicts the character (digit or alphabet) and displays the result on the web interface, including the confidence level of the prediction.

## Purpose and Applications

Rekognize serves as a demonstration of the capabilities of convolutional neural networks in image recognition tasks. It is an educational tool for understanding how machine learning models can be trained and deployed in real-world applications, extending beyond digit recognition to include alphabet recognition.

### Potential Applications:
- **Educational Tools:** Ideal for students and educators to learn and teach about machine learning and neural networks, with a focus on practical applications.
- **Character Recognition:** Useful in digitizing handwritten documents, forms, and notes, with extended capabilities to recognize handwritten alphabets.
- **Interactive Demos:** Perfect for workshops and presentations on AI and machine learning, providing an interactive experience that showcases the power of neural networks.

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
   This script will train the CNN on the MNIST and EMNIST datasets and save the model as `handwritten_alphabet_digit.model.keras`.

## Running the Application

1. **Start the Flask application:**
    ```sh
    python app.py
    ```

2. **Access the application:**
   Open a web browser and navigate to `http://0.0.0.0:8000` or `http://localhost:8000`.

## Using the Application

1. **Draw Character:**
   - Use the web interface to draw an image of a handwritten digit or alphabet. The image should be large and centered on the canvas for best results.

2. **Get Prediction:**
   - The application will process the image and display the predicted character along with the confidence score.

## Conclusion

Rekognize showcases the power of convolutional neural networks in processing and recognizing handwritten characters, including both digits and alphabets. By combining a robust machine learning model with an easy-to-use web interface, Rekognize serves as both an educational tool and a practical application, highlighting the seamless integration of AI with web technologies to create functional and interactive tools.
