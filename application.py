from flask import Flask, render_template, request, jsonify
import numpy as np
import tensorflow as tf
import cv2
import base64
from app.model.model import load_model
from PIL import Image, ImageOps
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'app')))

app = Flask(__name__, template_folder='app/templates', static_folder='app/static')

# Load models
digit_model = load_model('app/models/handwritten.model.keras')
alphabet_model = load_model('app/models/best_alphabet_model.keras')


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/digit')
def digit():
    return render_template('digit.html')

@app.route('/soon')
def soon():
    return render_template('soon.html')

@app.route('/alphabet')
def alphabet():
    return render_template('alphabet.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/howitworks')
def howitworks():
    return render_template('howitworks.html')


# Define a route in Flask application that listens for POST requests at '/predict_digit'.
@app.route('/predict_digit', methods=['POST'])
def predict_digit():
    # Get the JSON data from the incoming request.
    data = request.get_json()
    
    # Decode the base64 encoded image data from the JSON into bytes.
    img_data = base64.b64decode(data['image'])
    
    # Write the decoded image data to a file named 'digit.png' in binary mode.
    with open('digit.png', 'wb') as f:
        f.write(img_data)

    # Read the saved image file as a grayscale image using OpenCV.
    img = cv2.imread('digit.png', cv2.IMREAD_GRAYSCALE)
    
    # Resize the image to 28x28 pixels, which is the input size expected by the digit model.
    img = cv2.resize(img, (28, 28))
    
    # Invert the image colors if the image's average brightness is higher than 127.
    # This is done because some digit classifiers expect the background to be dark and the digit to be light.
    if np.mean(img) > 127:
        img = np.invert(img)
    
    # Normalize the pixel values to the range [0, 1] for better model performance.
    img = img / 255.0
    
    # Reshape the image array to match the input shape expected by the model (batch_size, height, width, channels).
    img = img.reshape(1, 28, 28, 1)

    # Use the digit model to predict the digit in the image.
    prediction = digit_model.predict(img)
    
    # Find the digit with the highest probability from the model's predictions.
    digit = np.argmax(prediction)
    
    # Calculate the confidence level of the prediction as the maximum probability.
    confidence = np.max(prediction)

    # Return the predicted digit and the confidence as a JSON response.
    return jsonify({'digit': int(digit), 'confidence': float(confidence)})



@app.route('/predict_alphabet', methods=['POST'])
def predict_alphabet():
    data = request.get_json()
    img_data = base64.b64decode(data['image'])

    with open('alphabet.png', 'wb') as f:
        f.write(img_data)

    img = cv2.imread('alphabet.png', cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (28, 28))
    if np.mean(img) > 127:
        img = np.invert(img)
    img = img / 255.0
    img = img.reshape(1, 28, 28, 1)

    prediction = alphabet_model.predict(img)
    alphabet = chr(np.argmax(prediction) + ord('A'))
    confidence = np.max(prediction)  # Calculate maximum probability

    return jsonify({'letter': alphabet, 'confidence': float(confidence)})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8080)
