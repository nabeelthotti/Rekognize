from flask import Flask, render_template, request, jsonify
import numpy as np
import tensorflow as tf
import cv2
import base64
from flask_cors import CORS
from model.model import load_model
import io
from PIL import Image, ImageOps

app = Flask(__name__)
CORS(app)

model = load_model()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    img_data = base64.b64decode(data['image'])
    
    with open('digit.png', 'wb') as f:
        f.write(img_data)
    
    # Read the image using OpenCV
    img = cv2.imread('digit.png', cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (28, 28))  # Resize to 28x28 pixels
    if np.mean(img) > 127:
        img = np.invert(img)
    img = img / 255.0  # Normalize the image
    img = img.reshape(1, 28, 28, 1)  # Reshape to match the input shape of the model

    prediction = model.predict(img)
    digit = np.argmax(prediction)

    return jsonify({'digit': int(digit)})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port = 8000)


