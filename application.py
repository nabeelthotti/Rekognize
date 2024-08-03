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

#app.config['SERVER_NAME'] = 'rekognize.co'

model = load_model()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/digit')
def digit():
    return render_template('digit.html')


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
    app.run(debug=True, host='0.0.0.0', port = 8080)


