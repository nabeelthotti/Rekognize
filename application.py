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

@app.route('/predict_digit', methods=['POST'])
def predict_digit():
    data = request.get_json()
    img_data = base64.b64decode(data['image'])

    with open('digit.png', 'wb') as f:
        f.write(img_data)

    img = cv2.imread('digit.png', cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (28, 28))
    if np.mean(img) > 127:
        img = np.invert(img)
    img = img / 255.0
    img = img.reshape(1, 28, 28, 1)

    prediction = digit_model.predict(img)
    digit = np.argmax(prediction)

    return jsonify({'digit': int(digit)})

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

    return jsonify({'letter': alphabet})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8080)
