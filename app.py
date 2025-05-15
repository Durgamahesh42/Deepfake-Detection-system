# app.py - Flask web application to serve the deepfake detection model
from flask import Flask, render_template, request, jsonify
import os
import tensorflow as tf
import numpy as np
from PIL import Image
import uuid
import time

app = Flask(__name__)

# Configuration
UPLOAD_FOLDER = 'data/uploads'
MODEL_PATH = 'final_deepfake_detector.h5'
IMG_SIZE = 128

# Create upload folder if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load the model when the application starts
print("Loading deepfake detection model...")
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No image selected'}), 400

    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500

    try:
        # Save the file temporarily with a unique filename
        filename = str(uuid.uuid4()) + os.path.splitext(file.filename)[1]
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)

        # Load and preprocess the image using PIL
        img = Image.open(filepath).convert('RGB')
        img = img.resize((IMG_SIZE, IMG_SIZE))
        img_array = np.array(img) / 255.0  # Normalize
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

        # Make prediction
        prediction = model.predict(img_array)[0][0]

        # Determine classification and confidence
        is_real = prediction > 0.5
        classification = "REAL" if is_real else "FAKE"
        confidence = float(prediction) if is_real else float(1 - prediction)

        # Prepare results
        results = {
            "classification": classification,
            "confidence": confidence,
            "raw_score": float(prediction),
            "image_path": os.path.join('uploads', filename)
        }

        return jsonify(results)

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
