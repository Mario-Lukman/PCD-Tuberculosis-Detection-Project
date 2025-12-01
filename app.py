"""
Flask Web Application for TB Detection System.
"""
import os
import json
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
import numpy as np
import cv2
import joblib
import base64

from config import *
from utils import preprocess_image, segment_lung, get_lbp_features, get_glcm_features

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = MAX_UPLOAD_SIZE
app.config['UPLOAD_FOLDER'] = UPLOAD_DIR

model = None
scaler = None
model_info = None


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def load_model_once():
    global model, scaler, model_info
    model_path = os.path.join(MODEL_DIR, 'best_model.pkl')
    scaler_path = os.path.join(MODEL_DIR, 'scaler.pkl')
    info_path = os.path.join(MODEL_DIR, 'model_info.json')
    
    if not os.path.exists(model_path):
        print("WARNING: Model not found. Please run train.py first.")
        return False
        
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    
    with open(info_path, 'r') as f:
        model_info = json.load(f)
        
    print(f"Loaded model: {model_info['model_name']}")
    return True


def image_to_base64(img):
    _, buffer = cv2.imencode('.png', img)
    img_str = base64.b64encode(buffer).decode('utf-8')
    return f"data:image/png;base64,{img_str}"


@app.route('/')
def index():
    return render_template('index.html', model_info=model_info)


@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Model not loaded. Please run train.py first.'}), 500
        
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
        
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
        
    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type.'}), 400
        
    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)
    
    try:
        img = preprocess_image(filepath)
        seg, mask = segment_lung(img)
        
        _, lbp_hist = get_lbp_features(seg, mask)
        glcm_vec = get_glcm_features(seg, mask)
        
        features = np.concatenate([lbp_hist, glcm_vec])
        features = features.reshape(1, -1)
        
        if scaler is not None:
            features = scaler.transform(features)
            
        prediction = model.predict(features)[0]
        probs = model.predict_proba(features)[0]
        
        label = CLASS_LABELS[prediction]
        confidence = float(probs[prediction] * 100)
        
        # Create overlay
        overlay = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        colored_mask = np.zeros_like(overlay)
        colored_mask[mask == 1] = [0, 255, 0]  # Green
        
        overlay = cv2.addWeighted(overlay, 0.7, colored_mask, 0.3, 0)
        
        result = {
            'prediction': label,
            'confidence': confidence,
            'probabilities': {
                'Normal': float(probs[0] * 100),
                'Tuberculosis': float(probs[1] * 100)
            },
            'images': {
                'original': image_to_base64(img),
                'mask': image_to_base64(mask * 255),
                'segmented': image_to_base64(seg),
                'overlay': image_to_base64(overlay)
            },
            'model_name': model_info['model_name']
        }
        
        os.remove(filepath)
        return jsonify(result)
        
    except Exception as e:
        if os.path.exists(filepath):
            os.remove(filepath)
        return jsonify({'error': f'Error processing image: {str(e)}'}), 500


if __name__ == '__main__':
    print("Starting server...")
    if load_model_once():
        app.run(host=FLASK_HOST, port=FLASK_PORT, debug=FLASK_DEBUG)
    else:
        print("Cannot start server: Model not found.")
