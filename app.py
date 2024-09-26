from flask import Flask, request, jsonify
from flask_cors import CORS
from tensorflow.keras.models import load_model
import cv2
import numpy as np
import base64
import os
import pickle
from skimage.feature import hog
import joblib
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model

# Confidence threshold
CONFIDENCE_THRESHOLD = 0.6

# Load pre-trained ResNet model for feature extraction
base_model = ResNet50(weights='imagenet', include_top=False, pooling='avg')
resnet_base_model = Model(inputs=base_model.input, outputs=base_model.output)

app = Flask(__name__)
CORS(app)

# Load models and scaler
hog_sgd_model = joblib.load('hog_sgd_model_2.pkl')
resnet_model = load_model('facial_recognition_classifier.h5')
scaler = joblib.load('scaler.pkl')

def preprocess_image_for_resnet(img):
    imgs = cv2.resize(img, (224, 224))
    img_array = image.img_to_array(imgs)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def extract_features_hog(img):
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    resized_image = cv2.resize(gray_image, (128, 128))
    features = hog(resized_image, block_norm='L2-Hys')
    return np.array([features], dtype=np.float32)

def extract_features_resnet(img, resnet_base_model, scaler):
    processed_image = preprocess_image_for_resnet(img)
    processed_image = preprocess_input(processed_image)  # Preprocess for ResNet50
    features = resnet_base_model.predict(processed_image)
    features = features.flatten()
    return scaler.transform([features])

def predict_face_label(face, hog_sgd_model, resnet_model, scaler):
    # For HOG model
    hog_features = extract_features_hog(face)
    hog_prediction = hog_sgd_model.predict(hog_features)
    # If your SGDClassifier supports `decision_function` or `predict_proba`:
    if hasattr(hog_sgd_model, 'predict_proba'):
        hog_confidence = hog_sgd_model.predict_proba(hog_features)
        hog_confidence = np.max(hog_confidence)  # Extract the highest probability
    else:
        # Alternatively, use `decision_function` if available (for binary classification)
        if hasattr(hog_sgd_model, 'decision_function'):
            hog_confidence = hog_sgd_model.decision_function(hog_features)
            hog_confidence = 1 / (1 + np.exp(-hog_confidence))  # Sigmoid for binary output
        else:
            hog_confidence = 1.0  # Fallback confidence if no confidence measurement is available
    
    # For ResNet model
    resnet_features = extract_features_resnet(face, resnet_base_model, scaler)
    resnet_prediction = resnet_model.predict(resnet_features)
    resnet_confidence = np.max(resnet_prediction)

    # Return 'no prediction' if confidence is too low
    if np.max(hog_confidence) < CONFIDENCE_THRESHOLD:
        hog_label = "no prediction"
    else:
        hog_label = str(hog_prediction[0])

    if resnet_confidence < CONFIDENCE_THRESHOLD:
        resnet_label = "no prediction"
    else:
        resnet_label = str(np.argmax(resnet_prediction))

    return {
        'hog_prediction': hog_label,
        'hog_confidence': str(hog_confidence),
        'resnet_prediction': resnet_label,
        'resnet_confidence': str(resnet_confidence)
    }

@app.route('/predict_face', methods=['POST'])
def predict_face():
    # Extract the image data from the request
    image_string = request.json['face']
    if image_string.startswith("data:image/"):
        header, base64_data = image_string.split(",", 1)
    else:
        base64_data = image_string
    image_data = base64.b64decode(base64_data)
    np_image = np.frombuffer(image_data, dtype=np.uint8)
    image = cv2.imdecode(np_image, cv2.IMREAD_COLOR)

    # Predict label using the provided image
    predictions = predict_face_label(image, hog_sgd_model, resnet_model, scaler)
    
    return jsonify(predictions)

if __name__ == '__main__':
    app.run(debug=True, use_reloader=False)
