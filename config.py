"""
Configuration file for TB Detection System.
"""
import os
import numpy as np

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BASE_DIR, "TB_Chest_Radiography_Database")
MODEL_DIR = os.path.join(BASE_DIR, "models")
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Image Processing
IMAGE_SIZE = (256, 256)
GRAYSCALE = True

# GLCM Parameters
GLCM_DISTANCES = [1]
GLCM_ANGLES = [0, np.pi/4, np.pi/2, 3*np.pi/4]

# LBP Parameters
LBP_P = 8
LBP_R = 1

# Model Training
TEST_SIZE = 0.2
RANDOM_STATE = 42
CV_FOLDS = 5

# Model Hyperparameters
SVM_PARAMS = {
    'kernel': 'rbf',
    'C': 1.0,
    'gamma': 'scale',
    'probability': True,
    'random_state': RANDOM_STATE,
    'class_weight': 'balanced'
}

RF_PARAMS = {
    'n_estimators': 100,
    'max_depth': 20,
    'min_samples_split': 10,
    'min_samples_leaf': 4,
    'random_state': RANDOM_STATE,
    'class_weight': 'balanced',
    'n_jobs': -1
}

KNN_PARAMS = {
    'n_neighbors': 7,
    'weights': 'distance',
    'n_jobs': -1
}

# Class Labels
CLASS_LABELS = {
    0: "Normal",
    1: "Tuberculosis"
}

# Web Server
FLASK_HOST = '0.0.0.0'
FLASK_PORT = 5000
FLASK_DEBUG = True
MAX_UPLOAD_SIZE = 16 * 1024 * 1024
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
