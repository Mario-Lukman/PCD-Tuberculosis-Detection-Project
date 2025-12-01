"""
Training script for TB Detection System.
Trains SVM, Random Forest, and KNN models with cross-validation.
"""
import os
import numpy as np
import json
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, f1_score
import joblib
from tqdm import tqdm

from config import *
from utils import extract_features


def load_dataset():
    """Load and extract features from the TB dataset."""
    X = []
    y = []
    classes = {"Normal": 0, "Tuberculosis": 1}
    
    print(f"Loading dataset from: {DATASET_PATH}")
    
    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError(f"Dataset path not found: {DATASET_PATH}")
    
    for cls_name, cls_label in classes.items():
        folder = os.path.join(DATASET_PATH, cls_name)
        if not os.path.exists(folder):
            print(f"WARNING: Folder not found: {folder}")
            continue
        
        print(f"Loading {cls_name} from: {folder}")
        file_list = [f for f in os.listdir(folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        count = 0
        
        for filename in tqdm(file_list, desc=f"Processing {cls_name}"):
            img_path = os.path.join(folder, filename)
            try:
                features = extract_features(img_path)
                X.append(features)
                y.append(cls_label)
                count += 1
            except Exception as e:
                print(f"Error processing {filename}: {e}")
        
        print(f"Loaded {count} images for class {cls_name}")
    
    return np.array(X), np.array(y)


def train_and_evaluate_model(name, model, X_train, y_train, X_test, y_test, scaler=None):
    """Train a model with cross-validation and evaluate on test set."""
    print(f"\nTraining {name}")
    print("-" * 30)
    
    if scaler is not None:
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
    else:
        X_train_scaled = X_train
        X_test_scaled = X_test
    
    print(f"Performing {CV_FOLDS}-fold cross-validation...")
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=cv, scoring='f1', n_jobs=-1)
    
    print(f"Mean CV F1-Score: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
    
    print("Training on full training set...")
    model.fit(X_train_scaled, y_train)
    
    y_pred = model.predict(X_test_scaled)
    test_f1 = f1_score(y_test, y_pred)
    
    print(f"\n{name} Results:")
    print(classification_report(y_test, y_pred, target_names=["Normal", "TB"]))
    print(f"Test F1-Score: {test_f1:.4f}")
    
    return {
        'name': name,
        'model': model,
        'scaler': scaler,
        'cv_f1_mean': cv_scores.mean(),
        'cv_f1_std': cv_scores.std(),
        'test_f1': test_f1,
        'cv_scores': cv_scores.tolist()
    }


def main():
    print("TB Detection System - Model Training")
    print("=" * 40)
    
    print("\nStep 1: Loading Dataset...")
    X, y = load_dataset()
    
    if len(X) == 0:
        print("ERROR: No data loaded.")
        return
    
    print(f"\nSuccessfully loaded {len(X)} images.")
    print(f"Class distribution: Normal={np.sum(y==0)}, TB={np.sum(y==1)}")
    
    print(f"\nStep 2: Splitting data (test_size={TEST_SIZE})...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )
    
    print(f"Training set: {len(X_train)} samples")
    print(f"Test set: {len(X_test)} samples")
    
    print("\nStep 3: Training models...")
    results = []
    
    # SVM
    svm_scaler = StandardScaler()
    svm_model = SVC(**SVM_PARAMS)
    results.append(train_and_evaluate_model(
        "SVM", svm_model, X_train, y_train, X_test, y_test, svm_scaler
    ))
    
    # Random Forest
    rf_model = RandomForestClassifier(**RF_PARAMS)
    results.append(train_and_evaluate_model(
        "Random Forest", rf_model, X_train, y_train, X_test, y_test, None
    ))
    
    # KNN
    knn_scaler = StandardScaler()
    knn_model = KNeighborsClassifier(**KNN_PARAMS)
    results.append(train_and_evaluate_model(
        "KNN", knn_model, X_train, y_train, X_test, y_test, knn_scaler
    ))
    
    print("\n" + "=" * 40)
    print("Model Selection")
    print("=" * 40)
    
    best_result = max(results, key=lambda x: x['cv_f1_mean'])
    
    print("\nModel Comparison (by CV F1-Score):")
    for r in sorted(results, key=lambda x: x['cv_f1_mean'], reverse=True):
        print(f"  {r['name']:15s}: {r['cv_f1_mean']:.4f} (+/- {r['cv_f1_std']:.4f}) | Test F1: {r['test_f1']:.4f}")
    
    print(f"\nBest Model: {best_result['name']}")
    print(f"CV F1-Score: {best_result['cv_f1_mean']:.4f}")
    
    print(f"\nStep 4: Saving best model to {MODEL_DIR}...")
    
    model_path = os.path.join(MODEL_DIR, 'best_model.pkl')
    scaler_path = os.path.join(MODEL_DIR, 'scaler.pkl')
    info_path = os.path.join(MODEL_DIR, 'model_info.json')
    
    joblib.dump(best_result['model'], model_path)
    print(f"Saved model: {model_path}")
    
    if best_result['scaler'] is not None:
        joblib.dump(best_result['scaler'], scaler_path)
        print(f"Saved scaler: {scaler_path}")
    else:
        joblib.dump(None, scaler_path)
        print(f"No scaler needed for {best_result['name']}")
    
    model_info = {
        'model_name': best_result['name'],
        'cv_f1_mean': best_result['cv_f1_mean'],
        'cv_f1_std': best_result['cv_f1_std'],
        'test_f1': best_result['test_f1'],
        'cv_scores': best_result['cv_scores'],
        'training_samples': len(X_train),
        'test_samples': len(X_test),
        'class_distribution': {
            'Normal': int(np.sum(y==0)),
            'TB': int(np.sum(y==1))
        }
    }
    
    with open(info_path, 'w') as f:
        json.dump(model_info, f, indent=2)
    print(f"Saved model info: {info_path}")
    
    print("\nTraining Complete!")
    print("You can now run the web app with: python app.py")


if __name__ == "__main__":
    main()
