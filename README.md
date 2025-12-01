# Tuberculosis Detection System

A machine learning system to detect Tuberculosis from chest X-rays using SVM, Random Forest, and KNN for PCD Group Project.
Authors: Mario Aloysius Lukman, Krisna Dwi, Naila Salma Yusroini, Aufa Sultan Majid Syach Putra Yuliyanto

## Setup

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Train the model (Optional, download the 'model' folder for trained model)**:
   ```bash
   python train.py
   ```
   *Note: Requires the `TB_Chest_Radiography_Database` folder.*

3. **Run the Web App**:
   ```bash
   python app.py
   ```
   Open **http://localhost:5000** in your browser.

## Dataset

The dataset is too large to be included in this repository. Please download it from Kaggle:
**[Tuberculosis (TB) Chest X-ray Database](https://www.kaggle.com/datasets/tawsifurrahman/tuberculosis-tb-chest-xray-dataset)**

**Instructions:**
1. Download the dataset.
2. Extract it.
3. Rename the folder to `TB_Chest_Radiography_Database`.
4. Place it in the root directory of this project.

## Project Structure

- **app.py**: Flask web application
- **train.py**: Script to train models
- **utils.py**: Image processing and feature extraction logic
- **config.py**: Configuration settings
- **models/**: Directory for saved models
- **static/** & **templates/**: Frontend files

