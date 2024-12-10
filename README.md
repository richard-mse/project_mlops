# Hiragana ML Pipeline

This repository contains a MLOps pipeline for recognizing Hiragana characters. Users can upload handwritten Hiragana images, which are then processed, labeled, and used to train and improve a model. The pipeline also allows continuous learning by retraining the model with new data. This project is done in the course "Machine Learning and Data in Operation" at HES-SO Provence, Lausanne.
---

## Pipeline Overview

1. **Image Processing**: Downloads and processes images from a Google Cloud Storage bucket.
2. **Dataset Management**: Maintains a dataset file (`dataset.h5`) that stores images, labels, and metadata.
3. **Data Preparation**: Splits the dataset into training and testing sets.
4. **Model Training**: Trains a convolutional neural network (CNN) model on the dataset.
5. **Evaluation**: Evaluates the trained model and generates performance metrics.
6. **Serving**: Deploys the trained model as a REST API for real-time predictions.
7. **Continuous Learning**: Retrains the model with new data and compares it with the previous version.

---

## Scripts Description

### **1. `dvc.yaml`**
Defines the pipeline stages and dependencies for DVC (Data Version Control).

- **Stages**:
  - `process`: Processes raw images into a structured dataset.
  - `update-dataset`: Updates or creates the dataset file (`dataset.h5`).
  - `prepare`: Prepares training and testing datasets from `dataset.h5`.
  - `train`: Trains the model on the prepared dataset.
  - `evaluate`: Evaluates the trained model's performance.
  - `serve`: Deploys the trained model as a REST API.

---

### **2. `updateDataset.py`**
Manages the dataset file (`dataset.h5`).

- **Functions**:
  - `checkDatasetExist`: Checks if the dataset file exists.
  - `createDataset`: Creates a new dataset with images, labels, and metadata.
  - `appendDataset`: Appends new data to the existing dataset.
  - `readDataset`: Reads and displays the dataset's contents.
  - `load_images_from_folders`: Loads and archives images from local folders.
  - `check_conditions`: Checks conditions to update the dataset.

---

### **3. `imageProcess.py`**
Handles image fetching and preprocessing.

- **Functions**:
  - `fetch_images_from_gcs`: Downloads images from Google Cloud Storage.
  - `process_images`: Resizes and organizes images into labeled folders.

---

### **4. `prepare.py`**
Prepares the data for training.

- **Functions**:
  - `extract_dataset`: Extracts images and labels from `dataset.h5`.
  - `split_new_data`: Separates new data from previously used data.
  - `select_old_data`: Selects a subset of old data for replay training.
  - `shuffle_dataset`: Randomly shuffles the dataset.
  - `split_dataset`: Splits the data into training and testing sets.
  - `add_train_to_dataset`: Saves training and testing sets in the dataset file.

---

### **5. `train.py`**
Trains a convolutional neural network (CNN) model.

- **Functions**:
  - `extract_dataset`: Loads the training data from `dataset.h5`.
  - `create_model`: Builds a CNN model.
  - `train_model`: Compiles and trains the model.
  - `plot_loss`: Plots the training and validation loss curves.
  - `record_model`: Saves the model using BentoML.
  - `export_model`: Exports the trained model for deployment.

---

### **6. `evaluate.py`**
Evaluates the trained model.

- **Functions**:
  - `extract_dataset`: Extracts testing data and class names from `dataset.h5`.
  - `evaluate_model`: Calculates test accuracy for the trained model.
  - `plot_confusion_matrix`: Generates and saves a confusion matrix.
  - `plot_loss`: Plots training and validation loss curves.
  - `calculate_prediction_variance`: Computes and visualizes prediction variances.

- **Usage**:
  - Evaluates the model on test data and saves visualizations (confusion matrix and variance histogram).
  - Automatically saves old evaluation results for version tracking.

---

### **7. `serve.py`**
Deploys the trained model as a REST API using BentoML.

- **Endpoints**:
  - `/predict`: Accepts a PNG image and a letter confirmation string as input and returns the predicted Hiragana character.

- **Features**:
  - Preprocesses input images and validates predictions.
  - Outputs predictions in JSON format for integration with applications.

## Authors
- Tyarks Richard-André
- Nuur Maxamed Maxamed
- Fournier Bruno
- Jouclard Charly
- Casey Benjamin
