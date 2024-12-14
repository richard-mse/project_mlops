import os
import shutil

import bentoml
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import load_img, img_to_array

import h5py
import tensorflow as tf
from PIL.Image import Image

from common.constant import MODEL_TITLE, MODEL_PATH


#Fonction for extract data from dataset.h5
def extract_dataset(base_file):
    with h5py.File(base_file + ".h5", 'r') as f:
        class_names = f.attrs["class_list"]
        images = f['X_train'][:]
        labels = f['y_train'][:]

    return images, labels, class_names


def create_model(input_shape=(64, 64, 1), num_classes=10):
    model = Sequential([
        Input(shape=input_shape),
        Conv2D(32, kernel_size=(3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(64, kernel_size=(3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    return model


def plot_loss(history, save_path=None):
    # Plot loss curve
    plt.figure(figsize=(8, 6))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid()

    if save_path:
        plt.savefig(save_path, dpi=300)  # Save loss plot
        print(f"Loss plot saved to {save_path}")
    else:
        plt.show()

def train_model(model, x_train, y_train, epochs=30, batch_size=32, learning_rate=0.001):
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    history = model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=0.2)
    model.save('model.h5') # Save the trained model
    return history

def record_model(model_to_save, optimizer=True):
    bentoml.keras.save_model(
        MODEL_TITLE,
        model_to_save,
        include_optimizer=optimizer,
    )


def export_model():
    if not os.path.exists(MODEL_PATH):
        os.makedirs(MODEL_PATH)

    bentoml.models.export_model(f"{MODEL_TITLE}:latest",
                                f"{MODEL_PATH}/{MODEL_TITLE}.bentomodel")
    np.save(f"{MODEL_PATH}/history.npy", model.history.history)
    print("Training history:", history.history)
    print(f"\nModel saved at {MODEL_PATH} folder")


if __name__ == "__main__":

    # Load the data from the dataset folder (dataset/trainset)
    #x_train, labels, class_names = load_images_from_folders("dataset/trainset")
    x_train, labels, class_names = extract_dataset(base_file="dataset")

    y_train = to_categorical(labels, num_classes=len(class_names))  # One-hot encode labels

    # Create and train the model
    model = create_model(input_shape=(64, 64, 1), num_classes=len(class_names))
    model.summary()
    history = train_model(model, x_train, y_train)

    # Define folder paths
    results_folder = 'results'
    old_results_folder = 'old_results'

    # Create results folder if it doesn't exist
    if not os.path.exists(results_folder):
        os.makedirs(results_folder)

    # Create old_results folder if it doesn't exist
    if not os.path.exists(old_results_folder):
        os.makedirs(old_results_folder)

    # Check if the old loss file exists and move it to old_results_folder
    loss_path = os.path.join(results_folder, 'loss.png')
    if os.path.exists(loss_path):
        old_loss_path = os.path.join(old_results_folder, 'loss_old.png')
        shutil.move(loss_path, old_loss_path)  # Move the old loss file

    # Save the new loss plot
    plot_loss(history, loss_path)

    # Record and export the model
    record_model(model)
    export_model()

    # Optionally, print the history of the training
    print("Training history:", history.history)