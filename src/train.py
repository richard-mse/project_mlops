import os

import bentoml
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import load_img, img_to_array

import h5py
import tensorflow as tf
from PIL.Image import Image

from common.constant import MODEL_TITLE, MODEL_PATH

#Fonction for extract data from dataset.h5
def extract_dataset(base_file):
    with h5py.File(base_file + ".h5", 'r') as f:
        class_names=f.attrs["class_list"]
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

def train_model(model, x_train, y_train, epochs=10, batch_size=128):
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    history = model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=0.2)
    model.save('model.h5')  # Save the trained model
    return history


def record_model(model_to_save, optimizer=True):
    def preprocess(x: Image):
        x = x.convert('L')  # Convert to grayscale (1 channel)
        x = x.resize((64, 64))
        x = np.array(x)
        x = x / 255.0
        x = np.expand_dims(x, axis=-1)  # Ensure it has 1 channel (shape will be (64, 64, 1))
        x = np.expand_dims(x, axis=0)  # Add batch dimension (shape will be (1, 64, 64, 1))
        return x

    def postprocess(x: Image):
        return {
            "prediction": labels[tf.argmax(x, axis=-1).numpy()[0]],
            "probabilities": {
                labels[i]: prob
                for i, prob in enumerate(tf.nn.softmax(x).numpy()[0].tolist())
            },
        }

    bentoml.keras.save_model(
        MODEL_TITLE,
        model_to_save,
        include_optimizer=optimizer,
        custom_objects={
            "preprocess": preprocess,
            "postprocess": postprocess,
        }
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

    # Create results folder if not exists
    results_folder = 'results'
    if not os.path.exists(results_folder):
        os.makedirs(results_folder)

    # Save loss plot
    loss_path = os.path.join(results_folder, 'loss.png')
    plot_loss(history, loss_path)

    record_model(model)
    export_model()

    # Optionally, print the history of the training
    print("Training history:", history.history)
