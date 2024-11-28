import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input
from tensorflow.keras.utils import to_categorical
import tensorflow as tf
import h5py

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

    # Optionally, print the history of the training
    print("Training history:", history.history)
