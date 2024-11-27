import os
import numpy as np
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

    # Optionally, print the history of the training
    print("Training history:", history.history)
