import os
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import tensorflow as tf


# Function to load images from directories in dataset/trainset/
def load_images_from_folders(base_dir="dataset/trainset"):
    images = []  
    labels = []  
    class_names = []  # This will hold the unique class names

    # Loop through the directories (subfolders) and gather images
    for folder_name in os.listdir(base_dir):
        folder_path = os.path.join(base_dir, folder_name)
        
        if os.path.isdir(folder_path):
            class_names.append(folder_name)  # Add folder name as class name
            # Iterate through each file in the folder
            for file_name in os.listdir(folder_path):
                if file_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                    file_path = os.path.join(folder_path, file_name)
                    image = load_img(file_path, target_size=(64, 64), color_mode='grayscale')  # Resize and convert to grayscale
                    image_array = img_to_array(image)  # No need to normalize here, assuming it's already done

                    images.append(image_array)
                    labels.append(folder_name)  # Folder name is the label

    # Convert lists to numpy arrays
    images = np.array(images)
    labels = np.array(labels)

    print(f"Loaded {len(images)} images.")
    print("Class names:", class_names)
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
    x_train, labels, class_names = load_images_from_folders("dataset/trainset")

    # Map labels to integers and one-hot encode them
    label_map = {class_name: index for index, class_name in enumerate(class_names)}
    y_train = np.array([label_map[label] for label in labels])
    y_train = to_categorical(y_train, num_classes=len(class_names))  # One-hot encode labels

    # Create and train the model
    model = create_model(input_shape=(64, 64, 1), num_classes=len(class_names))
    model.summary()
    history = train_model(model, x_train, y_train)

    # Optionally, print the history of the training
    print("Training history:", history.history)
