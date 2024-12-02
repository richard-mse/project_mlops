import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.utils import to_categorical
import bentoml

from common.constant import MODEL_TITLE, MODEL_PATH


# MODEL_TITLE = "hiragana_classifier_model"
# MODEL_PATH = "model"


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
                    image = load_img(file_path, target_size=(64, 64),
                                     color_mode='grayscale')  # Resize and convert to grayscale
                    image_array = img_to_array(image)  # Already normalized from the prepare.py step

                    images.append(image_array)
                    labels.append(folder_name)  # Folder name is the label

    # Convert lists to numpy arrays
    images = np.array(images)
    labels = np.array(labels)

    print(f"Loaded {len(images)} images for testing.")
    print("Class names:", class_names)
    return images, labels, class_names


def evaluate_model(model, x_test, y_test):
    test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
    print(f"Test Accuracy: {test_accuracy:.4f}")
    return test_accuracy


def plot_confusion_matrix(model, x_test, y_test, save_path=None):
    # Get model predictions
    y_pred = model.predict(x_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true = np.argmax(y_test, axis=1)

    # Calculate confusion matrix
    cm = confusion_matrix(y_true, y_pred_classes)

    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=range(len(np.unique(y_true))),
                yticklabels=range(len(np.unique(y_true))))
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')

    if save_path:
        plt.savefig(save_path, dpi=300)  # Save confusion matrix plot
        print(f"Confusion matrix saved to {save_path}")
    else:
        plt.show()


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


if __name__ == "__main__":
    x_test, labels, class_names = load_images_from_folders("dataset/testset")

    # Map labels to integers and one-hot encode them
    label_map = {class_name: index for index, class_name in enumerate(class_names)}
    y_test = np.array([label_map[label] for label in labels])
    y_test = to_categorical(y_test, num_classes=len(class_names))  # One-hot encode labels

    try:
        bentoml.models.import_model(f"{MODEL_PATH}/{MODEL_TITLE}.bentomodel")
    except bentoml.exceptions.BentoMLException:
        print("Model already exists in the model store - skipping import.")

    # Load the trained model
    model = bentoml.keras.load_model(MODEL_TITLE)

    # Evaluate the model and display accuracy
    evaluate_model(model, x_test, y_test)

    # Create results folder if not exists
    results_folder = 'results'
    if not os.path.exists(results_folder):
        os.makedirs(results_folder)

    # Save confusion matrix plot
    confusion_matrix_path = os.path.join(results_folder, 'confusion_matrix.png')
    plot_confusion_matrix(model, x_test, y_test, save_path=confusion_matrix_path)

    print("Custom objects:", model.__str__())
    print("Custom objects:", model.custom_objects)
