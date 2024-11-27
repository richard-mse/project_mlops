import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import load_model
from sklearn.metrics import confusion_matrix
from tensorflow.keras.utils import to_categorical
import h5py


def extract_dataset(base_file):
    with h5py.File(base_file + ".h5", 'r') as f:
        class_names=f.attrs["class_list"]
        images = f['X_test'][:]
        labels = f['y_test'][:]

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
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=range(len(np.unique(y_true))), yticklabels=range(len(np.unique(y_true))))
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
    x_test, labels, class_names = extract_dataset(base_file="dataset")

    # Map labels to integers and one-hot encode them
    y_test = to_categorical(labels, num_classes=len(class_names))  # One-hot encode labels

    # Load the trained model
    model = load_model('model.h5')  # The model trained in train.py
    
    # Evaluate the model and display accuracy
    evaluate_model(model, x_test, y_test)
    
    # Create results folder if not exists
    results_folder = 'results'
    if not os.path.exists(results_folder):
        os.makedirs(results_folder)

    # Save confusion matrix plot
    confusion_matrix_path = os.path.join(results_folder, 'confusion_matrix.png')
    plot_confusion_matrix(model, x_test, y_test, save_path=confusion_matrix_path)
