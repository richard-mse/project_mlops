import os
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.preprocessing import LabelEncoder
from PIL import Image
import shutil

def load_images_from_folders(base_dir="dataset/processed"):
    images = []  
    labels = []  
    class_names = []  # This will hold the unique class names

    # Loop through the directories and gather images
    for folder_name in os.listdir(base_dir):
        folder_path = os.path.join(base_dir, folder_name)
        
        if os.path.isdir(folder_path):
            class_names.append(folder_name)  # Add folder name as class name
            # Iterate through each file in the folder
            for file_name in os.listdir(folder_path):
                if file_name.lower().endswith(('.png', '.jpg', '.jpeg')):  # Image file extensions
                    file_path = os.path.join(folder_path, file_name)
                    image = load_img(file_path, target_size=(64, 64), color_mode='grayscale')  # Resize and convert to grayscale
                    image_array = img_to_array(image) / 255.0  # Normalize image

                    images.append(image_array)
                    labels.append(folder_name)  # Folder name is the label

    # Convert lists to numpy arrays
    images = np.array(images)
    labels = np.array(labels)

    print(f"Loaded {len(images)} images.")
    print("Class names:", class_names)
    return images, labels, class_names


def clear_and_create_dirs(*dirs):
    # Clear the directories if they exist and recreate them
    for dir_path in dirs:
        if os.path.exists(dir_path):
            shutil.rmtree(dir_path)  # Remove all files/folders inside
        os.makedirs(dir_path)  # Recreate the directory


def split_and_save_data(images, labels, class_names, train_dir="dataset/trainset", test_dir="dataset/testset", test_size=0.2):
    # Clear existing data in trainset and testset folders
    clear_and_create_dirs(train_dir, test_dir)

    # Encode labels to integer values
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(labels)
    y_encoded = np.array(y_encoded)

    # Split data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(images, y_encoded, test_size=test_size, random_state=42)

    # Save images to the train and test folders
    save_data_to_folders(X_train, y_train, train_dir, label_encoder)
    save_data_to_folders(X_test, y_test, test_dir, label_encoder)


def save_data_to_folders(X_data, y_data, data_dir, label_encoder):
    # Save images in the corresponding class folders
    for i, image in enumerate(X_data):
        label = label_encoder.inverse_transform([y_data[i]])[0]  # Get the original class name from label encoding
        label_folder = os.path.join(data_dir, label)
        
        # Create class folder if it does not exist
        if not os.path.exists(label_folder):
            os.makedirs(label_folder)
        
        # Format image filename as 00001.png, 00002.png, etc.
        output_filename = f"{i:05d}.png"
        output_path = os.path.join(label_folder, output_filename)
        
        # Convert NumPy array to PIL Image and save it
        img = (image * 255).astype(np.uint8)  # Convert back to original range (0-255)
        img = Image.fromarray(img.squeeze(), 'L')  # 'L' mode for grayscale image
        
        img.save(output_path, format='PNG')
        print(f"Saved {output_filename} in {label_folder}")


def main():
    # Load images from the dataset
    images, labels, class_names = load_images_from_folders("dataset/processed")
    
    # Split and save the dataset into train and test sets
    split_and_save_data(images, labels, class_names)

    print("Dataset splitting and saving completed.")


if __name__ == "__main__":
    main()
