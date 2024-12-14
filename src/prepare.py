import os
import numpy as np
from sklearn.model_selection import train_test_split
import h5py
import warnings

#Fonction for extract data from dataset.h5
def extract_dataset(base_file):

    file_path = base_file + ".h5"
    
    # Debugging the file path
    print(f"Looking for file at: {file_path}")
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file '{file_path}' does not exist.")

    with h5py.File(base_file + ".h5", 'r') as f:
        lastRevision = f.attrs["revision"]
        class_names = f.attrs["class_list"]
        images = f['images'][:]
        labels = f['labels'][:]
        updateId =  f['update-id'][:]

    labels = np.array([label.decode("utf-8") for label in labels])

    return images, labels, updateId, int(lastRevision), class_names

#Fonction for split new data from previous data
def split_new_data(all_images,all_labels, all_updateId, lastRevision):
    new_images = all_images[np.where(all_updateId==lastRevision)[0][0]:]
    new_labels = all_labels[np.where(all_updateId==lastRevision)[0][0]:]

    old_images = all_images[:np.where(all_updateId==lastRevision)[0][0]]
    old_labels = all_labels[:np.where(all_updateId==lastRevision)[0][0]]

    return new_images, new_labels, old_images, old_labels

#Fonction for select old data for training
def select_old_data(old_images,old_labels,replay_fraction):
    old_images_count = old_images.shape[0]
    replay_samples = int(replay_fraction*old_images_count)
    replay_indices = np.random.choice(old_images_count, replay_samples, replace=False)
    replay_images = old_images[replay_indices]
    replay_labels = old_labels[replay_indices]

    return replay_images, replay_labels

def shuffle_dataset(images, labels):
    shuffle_indices = np.random.permutation(images.shape[0])
    images = images[shuffle_indices]
    labels = labels[shuffle_indices]

    return images, labels

def load_dataset(base_file="dataset", replay_fraction = 1.0):
    all_images, all_labels, all_updateId, lastRevision, class_names = extract_dataset(base_file)

    new_images, new_labels, old_images, old_labels = split_new_data(all_images,all_labels, all_updateId, lastRevision)
    
    replay_images, replay_labels = select_old_data(old_images,old_labels,replay_fraction)

    images = np.concatenate((new_images,replay_images),axis=0)
    labels = np.concatenate((new_labels,replay_labels),axis=0)

    images, labels = shuffle_dataset(images,labels)

    return images, labels, class_names

def add_train_to_dataset(X_train, X_test, y_train, y_test, base_file="dataset"):
    with h5py.File(base_file+".h5", "a") as f:
        if "X_train" in f:
            del f["X_train"]
        f.create_dataset("X_train", data=X_train)
        if "X_test" in f:
                del f["X_test"]
        f.create_dataset("X_test", data=X_test)
        if "y_train" in f:
                del f["y_train"]
        f.create_dataset("y_train", data=y_train)
        if "y_test" in f:
                del f["y_test"]
        f.create_dataset("y_test", data=y_test)

#Fonction for split between train set and test set
def split_dataset(images, labels, class_names, base_file="dataset", test_size=0.2):

    # Encode labels to integer values
    # On utilise plus label_encoder car sinon ordre alphabetique latin
    label_map = {class_name: index for index, class_name in enumerate(class_names)}
    y_encoded = np.array([label_map[label] for label in labels])

    # Split data into training and test sets
    warnings.warn("Random_state initalisé !", UserWarning)
    X_train, X_test, y_train, y_test = train_test_split(images, y_encoded, test_size=test_size, random_state=42)

    add_train_to_dataset(X_train, X_test, y_train, y_test, base_file="dataset")

def main():
    images, labels, class_names = load_dataset(base_file="dataset",replay_fraction=1.0)
    split_dataset(images, labels, class_names, base_file="dataset", test_size=0.2)


if __name__ == "__main__":
    main()
