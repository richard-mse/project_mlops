import os
import shutil
import numpy as np
import h5py
from PIL import Image

def checkDatasetExist(datasetFileName):
    if not os.path.exists(datasetFileName + ".h5"):
        return False
    return True

def createDataset(images, labels, datasetFileName="dataset"):
    if checkDatasetExist(datasetFileName):
        print(f"Erreur : Le fichier {datasetFileName}.h5 existe déjà.")
        return

    with h5py.File(datasetFileName + ".h5", 'w') as f:
        # Créer un dataset pour les images
        f.create_dataset('images', data=images, maxshape=(None, images.shape[1], images.shape[2]), compression="gzip", chunks=True)
        
        # Créer un dataset pour les labels avec maxshape=(None,) pour permettre l'ajout de nouveaux labels
        f.create_dataset('labels', data=labels, maxshape=(None,), dtype="S10", compression="gzip", chunks=True)
        f.create_dataset('Times', data=labels, maxshape=(None,), dtype="S10", compression="gzip", chunks=True)

        # Optionnel : Ajouter des métadonnées pour mieux organiser
        f.attrs['description'] = 'Dataset d\'images pour classification'
        f.attrs['source'] = 'Hiragana écrit a la main'

    readDataset(datasetFileName)

def appendDataset(newImages, newLabels, datasetFileName="dataset"):
    if not checkDatasetExist(datasetFileName):
        print(f"Erreur : Le fichier {datasetFileName}.h5 n'existe pas.")
        return
    
    with h5py.File(datasetFileName + ".h5", 'a') as f:
        # Récupérer les datasets existants
        images = f['images']
        labels = f['labels']

        # Redimensionner les datasets pour ajouter les nouvelles données
        images.resize((images.shape[0] + newImages.shape[0]), axis=0)
        images[-newImages.shape[0]:] = newImages  # Ajouter les nouvelles images

        labels.resize((labels.shape[0] + newLabels.shape[0]), axis=0)
        labels[-newLabels.shape[0]:] = newLabels  # Ajouter les nouveaux labels

    readDataset(datasetFileName)

def readDataset(datasetFileName="dataset"):
    with h5py.File(datasetFileName + ".h5", 'r') as f:
        print("Contenu du fichier HDF5 :")
        print(f.keys())

        images = f['images'][:]
        labels = f['labels'][:]
        times = f['Times'][:]

        print(images)
        print("Images shape:", images.shape)
        print("Labels:", labels)
        print("Times",times)

def load_images_from_folders(base_dir="./dataset/processed",archive_dir="./dataset/archive"):
    images = []  
    labels = []  

    # Parcourir tous les sous-dossiers dans le dossier de base
    for folder_name in os.listdir(base_dir):
        folder_path = os.path.join(base_dir, folder_name)
        
        if os.path.isdir(folder_path):
            # Parcourir tous les fichiers dans le dossier
            for file_name in os.listdir(folder_path):
                print(file_name)
                file_path = os.path.join(folder_path, file_name)
                
                if file_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    image = Image.open(file_path)
                    # Convertir en niveau de gris
                    image = image.convert("L")

                    image_array = np.array(image)

                    images.append(image_array)
                    labels.append(folder_name)

                    archive_folder = os.path.join(archive_dir, folder_name)
                    if not os.path.exists(archive_folder):
                        os.makedirs(archive_folder)

                    # Déplacer l'image vers le dossier "archive"
                    archived_image_path = os.path.join(archive_folder, file_name)
                    shutil.move(file_path, archived_image_path)

    images = np.array(images)
    labels = np.array(labels, dtype="S10")

    return images, labels


images, labels = load_images_from_folders()
if len(images)<=10:
    print("Rien à ajouter au dataset.")
else:
    createDataset(images, labels)   # Crée le dataset initial
    #appendDataset(images, labels)   # Ajoute de nouvelles données

readDataset()     # Lis et affiche le contenu du dataset initial