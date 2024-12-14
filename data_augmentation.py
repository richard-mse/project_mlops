import os
import shutil
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
import numpy as np

# Chemins de base
base_dir = "dataset/unprocessed/downloaded/archive"
output_dir = "augmented_balanced_dataset"

# Nombre d'images cible par classe
target_image_count = 200

# Créer le dossier de sortie s'il n'existe pas
os.makedirs(output_dir, exist_ok=True)

# Paramètres d'augmentation
datagen = ImageDataGenerator(
    rotation_range=20,        # Rotation aléatoire de 0 à 20 degrés
    width_shift_range=0.2,    # Décalage horizontal
    height_shift_range=0.2,   # Décalage vertical
    shear_range=0.15,         # Transformation en cisaillement
    zoom_range=0.2,           # Zoom avant ou arrière
    horizontal_flip=False,    # Retourner horizontalement
    fill_mode="nearest"       # Mode de remplissage des pixels manquants
)

# Étape 1 : Génération pour chaque dossier
for folder_name in os.listdir(base_dir):
    folder_path = os.path.join(base_dir, folder_name)
    if os.path.isdir(folder_path):
        output_folder = os.path.join(output_dir, folder_name)
        os.makedirs(output_folder, exist_ok=True)
        
        # Comptage des images actuelles
        current_images = [img for img in os.listdir(folder_path) if img.endswith(('png', 'jpg', 'jpeg'))]
        current_image_count = len(current_images)
        images_to_generate = max(0, target_image_count - current_image_count)

        if images_to_generate > 0:
            print(f"Génération de {images_to_generate} images augmentées pour {folder_name}")
            current_image_index = 0  # Pour parcourir les images en rotation
            
            # Ajouter des images augmentées uniquement
            while images_to_generate > 0:
                image_name = current_images[current_image_index]
                image_path = os.path.join(folder_path, image_name)
                try:
                    # Charger l'image
                    img = load_img(image_path)
                    img_array = img_to_array(img)
                    img_array = np.expand_dims(img_array, axis=0)
                    
                    # Générer des images augmentées
                    for batch in datagen.flow(img_array, batch_size=1,
                                              save_to_dir=output_folder,
                                              save_prefix="aug",
                                              save_format="png"):
                        images_to_generate -= 1
                        if images_to_generate <= 0:
                            break
                except Exception as e:
                    print(f"Erreur avec l'image {image_path}: {e}")
                
                # Passer à l'image suivante
                current_image_index = (current_image_index + 1) % len(current_images)

        print(f"{folder_name} contient maintenant environ {target_image_count} images (originales + augmentées).")

print(f"Augmentation et équilibrage terminés. Chaque dossier contient environ {target_image_count} images.")
