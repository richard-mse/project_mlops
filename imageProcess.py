import os
import json
from PIL import Image

def process_images(input_dir='rawImage', output_dir='processedImage', target_size=(64, 64), label='a'):
    """
    Génère une copie redimensionnée des images du dossier d'entrée dans un nouveau dossier au format PNG 
    et crée un fichier JSON contenant une liste des labels associés.

    Arguments :
    - input_dir (str) : Dossier contenant les images d'origine. Par défaut, 'rawImage'.
    - output_dir (str) : Dossier de destination pour les images traitées. Par défaut, 'processedImage'.
    - target_size (tuple) : Dimensions cibles des images (largeur, hauteur). Par défaut, (64, 64).
    - label (str) : Label à associer à chaque image traitée, stocké dans le fichier JSON. Par défaut, 'a'.

    Résultat :
    - Les images sont redimensionnées à la taille spécifiée et enregistrées en PNG dans le dossier de destination.
    - Un fichier JSON contenant une liste de labels est créé dans le dossier de destination.
    """
    # Créer le dossier de destination s'il n'existe pas
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    labelList = []

    # Parcourir toutes les images du dossier d'entrée
    for i, filename in enumerate(os.listdir(input_dir)):
        # Chemin complet vers l'image source
        input_path = os.path.join(input_dir, filename)

        # Vérifie si le fichier est une image
        if os.path.isfile(input_path) and filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
            
            with Image.open(input_path) as img:
                img_resized = img.resize(target_size)

                # Format de nommage : 0000x.png
                output_filename = f"{i:05d}.png"
                output_path = os.path.join(output_dir, output_filename)

                img_resized.save(output_path, format='PNG')

                labelList.append(label)

    # Enregistrer la liste de statuts dans un fichier JSON
    status_file_path = os.path.join(output_dir, "labels.json")
    with open(status_file_path, 'w') as json_file:
        json.dump(labelList, json_file)

    print(f"Toutes les images ont été traitées et enregistrées dans le dossier '{output_dir}'.")
    print(f"Le fichier de labels a été enregistré sous '{status_file_path}'.")

# Exécution de la fonction
process_images()
