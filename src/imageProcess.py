import os
from datetime import datetime
from PIL import Image

def process_images(input_dir='./dataset/unprocessed/rawImage', output_dir='./dataset/processed', target_size=(64, 64), label='a'):
    """
    Génère une copie redimensionnée des images du dossier d'entrée dans un nouveau dossier au format PNG.

    Arguments :
    - input_dir (str) : Dossier contenant les images d'origine. Par défaut, 'rawImage'.
    - output_dir (str) : Dossier de destination parent pour les images traitées. Par défaut, 'processed'.
    - target_size (tuple) : Dimensions cibles des images (largeur, hauteur). Par défaut, (64, 64).
    - label (str) : Label à associer à chaque image traitée. Par défaut, 'a'.

    Résultat :
    - Les images sont redimensionnées à la taille spécifiée et enregistrées en PNG dans le dossier de destination.
    """
    # Créer le dossier de destination s'il n'existe pas
    if not os.path.exists(os.path.join(output_dir, label)):
        os.makedirs(os.path.join(output_dir, label))

    # Parcourir toutes les images du dossier d'entrée
    for i, filename in enumerate(os.listdir(input_dir)):
        # Chemin complet vers l'image source
        input_path = os.path.join(input_dir, filename)

        # Vérifie si le fichier est une image
        if os.path.isfile(input_path) and filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
            
            with Image.open(input_path) as img:
                img_resized = img.resize(target_size)

                # Format de nommage : YYYYMMDD_HHMMSS_0000x.png
                current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_filename = f"{current_time}_{i:05d}.png"
                output_path = os.path.join(os.path.join(output_dir, label), output_filename)

                img_resized.save(output_path, format='PNG')

    print(f"Toutes les images ont été traitées et enregistrées dans le dossier '{os.path.join(output_dir, label)}'.")

# Exécution de la fonction
process_images(label="ka")