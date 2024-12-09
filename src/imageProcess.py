from google.cloud import storage
from google.oauth2 import service_account
import os
from datetime import datetime
from PIL import Image

# Path to your service account key file
key_file_path = r'C:\Users\Benja\OneDrive - HESSO\1er semestre\Machine Learning and Data in Operation\mlops-project-machledata-3657a44f4ecf.json'

# Load the credentials
credentials = service_account.Credentials.from_service_account_file(key_file_path)
project_id = "mlops-project-machledata"
bucket_name = "png-dataset-processed"

def fetch_images_from_gcs(bucket_name, remote_folder, local_folder):
    """
    Downloads all PNG images from a specific folder in a Google Cloud Storage bucket into corresponding local folders.
    
    :param bucket_name: Name of the GCS bucket
    :param remote_folder: Folder path in the bucket (e.g., 'image/pending/')
    :param local_folder: Local folder to store the downloaded images
    """
    client = storage.Client(credentials=credentials, project=project_id)
    bucket = client.get_bucket(bucket_name)

    # List all files in the specified folder
    blobs = bucket.list_blobs(prefix=remote_folder)  # List all files with the specified prefix
    for blob in blobs:
        if blob.name.endswith('.png'):  # Only download PNG files
            # Determine the label folder name (i.e., 'a', 'e', 'i', etc.) from the remote path
            parts = blob.name.split("/")
            folder_name = parts[1]  # 'pending' or 'archive', and then the label (e.g., 'i')

            # Construct the local subfolder path
            local_subdir = os.path.join(local_folder, folder_name, parts[2])  # Example: downloaded/archive/i/
            os.makedirs(local_subdir, exist_ok=True)

            # Local path to save the image
            local_path = os.path.join(local_subdir, os.path.basename(blob.name))  # Save with original filename
            blob.download_to_filename(local_path)
            print(f"Downloaded {blob.name} to {local_path}.")

def process_images(input_dir, output_dir='./dataset/processed', target_size=(64, 64)):
    """
    Processes images by resizing and saving them in structured folders.
    Ensures that specific folders ('a', 'e', 'i', 'o', 'u', 'ka', 'ke', 'ki', 'ko', 'ku') are created,
    even if no images are present in them.
    """
    # List of manual folder names to create (for Hiragana)
    manual_folders = ['a', 'e', 'i', 'o', 'u', 'ka', 'ke', 'ki', 'ko', 'ku']
    
    # Manually create the output label directories if they do not exist
    for folder in manual_folders:
        output_label_dir = os.path.join(output_dir, folder)
        os.makedirs(output_label_dir, exist_ok=True)  # Ensure folders are created even if they are empty

    # Iterate through all folders in the input directory (e.g., 'archive', 'pending')
    for folder in os.listdir(input_dir):  # 'archive', 'pending', etc.
        folder_dir = os.path.join(input_dir, folder)
        
        if os.path.isdir(folder_dir):  # Only process subdirectories
            for label in os.listdir(folder_dir):  # 'a', 'e', 'i', etc.
                label_dir = os.path.join(folder_dir, label)

                if os.path.isdir(label_dir):  # Only process label directories
                    for filename in os.listdir(label_dir):
                        input_path = os.path.join(label_dir, filename)

                        if os.path.isfile(input_path) and filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):

                            # Ensure the label folder exists in the output directory
                            output_label_dir = os.path.join(output_dir, label)
                            os.makedirs(output_label_dir, exist_ok=True)  # Create the label folder

                            # Check if the input file exists
                            if os.path.exists(input_path):
                                with Image.open(input_path) as img:
                                    img_resized = img.resize(target_size)
                                    output_path = os.path.join(output_label_dir, filename)
                                    img_resized.save(output_path, format='PNG')
                            else:
                                print(f"Warning: Input file {input_path} does not exist.")
                        else:
                            print(f"Skipping non-image file {input_path}")

    print(f"Images processed and saved to '{output_dir}'.")


if __name__ == "__main__":
    # Local folder to store all downloaded images from both 'pending' and 'archive'
    local_download_dir = "./dataset/unprocessed/downloaded"

    # Fetch images from GCS
    print("Fetching pending images...")
    fetch_images_from_gcs(bucket_name, "image/pending/", local_download_dir)
    print("Fetching archive images...")
    fetch_images_from_gcs(bucket_name, "image/archive/", local_download_dir)

    # Process images
    print("Processing images...")
    process_images(local_download_dir)
