from google.cloud import storage
import os
from google.oauth2 import service_account

# Path to your service account key file
key_file_path = r'C:\...\mlops-project-machledata-3657a44f4ecf.json'

# Load the credentials from the service account JSON file
credentials = service_account.Credentials.from_service_account_file(key_file_path)

project_id = "mlops-project-machledata"

def create_bucket(bucket_name, location="us-central1"):
    """
    Creates a new bucket in Google Cloud Storage
    :param bucket_name: Name of the new bucket
    :param location: Location where the bucket will be created, default is US
    """
    # Initialize a client for interacting with GCS, pass the credentials and project ID
    client = storage.Client(credentials=credentials, project=project_id)

    # Create the bucket
    bucket = client.bucket(bucket_name)
    if not bucket.exists():
        new_bucket = client.create_bucket(bucket_name, location=location)
        print(f"Bucket {new_bucket.name} created in {location}.")
    else:
        print(f"Bucket {bucket_name} already exists.")


def upload_pngs_to_bucket(bucket_name, root_directory):
    """
    Uploads all PNG files in a directory (including subdirectories) to the specified bucket in GCS
    :param bucket_name: Name of the bucket
    :param root_directory: Local path to the root directory containing PNG files
    """
    # Initialize a client for interacting with GCS, pass the credentials and project ID
    client = storage.Client(credentials=credentials, project=project_id)

    # Get the bucket
    bucket = client.get_bucket(bucket_name)

    # Walk through all directories and files starting from the root directory
    for subdir, _, files in os.walk(root_directory):
        for filename in files:
            # Check if the file is a PNG
            if filename.endswith(".png"):
                # Get the full path to the file
                file_path = os.path.join(subdir, filename)
                
                # Use relative path to preserve folder structure in the bucket
                relative_path = os.path.relpath(file_path, root_directory)
                
                # Create a new blob and upload the PNG
                blob = bucket.blob(relative_path)  # Use relative path as the object name
                blob.upload_from_filename(file_path)

                print(f"File {relative_path} uploaded to bucket {bucket_name}.")


if __name__ == "__main__":

    bucket_name = "png-dataset-processed"
    
    # Local path to the PNGs you want to upload
    png_directory_path = r"C:\...\project_mlops\dataset\processed"
    
    # Create a new bucket
    create_bucket(bucket_name)

    # Upload the PNGs to the created bucket
    upload_pngs_to_bucket(bucket_name, png_directory_path)
