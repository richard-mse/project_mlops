from __future__ import annotations

import io
import json
from typing import Annotated, List

import bentoml
from pathlib import Path
from PIL.Image import Image as PILImage
from bentoml.validators import ContentType
from google.cloud import storage
from google.oauth2 import service_account
from pydantic import Field, BaseModel
from typing_extensions import Annotated
from datetime import datetime

from common.constant import MODEL_TITLE, PATH_CREDENTIAL, BUCKET_NAME, PROJECT_ID

credentials = service_account.Credentials.from_service_account_file(f"../{PATH_CREDENTIAL}")
client = storage.Client(credentials=credentials, project=PROJECT_ID)

# bentoml containerize hiragana_classifier_service:latest --image-tag hiragana_classifier_service:latest

def upload_to_gcs(file_name, file_data, character):
    bucket = client.get_bucket(BUCKET_NAME)
    dest_path = f"image/archive/{character}/{file_name}"
    blob = bucket.blob(dest_path)
    blob.upload_from_file(file_data)
    return f"File : {file_data}, uploaded to path : gs://{BUCKET_NAME}/{dest_path}"

# bentoml serve --working-dir src

@bentoml.service
class HiraganaClassifierService:

    @bentoml.api()
    def upload_images(self, image: Annotated[PILImage, ContentType("image/png")], label: str) -> str:
        """
        Handle upload of image in google cloud.
        """
        uploaded_url = ""

        # Generate a file name for the image
        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_name = f"img_{current_time}_{label}.png"

        # Save image to a buffer
        with io.BytesIO() as buffer:
            image.save(buffer, format="PNG")
            buffer.seek(0)

            # Upload image to GCS
            uploaded_url = upload_to_gcs(file_name, buffer, label)

        return uploaded_url