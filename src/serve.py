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

    # bento_model = bentoml.keras.get(MODEL_TITLE)
    # # TODO : crash here
    #
    # def __init__(self) -> None:
    #     self.preprocess = self.bento_model.custom_objects["preprocess"]
    #     self.postprocess = self.bento_model.custom_objects["postprocess"]
    #     self.model = self.bento_model.load_model()
    #
    # @bentoml.api()
    # def predict(
    #         self,
    #         image: Annotated[PILImage, ContentType("image/png")] = Field(description="Hiragana image"),
    #         letter_confirmation: str = Field(..., description="write which letter you drew"),
    # ) -> Annotated[str, ContentType("application/json")]:
    #     image = self.preprocess(image)
    #     print(f"Hiragana drew : {letter_confirmation}")
    #
    #     predictions = self.model.predict(image)
    #
    #     return json.dumps(self.postprocess(predictions))

    @bentoml.api()
    def upload_images(self, inputs: List[PILImage], label: str) -> List[str]:
        """
        Handle batch uploads of images.
        """
        uploaded_urls = []
        for i, image in enumerate(inputs, start=1):

            # Generate a file name for the image
            current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
            file_name = f"img_{current_time}_{label}.png"

            # Save image to a buffer
            with io.BytesIO() as buffer:
                image.save(buffer, format="PNG")
                buffer.seek(0)

                # Upload image to GCS
                uploaded_url = upload_to_gcs(file_name, buffer, label)
                uploaded_urls.append(uploaded_url)

        return uploaded_urls