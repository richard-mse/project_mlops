from __future__ import annotations

import io
import json
from typing import Annotated, List

import bentoml
import numpy as np
import tensorflow as tf

from pathlib import Path
from PIL.Image import Image as PILImage
from bentoml.validators import ContentType
from google.cloud import storage
from google.oauth2 import service_account
from pydantic import Field, BaseModel
from typing_extensions import Annotated
from datetime import datetime


from common.constant import MODEL_TITLE, FILE_CREDENTIAL, BUCKET_NAME, PROJECT_ID, LABELS

credentials = service_account.Credentials.from_service_account_file(f"./{FILE_CREDENTIAL}")
client = storage.Client(credentials=credentials, project=PROJECT_ID)

# bentoml containerize hiragana_classifier_service:latest --image-tag hiragana_classifier_service:latest

def upload_to_gcs(file_name, file_data, character):
    bucket = client.get_bucket(BUCKET_NAME)
    dest_path = f"image/archive/{character}/{file_name}"
    blob = bucket.blob(dest_path)
    blob.upload_from_file(file_data)
    return f"File : {file_data}, uploaded to path : gs://{BUCKET_NAME}/{dest_path}"

# bentoml serve --working-dir src

def preprocess(x: Image):
    x = x.convert('L')  # Convert to grayscale (1 channel)
    x = x.resize((64, 64))
    x = np.array(x)
    x = x / 255.0
    x = np.expand_dims(x, axis=-1)  # Ensure it has 1 channel (shape will be (64, 64, 1))
    x = np.expand_dims(x, axis=0)  # Add batch dimension (shape will be (1, 64, 64, 1))
    return x

def postprocess(x: Image):
    return {
        "prediction": LABELS[tf.argmax(x, axis=-1).numpy()[0]],
        "probabilities": {
            LABELS[i]: prob
            for i, prob in enumerate(tf.nn.softmax(x).numpy()[0].tolist())
        },
    }


@bentoml.service
class HiraganaClassifierService:

    bento_model = bentoml.keras.get(MODEL_TITLE)

    def __init__(self) -> None:
        self.model = self.bento_model.load_model()

    @bentoml.api()
    def predict(
            self,
            image: Annotated[PILImage,
            ContentType("image/png")] = Field(description="Hiragana image")) -> Annotated[str, ContentType("application/json")]:
        """
        Predict the class of the image
        """
        image = preprocess(image)
        predictions = self.model.predict(image)
        return json.dumps(postprocess(predictions))

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