from __future__ import annotations

import json
from typing import Annotated

import bentoml
from PIL.Image import Image as PILImage
from bentoml.validators import ContentType
from pydantic import Field
from typing_extensions import Annotated

from common.constant import MODEL_TITLE


@bentoml.service
class HiraganaClassifierService:
    print(MODEL_TITLE)
    bento_model = bentoml.keras.get(MODEL_TITLE)
    # TODO : crash here
    print("Custom objects:", bento_model.__str__())
    print("Custom objects:", bento_model.custom_objects)

    def __init__(self) -> None:
        self.preprocess = self.bento_model.custom_objects["preprocess"]
        self.postprocess = self.bento_model.custom_objects["postprocess"]
        self.model = self.bento_model.load_model()

    @bentoml.api()
    def predict(
            self,
            image: Annotated[PILImage, ContentType("image/png")] = Field(description="Hiragana image"),
            letter_confirmation: str = Field(..., description="write which letter you drew"),
    ) -> Annotated[str, ContentType("application/json")]:
        image = self.preprocess(image)
        print(f"Hiragana drew : {letter_confirmation}")

        predictions = self.model.predict(image)

        return json.dumps(self.postprocess(predictions))
