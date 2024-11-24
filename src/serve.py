from __future__ import annotations

import bentoml

from common.constants import MODEL_CLASSIFIER, MODEL_TITLE


@bentoml.service(name=MODEL_CLASSIFIER)
class HiraganaClassifierService:
    bento_model = bentoml.keras.get(MODEL_TITLE)

    def __init__(self) -> None:
        self.model = self.bento_model.load_model()

    @bentoml.api
    def hello_world(self) -> str:
        return f"Hello world! model : {MODEL_TITLE}"

# bentoml serve --working-dir ./src serve:HiraganaClassifierService
