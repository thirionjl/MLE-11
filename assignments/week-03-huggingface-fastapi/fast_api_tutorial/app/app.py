# It is expected that:
# - A model already exists in local system folder model/my_translation_model (relative path to this file)
# - Uvicorn is launched from folder app/ to load our model from local filesystem
from enum import Enum
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline
from typing import List


class TextToTranslate(BaseModel):
    input_text: str


class TextsToTranslate(BaseModel):
    input_texts: List[str]


class TargetLang(str, Enum):
    FR = "fr"
    DE = "de"
    RO = "ro"

# Load model from local filesystem and reuse for translation pipelines to translate from english to french,german and romanian
translation_functions = {
    lang: pipeline(
        f"translation_en_to_{lang.value}", model="model/my_translation_model"
    )
    for lang in TargetLang
}

app = FastAPI()


@app.post("/translate")
def translate(body: TextToTranslate, lang: TargetLang = TargetLang.DE):
    """
    Translates a single string from english to a target language. Target language is expressed as a query param
    """
    translations = _translate([body.input_text], lang)
    return {"output_text": translations[0]}


@app.post("/translate-all")
def translate_all(body: TextsToTranslate, lang: TargetLang = TargetLang.DE):
    """
    Translates a list of strings from english to a target language. Target language is expressed as a query param
    """
    return {"output_texts": _translate(body.input_texts, lang)}


def _translate(texts: List[str], lang: TargetLang) -> List[str]:
    """
    Removes the wrapping the pipeline puts around the strings
    """    
    f = translation_functions[lang]
    translations = f(texts)
    return [trans["translation_text"] for trans in translations]


@app.get("/")
def index():
    return {"message": "Hello World"}


@app.get("/ping")
def ping():
    return {"message": "pong"}


@app.post("/echo")
def echo(text_to_translate: TextToTranslate):
    return {"message": text_to_translate.input_text}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
