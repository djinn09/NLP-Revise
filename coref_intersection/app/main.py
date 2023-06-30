import os
from functools import lru_cache

import neuralcoref
import spacy
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from starlette.middleware import Middleware

from app.base_model import RequestBody, setting
from app.allennlp_coref import get_allennlp_coref, get_coref_object
from app.middleware import ResponseTimeMiddleWare
from app.utils import get_neural_reference_resolved
from concurrent.futures import ThreadPoolExecutor

# middlewares
middlewares = (
    Middleware(ResponseTimeMiddleWare),
    Middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    ),
)

os.environ["NEURALCOREF_CACHE"] = "{}/{}".format(os.getcwd(), setting.NEURALCOREF_CACHE)
print(os.getenv("NEURALCOREF_CACHE"))
nlp = spacy.load("en_core_web_sm")
print("Loaded model spacy model.....")
neuralcoref.add_to_pipe(
    nlp,
    max_dist=200,
    max_dist_match=200,
    conv_dict={
        "Deepika": ["woman", "actress"],
        "Shivaji Bhonsale": [
            "Chhatrapati Shivaji",
            "king",
            "Marathi ruler",
            "Shivaji Bhonsale",
        ],
    },
)
print("Loaded model neuralcoref......")
predictor = get_coref_object(setting.ALLEN_NLP_MODEL_URL)
print("Loaded model allen nlp")


@lru_cache()
def get_app() -> FastAPI:
    server = FastAPI(
        title=setting.app_name,
        debug=setting.DEBUG,
        middleware=middlewares,
        port=setting.PORT,
        host=setting.HOST,
    )

    @server.get("/")
    async def root_get() -> RedirectResponse:
        return RedirectResponse("/docs")

    return server


app = get_app()


@app.get("/ping")
async def ping() -> str:
    return {"msg": "pong"}


@app.post("/coref")
async def coref(data: RequestBody):
    text = data.text
    if len(text) < 1:
        return {"msg": "No text provided"}

    doc = nlp(text)
    response = {"msg": "Success", "text": text}
    if doc._.has_coref:
        neural_response =  get_neural_reference_resolved(doc)
        response["neural_response"] = neural_response
    else:
        response["neural_response"] = {"msg": "No coref found"}
    response["nlp_coref"] = get_allennlp_coref(predictor, nlp, text)
    return response


if __name__ == "__main__":
    port = os.getenv("PORT", 5000)
    host = os.getenv("HOST", "0.0.0.0")
    uvicorn.run(app=app, host=host, port=port, log_level="info")
