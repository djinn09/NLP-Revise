"""Main application module for coreference resolution.

This module initializes and configures a FastAPI application with endpoints
for coreference resolution using SpaCy, NeuralCoref, and AllenNLP models.
It also includes middleware, logging, and utility functions.
"""

import logging
import os
from functools import lru_cache

import neuralcoref
import spacy
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from starlette.middleware import Middleware

from app.allennlp_coref import get_allennlp_coref, get_coref_object
from app.base_model import RequestBody, setting
from app.middleware import ResponseTimeMiddleWare
from app.utils import get_neural_reference_resolved

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("app.log"),
    ],
)
logger = logging.getLogger(__name__)
logger.info("Starting the application...")
logger.info("Loading environment variables...")

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

# Load the neuralcoref model
logger.info("NEURALCOREF_CACHE:%s", os.getenv("NEURALCOREF_CACHE"))
logger.info("Loading Spacy model...")
nlp = spacy.load("en_core_web_sm")
logger.info("Loaded spacy model.....")
logger.info("Loading neuralcoref model...")
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
logger.info("Loaded model neuralcoref......")
logger.info("Loading allen-nlp model...")
# Load the allen-nlp model
predictor = get_coref_object(setting.ALLEN_NLP_MODEL_URL)
logger.info("Loaded model allen nlp")


@lru_cache()
def get_app() -> FastAPI:
    """Create and configure a FastAPI application instance with specified settings.

    This function initializes the FastAPI application with middleware and
    configurations based on the provided settings and middleware.

    Returns:
        FastAPI: The configured FastAPI application instance.

    """
    logger.info("Creating FastAPI application...")
    server = FastAPI(
        title=setting.app_name,
        debug=setting.DEBUG,
        middleware=middlewares,
        port=setting.PORT,
        host=setting.HOST,
    )
    logger.info("FastAPI application created.")

    @server.get("/")
    async def root_get() -> RedirectResponse:
        """Redirect to the docs page.

        Returns:
            RedirectResponse: The redirect response to the docs page.

        """
        return RedirectResponse("/docs")

    return server


app = get_app()

logger.info("Starting server on %s:%s", setting.HOST, setting.PORT)


@app.get("/ping")
async def ping() -> dict:
    """Handle GET requests to the /ping endpoint.

    Returns:
        str: A response message indicating success with the message "pong".

    """
    return {"msg": "pong"}


@app.post("/coref")
async def coref(data: RequestBody) -> dict:
    """Handle POST requests to the /coref endpoint.

    Args:
        data (RequestBody): The request body containing the text to process.

    Returns:
        dict: A response dictionary containing the original text, neural coreference response,
              and NLP coreference information.

    """
    text: str = data.text

    # Validate input text
    if len(text) < 1:
        return {"msg": "No text provided"}

    # Process the text with SpaCy
    doc = nlp(text)
    response: dict = {"msg": "Success", "text": text}

    # Check for neuralcoref coreferences
    if doc._.has_coref:
        neural_response: dict = get_neural_reference_resolved(doc)
        response["neural_response"] = neural_response
    else:
        response["neural_response"] = {"msg": "No coref found"}

    # Get AllenNLP coreference resolution
    response["nlp_coref"] = get_allennlp_coref(predictor, nlp, text)

    return response


if __name__ == "__main__":
    port = int(os.getenv("PORT", setting.PORT))
    host = os.getenv("HOST", setting.HOST)
    logger.info("Starting server on %s:%s", host, port)
    # Start the server
    uvicorn.run(app=app, host=host, port=port, log_level="info")
