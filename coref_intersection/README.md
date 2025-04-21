# Coref Intersection

Coref Intersection is a [FastAPI](https://fastapi.tiangolo.com/)-based application designed for coreference resolution. It leverages libraries like [SpaCy](https://spacy.io/), [NeuralCoref](https://github.com/huggingface/neuralcoref), and potentially [AllenNLP](https://allennlp.org/) (if integrated) to provide API endpoints for processing text and resolving coreferences, aiming to enhance text analysis and understanding.


## Features

- Coreference resolution using SpaCy integrated with NeuralCoref.
- (Optional/Potential) Advanced coreference resolution using AllenNLP.
- REST API endpoints for straightforward integration.
- Middleware for CORS support and response time tracking.

## Requirements

- Python 3.7+
- [Poetry](https://python-poetry.org/) (for dependency management)
- [Docker](https://www.docker.com/) (optional, for containerized deployment)

## Installation

1.  **Clone the repository:**
    ```bash
    git clone <repository-url> # Replace <repository-url> with the actual URL
    cd coref_intersection
    ```

2.  **Install dependencies using Poetry:**
    This installs the main dependencies defined in `pyproject.toml`.
    ```bash
    poetry install
    ```

3.  **Download required SpaCy model:**
    NeuralCoref often works well with specific SpaCy models. `en_core_web_sm` is a common choice.
    ```bash
    python -m spacy download en_core_web_sm
    ```

4.  **Install NeuralCoref:**
    NeuralCoref often requires specific build steps, hence the separate pip install with `--no-binary`.
    ```bash
    pip install neuralcoref --no-binary neuralcoref
    ```
    *Note: Ensure this step is performed within the environment managed by Poetry (e.g., after activating the virtual environment using `poetry shell` or by running `poetry run pip install ...`)*

5.  **(If using AllenNLP) Install AllenNLP and download models:**
    *If AllenNLP features are implemented, add specific instructions here for its installation and model downloads.*

## Running the Application

### Using Poetry

1.  **Activate the virtual environment (optional but recommended):**
    ```bash
    poetry shell
    ```
2.  **Start the FastAPI server:**
    ```bash
    # If inside the shell activated by `poetry shell`
    uvicorn app.main:app --host 0.0.0.0 --port 5000 --reload

    # Or run directly using poetry run
    poetry run uvicorn app.main:app --host 0.0.0.0 --port 5000 --reload
    ```
    *(Added `--reload` for development convenience)*

Access the API documentation (Swagger UI) at `http://127.0.0.1:5000/docs`.

### Using Docker

1.  **Build the Docker image:**
    ```bash
    docker build -t coref_intersection .
    ```

2.  **Run the container:**
    ```bash
    docker run -d --name coref_app -p 5000:5000 coref_intersection
    ```
    *(Added `-d` to run detached and `--name` for easier management)*

Access the API documentation (Swagger UI) at `http://127.0.0.1:5000/docs`.

## API Endpoints

### `/ping`

*   **Method:** `GET`
*   **Description:** Health check endpoint. Verifies if the server is running and responsive.
*   **Response:**
    ```json
    {
      "ping": "pong"
    }
    ```

### `/coref`

*   **Method:** `POST`
*   **Description:** Processes the input text to perform coreference resolution using the configured models (NeuralCoref and potentially others).
*   **Request Body:**
    ```json
    {
      "text": "Your input text here. For example: Anna went to the park. She enjoyed her time there."
    }
    ```
*   **Response Body (Example Structure):**
    ```json
    {
      "msg": "Success",
      "text": "Original input text",
      "neural_response": {
        "resolved_text": "Anna went to the park. Anna enjoyed Anna's time there.",
        "clusters": [ /* List of coreference clusters found by NeuralCoref */ ]
        // ... other potential fields from NeuralCoref
      },
      "nlp_coref": {
         // Results from AllenNLP or another coref model if integrated
         "resolved_text": "...",
         "clusters": [ /* ... */ ]
         // ...
      }
    }
    ```
    *Note: The exact structure of `neural_response` and `nlp_coref` depends on the implementation details in `app/services/coref_service.py`.*

## Development

### Linting and Formatting

This project uses Ruff for linting and formatting.

*   **Check for issues:**
    ```bash
    poetry run ruff check .
    ```
*   **Format code:**
    ```bash
    poetry run ruff format .
    ```

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.
