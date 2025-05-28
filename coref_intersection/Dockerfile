FROM python:3.7.12-slim-buster AS builder
ENV HOME=/home
WORKDIR $HOME


# set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1
ENV NEURALCOREF_CACHE $HOME/neuralcoref
RUN addgroup --system app && adduser --system app && adduser app app



# install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc g++ make && \
    rm -rf /var/lib/apt/lists/*

# Install poetry
RUN pip install --upgrade pip
RUN pip install poetry

COPY app $HOME/app
COPY pyproject.toml poetry.lock* ./
COPY init.py init.py
COPY script.sh $HOME/script.sh

# Ensure poetry is in PATH - this might vary depending on how poetry is installed.
# Adding /root/.local/bin to PATH as a common location for pip user installs.
ENV PATH="/root/.local/bin:$PATH"

RUN poetry install --no-dev --no-root

RUN chmod +x $HOME/script.sh
RUN ./script.sh
WORKDIR $HOME/app

CMD [ "uvicorn", "main:app", "--port", "5000", "--host", "0.0.0.0" ]
