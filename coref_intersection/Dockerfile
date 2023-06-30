FROM python:3.7.12-slim-buster AS builder
ENV HOME=/home
WORKDIR $HOME


# set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1
RUN addgroup --system app && adduser --system app && adduser app app



# install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc g++ make && \
    rm -rf /var/lib/apt/lists/*

COPY app $HOME/app
COPY requirements.txt .
COPY init.py init.py
COPY script.sh $HOME/script.sh

RUN pip install --upgrade pip

RUN pip install -r requirements.txt
RUN chmod +x $HOME/script.sh
RUN ./script.sh
WORKDIR $HOME/app

CMD [ "uvicorn", "main:app", "--port", "5000", "--host", "0.0.0.0" ]
