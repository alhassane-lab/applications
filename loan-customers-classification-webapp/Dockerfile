FROM python:3.9-slim as python

WORKDIR /app

RUN apt-get update -y && apt-get install git -y && \
    apt-get -y -q clean all && \
    rm -rf /var/lib/apt/lists/*

RUN apt-get update && apt-get install -y --no-install-recommends apt-utils
RUN apt-get -y install curl
RUN apt-get install libgomp1

RUN useradd -ms /bin/bash user && \
    chown -R 1000:3000 /app

# 1. Install Poetry

## Poetry env variables
ENV PYTHONUNBUFFERED=true \
    POETRY_HOME=/opt/poetry \
    POETRY_VIRTUALENVS_CREATE=false \
    POETRY_VERSION="1.2.2"

## install poetry
RUN python -c 'from urllib.request import urlopen; print(urlopen("https://install.python-poetry.org").read().decode())' | python -

## Add poetry to Path
ENV PATH="$POETRY_HOME/bin:$PATH"

# 2. Install analytics package
COPY pyproject.toml poetry.lock ./
RUN poetry install 

EXPOSE 8080

COPY . .

ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8080", "--server.address=0.0.0.0"]