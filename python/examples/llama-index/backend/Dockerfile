# Use an official Python runtime as a parent image
FROM python:3.11-slim

# install wget for healthchecks
RUN apt-get update \
    && apt-get install -y --no-install-recommends wget \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /llama

# Install poetry
RUN pip install poetry

# Copy the pyproject.toml (and possibly poetry.lock) file to the container
COPY pyproject.toml poetry.lock* /llama/

ARG OPENAI_API_KEY
ENV OPENAI_API_KEY=$OPENAI_API_KEY

# Install dependencies via poetry
RUN poetry config virtualenvs.create false \
&& poetry install --only main

# Copy the rest of your application's code
COPY . /llama/

# Build indexes
RUN ["python", "-m", "app.engine.generate"]

EXPOSE 8000

# Run the main.py script when the container starts
CMD ["python", "main.py"]
