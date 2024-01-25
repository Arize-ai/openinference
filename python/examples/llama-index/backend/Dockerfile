# Use an official Python runtime as a parent image
FROM python:3.11-slim as builder

RUN apt-get update \
    && apt-get install -y --no-install-recommends wget \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install poetry
RUN pip install poetry
RUN pip install openinference-semantic-conventions
RUN pip install openinference-instrumentation-llama-index
RUN pip install opentelemetry-exporter-otlp 

# Copy the pyproject.toml (and possibly poetry.lock) file to the container
COPY pyproject.toml poetry.lock* /app/

# Install dependencies via poetry
RUN poetry config virtualenvs.create false \
 && poetry install --no-dev

 # Copy the rest of your application's code
COPY . /app/

EXPOSE 8000

# Run the main.py script when the container starts
CMD ["python", "main.py"]
