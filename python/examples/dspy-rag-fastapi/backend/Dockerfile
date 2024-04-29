# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Install gcc
RUN apt-get update -y \
    && apt-get install -y python3-dev \
    gcc \
    libc-dev \
    libffi-dev

WORKDIR /rag

# Install poetry
RUN pip install poetry

# Copy the pyproject.toml (and possibly poetry.lock) file to the container
COPY pyproject.toml /rag/

# Install dependencies via poetry
RUN poetry config virtualenvs.create false \
    && poetry install

# Copy the rest of your application's code
COPY . /rag/

EXPOSE 8000

# Run the main.py script when the container starts
CMD ["python", "main.py"]