# Use an official Python runtime as a parent image
FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:0.9.30 /uv /uvx /bin/

# Install gcc
RUN apt-get update -y \
    && apt-get install -y python3-dev \
    gcc \
    libc-dev \
    libffi-dev

WORKDIR /rag

# Copy the project metadata to leverage Docker layer caching
COPY pyproject.toml /rag/
COPY uv.lock /rag/

RUN uv sync --locked

# Copy the rest of your application's code
COPY . /rag/

EXPOSE 8000

# Run the main.py script when the container starts
CMD ["uv", "run", "python", "main.py"]
