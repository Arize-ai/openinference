# Use an official Python runtime as a parent image
FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:0.9.30 /uv /uvx /bin/

# Install gcc
RUN apt-get update -y \
    && apt-get install -y python3-dev \
    gcc \
    libc-dev \
    libffi-dev

WORKDIR /front

# Copy the project metadata to leverage Docker layer caching
COPY pyproject.toml /front/
COPY uv.lock /front/

RUN uv sync --locked

# Copy the rest of your application's code
COPY . /front/

EXPOSE 8501

CMD ["uv", "run", "streamlit", "run", "about.py", "--server.port=8501", "--server.address=0.0.0.0"]
