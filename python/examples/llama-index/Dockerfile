# Use an official Python runtime as a parent image
FROM python:3.11-slim as builder

# Install poetry
RUN pip install poetry

# Install nodejs
RUN apt-get update && apt-get upgrade -y &&\
    curl -fsSL https://deb.nodesource.com/setup_20.x | bash - &&\
    apt-get install -y nodejs npm

# Set the working directory in the container to /llama
WORKDIR /llama

# Add the current directory contents into the container at /phoenix
ADD . /llama

# Install the frontend by building the typescript package
RUN cd ./frontend && npm install && npm run build 

# Install any needed packages 
RUN cd ./backend && pip install . && python app/engine/generate.py

# Make port 3000 available to the world outside this container
EXPOSE 3000
EXPOSE 8000

# Start the backend and frontend
CMD ["./run.sh", "&", "sleep", "infinity"]