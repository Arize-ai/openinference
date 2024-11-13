# Overview

This example shows how to use langchain with express to create a chat bot. The chat bot performance Retrieval Augmented Generation (RAG) to answer questions about using Phoenix and OpenInference with TypeScript / JavaScript.

Our example will export spans data simultaneously on `Console` and [arize-phoenix](https://github.com/Arize-ai/phoenix), however you can run your code anywhere and can use any exporter that OpenTelemetry supports.

## Getting Started With Local Development

Make sure you have `OPENAI_API_KEY` set as an environment variable.

First, startup the backend as described in the [backend README](./backend/README.md).

Second, run the development server of the frontend as described in the [frontend README](./frontend/README.md).

Open [http://localhost:3000](http://localhost:3000) with your browser to see the result.

## Getting Started With Docker-Compose

Copy the `.env.example` file to `.env` and set your `OPENAI_API_KEY`.

Ensure that Docker is installed and running. Run the command `docker compose up` to spin up services for the frontend, backend, and Phoenix. Once those services are running, open [http://localhost:3000](http://localhost:3000) to use the chat interface and [http://localhost:6006](http://localhost:6006) to view the Phoenix UI. When you're finished, run `docker compose down` to spin down the services.

## Learn More

To learn more about Arize Phoenix, take a look at the following resources:

You can check out [the Phoenix GitHub repository](https://github.com/Arize-ai/phoenix) - your feedback and contributions are welcome!
