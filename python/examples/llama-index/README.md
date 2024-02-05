# llama-index OpenInference instrumentation example

This is a [LlamaIndex](https://www.llamaindex.ai/) project bootstrapped with [`create-llama`](https://github.com/run-llama/LlamaIndexTS/tree/main/packages/create-llama) and instrumented using OpenInference.

This example integrates three components:
- A NextJS frontend that provides an interface to a basic RAG chat application
- A Python FastAPI backend that serves a simple LlamaIndex RAG application. The LlamaIndex framework is instrumented using OpenInference to produce traces.
- A [Phoenix](https://github.com/Arize-ai/phoenix) server that acts as both a collector for OpenInference traces and UI for viewing traces.

## Getting Started with Local Development

First, startup the backend as described in the [backend README](./backend/README.md).
- If you'd like, include your own data to build an index in [the data directory](./backend/data/)
- Build a simple index using LlamaIndex
- Ensure that your OpenAI API key is available to the application, either via the `OPENAI_API_KEY` environment variable or a `.env` file
- Start the backend server

Second, run the development server of the frontend as described in the [frontend README](./frontend/README.md).

Open [http://localhost:3000](http://localhost:3000) with your browser to use the chat interface to your RAG application.

Traces can be viewed using the [Phoenix UI](http://localhost:6006).

## Getting Started with Docker-Compose

If you'd like, add your own PDFs to `./backend/data` to build indexes over.

Ensure that your OpenAI API key is available to the application, either via the `OPENAI_API_KEY` environment variable or a `.env` file.

Ensure that Docker is installed and running. Run the command `docker compose up` to spin up services for the frontend, backend, and Phoenix. Once those services are running, open [http://localhost:3000](http://localhost:3000) to use the chat interface. When you're finished, run `docker compose down` to spin down the services.

Traces can be viewed using the [Phoenix UI](http://localhost:6006).

## Learn More

This application is automatically instrumented using OpenInference entirely with one [instrumentation call](./backend/instrument.py). By running the instrumentation code LlamaIndex prior to starting the FastAPI server, LlamaIndex will automatically send traces to the configured Phoenix server that acts as a trace collector, providing deep observability into the underlying behavior of the application. This includes information about all of the retrieved context for a given query, and other relevant information such as reranking and synthesis steps that might occur prior to returning the final LLM response to the user.

To learn more about LlamaIndex, take a look at the following resources:

-   [LlamaIndex Documentation](https://docs.llamaindex.ai) - learn about LlamaIndex (Python features).
-   [LlamaIndexTS Documentation](https://ts.llamaindex.ai) - learn about LlamaIndex (Typescript features).

You can check out [the LlamaIndexTS GitHub repository](https://github.com/run-llama/LlamaIndexTS) - your feedback and contributions are welcome!
