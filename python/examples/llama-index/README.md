# llama-index OpenInference instrumentation example

This is a [LlamaIndex](https://www.llamaindex.ai/) project bootstrapped with [`create-llama`](https://github.com/run-llama/LlamaIndexTS/tree/main/packages/create-llama) and instrumented using OpenInference.

## Getting Started

First, startup the backend as described in the [backend README](./backend/README.md).

Second, run the development server of the frontend as described in the [frontend README](./frontend/README.md).

Open [http://localhost:3000](http://localhost:3000) with your browser to see the result.

## Running as a Docker Container

Build the docker image:

```shell
docker build -t llama-index-openinference-example .
```

Run the docker container:

```shell
docker run -p 3000:3000 -p 8000:8000 llama-index-openinference-example
```

Access the UI at [http://localhost:3000](http://localhost:3000).

## Learn More

To learn more about LlamaIndex, take a look at the following resources:

-   [LlamaIndex Documentation](https://docs.llamaindex.ai) - learn about LlamaIndex (Python features).
-   [LlamaIndexTS Documentation](https://ts.llamaindex.ai) - learn about LlamaIndex (Typescript features).

You can check out [the LlamaIndexTS GitHub repository](https://github.com/run-llama/LlamaIndexTS) - your feedback and contributions are welcome!
