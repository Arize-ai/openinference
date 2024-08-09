# Overview

This example shows how to use [@arizeai/openinference-instrumentation-openai](https://github.com/Arize-ai/openinference/tree/main/js/packages/openinference-instrumentation-openai) to instrument a simple Node.js application with OpenAI

Our example will export spans data simultaneously on `Console` and [arize-phoenix](https://github.com/Arize-ai/phoenix), however you can run your code anywhere and can use any exporter that OpenTelemetry supports.

If running and exporting to phoenix. You will also be able to add feedback to your spans via the chat interface.

## Getting Started With Local Development

Make sure you have `OPENAI_API_KEY` set as an environment variable.

First, startup the backend as described in the [backend README](./backend/README.md).

Second, run the development server of the frontend as described in the [frontend README](./frontend/README.md).

Open [http://localhost:3000](http://localhost:3000) with your browser to see the result.

## Getting Started With Docker-Compose

Ensure that Docker is installed and running. Run the command `docker compose up` to spin up services for the frontend, backend, and Phoenix. Once those services are running, open [http://localhost:3000](http://localhost:3000) to use the chat interface. When you're finished, run `docker compose down` to spin down the services.

Open up [http://localhost:6006](http://localhost:6006) to view spans and feedback in Phoenix.

## Learn More

To learn more about Arize Phoenix, take a look at the following resources:

You can check out [the Phoenix GitHub repository](https://github.com/Arize-ai/phoenix) - your feedback and contributions are welcome!
