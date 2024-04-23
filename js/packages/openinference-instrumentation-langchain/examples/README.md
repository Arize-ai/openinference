# Overview
This is a very simple example of how to setup LangChain.js auto instrumentation in a node application. 

## Instrumentation
Checkout the [instrumentation.ts](./instrumentation.ts) file to see how to auto-instrument LangChain.js and export spans to a locally running (Phoenix)[https://github.com/Arize-ai/phoenix] server.

## Chat

Checkout the [chat.ts](./chat.ts) file to see how to send a simple message to OpenAI with langchain.

## Instructions
Please use node version >= 18 to run this example.
1. (Optional) In order to see spans in [Phoenix](https://github.com/Arize-ai/phoenix), begin running a Phoenix server. This can be done in one command using docker.
```
docker run -p 6006:6006 -i -t arizephoenix/phoenix
```
2. Make sure you have and OpenAI API key set in your environment variables as `OPENAI_API_KEY`.
3. Go to the root of the JS directory of this repo and run the following commands to install dependencies and build the latest packages in the repo. This ensures any dependencies within the repo are at their latest versions.
```shell
pnpm -r install
```
```shell
pnpm -r prebuild
```  
```shell
pnpm -r build
```
4. Run the following command from the `openinference-instrumentation-langchain` package directory
```shell
node dist/examples/chat.js
```

You should see your spans exported in your console. If you've set up a locally running Phoenix server, head over to `localhost:6006` to see your spans.
