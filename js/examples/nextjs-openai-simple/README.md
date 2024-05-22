# Nextjs + OpenInference Minimum Working Example

This is a [Next.js](https://nextjs.org/) project bootstrapped with [`create-next-app`](https://github.com/vercel/next.js/tree/canary/packages/create-next-app) that uses OpenAI to generate text and is traced using [Phoenix](https://github.com/Arize-ai/phoenix).

> [!NOTE]  
> This example explicitly uses Next.js 13. OpenTelemetry support for Next.js 14 is still under active development. If you are using Next.js 14 and need workarounds, please contact us at oss@arize.com

## Installation

Install the dependencies:

```shell
npm install
```

## Getting Started

Make sure you have `OPENAI_API_KEY` set as an environment variable.

First, start phoenix to trace the application.

```shell
docker run -p 6006:6006 arizephoenix/phoenix:latest
```

Start the Next.js application:

```shell
npm run dev
```

View the application at `http://localhost:3000` and the traces in the Phoenix UI at `http://localhost:6006`.

## Learn More

You can check out [the Phoenix GitHub repository](https://github.com/Arize-ai/phoenix) - your feedback and contributions are welcome!
