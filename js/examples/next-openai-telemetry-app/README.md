# Phoenix, Vercel AI SDK, Next.js, and OpenAI Generative Example with Telemetry

This example shows how to use the [Vercel AI SDK](https://sdk.vercel.ai/docs) with [Next.js](https://nextjs.org/) and [OpenAI](https://openai.com) to create a sample generative application with [OpenTelemetry support](https://sdk.vercel.ai/docs/ai-sdk-core/telemetry).

OTEL spans are processed by the [@arizeai/openinference-vercel](https://github.com/Arize-ai/openinference/tree/main/js/packages/openinference-vercel) package and collected and viewable in [Arize-Phoenix](https://github.com/Arize-ai/phoenix).

## Deploy your own

Deploy the example using [Vercel](https://vercel.com?utm_source=github&utm_medium=readme&utm_campaign=ai-sdk-example):

[![Deploy with Vercel](https://vercel.com/button)](https://vercel.com/new/clone?repository-url=https%3A%2F%2Fgithub.com%2Fvercel%2Fai%2Ftree%2Fmain%2Fexamples%2Fnext-openai-telemetry&env=OPENAI_API_KEY&envDescription=OpenAI%20API%20Key&envLink=https%3A%2F%2Fplatform.openai.com%2Faccount%2Fapi-keys&project-name=vercel-ai-chat-openai-telemetry&repository-name=vercel-ai-chat-openai-telemetry)

## How to use

If you don't already have an OpenAI API key do the following to create one:

1. Sign up at [OpenAI's Developer Platform](https://platform.openai.com/signup).
2. Go to [OpenAI's dashboard](https://platform.openai.com/account/api-keys) and create an API KEY.

### Local

To run the example locally you need to:

1. Set the required OpenAI environment variable as the token value as shown [the example env file](./.env.local.example) but in a new file called `.env.local`.
2. To run [Arize-Phoenix](https://github.com/Arize-ai/phoenix) locally run `docker run -p 6006:6006 -i -t arizephoenix/phoenix`
3. `npm i` to install the required dependencies.
4. Use node version >=18.18.0 see [nvm](https://github.com/nvm-sh/nvm) for docs on how to install and manage node versions
4. `npm run dev` to launch the development server.

To view [Arize-Phoenix](https://github.com/Arize-ai/phoenix) and the example app visit:

- **App:** [localhost:3000](http://localhost:3000)

- **Phoenix:** [localhost:6006](http://localhost:6006) or [app.phoenix.arize.com](https://app.phoenix.arize.com)

## Learn More

To learn more about Phoenix, OpenAI, Next.js, and the Vercel AI SDK take a look at the following resources:

- [Phoenix repository](https://github.com/Arize-ai/phoenix) - learn about LLM observability with Phoenix
- [Phoenix docs](https://docs.arize.com/phoenix)
- [@arizeai/openinference-vercel](https://github.com/Arize-ai/openinference/tree/main/js/packages/openinference-vercel) - Check out the OpenInference Vercel support.
- [Vercel AI SDK docs](https://sdk.vercel.ai/docs)
- [Vercel AI SDK telemetry support](https://sdk.vercel.ai/docs/ai-sdk-core/telemetry)
- [Vercel AI Playground](https://play.vercel.ai)
- [OpenAI Documentation](https://platform.openai.com/docs) - learn about OpenAI features and API.
- [Next.js Documentation](https://nextjs.org/docs) - learn about Next.js features and API.
