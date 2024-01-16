import { OpenAIInstrumentation } from "../src";
import {
  InMemorySpanExporter,
  SimpleSpanProcessor,
} from "@opentelemetry/sdk-trace-base";
import { NodeTracerProvider } from "@opentelemetry/sdk-trace-node";
const tracerProvider = new NodeTracerProvider();
tracerProvider.register();

const instrumentation = new OpenAIInstrumentation();
instrumentation.disable();

import * as OpenAI from "openai";
import { Stream } from "openai/streaming";

describe("OpenAIInstrumentation", () => {
  let openai: OpenAI.OpenAI;

  const memoryExporter = new InMemorySpanExporter();
  const provider = new NodeTracerProvider();
  provider.getTracer("default");

  instrumentation.setTracerProvider(tracerProvider);
  tracerProvider.addSpanProcessor(new SimpleSpanProcessor(memoryExporter));
  // @ts-expect-error the moduleExports property is private. This is needed to make the test work with auto-mocking
  instrumentation._modules[0].moduleExports = OpenAI;

  beforeAll(() => {
    instrumentation.enable();
    openai = new OpenAI.OpenAI({
      apiKey: "fake-api-key",
    });
  });
  afterAll(() => {
    instrumentation.disable();
  });
  beforeEach(() => {
    memoryExporter.reset();
  });
  afterEach(() => {
    jest.clearAllMocks();
  });
  it("is patched", () => {
    expect(
      (OpenAI as { openInferencePatched?: boolean }).openInferencePatched,
    ).toBe(true);
  });
  it("creates a span for chat completions", async () => {
    const response = {
      id: "chatcmpl-8adq9JloOzNZ9TyuzrKyLpGXexh6p",
      object: "chat.completion",
      created: 1703743645,
      model: "gpt-3.5-turbo-0613",
      choices: [
        {
          index: 0,
          message: {
            role: "assistant",
            content: "This is a test.",
          },
          logprobs: null,
          finish_reason: "stop",
        },
      ],
      usage: {
        prompt_tokens: 12,
        completion_tokens: 5,
        total_tokens: 17,
      },
    };
    // Mock out the chat completions endpoint
    jest.spyOn(openai, "post").mockImplementation(
      // @ts-expect-error the response type is not correct - this is just for testing
      async (): Promise<unknown> => {
        return response;
      },
    );
    await openai.chat.completions.create({
      messages: [{ role: "user", content: "Say this is a test" }],
      model: "gpt-3.5-turbo",
    });
    const spans = memoryExporter.getFinishedSpans();
    expect(spans.length).toBe(1);
    const span = spans[0];
    expect(span.name).toBe("OpenAI Chat Completions");
    expect(span.attributes).toMatchInlineSnapshot(`
      {
        "input.mime_type": "application/json",
        "input.value": "{"messages":[{"role":"user","content":"Say this is a test"}],"model":"gpt-3.5-turbo"}",
        "llm.input_messages.0.message.content": "Say this is a test",
        "llm.input_messages.0.message.role": "user",
        "llm.invocation_parameters": "{"model":"gpt-3.5-turbo"}",
        "llm.model_name": "gpt-3.5-turbo-0613",
        "llm.output_messages.0.message.content": "This is a test.",
        "llm.output_messages.0.message.role": "assistant",
        "llm.token_count.completion": 5,
        "llm.token_count.prompt": 12,
        "llm.token_count.total": 17,
        "openinference.span.kind": "llm",
        "output.mime_type": "application/json",
        "output.value": "{"id":"chatcmpl-8adq9JloOzNZ9TyuzrKyLpGXexh6p","object":"chat.completion","created":1703743645,"model":"gpt-3.5-turbo-0613","choices":[{"index":0,"message":{"role":"assistant","content":"This is a test."},"logprobs":null,"finish_reason":"stop"}],"usage":{"prompt_tokens":12,"completion_tokens":5,"total_tokens":17}}",
      }
    `);
  });
  it("creates a span for completions", async () => {
    const response = {
      id: "cmpl-8fZu1H3VijJUWev9asnxaYyQvJTC9",
      object: "text_completion",
      created: 1704920149,
      model: "gpt-3.5-turbo-instruct",
      choices: [
        {
          text: "This is a test",
          index: 0,
          logprobs: null,
          finish_reason: "stop",
        },
      ],
      usage: { prompt_tokens: 12, completion_tokens: 5, total_tokens: 17 },
    };
    // Mock out the completions endpoint
    jest.spyOn(openai, "post").mockImplementation(
      // @ts-expect-error the response type is not correct - this is just for testing
      async (): Promise<unknown> => {
        return response;
      },
    );
    await openai.completions.create({
      prompt: "Say this is a test",
      model: "gpt-3.5-turbo-instruct",
    });
    const spans = memoryExporter.getFinishedSpans();
    expect(spans.length).toBe(1);
    const span = spans[0];
    expect(span.name).toBe("OpenAI Completions");
    expect(span.attributes).toMatchInlineSnapshot(`
      {
        "input.mime_type": "text/plain",
        "input.value": "Say this is a test",
        "llm.invocation_parameters": "{"model":"gpt-3.5-turbo-instruct"}",
        "llm.model_name": "gpt-3.5-turbo-instruct",
        "llm.token_count.completion": 5,
        "llm.token_count.prompt": 12,
        "llm.token_count.total": 17,
        "openinference.span.kind": "llm",
        "output.mime_type": "text/plain",
        "output.value": "This is a test",
      }
    `);
  });
  it("creates a span for embedding create", async () => {
    const response = {
      object: "list",
      data: [{ object: "embedding", index: 0, embedding: [1, 2, 3] }],
    };
    // Mock out the embedding create endpoint
    jest.spyOn(openai, "post").mockImplementation(
      // @ts-expect-error the response type is not correct - this is just for testing
      async (): Promise<unknown> => {
        return response;
      },
    );
    await openai.embeddings.create({
      input: "A happy moment",
      model: "text-embedding-ada-002",
    });
    const spans = memoryExporter.getFinishedSpans();
    expect(spans.length).toBe(1);
    const span = spans[0];
    expect(span.name).toBe("OpenAI Embeddings");
    expect(span.attributes).toMatchInlineSnapshot(`
      {
        "embedding.embeddings.0.embedding.text": "A happy moment",
        "embedding.embeddings.0.embedding.vector": [
          1,
          2,
          3,
        ],
        "embedding.model_name": "text-embedding-ada-002",
        "input.mime_type": "text/plain",
        "input.value": "A happy moment",
        "openinference.span.kind": "embedding",
      }
    `);
  });
  it("can handle streaming responses", async () => {
    // Mock out the embedding create endpoint
    jest.spyOn(openai, "post").mockImplementation(
      // @ts-expect-error the response type is not correct - this is just for testing
      async (): Promise<unknown> => {
        const iterator = () =>
          (async function* () {
            yield { choices: [{ delta: { content: "This is " } }] };
            yield { choices: [{ delta: { content: "a test." } }] };
            yield { choices: [{ delta: { finish_reason: "stop" } }] };
          })();
        const controller = new AbortController();
        return new Stream(iterator, controller);
      },
    );
    const stream = await openai.chat.completions.create({
      messages: [{ role: "user", content: "Say this is a test" }],
      model: "gpt-3.5-turbo",
      stream: true,
    });

    let response = "";
    for await (const chunk of stream) {
      if (chunk.choices[0].delta.content)
        response += chunk.choices[0].delta.content;
    }
    expect(response).toBe("This is a test.");
    const spans = memoryExporter.getFinishedSpans();
    expect(spans.length).toBe(1);
    const span = spans[0];
    expect(span.name).toBe("OpenAI Chat Completions");
    expect(span.attributes).toMatchInlineSnapshot(`
      {
        "input.mime_type": "application/json",
        "input.value": "{"messages":[{"role":"user","content":"Say this is a test"}],"model":"gpt-3.5-turbo","stream":true}",
        "llm.input_messages.0.message.content": "Say this is a test",
        "llm.input_messages.0.message.role": "user",
        "llm.invocation_parameters": "{"model":"gpt-3.5-turbo","stream":true}",
        "llm.model_name": "gpt-3.5-turbo",
        "llm.output_messages.0.message.content": "This is a test.",
        "llm.output_messages.0.message.role": "assistant",
        "openinference.span.kind": "llm",
        "output.mime_type": "text/plain",
        "output.value": "This is a test.",
      }
    `);
  });
  it("should capture tool calls", async () => {
    // Mock out the embedding create endpoint
    const response1 = {
      id: "chatcmpl-8hhqZDFTRD0vzExhqWnMLE7viVl7E",
      object: "chat.completion",
      created: 1705427343,
      model: "gpt-3.5-turbo-0613",
      choices: [
        {
          index: 0,
          message: {
            role: "assistant",
            content: null,
            tool_calls: [
              {
                id: "call_5ERYvu4iTGSvDlcDQjDP3g3J",
                type: "function",
                function: { name: "getCurrentLocation", arguments: "{}" },
              },
            ],
          },
          logprobs: null,
          finish_reason: "tool_calls",
        },
      ],
      usage: { prompt_tokens: 70, completion_tokens: 7, total_tokens: 77 },
      system_fingerprint: null,
    };
    const response2 = {
      id: "chatcmpl-8hhsP9eAplUFYB3mHUJxBkq7IwnjZ",
      object: "chat.completion",
      created: 1705427457,
      model: "gpt-3.5-turbo-0613",
      choices: [
        {
          index: 0,
          message: {
            role: "assistant",
            content: null,
            tool_calls: [
              {
                id: "call_0LCdYLkdRUt3rV3dawoIFHBf",
                type: "function",
                function: {
                  name: "getWeather",
                  arguments: '{\n  "location": "Boston"\n}',
                },
              },
            ],
          },
          logprobs: null,
          finish_reason: "tool_calls",
        },
      ],
      usage: { prompt_tokens: 86, completion_tokens: 15, total_tokens: 101 },
      system_fingerprint: null,
    };
    const response3 = {
      id: "chatcmpl-8hhtfzSD33tsG7XJiBg4F9MqnXKDp",
      object: "chat.completion",
      created: 1705427535,
      model: "gpt-3.5-turbo-0613",
      choices: [
        {
          index: 0,
          message: {
            role: "assistant",
            content:
              "The weather in Boston this week is expected to be rainy with a temperature of 52 degrees.",
          },
          logprobs: null,
          finish_reason: "stop",
        },
      ],
      usage: { prompt_tokens: 121, completion_tokens: 20, total_tokens: 141 },
      system_fingerprint: null,
    };
    jest
      .spyOn(openai, "post")
      .mockImplementationOnce(
        // @ts-expect-error the response type is not correct - this is just for testing
        async (): Promise<unknown> => {
          return response1;
        },
      )
      .mockImplementationOnce(
        // @ts-expect-error the response type is not correct - this is just for testing
        async (): Promise<unknown> => {
          return response2;
        },
      )
      .mockImplementationOnce(
        // @ts-expect-error the response type is not correct - this is just for testing
        async (): Promise<unknown> => {
          return response3;
        },
      );
    // Function tools
    async function getCurrentLocation() {
      return "Boston"; // Simulate lookup
    }

    async function getWeather(_args: { location: string }) {
      return { temperature: 52, precipitation: "rainy" };
    }

    const messages = [];
    const runner = openai.beta.chat.completions
      .runTools({
        model: "gpt-3.5-turbo",
        messages: [{ role: "user", content: "How is the weather this week?" }],
        tools: [
          {
            type: "function",
            function: {
              function: getCurrentLocation,
              parameters: { type: "object", properties: {} },
              description: "Get the current location of the user.",
            },
          },
          {
            type: "function",
            function: {
              function: getWeather,
              parse: JSON.parse, // or use a validation library like zod for typesafe parsing.
              description: "Get the weather for a location.",
              parameters: {
                type: "object",
                properties: {
                  location: { type: "string" },
                },
              },
            },
          },
        ],
      })
      .on("message", (message) => messages.push(message));

    const _finalContent = await runner.finalContent();
    const spans = memoryExporter.getFinishedSpans();
    expect(spans.length).toBe(3);
    const [span1, span2, span3] = spans;
    expect(span1.name).toBe("OpenAI Chat Completions");
    expect(span1.attributes).toMatchInlineSnapshot(`
      {
        "input.mime_type": "application/json",
        "input.value": "{"model":"gpt-3.5-turbo","messages":[{"role":"user","content":"How is the weather this week?"}],"tools":[{"type":"function","function":{"name":"getCurrentLocation","parameters":{"type":"object","properties":{}},"description":"Get the current location of the user."}},{"type":"function","function":{"name":"getWeather","parameters":{"type":"object","properties":{"location":{"type":"string"}}},"description":"Get the weather for a location."}}],"tool_choice":"auto","stream":false}",
        "llm.input_messages.0.message.content": "How is the weather this week?",
        "llm.input_messages.0.message.role": "user",
        "llm.invocation_parameters": "{"model":"gpt-3.5-turbo","tools":[{"type":"function","function":{"name":"getCurrentLocation","parameters":{"type":"object","properties":{}},"description":"Get the current location of the user."}},{"type":"function","function":{"name":"getWeather","parameters":{"type":"object","properties":{"location":{"type":"string"}}},"description":"Get the weather for a location."}}],"tool_choice":"auto","stream":false}",
        "llm.model_name": "gpt-3.5-turbo-0613",
        "llm.output_messages.0.message.role": "assistant",
        "llm.output_messages.0.message.tool_calls.0.tool_call.function.arguments": "{}",
        "llm.output_messages.0.message.tool_calls.0.tool_call.function.name": "getCurrentLocation",
        "llm.token_count.completion": 7,
        "llm.token_count.prompt": 70,
        "llm.token_count.total": 77,
        "openinference.span.kind": "llm",
        "output.mime_type": "application/json",
        "output.value": "{"id":"chatcmpl-8hhqZDFTRD0vzExhqWnMLE7viVl7E","object":"chat.completion","created":1705427343,"model":"gpt-3.5-turbo-0613","choices":[{"index":0,"message":{"role":"assistant","content":null,"tool_calls":[{"id":"call_5ERYvu4iTGSvDlcDQjDP3g3J","type":"function","function":{"name":"getCurrentLocation","arguments":"{}"}}]},"logprobs":null,"finish_reason":"tool_calls"}],"usage":{"prompt_tokens":70,"completion_tokens":7,"total_tokens":77},"system_fingerprint":null}",
      }
    `);
    expect(span2.name).toBe("OpenAI Chat Completions");
    expect(span2.attributes).toMatchInlineSnapshot(`
      {
        "input.mime_type": "application/json",
        "input.value": "{"model":"gpt-3.5-turbo","messages":[{"role":"user","content":"How is the weather this week?"},{"role":"assistant","content":null,"tool_calls":[{"id":"call_5ERYvu4iTGSvDlcDQjDP3g3J","type":"function","function":{"name":"getCurrentLocation","arguments":"{}"}}]},{"role":"tool","tool_call_id":"call_5ERYvu4iTGSvDlcDQjDP3g3J","content":"Boston"}],"tools":[{"type":"function","function":{"name":"getCurrentLocation","parameters":{"type":"object","properties":{}},"description":"Get the current location of the user."}},{"type":"function","function":{"name":"getWeather","parameters":{"type":"object","properties":{"location":{"type":"string"}}},"description":"Get the weather for a location."}}],"tool_choice":"auto","stream":false}",
        "llm.input_messages.0.message.content": "How is the weather this week?",
        "llm.input_messages.0.message.role": "user",
        "llm.input_messages.1.message.role": "assistant",
        "llm.input_messages.1.message.tool_calls.0.tool_call.function.arguments": "{}",
        "llm.input_messages.1.message.tool_calls.0.tool_call.function.name": "getCurrentLocation",
        "llm.input_messages.2.message.content": "Boston",
        "llm.input_messages.2.message.role": "tool",
        "llm.invocation_parameters": "{"model":"gpt-3.5-turbo","tools":[{"type":"function","function":{"name":"getCurrentLocation","parameters":{"type":"object","properties":{}},"description":"Get the current location of the user."}},{"type":"function","function":{"name":"getWeather","parameters":{"type":"object","properties":{"location":{"type":"string"}}},"description":"Get the weather for a location."}}],"tool_choice":"auto","stream":false}",
        "llm.model_name": "gpt-3.5-turbo-0613",
        "llm.output_messages.0.message.role": "assistant",
        "llm.output_messages.0.message.tool_calls.0.tool_call.function.arguments": "{
        "location": "Boston"
      }",
        "llm.output_messages.0.message.tool_calls.0.tool_call.function.name": "getWeather",
        "llm.token_count.completion": 15,
        "llm.token_count.prompt": 86,
        "llm.token_count.total": 101,
        "openinference.span.kind": "llm",
        "output.mime_type": "application/json",
        "output.value": "{"id":"chatcmpl-8hhsP9eAplUFYB3mHUJxBkq7IwnjZ","object":"chat.completion","created":1705427457,"model":"gpt-3.5-turbo-0613","choices":[{"index":0,"message":{"role":"assistant","content":null,"tool_calls":[{"id":"call_0LCdYLkdRUt3rV3dawoIFHBf","type":"function","function":{"name":"getWeather","arguments":"{\\n  \\"location\\": \\"Boston\\"\\n}"}}]},"logprobs":null,"finish_reason":"tool_calls"}],"usage":{"prompt_tokens":86,"completion_tokens":15,"total_tokens":101},"system_fingerprint":null}",
      }
    `);
    expect(span3.name).toBe("OpenAI Chat Completions");
    expect(span3.attributes).toMatchInlineSnapshot(`
{
  "input.mime_type": "application/json",
  "input.value": "{"model":"gpt-3.5-turbo","messages":[{"role":"user","content":"How is the weather this week?"},{"role":"assistant","content":null,"tool_calls":[{"id":"call_5ERYvu4iTGSvDlcDQjDP3g3J","type":"function","function":{"name":"getCurrentLocation","arguments":"{}"}}]},{"role":"tool","tool_call_id":"call_5ERYvu4iTGSvDlcDQjDP3g3J","content":"Boston"},{"role":"assistant","content":null,"tool_calls":[{"id":"call_0LCdYLkdRUt3rV3dawoIFHBf","type":"function","function":{"name":"getWeather","arguments":"{\\n  \\"location\\": \\"Boston\\"\\n}"}}]},{"role":"tool","tool_call_id":"call_0LCdYLkdRUt3rV3dawoIFHBf","content":"{\\"temperature\\":52,\\"precipitation\\":\\"rainy\\"}"}],"tools":[{"type":"function","function":{"name":"getCurrentLocation","parameters":{"type":"object","properties":{}},"description":"Get the current location of the user."}},{"type":"function","function":{"name":"getWeather","parameters":{"type":"object","properties":{"location":{"type":"string"}}},"description":"Get the weather for a location."}}],"tool_choice":"auto","stream":false}",
  "llm.input_messages.0.message.content": "How is the weather this week?",
  "llm.input_messages.0.message.role": "user",
  "llm.input_messages.1.message.role": "assistant",
  "llm.input_messages.1.message.tool_calls.0.tool_call.function.arguments": "{}",
  "llm.input_messages.1.message.tool_calls.0.tool_call.function.name": "getCurrentLocation",
  "llm.input_messages.2.message.content": "Boston",
  "llm.input_messages.2.message.role": "tool",
  "llm.input_messages.3.message.role": "assistant",
  "llm.input_messages.3.message.tool_calls.0.tool_call.function.arguments": "{
  "location": "Boston"
}",
  "llm.input_messages.3.message.tool_calls.0.tool_call.function.name": "getWeather",
  "llm.input_messages.4.message.content": "{"temperature":52,"precipitation":"rainy"}",
  "llm.input_messages.4.message.role": "tool",
  "llm.invocation_parameters": "{"model":"gpt-3.5-turbo","tools":[{"type":"function","function":{"name":"getCurrentLocation","parameters":{"type":"object","properties":{}},"description":"Get the current location of the user."}},{"type":"function","function":{"name":"getWeather","parameters":{"type":"object","properties":{"location":{"type":"string"}}},"description":"Get the weather for a location."}}],"tool_choice":"auto","stream":false}",
  "llm.model_name": "gpt-3.5-turbo-0613",
  "llm.output_messages.0.message.content": "The weather in Boston this week is expected to be rainy with a temperature of 52 degrees.",
  "llm.output_messages.0.message.role": "assistant",
  "llm.token_count.completion": 20,
  "llm.token_count.prompt": 121,
  "llm.token_count.total": 141,
  "openinference.span.kind": "llm",
  "output.mime_type": "application/json",
  "output.value": "{"id":"chatcmpl-8hhtfzSD33tsG7XJiBg4F9MqnXKDp","object":"chat.completion","created":1705427535,"model":"gpt-3.5-turbo-0613","choices":[{"index":0,"message":{"role":"assistant","content":"The weather in Boston this week is expected to be rainy with a temperature of 52 degrees."},"logprobs":null,"finish_reason":"stop"}],"usage":{"prompt_tokens":121,"completion_tokens":20,"total_tokens":141},"system_fingerprint":null}",
}
`);
  });
});
