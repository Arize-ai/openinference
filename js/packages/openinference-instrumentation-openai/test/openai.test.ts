import { isPatched, OpenAIInstrumentation } from "../src";
import {
  InMemorySpanExporter,
  SimpleSpanProcessor,
} from "@opentelemetry/sdk-trace-base";
import { NodeTracerProvider } from "@opentelemetry/sdk-trace-node";
import { suppressTracing } from "@opentelemetry/core";
import { context } from "@opentelemetry/api";

import OpenAI, { APIPromise, AzureOpenAI } from "openai";
import { Stream } from "openai/streaming";
import { setPromptTemplate, setSession } from "@arizeai/openinference-core";
import { CreateEmbeddingResponse } from "openai/resources/embeddings";
import { z } from "zod";
import { zodResponseFormat } from "openai/helpers/zod";
import { registerInstrumentations } from "@opentelemetry/instrumentation";
import { vi } from "vitest";

// Function tools
async function getCurrentLocation() {
  return "Boston"; // Simulate lookup
}

async function getWeather(_args: { location: string }) {
  return { temperature: 52, precipitation: "rainy" };
}

process.env.OPENAI_API_KEY = "fake-api-key";

const memoryExporter = new InMemorySpanExporter();

describe("OpenAIInstrumentation", () => {
  const tracerProvider = new NodeTracerProvider();
  tracerProvider.register();
  const instrumentation = new OpenAIInstrumentation();
  instrumentation.disable();
  let openai: OpenAI;

  instrumentation.setTracerProvider(tracerProvider);
  tracerProvider.addSpanProcessor(new SimpleSpanProcessor(memoryExporter));
  // @ts-expect-error the moduleExports property is private. This is needed to make the test work with auto-mocking
  instrumentation._modules[0].moduleExports = OpenAI;

  beforeAll(() => {
    instrumentation.enable();
    openai = new OpenAI({
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
    vi.clearAllMocks();
  });
  it("is patched", () => {
    expect(
      (OpenAI as { openInferencePatched?: boolean }).openInferencePatched,
    ).toBe(true);
    expect(isPatched()).toBe(true);
  });
  it("sets a patched flag correctly to track whether or not openai is instrumented", () => {
    instrumentation.disable();
    expect(isPatched()).toBe(false);
    instrumentation.enable();
    expect(isPatched()).toBe(true);
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
    vi.spyOn(openai, "post").mockImplementation(
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
  "llm.provider": "openai",
  "llm.system": "openai",
  "llm.token_count.completion": 5,
  "llm.token_count.prompt": 12,
  "llm.token_count.total": 17,
  "openinference.span.kind": "LLM",
  "output.mime_type": "application/json",
  "output.value": "{"id":"chatcmpl-8adq9JloOzNZ9TyuzrKyLpGXexh6p","object":"chat.completion","created":1703743645,"model":"gpt-3.5-turbo-0613","choices":[{"index":0,"message":{"role":"assistant","content":"This is a test."},"logprobs":null,"finish_reason":"stop"}],"usage":{"prompt_tokens":12,"completion_tokens":5,"total_tokens":17}}",
}
`);
  });
  it("captures the token count details for caching", async () => {
    const response = {
      id: "chatcmpl-8adq9JloOzNZ9TyuzrKyLpGXexh6p",
      object: "chat.completion",
      created: 1703743645,
      model: "gpt-4o-mini",
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
        prompt_tokens_details: {
          cached_tokens: 1,
        },
      },
    };
    // Mock out the chat completions endpoint
    vi.spyOn(openai, "post").mockImplementation(
      // @ts-expect-error the response type is not correct - this is just for testing
      async (): Promise<unknown> => {
        return response;
      },
    );
    await openai.chat.completions.create({
      messages: [{ role: "user", content: "Say this is a test" }],
      model: "gpt-4o-mini",
    });
    const spans = memoryExporter.getFinishedSpans();
    expect(spans.length).toBe(1);
    const span = spans[0];
    expect(span.name).toBe("OpenAI Chat Completions");
    expect(span.attributes).toMatchInlineSnapshot(`
    {
      "input.mime_type": "application/json",
      "input.value": "{"messages":[{"role":"user","content":"Say this is a test"}],"model":"gpt-4o-mini"}",
      "llm.input_messages.0.message.content": "Say this is a test",
      "llm.input_messages.0.message.role": "user",
      "llm.invocation_parameters": "{"model":"gpt-4o-mini"}",
      "llm.model_name": "gpt-4o-mini",
      "llm.output_messages.0.message.content": "This is a test.",
      "llm.output_messages.0.message.role": "assistant",
      "llm.provider": "openai",
      "llm.system": "openai",
      "llm.token_count.completion": 5,
      "llm.token_count.prompt": 12,
      "llm.token_count.prompt_details.cache_read": 1,
      "llm.token_count.total": 17,
      "openinference.span.kind": "LLM",
      "output.mime_type": "application/json",
      "output.value": "{"id":"chatcmpl-8adq9JloOzNZ9TyuzrKyLpGXexh6p","object":"chat.completion","created":1703743645,"model":"gpt-4o-mini","choices":[{"index":0,"message":{"role":"assistant","content":"This is a test."},"logprobs":null,"finish_reason":"stop"}],"usage":{"prompt_tokens":12,"completion_tokens":5,"total_tokens":17,"prompt_tokens_details":{"cached_tokens":1}}}",
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
    vi.spyOn(openai, "post").mockImplementation(
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
  "llm.provider": "openai",
  "llm.system": "openai",
  "llm.token_count.completion": 5,
  "llm.token_count.prompt": 12,
  "llm.token_count.total": 17,
  "openinference.span.kind": "LLM",
  "output.mime_type": "text/plain",
  "output.value": "This is a test",
}
`);
  });
  it("creates a span for embedding create", async () => {
    const response: CreateEmbeddingResponse = {
      object: "list",
      data: [{ object: "embedding", index: 0, embedding: [1, 2, 3] }],
      model: "text-embedding-ada-003-small",
      usage: { prompt_tokens: 0, total_tokens: 0 },
    };

    // Mock out the embedding create endpoint
    vi.spyOn(openai, "post").mockImplementation(() => {
      return new APIPromise(
        openai,
        Promise.resolve({
          requestLogID: "123",
          retryOfRequestLogID: "123",
          startTime: 123,
          response: {
            json: () => Promise.resolve(response),
            text: () => Promise.resolve(JSON.stringify(response)),
            clone: () => ({
              json: () => Promise.resolve(response),
              text: () => Promise.resolve(JSON.stringify(response)),
            }),
            headers: new Headers({ "content-type": "application/json" }),
            status: 200,
            statusText: "OK",
            ok: true,
          } as Response,
          options: {
            method: "post",
            path: "/embeddings",
          },
          controller: new AbortController(),
        }),
      );
    });

    await openai.embeddings.create({
      input: "A happy moment",
      model: "text-embedding-ada-003-small",
    });

    const spans = memoryExporter.getFinishedSpans();
    expect(spans.length).toBe(1);
    const span = spans[0];
    expect(span.name).toBe("OpenAI Embeddings");
    // Check the attributes
    expect(span.attributes["embedding.embeddings.0.embedding.text"]).toBe(
      "A happy moment",
    );
    expect(span.attributes["embedding.model_name"]).toBe(
      "text-embedding-ada-003-small",
    );
    expect(span.attributes["input.mime_type"]).toBe("text/plain");
    expect(span.attributes["input.value"]).toBe("A happy moment");
    expect(span.attributes["openinference.span.kind"]).toBe("EMBEDDING");
  });
  it("can handle streaming responses", async () => {
    // Mock out the post endpoint to return a stream
    vi.spyOn(openai, "post").mockImplementation(
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
  "llm.provider": "openai",
  "llm.system": "openai",
  "openinference.span.kind": "LLM",
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
    vi.spyOn(openai, "post")
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

    const messages = [];
    const runner = openai.chat.completions
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
  "llm.output_messages.0.message.tool_calls.0.tool_call.id": "call_5ERYvu4iTGSvDlcDQjDP3g3J",
  "llm.provider": "openai",
  "llm.system": "openai",
  "llm.token_count.completion": 7,
  "llm.token_count.prompt": 70,
  "llm.token_count.total": 77,
  "llm.tools.0.tool.json_schema": "{"type":"function","function":{"name":"getCurrentLocation","parameters":{"type":"object","properties":{}},"description":"Get the current location of the user."}}",
  "llm.tools.1.tool.json_schema": "{"type":"function","function":{"name":"getWeather","parameters":{"type":"object","properties":{"location":{"type":"string"}}},"description":"Get the weather for a location."}}",
  "openinference.span.kind": "LLM",
  "output.mime_type": "application/json",
  "output.value": "{"id":"chatcmpl-8hhqZDFTRD0vzExhqWnMLE7viVl7E","object":"chat.completion","created":1705427343,"model":"gpt-3.5-turbo-0613","choices":[{"index":0,"message":{"role":"assistant","content":null,"tool_calls":[{"id":"call_5ERYvu4iTGSvDlcDQjDP3g3J","type":"function","function":{"name":"getCurrentLocation","arguments":"{}"}}]},"logprobs":null,"finish_reason":"tool_calls"}],"usage":{"prompt_tokens":70,"completion_tokens":7,"total_tokens":77},"system_fingerprint":null}",
}
`);
    expect(span2.name).toBe("OpenAI Chat Completions");
    expect(span2.attributes).toMatchInlineSnapshot(`
{
  "input.mime_type": "application/json",
  "input.value": "{"model":"gpt-3.5-turbo","messages":[{"role":"user","content":"How is the weather this week?"},{"role":"assistant","content":null,"tool_calls":[{"id":"call_5ERYvu4iTGSvDlcDQjDP3g3J","type":"function","function":{"name":"getCurrentLocation","arguments":"{}","parsed_arguments":null}}],"parsed":null},{"role":"tool","tool_call_id":"call_5ERYvu4iTGSvDlcDQjDP3g3J","content":"Boston"}],"tools":[{"type":"function","function":{"name":"getCurrentLocation","parameters":{"type":"object","properties":{}},"description":"Get the current location of the user."}},{"type":"function","function":{"name":"getWeather","parameters":{"type":"object","properties":{"location":{"type":"string"}}},"description":"Get the weather for a location."}}],"tool_choice":"auto","stream":false}",
  "llm.input_messages.0.message.content": "How is the weather this week?",
  "llm.input_messages.0.message.role": "user",
  "llm.input_messages.1.message.role": "assistant",
  "llm.input_messages.1.message.tool_calls.0.tool_call.function.arguments": "{}",
  "llm.input_messages.1.message.tool_calls.0.tool_call.function.name": "getCurrentLocation",
  "llm.input_messages.1.message.tool_calls.0.tool_call.id": "call_5ERYvu4iTGSvDlcDQjDP3g3J",
  "llm.input_messages.2.message.content": "Boston",
  "llm.input_messages.2.message.role": "tool",
  "llm.input_messages.2.message.tool_call_id": "call_5ERYvu4iTGSvDlcDQjDP3g3J",
  "llm.invocation_parameters": "{"model":"gpt-3.5-turbo","tools":[{"type":"function","function":{"name":"getCurrentLocation","parameters":{"type":"object","properties":{}},"description":"Get the current location of the user."}},{"type":"function","function":{"name":"getWeather","parameters":{"type":"object","properties":{"location":{"type":"string"}}},"description":"Get the weather for a location."}}],"tool_choice":"auto","stream":false}",
  "llm.model_name": "gpt-3.5-turbo-0613",
  "llm.output_messages.0.message.role": "assistant",
  "llm.output_messages.0.message.tool_calls.0.tool_call.function.arguments": "{
  "location": "Boston"
}",
  "llm.output_messages.0.message.tool_calls.0.tool_call.function.name": "getWeather",
  "llm.output_messages.0.message.tool_calls.0.tool_call.id": "call_0LCdYLkdRUt3rV3dawoIFHBf",
  "llm.provider": "openai",
  "llm.system": "openai",
  "llm.token_count.completion": 15,
  "llm.token_count.prompt": 86,
  "llm.token_count.total": 101,
  "llm.tools.0.tool.json_schema": "{"type":"function","function":{"name":"getCurrentLocation","parameters":{"type":"object","properties":{}},"description":"Get the current location of the user."}}",
  "llm.tools.1.tool.json_schema": "{"type":"function","function":{"name":"getWeather","parameters":{"type":"object","properties":{"location":{"type":"string"}}},"description":"Get the weather for a location."}}",
  "openinference.span.kind": "LLM",
  "output.mime_type": "application/json",
  "output.value": "{"id":"chatcmpl-8hhsP9eAplUFYB3mHUJxBkq7IwnjZ","object":"chat.completion","created":1705427457,"model":"gpt-3.5-turbo-0613","choices":[{"index":0,"message":{"role":"assistant","content":null,"tool_calls":[{"id":"call_0LCdYLkdRUt3rV3dawoIFHBf","type":"function","function":{"name":"getWeather","arguments":"{\\n  \\"location\\": \\"Boston\\"\\n}"}}]},"logprobs":null,"finish_reason":"tool_calls"}],"usage":{"prompt_tokens":86,"completion_tokens":15,"total_tokens":101},"system_fingerprint":null}",
}
`);
    expect(span3.name).toBe("OpenAI Chat Completions");
    expect(span3.attributes).toMatchInlineSnapshot(`
{
  "input.mime_type": "application/json",
  "input.value": "{"model":"gpt-3.5-turbo","messages":[{"role":"user","content":"How is the weather this week?"},{"role":"assistant","content":null,"tool_calls":[{"id":"call_5ERYvu4iTGSvDlcDQjDP3g3J","type":"function","function":{"name":"getCurrentLocation","arguments":"{}","parsed_arguments":null}}],"parsed":null},{"role":"tool","tool_call_id":"call_5ERYvu4iTGSvDlcDQjDP3g3J","content":"Boston"},{"role":"assistant","content":null,"tool_calls":[{"id":"call_0LCdYLkdRUt3rV3dawoIFHBf","type":"function","function":{"name":"getWeather","arguments":"{\\n  \\"location\\": \\"Boston\\"\\n}","parsed_arguments":null}}],"parsed":null},{"role":"tool","tool_call_id":"call_0LCdYLkdRUt3rV3dawoIFHBf","content":"{\\"temperature\\":52,\\"precipitation\\":\\"rainy\\"}"}],"tools":[{"type":"function","function":{"name":"getCurrentLocation","parameters":{"type":"object","properties":{}},"description":"Get the current location of the user."}},{"type":"function","function":{"name":"getWeather","parameters":{"type":"object","properties":{"location":{"type":"string"}}},"description":"Get the weather for a location."}}],"tool_choice":"auto","stream":false}",
  "llm.input_messages.0.message.content": "How is the weather this week?",
  "llm.input_messages.0.message.role": "user",
  "llm.input_messages.1.message.role": "assistant",
  "llm.input_messages.1.message.tool_calls.0.tool_call.function.arguments": "{}",
  "llm.input_messages.1.message.tool_calls.0.tool_call.function.name": "getCurrentLocation",
  "llm.input_messages.1.message.tool_calls.0.tool_call.id": "call_5ERYvu4iTGSvDlcDQjDP3g3J",
  "llm.input_messages.2.message.content": "Boston",
  "llm.input_messages.2.message.role": "tool",
  "llm.input_messages.2.message.tool_call_id": "call_5ERYvu4iTGSvDlcDQjDP3g3J",
  "llm.input_messages.3.message.role": "assistant",
  "llm.input_messages.3.message.tool_calls.0.tool_call.function.arguments": "{
  "location": "Boston"
}",
  "llm.input_messages.3.message.tool_calls.0.tool_call.function.name": "getWeather",
  "llm.input_messages.3.message.tool_calls.0.tool_call.id": "call_0LCdYLkdRUt3rV3dawoIFHBf",
  "llm.input_messages.4.message.content": "{"temperature":52,"precipitation":"rainy"}",
  "llm.input_messages.4.message.role": "tool",
  "llm.input_messages.4.message.tool_call_id": "call_0LCdYLkdRUt3rV3dawoIFHBf",
  "llm.invocation_parameters": "{"model":"gpt-3.5-turbo","tools":[{"type":"function","function":{"name":"getCurrentLocation","parameters":{"type":"object","properties":{}},"description":"Get the current location of the user."}},{"type":"function","function":{"name":"getWeather","parameters":{"type":"object","properties":{"location":{"type":"string"}}},"description":"Get the weather for a location."}}],"tool_choice":"auto","stream":false}",
  "llm.model_name": "gpt-3.5-turbo-0613",
  "llm.output_messages.0.message.content": "The weather in Boston this week is expected to be rainy with a temperature of 52 degrees.",
  "llm.output_messages.0.message.role": "assistant",
  "llm.provider": "openai",
  "llm.system": "openai",
  "llm.token_count.completion": 20,
  "llm.token_count.prompt": 121,
  "llm.token_count.total": 141,
  "llm.tools.0.tool.json_schema": "{"type":"function","function":{"name":"getCurrentLocation","parameters":{"type":"object","properties":{}},"description":"Get the current location of the user."}}",
  "llm.tools.1.tool.json_schema": "{"type":"function","function":{"name":"getWeather","parameters":{"type":"object","properties":{"location":{"type":"string"}}},"description":"Get the weather for a location."}}",
  "openinference.span.kind": "LLM",
  "output.mime_type": "application/json",
  "output.value": "{"id":"chatcmpl-8hhtfzSD33tsG7XJiBg4F9MqnXKDp","object":"chat.completion","created":1705427535,"model":"gpt-3.5-turbo-0613","choices":[{"index":0,"message":{"role":"assistant","content":"The weather in Boston this week is expected to be rainy with a temperature of 52 degrees."},"logprobs":null,"finish_reason":"stop"}],"usage":{"prompt_tokens":121,"completion_tokens":20,"total_tokens":141},"system_fingerprint":null}",
}
`);
  });
  it("should capture tool calls with streaming", async () => {
    vi.spyOn(openai, "post").mockImplementation(
      // @ts-expect-error the response type is not correct - this is just for testing
      async (): Promise<unknown> => {
        const iterator = () =>
          (async function* () {
            yield {
              id: "chatcmpl-8iA39kCtuVHIVDr9AnBdJZjgSjNWL",
              object: "chat.completion.chunk",
              created: 1705535755,
              model: "gpt-3.5-turbo-0613",
              system_fingerprint: null,
              choices: [
                {
                  index: 0,
                  delta: {
                    role: "assistant",
                    content: null,
                    tool_calls: [
                      {
                        index: 0,
                        id: "call_PGkcUg2u6vYrCpTn0e9ofykY",
                        type: "function",
                        function: { name: "getWeather", arguments: "" },
                      },
                    ],
                  },
                  logprobs: null,
                  finish_reason: null,
                },
              ],
            };
            yield {
              id: "chatcmpl-8iA39kCtuVHIVDr9AnBdJZjgSjNWL",
              object: "chat.completion.chunk",
              created: 1705535755,
              model: "gpt-3.5-turbo-0613",
              system_fingerprint: null,
              choices: [
                {
                  index: 0,
                  delta: {
                    tool_calls: [{ index: 0, function: { arguments: "{}" } }],
                  },
                  logprobs: null,
                  finish_reason: null,
                },
              ],
            };
            yield {
              id: "chatcmpl-8iA39kCtuVHIVDr9AnBdJZjgSjNWL",
              object: "chat.completion.chunk",
              created: 1705535755,
              model: "gpt-3.5-turbo-0613",
              system_fingerprint: null,
              choices: [
                {
                  index: 0,
                  delta: {},
                  logprobs: null,
                  finish_reason: "tool_calls",
                },
              ],
            };
          })();
        const controller = new AbortController();
        return new Stream(iterator, controller);
      },
    );
    const stream = await openai.chat.completions.create({
      messages: [{ role: "user", content: "What's the weather today?" }],
      model: "gpt-3.5-turbo",
      tools: [
        {
          type: "function",
          function: {
            name: "getCurrentLocation",
            parameters: { type: "object", properties: {} },
            description: "Get the current location of the user.",
          },
        },
        {
          type: "function",
          function: {
            name: "getWeather",
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
      stream: true,
    });

    let response = "";
    for await (const chunk of stream) {
      if (chunk.choices[0].delta.content)
        response += chunk.choices[0].delta.content;
    }
    // When a tool is called, the content is empty
    expect(response).toBe("");
    const spans = memoryExporter.getFinishedSpans();
    expect(spans.length).toBe(1);
    const span = spans[0];
    expect(span.name).toBe("OpenAI Chat Completions");
    expect(span.attributes).toMatchInlineSnapshot(`
{
  "input.mime_type": "application/json",
  "input.value": "{"messages":[{"role":"user","content":"What's the weather today?"}],"model":"gpt-3.5-turbo","tools":[{"type":"function","function":{"name":"getCurrentLocation","parameters":{"type":"object","properties":{}},"description":"Get the current location of the user."}},{"type":"function","function":{"name":"getWeather","description":"Get the weather for a location.","parameters":{"type":"object","properties":{"location":{"type":"string"}}}}}],"stream":true}",
  "llm.input_messages.0.message.content": "What's the weather today?",
  "llm.input_messages.0.message.role": "user",
  "llm.invocation_parameters": "{"model":"gpt-3.5-turbo","tools":[{"type":"function","function":{"name":"getCurrentLocation","parameters":{"type":"object","properties":{}},"description":"Get the current location of the user."}},{"type":"function","function":{"name":"getWeather","description":"Get the weather for a location.","parameters":{"type":"object","properties":{"location":{"type":"string"}}}}}],"stream":true}",
  "llm.model_name": "gpt-3.5-turbo",
  "llm.output_messages.0.message.content": "",
  "llm.output_messages.0.message.role": "assistant",
  "llm.output_messages.0.message.tool_calls.0.tool_call.function.arguments": "{}",
  "llm.output_messages.0.message.tool_calls.0.tool_call.function.name": "getWeather",
  "llm.output_messages.0.message.tool_calls.0.tool_call.id": "call_PGkcUg2u6vYrCpTn0e9ofykY",
  "llm.provider": "openai",
  "llm.system": "openai",
  "llm.tools.0.tool.json_schema": "{"type":"function","function":{"name":"getCurrentLocation","parameters":{"type":"object","properties":{}},"description":"Get the current location of the user."}}",
  "llm.tools.1.tool.json_schema": "{"type":"function","function":{"name":"getWeather","description":"Get the weather for a location.","parameters":{"type":"object","properties":{"location":{"type":"string"}}}}}",
  "openinference.span.kind": "LLM",
  "output.mime_type": "text/plain",
  "output.value": "",
}
`);
  });
  it("should capture a function call with streaming", async () => {
    vi.spyOn(openai, "post").mockImplementation(
      // @ts-expect-error the response type is not correct - this is just for testing
      async (): Promise<unknown> => {
        const iterator = () =>
          (async function* () {
            yield {
              id: "chatcmpl-8iA39kCtuVHIVDr9AnBdJZjgSjNWL",
              object: "chat.completion.chunk",
              created: 1705535755,
              model: "gpt-3.5-turbo-0613",
              system_fingerprint: null,
              choices: [
                {
                  index: 0,
                  delta: {
                    role: "assistant",
                    content: null,
                    function_call: { name: "getWeather", arguments: "" },
                  },
                  logprobs: null,
                  finish_reason: null,
                },
              ],
            };
            yield {
              id: "chatcmpl-8iA39kCtuVHIVDr9AnBdJZjgSjNWL",
              object: "chat.completion.chunk",
              created: 1705535755,
              model: "gpt-3.5-turbo-0613",
              system_fingerprint: null,
              choices: [
                {
                  index: 0,
                  delta: { function_call: { arguments: "{}" } },
                  logprobs: null,
                  finish_reason: null,
                },
              ],
            };
            yield {
              id: "chatcmpl-8iA39kCtuVHIVDr9AnBdJZjgSjNWL",
              object: "chat.completion.chunk",
              created: 1705535755,
              model: "gpt-3.5-turbo-0613",
              system_fingerprint: null,
              choices: [
                {
                  index: 0,
                  delta: {},
                  logprobs: null,
                  finish_reason: "function_call",
                },
              ],
            };
          })();
        const controller = new AbortController();
        return new Stream(iterator, controller);
      },
    );
    const stream = await openai.chat.completions.create({
      messages: [{ role: "user", content: "What's the weather today?" }],
      model: "gpt-3.5-turbo",
      functions: [
        {
          name: "getWeather",
          description: "Get the weather for a location.",
          parameters: {
            type: "object",
            properties: {
              location: { type: "string" },
            },
          },
        },
        {
          name: "getCurrentLocation",
          description: "Get the current location of the user.",
          parameters: { type: "object", properties: {} },
        },
      ],
      stream: true,
    });

    let response = "";
    for await (const chunk of stream) {
      if (chunk.choices[0].delta.content)
        response += chunk.choices[0].delta.content;
    }
    // When a tool is called, the content is empty
    expect(response).toBe("");
    const spans = memoryExporter.getFinishedSpans();
    expect(spans.length).toBe(1);
    const span = spans[0];
    expect(span.name).toBe("OpenAI Chat Completions");
    expect(span.attributes).toMatchInlineSnapshot(`
      {
        "input.mime_type": "application/json",
        "input.value": "{"messages":[{"role":"user","content":"What's the weather today?"}],"model":"gpt-3.5-turbo","functions":[{"name":"getWeather","description":"Get the weather for a location.","parameters":{"type":"object","properties":{"location":{"type":"string"}}}},{"name":"getCurrentLocation","description":"Get the current location of the user.","parameters":{"type":"object","properties":{}}}],"stream":true}",
        "llm.input_messages.0.message.content": "What's the weather today?",
        "llm.input_messages.0.message.role": "user",
        "llm.invocation_parameters": "{"model":"gpt-3.5-turbo","functions":[{"name":"getWeather","description":"Get the weather for a location.","parameters":{"type":"object","properties":{"location":{"type":"string"}}}},{"name":"getCurrentLocation","description":"Get the current location of the user.","parameters":{"type":"object","properties":{}}}],"stream":true}",
        "llm.model_name": "gpt-3.5-turbo",
        "llm.output_messages.0.message.content": "",
        "llm.output_messages.0.message.function_call_arguments_json": "{}",
        "llm.output_messages.0.message.function_call_name": "getWeather",
        "llm.output_messages.0.message.role": "assistant",
        "llm.provider": "openai",
        "llm.system": "openai",
        "openinference.span.kind": "LLM",
        "output.mime_type": "text/plain",
        "output.value": "",
      }
`);
  });
  it("should not emit a span if tracing is suppressed", async () => {
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
    vi.spyOn(openai, "post").mockImplementation(
      // @ts-expect-error the response type is not correct - this is just for testing
      async (): Promise<unknown> => {
        return response;
      },
    );
    const _response = await new Promise((resolve, _reject) => {
      context.with(suppressTracing(context.active()), () => {
        resolve(
          openai.chat.completions.create({
            messages: [{ role: "user", content: "Say this is a test" }],
            model: "gpt-3.5-turbo",
          }),
        );
      });
    });
    const spans = memoryExporter.getFinishedSpans();
    expect(spans.length).toBe(0);
  });
  it("should capture image in request", async () => {
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
    vi.spyOn(openai, "post").mockImplementation(
      // @ts-expect-error the response type is not correct - this is just for testing
      async (): Promise<unknown> => {
        return response;
      },
    );
    await openai.chat.completions.create({
      messages: [
        {
          role: "user",
          content: [
            { type: "text", text: "Say this is a test" },
            {
              type: "image_url",
              image_url: {
                url: "data:image/gif;base64,R0lGODlhAQABAIAAAP///wAAACH5BAEAAAAALAAAAAABAAEAAAICRAEAOw==",
              },
            },
          ],
        },
      ],
      model: "gpt-3.5-turbo",
    });
    const spans = memoryExporter.getFinishedSpans();
    expect(spans.length).toBe(1);
    const span = spans[0];
    expect(span.name).toBe("OpenAI Chat Completions");
    expect(span.attributes).toMatchInlineSnapshot(`
{
  "input.mime_type": "application/json",
  "input.value": "{"messages":[{"role":"user","content":[{"type":"text","text":"Say this is a test"},{"type":"image_url","image_url":{"url":"data:image/gif;base64,R0lGODlhAQABAIAAAP///wAAACH5BAEAAAAALAAAAAABAAEAAAICRAEAOw=="}}]}],"model":"gpt-3.5-turbo"}",
  "llm.input_messages.0.message.contents.0.message_content.text": "Say this is a test",
  "llm.input_messages.0.message.contents.0.message_content.type": "text",
  "llm.input_messages.0.message.contents.1.message_content.image.image.url": "data:image/gif;base64,R0lGODlhAQABAIAAAP///wAAACH5BAEAAAAALAAAAAABAAEAAAICRAEAOw==",
  "llm.input_messages.0.message.contents.1.message_content.type": "image",
  "llm.input_messages.0.message.role": "user",
  "llm.invocation_parameters": "{"model":"gpt-3.5-turbo"}",
  "llm.model_name": "gpt-3.5-turbo-0613",
  "llm.output_messages.0.message.content": "This is a test.",
  "llm.output_messages.0.message.role": "assistant",
  "llm.provider": "openai",
  "llm.system": "openai",
  "llm.token_count.completion": 5,
  "llm.token_count.prompt": 12,
  "llm.token_count.total": 17,
  "openinference.span.kind": "LLM",
  "output.mime_type": "application/json",
  "output.value": "{"id":"chatcmpl-8adq9JloOzNZ9TyuzrKyLpGXexh6p","object":"chat.completion","created":1703743645,"model":"gpt-3.5-turbo-0613","choices":[{"index":0,"message":{"role":"assistant","content":"This is a test."},"logprobs":null,"finish_reason":"stop"}],"usage":{"prompt_tokens":12,"completion_tokens":5,"total_tokens":17}}",
}
`);
  });

  it("should capture context attributes and add them to spans", async () => {
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
    vi.spyOn(openai, "post").mockImplementation(
      // @ts-expect-error the response type is not correct - this is just for testing
      async (): Promise<unknown> => {
        return response;
      },
    );
    await context.with(
      setSession(
        setPromptTemplate(context.active(), {
          template: "hello {name}",
          variables: { name: "world" },
          version: "V1.0",
        }),
        { sessionId: "session-id" },
      ),
      async () => {
        await openai.completions.create({
          prompt: "Say this is a test",
          model: "gpt-3.5-turbo-instruct",
        });
      },
    );
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
  "llm.prompt_template.template": "hello {name}",
  "llm.prompt_template.variables": "{"name":"world"}",
  "llm.prompt_template.version": "V1.0",
  "llm.provider": "openai",
  "llm.system": "openai",
  "llm.token_count.completion": 5,
  "llm.token_count.prompt": 12,
  "llm.token_count.total": 17,
  "openinference.span.kind": "LLM",
  "output.mime_type": "text/plain",
  "output.value": "This is a test",
  "session.id": "session-id",
}
`);
  });
  it("creates a span for chat completions parse", async () => {
    const response = {
      id: "chatcmpl-parseTest",
      object: "chat.completion",
      created: 1706000000,
      model: "gpt-4o-2024-08-06",
      choices: [
        {
          index: 0,
          message: {
            role: "assistant",
            content:
              '{"name":"science fair","date":"Friday","participants":["Alice","Bob"]}',
          },
          logprobs: null,
          finish_reason: "stop",
        },
      ],
      usage: { prompt_tokens: 20, completion_tokens: 10, total_tokens: 30 },
    };

    // Mock out the post method that chat completions uses internally
    vi.spyOn(openai, "post").mockImplementation(() => {
      // Create a full APIPromise-like object that satisfies all OpenAI SDK expectations
      const apiPromise = new APIPromise(
        openai,
        Promise.resolve({
          requestLogID: "123",
          retryOfRequestLogID: "123",
          startTime: 123,
          response: {
            json: () => Promise.resolve(response),
            text: () => Promise.resolve(JSON.stringify(response)),
            clone: () => ({
              json: () => Promise.resolve(response),
              text: () => Promise.resolve(JSON.stringify(response)),
            }),
            headers: new Headers({ "content-type": "application/json" }),
            status: 200,
            statusText: "OK",
            ok: true,
          } as Response,
          options: {
            method: "post",
            path: "/chat/completions",
          },
          controller: new AbortController(),
        }),
      );
      return apiPromise;
    });

    const CalendarEvent = z.object({
      name: z.string(),
      date: z.string(),
      participants: z.array(z.string()),
    });

    // Invoke the helper method under test
    await openai.chat.completions.parse({
      model: "gpt-4o-2024-08-06",
      messages: [
        { role: "system", content: "Extract the event information." },
        {
          role: "user",
          content: "Alice and Bob are going to a science fair on Friday.",
        },
      ],
      response_format: zodResponseFormat(CalendarEvent, "event"),
    });

    const spans = memoryExporter.getFinishedSpans();
    expect(spans.length).toBe(1);
    const span = spans[0];
    expect(span.name).toBe("OpenAI Chat Completions");
    expect(span.attributes).toMatchInlineSnapshot(`
{
  "input.mime_type": "application/json",
  "input.value": "{"model":"gpt-4o-2024-08-06","messages":[{"role":"system","content":"Extract the event information."},{"role":"user","content":"Alice and Bob are going to a science fair on Friday."}],"response_format":{"type":"json_schema","json_schema":{"name":"event","strict":true,"schema":{"type":"object","properties":{"name":{"type":"string"},"date":{"type":"string"},"participants":{"type":"array","items":{"type":"string"}}},"required":["name","date","participants"],"additionalProperties":false,"$schema":"http://json-schema.org/draft-07/schema#"}}}}",
  "llm.input_messages.0.message.content": "Extract the event information.",
  "llm.input_messages.0.message.role": "system",
  "llm.input_messages.1.message.content": "Alice and Bob are going to a science fair on Friday.",
  "llm.input_messages.1.message.role": "user",
  "llm.invocation_parameters": "{"model":"gpt-4o-2024-08-06","response_format":{"type":"json_schema","json_schema":{"name":"event","strict":true,"schema":{"type":"object","properties":{"name":{"type":"string"},"date":{"type":"string"},"participants":{"type":"array","items":{"type":"string"}}},"required":["name","date","participants"],"additionalProperties":false,"$schema":"http://json-schema.org/draft-07/schema#"}}}}",
  "llm.model_name": "gpt-4o-2024-08-06",
  "llm.output_messages.0.message.content": "{"name":"science fair","date":"Friday","participants":["Alice","Bob"]}",
  "llm.output_messages.0.message.role": "assistant",
  "llm.provider": "openai",
  "llm.system": "openai",
  "llm.token_count.completion": 10,
  "llm.token_count.prompt": 20,
  "llm.token_count.total": 30,
  "openinference.span.kind": "LLM",
  "output.mime_type": "application/json",
  "output.value": "{"id":"chatcmpl-parseTest","object":"chat.completion","created":1706000000,"model":"gpt-4o-2024-08-06","choices":[{"index":0,"message":{"role":"assistant","content":"{\\"name\\":\\"science fair\\",\\"date\\":\\"Friday\\",\\"participants\\":[\\"Alice\\",\\"Bob\\"]}"},"logprobs":null,"finish_reason":"stop"}],"usage":{"prompt_tokens":20,"completion_tokens":10,"total_tokens":30}}",
}
`);
  });
});

describe("OpenAIInstrumentation with TraceConfig", () => {
  const tracerProvider = new NodeTracerProvider();
  tracerProvider.register();
  const instrumentation = new OpenAIInstrumentation({
    traceConfig: { hideInputs: true },
  });
  instrumentation.disable();
  let openai: OpenAI;

  instrumentation.setTracerProvider(tracerProvider);
  tracerProvider.addSpanProcessor(new SimpleSpanProcessor(memoryExporter));
  // @ts-expect-error the moduleExports property is private. This is needed to make the test work with auto-mocking
  instrumentation._modules[0].moduleExports = OpenAI;

  beforeAll(() => {
    instrumentation.enable();
    openai = new OpenAI({
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
    vi.clearAllMocks();
  });
  it("is patched", () => {
    expect(
      (OpenAI as { openInferencePatched?: boolean }).openInferencePatched,
    ).toBe(true);
    expect(isPatched()).toBe(true);
  });
  it("should respect a trace config and mask attributes accordingly", async () => {
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
    vi.spyOn(openai, "post").mockImplementation(
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
  "input.value": "__REDACTED__",
  "llm.invocation_parameters": "{"model":"gpt-3.5-turbo-instruct"}",
  "llm.model_name": "gpt-3.5-turbo-instruct",
  "llm.provider": "openai",
  "llm.system": "openai",
  "llm.token_count.completion": 5,
  "llm.token_count.prompt": 12,
  "llm.token_count.total": 17,
  "openinference.span.kind": "LLM",
  "output.mime_type": "text/plain",
  "output.value": "This is a test",
}
`);
  });
});

describe("AzureOpenAIInstrumentation", () => {
  const tracerProvider = new NodeTracerProvider();
  tracerProvider.register();
  const instrumentation = new OpenAIInstrumentation();
  instrumentation.disable();
  let azureOpenai: AzureOpenAI;

  instrumentation.setTracerProvider(tracerProvider);
  tracerProvider.addSpanProcessor(new SimpleSpanProcessor(memoryExporter));
  // @ts-expect-error the moduleExports property is private. This is needed to make the test work with auto-mocking
  instrumentation._modules[0].moduleExports = OpenAI;

  beforeAll(() => {
    instrumentation.enable();
    azureOpenai = new AzureOpenAI({
      apiKey: "fake-api-key",
      endpoint: "https://my-azure-openai.openai.azure.com",
      apiVersion: "2024-02-15-preview",
    });
  });
  afterAll(() => {
    instrumentation.disable();
  });
  beforeEach(() => {
    memoryExporter.reset();
  });
  afterEach(() => {
    vi.clearAllMocks();
  });

  it("is patched", () => {
    expect(
      (OpenAI as { openInferencePatched?: boolean }).openInferencePatched,
    ).toBe(true);
    expect(isPatched()).toBe(true);
  });

  it("creates a span for chat completions", async () => {
    const response = {
      id: "chatcmpl-8adq9JloOzNZ9TyuzrKyLpGXexh6p",
      object: "chat.completion",
      created: 1703743645,
      model: "gpt-35-turbo",
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
    vi.spyOn(azureOpenai, "post").mockImplementation(
      // @ts-expect-error the response type is not correct - this is just for testing
      async (): Promise<unknown> => {
        return response;
      },
    );
    await azureOpenai.chat.completions.create({
      messages: [{ role: "user", content: "Say this is a test" }],
      model: "gpt-35-turbo",
    });
    const spans = memoryExporter.getFinishedSpans();
    expect(spans.length).toBe(1);
    const span = spans[0];
    expect(span.name).toBe("OpenAI Chat Completions");
    expect(span.attributes).toMatchInlineSnapshot(`
{
  "input.mime_type": "application/json",
  "input.value": "{"messages":[{"role":"user","content":"Say this is a test"}],"model":"gpt-35-turbo"}",
  "llm.input_messages.0.message.content": "Say this is a test",
  "llm.input_messages.0.message.role": "user",
  "llm.invocation_parameters": "{"model":"gpt-35-turbo"}",
  "llm.model_name": "gpt-35-turbo",
  "llm.output_messages.0.message.content": "This is a test.",
  "llm.output_messages.0.message.role": "assistant",
  "llm.provider": "azure",
  "llm.system": "openai",
  "llm.token_count.completion": 5,
  "llm.token_count.prompt": 12,
  "llm.token_count.total": 17,
  "openinference.span.kind": "LLM",
  "output.mime_type": "application/json",
  "output.value": "{"id":"chatcmpl-8adq9JloOzNZ9TyuzrKyLpGXexh6p","object":"chat.completion","created":1703743645,"model":"gpt-35-turbo","choices":[{"index":0,"message":{"role":"assistant","content":"This is a test."},"logprobs":null,"finish_reason":"stop"}],"usage":{"prompt_tokens":12,"completion_tokens":5,"total_tokens":17}}",
}
`);
  });

  it("creates a span for embeddings", async () => {
    const response: CreateEmbeddingResponse = {
      object: "list",
      data: [{ object: "embedding", index: 0, embedding: [1, 2, 3] }],
      model: "text-embedding-ada-002",
      usage: { prompt_tokens: 0, total_tokens: 0 },
    };

    // Mock out the embedding create endpoint
    vi.spyOn(azureOpenai, "post").mockImplementation(() => {
      return new APIPromise(
        azureOpenai,
        Promise.resolve({
          requestLogID: "123",
          retryOfRequestLogID: "123",
          startTime: 123,
          response: {
            json: () => Promise.resolve(response),
            text: () => Promise.resolve(JSON.stringify(response)),
            clone: () => ({
              json: () => Promise.resolve(response),
              text: () => Promise.resolve(JSON.stringify(response)),
            }),
            headers: new Headers({ "content-type": "application/json" }),
            status: 200,
            statusText: "OK",
            ok: true,
          } as Response,
          options: {
            method: "post",
            path: "/embeddings",
          },
          controller: new AbortController(),
        }),
      );
    });

    await azureOpenai.embeddings.create({
      input: "A happy moment",
      model: "text-embedding-ada-002",
    });

    const spans = memoryExporter.getFinishedSpans();
    expect(spans.length).toBe(1);
    const span = spans[0];
    expect(span.name).toBe("OpenAI Embeddings");
    // Check the attributes
    expect(span.attributes["embedding.embeddings.0.embedding.text"]).toBe(
      "A happy moment",
    );
    expect(span.attributes["embedding.model_name"]).toBe(
      "text-embedding-ada-002",
    );
    expect(span.attributes["input.mime_type"]).toBe("text/plain");
    expect(span.attributes["input.value"]).toBe("A happy moment");
    expect(span.attributes["openinference.span.kind"]).toBe("EMBEDDING");
  });

  it("can handle streaming responses", async () => {
    // Mock out the post endpoint to return a stream
    vi.spyOn(azureOpenai, "post").mockImplementation(
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
    const stream = await azureOpenai.chat.completions.create({
      messages: [{ role: "user", content: "Say this is a test" }],
      model: "gpt-35-turbo",
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
  "input.value": "{"messages":[{"role":"user","content":"Say this is a test"}],"model":"gpt-35-turbo","stream":true}",
  "llm.input_messages.0.message.content": "Say this is a test",
  "llm.input_messages.0.message.role": "user",
  "llm.invocation_parameters": "{"model":"gpt-35-turbo","stream":true}",
  "llm.model_name": "gpt-35-turbo",
  "llm.output_messages.0.message.content": "This is a test.",
  "llm.output_messages.0.message.role": "assistant",
  "llm.provider": "azure",
  "llm.system": "openai",
  "openinference.span.kind": "LLM",
  "output.mime_type": "text/plain",
  "output.value": "This is a test.",
}
`);
  });
});

describe("OpenAIInstrumentation with a custom tracer provider", () => {
  describe("OpenAIInstrumentation with custom TracerProvider passed in", () => {
    const customTracerProvider = new NodeTracerProvider();
    const customMemoryExporter = new InMemorySpanExporter();
    let openai: OpenAI;

    // Note: We don't register this provider globally.
    customTracerProvider.addSpanProcessor(
      new SimpleSpanProcessor(customMemoryExporter),
    );

    // Instantiate instrumentation with the custom provider
    const instrumentation = new OpenAIInstrumentation({
      tracerProvider: customTracerProvider,
    });
    instrumentation.disable();

    // Mock the module exports like in other tests
    // @ts-expect-error the moduleExports property is private. This is needed to make the test work with auto-mocking
    instrumentation._modules[0].moduleExports = OpenAI;

    beforeAll(() => {
      instrumentation.enable();
      openai = new OpenAI({
        apiKey: "fake-api-key",
      });
    });

    afterAll(() => {
      instrumentation.disable();
    });

    beforeEach(() => {
      memoryExporter.reset();
      customMemoryExporter.reset();
    });

    afterEach(() => {
      vi.resetAllMocks();
      vi.clearAllMocks();
    });

    it("should use the provided tracer provider instead of the global one", async () => {
      const response = {
        id: "chatcmpl-test",
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

      vi.spyOn(openai, "post").mockImplementation(
        // @ts-expect-error the response type is not correct - this is just for testing
        async (): Promise<unknown> => {
          return response;
        },
      );

      await openai.chat.completions.create({
        messages: [{ role: "user", content: "Say this is a test" }],
        model: "gpt-3.5-turbo",
      });

      const spans = customMemoryExporter.getFinishedSpans();
      const globalSpans = memoryExporter.getFinishedSpans();
      expect(spans.length).toBe(1);
      expect(globalSpans.length).toBe(0);
      const span = spans[0];
      expect(span.name).toBe("OpenAI Chat Completions");
      expect(span.attributes["llm.provider"]).toBe("openai");
      expect(span.attributes["llm.model_name"]).toBe("gpt-3.5-turbo-0613");
    });
  });

  describe("OpenAIInstrumentation with custom TracerProvider set", () => {
    const customTracerProvider = new NodeTracerProvider();
    const customMemoryExporter = new InMemorySpanExporter();
    let openai: OpenAI;

    // Note: We don't register this provider globally.
    customTracerProvider.addSpanProcessor(
      new SimpleSpanProcessor(customMemoryExporter),
    );

    // Instantiate instrumentation with the custom provider
    const instrumentation = new OpenAIInstrumentation();
    instrumentation.setTracerProvider(customTracerProvider);
    instrumentation.disable();

    // Mock the module exports like in other tests
    // @ts-expect-error the moduleExports property is private. This is needed to make the test work with auto-mocking
    instrumentation._modules[0].moduleExports = OpenAI;

    beforeAll(() => {
      instrumentation.enable();
      openai = new OpenAI({
        apiKey: "fake-api-key",
      });
    });

    afterAll(() => {
      instrumentation.disable();
    });

    beforeEach(() => {
      memoryExporter.reset();
      customMemoryExporter.reset();
    });

    afterEach(() => {
      vi.resetAllMocks();
      vi.clearAllMocks();
    });

    it("should use the provided tracer provider instead of the global one", async () => {
      const response = {
        id: "chatcmpl-test",
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

      vi.spyOn(openai, "post").mockImplementation(
        // @ts-expect-error the response type is not correct - this is just for testing
        async (): Promise<unknown> => {
          return response;
        },
      );

      await openai.chat.completions.create({
        messages: [{ role: "user", content: "Say this is a test" }],
        model: "gpt-3.5-turbo",
      });

      const spans = customMemoryExporter.getFinishedSpans();
      const globalSpans = memoryExporter.getFinishedSpans();
      expect(spans.length).toBe(1);
      expect(globalSpans.length).toBe(0);
      const span = spans[0];
      expect(span.name).toBe("OpenAI Chat Completions");
      expect(span.attributes["llm.provider"]).toBe("openai");
      expect(span.attributes["llm.model_name"]).toBe("gpt-3.5-turbo-0613");
    });
  });

  describe("OpenAIInstrumentation with custom TracerProvider set via registerInstrumentations", () => {
    const customTracerProvider = new NodeTracerProvider();
    const customMemoryExporter = new InMemorySpanExporter();
    let openai: OpenAI;

    // Note: We don't register this provider globally.
    customTracerProvider.addSpanProcessor(
      new SimpleSpanProcessor(customMemoryExporter),
    );

    // Instantiate instrumentation with the custom provider
    const instrumentation = new OpenAIInstrumentation();
    registerInstrumentations({
      instrumentations: [instrumentation],
      tracerProvider: customTracerProvider,
    });
    instrumentation.disable();

    // Mock the module exports like in other tests
    // @ts-expect-error the moduleExports property is private. This is needed to make the test work with auto-mocking
    instrumentation._modules[0].moduleExports = OpenAI;

    beforeAll(() => {
      instrumentation.enable();
      openai = new OpenAI({
        apiKey: "fake-api-key",
      });
    });

    afterAll(() => {
      instrumentation.disable();
    });

    beforeEach(() => {
      memoryExporter.reset();
      customMemoryExporter.reset();
    });

    afterEach(() => {
      vi.resetAllMocks();
      vi.clearAllMocks();
    });

    it("should use the provided tracer provider instead of the global one", async () => {
      const response = {
        id: "chatcmpl-test",
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

      vi.spyOn(openai, "post").mockImplementation(
        // @ts-expect-error the response type is not correct - this is just for testing
        async (): Promise<unknown> => {
          return response;
        },
      );

      await openai.chat.completions.create({
        messages: [{ role: "user", content: "Say this is a test" }],
        model: "gpt-3.5-turbo",
      });

      const spans = customMemoryExporter.getFinishedSpans();
      const globalSpans = memoryExporter.getFinishedSpans();
      expect(spans.length).toBe(1);
      expect(globalSpans.length).toBe(0);
      const span = spans[0];
      expect(span.name).toBe("OpenAI Chat Completions");
      expect(span.attributes["llm.provider"]).toBe("openai");
      expect(span.attributes["llm.model_name"]).toBe("gpt-3.5-turbo-0613");
    });
  });
});
