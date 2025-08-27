import { OpenAIInstrumentation } from "../src";
import {
  InMemorySpanExporter,
  SimpleSpanProcessor,
} from "@opentelemetry/sdk-trace-base";
import { NodeTracerProvider } from "@opentelemetry/sdk-trace-node";

import OpenAI, { APIPromise } from "openai";
import { Stream } from "openai/streaming";
import { Response as ResponseType } from "openai/resources/responses/responses";
import { zodTextFormat } from "openai/helpers/zod";
import { z } from "zod";
import { vi } from "vitest";

const memoryExporter = new InMemorySpanExporter();

// Add new describe block for OpenAI Responses tests
describe("OpenAIInstrumentation - Responses", () => {
  const tracerProvider = new NodeTracerProvider();
  tracerProvider.register();
  const instrumentation = new OpenAIInstrumentation();
  instrumentation.disable();
  let openai: OpenAI;

  instrumentation.setTracerProvider(tracerProvider);
  tracerProvider.addSpanProcessor(new SimpleSpanProcessor(memoryExporter));
  // @ts-expect-error the moduleExports property is private. This is needed to make the test work with auto-mocking
  instrumentation._modules[0].moduleExports = OpenAI;

  const responseBase = {
    status: "completed",
    error: null,
    incomplete_details: null,
    instructions: null,
    max_output_tokens: null,
    tools: [],
    tool_choice: "auto",
    text: { format: { type: "text" } },
    parallel_tool_calls: true,
    previous_response_id: null,
    reasoning: { effort: null, summary: null },
    service_tier: "default",
    metadata: {},
    object: "response",
    created_at: 1744987785,
    temperature: 1,
    top_p: 1,
    truncation: "disabled",
    usage: {
      input_tokens: 12,
      input_tokens_details: { cached_tokens: 0 },
      output_tokens: 6,
      output_tokens_details: { reasoning_tokens: 0 },
      total_tokens: 18,
    },
  } satisfies Partial<ResponseType>;

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

  it("creates a span for responses", async () => {
    const response = {
      ...responseBase,
      id: "resp_245",
      output: [
        {
          id: "msg_example",
          type: "message",
          status: "completed",
          content: [
            {
              type: "output_text",
              annotations: [],
              text: "This is a test.",
            },
          ],
          role: "assistant",
        },
      ],
      model: "gpt-4.1",
      output_text: "This is a test.",
    } satisfies ResponseType;
    // Mock out the responses endpoint
    vi.spyOn(openai, "post").mockImplementation(() => {
      return new APIPromise(
        new OpenAI({ apiKey: "fake-api-key" }),
        new Promise((resolve) => {
          resolve({
            response: new Response(),
            // @ts-expect-error the response type is not correct - this is just for testing
            options: {},
            controller: new AbortController(),
          });
        }),
        () => response,
      );
    });
    await openai.responses.create({
      input: "Say this is a test",
      model: "gpt-4.1",
    });
    const spans = memoryExporter.getFinishedSpans();
    expect(spans.length).toBe(1);
    const span = spans[0];
    expect(span.name).toBe("OpenAI Responses");
    expect(span.attributes).toMatchInlineSnapshot(`
{
  "input.mime_type": "application/json",
  "input.value": "{"input":"Say this is a test","model":"gpt-4.1"}",
  "llm.input_messages.0.message.content": "Say this is a test",
  "llm.input_messages.0.message.role": "user",
  "llm.invocation_parameters": "{"model":"gpt-4.1"}",
  "llm.model_name": "gpt-4.1",
  "llm.output_messages.0.message.contents.0.message_content.text": "This is a test.",
  "llm.output_messages.0.message.contents.0.message_content.type": "output_text",
  "llm.output_messages.0.message.role": "assistant",
  "llm.provider": "openai",
  "llm.system": "openai",
  "llm.token_count.completion": 6,
  "llm.token_count.completion_details.reasoning": 0,
  "llm.token_count.prompt": 12,
  "llm.token_count.prompt_details.cache_read": 0,
  "llm.token_count.total": 18,
  "openinference.span.kind": "LLM",
  "output.mime_type": "application/json",
  "output.value": "{"status":"completed","error":null,"incomplete_details":null,"instructions":null,"max_output_tokens":null,"tools":[],"tool_choice":"auto","text":{"format":{"type":"text"}},"parallel_tool_calls":true,"previous_response_id":null,"reasoning":{"effort":null,"summary":null},"service_tier":"default","metadata":{},"object":"response","created_at":1744987785,"temperature":1,"top_p":1,"truncation":"disabled","usage":{"input_tokens":12,"input_tokens_details":{"cached_tokens":0},"output_tokens":6,"output_tokens_details":{"reasoning_tokens":0},"total_tokens":18},"id":"resp_245","output":[{"id":"msg_example","type":"message","status":"completed","content":[{"type":"output_text","annotations":[],"text":"This is a test."}],"role":"assistant"}],"model":"gpt-4.1","output_text":"This is a test."}",
}
`);
  });

  it("creates a span for responses with multiple messages and instructions", async () => {
    const response = {
      ...responseBase,
      id: "resp_245",
      output: [
        {
          id: "msg_example",
          type: "message",
          status: "completed",
          content: [
            {
              type: "output_text",
              annotations: [],
              text: "This is a test.",
            },
          ],
          role: "assistant",
        },
      ],
      model: "gpt-4.1",
      output_text: "This is a test.",
    } satisfies ResponseType;
    // Mock out the responses endpoint
    vi.spyOn(openai, "post").mockImplementation(() => {
      return new APIPromise(
        new OpenAI({ apiKey: "fake-api-key" }),
        new Promise((resolve) => {
          resolve({
            response: new Response(),
            // @ts-expect-error the response type is not correct - this is just for testing
            options: {},
            controller: new AbortController(),
          });
        }),
        () => response,
      );
    });
    await openai.responses.create({
      input: [
        {
          type: "message",
          content: "say this is a test",
          role: "user",
        },
        {
          type: "message",
          content: "remember to say this is a test",
          role: "user",
        },
      ],
      model: "gpt-4.1",
      instructions: "You are a helpful assistant.",
    });
    const spans = memoryExporter.getFinishedSpans();
    expect(spans.length).toBe(1);
    const span = spans[0];
    expect(span.name).toBe("OpenAI Responses");
    expect(span.attributes).toMatchInlineSnapshot(`
{
  "input.mime_type": "application/json",
  "input.value": "{"input":[{"type":"message","content":"say this is a test","role":"user"},{"type":"message","content":"remember to say this is a test","role":"user"}],"model":"gpt-4.1","instructions":"You are a helpful assistant."}",
  "llm.input_messages.0.message.content": "You are a helpful assistant.",
  "llm.input_messages.0.message.role": "system",
  "llm.input_messages.1.message.content": "say this is a test",
  "llm.input_messages.1.message.role": "user",
  "llm.input_messages.2.message.content": "remember to say this is a test",
  "llm.input_messages.2.message.role": "user",
  "llm.invocation_parameters": "{"model":"gpt-4.1","instructions":"You are a helpful assistant."}",
  "llm.model_name": "gpt-4.1",
  "llm.output_messages.0.message.contents.0.message_content.text": "This is a test.",
  "llm.output_messages.0.message.contents.0.message_content.type": "output_text",
  "llm.output_messages.0.message.role": "assistant",
  "llm.provider": "openai",
  "llm.system": "openai",
  "llm.token_count.completion": 6,
  "llm.token_count.completion_details.reasoning": 0,
  "llm.token_count.prompt": 12,
  "llm.token_count.prompt_details.cache_read": 0,
  "llm.token_count.total": 18,
  "openinference.span.kind": "LLM",
  "output.mime_type": "application/json",
  "output.value": "{"status":"completed","error":null,"incomplete_details":null,"instructions":null,"max_output_tokens":null,"tools":[],"tool_choice":"auto","text":{"format":{"type":"text"}},"parallel_tool_calls":true,"previous_response_id":null,"reasoning":{"effort":null,"summary":null},"service_tier":"default","metadata":{},"object":"response","created_at":1744987785,"temperature":1,"top_p":1,"truncation":"disabled","usage":{"input_tokens":12,"input_tokens_details":{"cached_tokens":0},"output_tokens":6,"output_tokens_details":{"reasoning_tokens":0},"total_tokens":18},"id":"resp_245","output":[{"id":"msg_example","type":"message","status":"completed","content":[{"type":"output_text","annotations":[],"text":"This is a test."}],"role":"assistant"}],"model":"gpt-4.1","output_text":"This is a test."}",
}
`);
  });

  it("can handle streaming responses", async () => {
    // Mock out the post endpoint to return a stream
    vi.spyOn(openai, "post").mockImplementation(
      // @ts-expect-error the response type is not correct - this is just for testing
      (): Promise<unknown> => {
        const iterator = () =>
          (async function* () {
            yield {
              type: "response.output_text.delta",
              delta: "I ",
            };
            yield {
              type: "response.output_text.delta",
              delta: "am streaming!",
            };
            yield {
              type: "response.completed",
              response: {
                ...responseBase,
                id: "resp-567",
                model: "gpt-4.1",
                output: [
                  {
                    type: "output_text",
                    text: "I am streaming!",
                    annotations: [],
                  },
                ],
              },
            };
          })();
        const controller = new AbortController();
        const stream = new Stream(iterator, controller);
        return new APIPromise(
          new OpenAI({ apiKey: "fake-api-key" }),
          // @ts-expect-error the response type is not correct - this is just for testing
          Promise.resolve({
            response: new Response(),
            options: {},
            controller,
          }),
          () => stream,
        );
      },
    );
    const stream = await openai.responses.create({
      input: "Say I am streaming!",
      model: "gpt-4.1",
      stream: true,
    });

    let responseText = "";
    for await (const event of stream) {
      if (event.type === "response.output_text.delta") {
        responseText += event.delta;
      }
    }
    expect(responseText).toBe("I am streaming!");
    // Wait for the span to be exported
    await new Promise((resolve) => setTimeout(resolve));
    const spans = memoryExporter.getFinishedSpans();
    expect(spans.length).toBe(1);
    const span = spans[0];
    expect(span.name).toBe("OpenAI Responses");
    expect(span.attributes).toMatchInlineSnapshot(`
{
  "input.mime_type": "application/json",
  "input.value": "{"input":"Say I am streaming!","model":"gpt-4.1","stream":true}",
  "llm.input_messages.0.message.content": "Say I am streaming!",
  "llm.input_messages.0.message.role": "user",
  "llm.invocation_parameters": "{"model":"gpt-4.1","stream":true}",
  "llm.model_name": "gpt-4.1",
  "llm.provider": "openai",
  "llm.system": "openai",
  "llm.token_count.completion": 6,
  "llm.token_count.completion_details.reasoning": 0,
  "llm.token_count.prompt": 12,
  "llm.token_count.prompt_details.cache_read": 0,
  "llm.token_count.total": 18,
  "openinference.span.kind": "LLM",
  "output.mime_type": "application/json",
  "output.value": "{"status":"completed","error":null,"incomplete_details":null,"instructions":null,"max_output_tokens":null,"tools":[],"tool_choice":"auto","text":{"format":{"type":"text"}},"parallel_tool_calls":true,"previous_response_id":null,"reasoning":{"effort":null,"summary":null},"service_tier":"default","metadata":{},"object":"response","created_at":1744987785,"temperature":1,"top_p":1,"truncation":"disabled","usage":{"input_tokens":12,"input_tokens_details":{"cached_tokens":0},"output_tokens":6,"output_tokens_details":{"reasoning_tokens":0},"total_tokens":18},"id":"resp-567","model":"gpt-4.1","output":[{"type":"output_text","text":"I am streaming!","annotations":[]}]}",
}
`);
  });

  it("should capture tool calls with streaming responses", async () => {
    // Mock out the post endpoint to return a stream with tool calls
    vi.spyOn(openai, "post").mockImplementation(
      // @ts-expect-error the response type is not correct - this is just for testing
      (): Promise<unknown> => {
        const iterator = () =>
          (async function* () {
            yield {
              type: "response.output_item.added",
              item: {
                type: "function_call",
                id: "call_abc123",
                name: "get_weather",
                arguments: "",
              },
            };
            yield {
              type: "response.function_call_arguments.delta",
              item_id: "call_abc123",
              delta: '{"locati',
            };
            yield {
              type: "response.function_call_arguments.delta",
              item_id: "call_abc123",
              delta: 'on": "Boston"}',
            };
            yield {
              type: "response.completed",
              response: {
                id: "resp-890",
                object: "response",
                created: 1705535755,
                model: "gpt-4.1",
                output_text: null,
                output: [
                  {
                    id: "fc_1234",
                    type: "function_call",
                    status: "completed",
                    arguments: '{"location":"boston"}',
                    call_id: "call_abc123",
                    name: "get_weather",
                  },
                ],
                tools: [
                  {
                    type: "function",
                    description: null,
                    name: "get_weather",
                    parameters: {
                      type: "object",
                      properties: { location: { type: "string" } },
                      additionalProperties: false,
                      required: ["location"],
                    },
                    strict: true,
                  },
                ],
                usage: {
                  prompt_tokens: 86,
                  completion_tokens: 15,
                  total_tokens: 101,
                },
              },
            };
          })();
        const controller = new AbortController();
        const stream = new Stream(iterator, controller);
        return new APIPromise(
          new OpenAI({ apiKey: "fake-api-key" }),
          // @ts-expect-error the response type is not correct - this is just for testing
          Promise.resolve({
            response: new Response(),
            options: {},
            controller,
          }),
          () => stream,
        );
      },
    );
    const stream = await openai.responses.create({
      input: "What's the weather in Boston?",
      model: "gpt-4.1",
      tools: [
        {
          name: "get_weather",
          type: "function",
          parameters: {
            type: "object",
            properties: {
              location: { type: "string" },
            },
            additionalProperties: false,
            required: ["location"],
          },
          strict: true,
        },
      ],
      stream: true,
    });

    let functionName = "";
    let functionArgs = "";
    for await (const event of stream) {
      if (
        event.type === "response.output_item.added" &&
        event.item.type === "function_call"
      ) {
        if (event.item.name) {
          functionName = event.item.name;
        }
      }
      if (event.type === "response.function_call_arguments.delta") {
        functionArgs += event.delta;
      }
    }
    expect(functionName).toBe("get_weather");
    expect(functionArgs).toBe('{"location": "Boston"}');
    // Wait for the span to be exported
    await new Promise((resolve) => setTimeout(resolve));
    const spans = memoryExporter.getFinishedSpans();
    expect(spans.length).toBe(1);
    const span = spans[0];
    expect(span.name).toBe("OpenAI Responses");
    expect(span.attributes).toMatchInlineSnapshot(`
{
  "input.mime_type": "application/json",
  "input.value": "{"input":"What's the weather in Boston?","model":"gpt-4.1","tools":[{"name":"get_weather","type":"function","parameters":{"type":"object","properties":{"location":{"type":"string"}},"additionalProperties":false,"required":["location"]},"strict":true}],"stream":true}",
  "llm.input_messages.0.message.content": "What's the weather in Boston?",
  "llm.input_messages.0.message.role": "user",
  "llm.invocation_parameters": "{"model":"gpt-4.1","tools":[{"name":"get_weather","type":"function","parameters":{"type":"object","properties":{"location":{"type":"string"}},"additionalProperties":false,"required":["location"]},"strict":true}],"stream":true}",
  "llm.model_name": "gpt-4.1",
  "llm.output_messages.0.message.role": "assistant",
  "llm.output_messages.0.message.tool_calls.0.tool_call.function.arguments": "{"location":"boston"}",
  "llm.output_messages.0.message.tool_calls.0.tool_call.function.name": "get_weather",
  "llm.output_messages.0.message.tool_calls.0.tool_call.id": "call_abc123",
  "llm.provider": "openai",
  "llm.system": "openai",
  "llm.token_count.total": 101,
  "llm.tools.0.tool.json_schema": "{"name":"get_weather","type":"function","parameters":{"type":"object","properties":{"location":{"type":"string"}},"additionalProperties":false,"required":["location"]},"strict":true}",
  "openinference.span.kind": "LLM",
  "output.mime_type": "application/json",
  "output.value": "{"id":"resp-890","object":"response","created":1705535755,"model":"gpt-4.1","output_text":null,"output":[{"id":"fc_1234","type":"function_call","status":"completed","arguments":"{\\"location\\":\\"boston\\"}","call_id":"call_abc123","name":"get_weather"}],"tools":[{"type":"function","description":null,"name":"get_weather","parameters":{"type":"object","properties":{"location":{"type":"string"}},"additionalProperties":false,"required":["location"]},"strict":true}],"usage":{"prompt_tokens":86,"completion_tokens":15,"total_tokens":101}}",
}
`);
  });

  it("should capture tool calls, instructions, and multiple messages with streaming responses", async () => {
    // Mock out the post endpoint to return a stream with tool calls
    vi.spyOn(openai, "post").mockImplementation(
      // @ts-expect-error the response type is not correct - this is just for testing
      (): Promise<unknown> => {
        const iterator = () =>
          (async function* () {
            yield {
              type: "response.output_item.added",
              item: {
                type: "function_call",
                id: "call_abc123",
                name: "get_weather",
                arguments: "",
              },
            };
            yield {
              type: "response.function_call_arguments.delta",
              item_id: "call_abc123",
              delta: '{"locati',
            };
            yield {
              type: "response.function_call_arguments.delta",
              item_id: "call_abc123",
              delta: 'on": "Boston"}',
            };
            yield {
              type: "response.completed",
              response: {
                id: "resp-890",
                object: "response",
                created: 1705535755,
                model: "gpt-4.1",
                output_text: null,
                output: [
                  {
                    id: "fc_1234",
                    type: "function_call",
                    status: "completed",
                    arguments: '{"location":"boston"}',
                    call_id: "call_abc123",
                    name: "get_weather",
                  },
                ],
                tools: [
                  {
                    type: "function",
                    description: null,
                    name: "get_weather",
                    parameters: {
                      type: "object",
                      properties: { location: { type: "string" } },
                      additionalProperties: false,
                      required: ["location"],
                    },
                    strict: true,
                  },
                ],
                usage: {
                  prompt_tokens: 86,
                  completion_tokens: 15,
                  total_tokens: 101,
                },
              },
            };
          })();
        const controller = new AbortController();
        const stream = new Stream(iterator, controller);
        return new APIPromise(
          new OpenAI({ apiKey: "fake-api-key" }),
          // @ts-expect-error the response type is not correct - this is just for testing
          Promise.resolve({
            response: new Response(),
            options: {},
            controller,
          }),
          () => stream,
        );
      },
    );
    const stream = await openai.responses.create({
      instructions: "You are a helpful weather assistant.",
      input: [
        {
          type: "message",
          content: "What's the weather in Boston?",
          role: "user",
        },
      ],
      model: "gpt-4.1",
      tools: [
        {
          name: "get_weather",
          type: "function",
          parameters: {
            type: "object",
            properties: {
              location: { type: "string" },
            },
            additionalProperties: false,
            required: ["location"],
          },
          strict: true,
        },
      ],
      stream: true,
    });

    let functionName = "";
    let functionArgs = "";
    for await (const event of stream) {
      if (
        event.type === "response.output_item.added" &&
        event.item.type === "function_call"
      ) {
        if (event.item.name) {
          functionName = event.item.name;
        }
      }
      if (event.type === "response.function_call_arguments.delta") {
        functionArgs += event.delta;
      }
    }
    expect(functionName).toBe("get_weather");
    expect(functionArgs).toBe('{"location": "Boston"}');
    // Wait for the span to be exported
    await new Promise((resolve) => setTimeout(resolve));
    const spans = memoryExporter.getFinishedSpans();
    expect(spans.length).toBe(1);
    const span = spans[0];
    expect(span.name).toBe("OpenAI Responses");
    expect(span.attributes).toMatchInlineSnapshot(`
{
  "input.mime_type": "application/json",
  "input.value": "{"instructions":"You are a helpful weather assistant.","input":[{"type":"message","content":"What's the weather in Boston?","role":"user"}],"model":"gpt-4.1","tools":[{"name":"get_weather","type":"function","parameters":{"type":"object","properties":{"location":{"type":"string"}},"additionalProperties":false,"required":["location"]},"strict":true}],"stream":true}",
  "llm.input_messages.0.message.content": "You are a helpful weather assistant.",
  "llm.input_messages.0.message.role": "system",
  "llm.input_messages.1.message.content": "What's the weather in Boston?",
  "llm.input_messages.1.message.role": "user",
  "llm.invocation_parameters": "{"instructions":"You are a helpful weather assistant.","model":"gpt-4.1","tools":[{"name":"get_weather","type":"function","parameters":{"type":"object","properties":{"location":{"type":"string"}},"additionalProperties":false,"required":["location"]},"strict":true}],"stream":true}",
  "llm.model_name": "gpt-4.1",
  "llm.output_messages.0.message.role": "assistant",
  "llm.output_messages.0.message.tool_calls.0.tool_call.function.arguments": "{"location":"boston"}",
  "llm.output_messages.0.message.tool_calls.0.tool_call.function.name": "get_weather",
  "llm.output_messages.0.message.tool_calls.0.tool_call.id": "call_abc123",
  "llm.provider": "openai",
  "llm.system": "openai",
  "llm.token_count.total": 101,
  "llm.tools.0.tool.json_schema": "{"name":"get_weather","type":"function","parameters":{"type":"object","properties":{"location":{"type":"string"}},"additionalProperties":false,"required":["location"]},"strict":true}",
  "openinference.span.kind": "LLM",
  "output.mime_type": "application/json",
  "output.value": "{"id":"resp-890","object":"response","created":1705535755,"model":"gpt-4.1","output_text":null,"output":[{"id":"fc_1234","type":"function_call","status":"completed","arguments":"{\\"location\\":\\"boston\\"}","call_id":"call_abc123","name":"get_weather"}],"tools":[{"type":"function","description":null,"name":"get_weather","parameters":{"type":"object","properties":{"location":{"type":"string"}},"additionalProperties":false,"required":["location"]},"strict":true}],"usage":{"prompt_tokens":86,"completion_tokens":15,"total_tokens":101}}",
}
`);
  });

  it("should handle structured outputs", async () => {
    const response = {
      id: "resp-890",
      object: "response",
      created: 1705535755,
      model: "gpt-4.1",
      output: [
        {
          type: "message",
          status: "completed",
          content: [
            {
              type: "output_text",
              text: '{"name":"science fair","date":"Friday","participants":["Alice","Bob"]}',
            },
          ],
        },
      ],
    };
    vi.spyOn(openai, "post").mockImplementation(() => {
      return new APIPromise(
        new OpenAI({ apiKey: "fake-api-key" }),
        new Promise((resolve) => {
          resolve({
            response: new Response(JSON.stringify(response)),
            // @ts-expect-error the response type is not correct - this is just for testing
            options: {},
            controller: new AbortController(),
          });
        }),
        () => response,
      );
    });
    const CalendarEvent = z.object({
      name: z.string(),
      date: z.string(),
      participants: z.array(z.string()),
    });
    const parsed = await openai.responses.parse({
      input: [
        { role: "system", content: "Extract the event information." },
        {
          role: "user",
          content: "Alice and Bob are going to a science fair on Friday.",
        },
      ],
      model: "gpt-4.1",
      text: {
        format: zodTextFormat(CalendarEvent, "event"),
      },
    });
    expect(parsed.output_parsed).toMatchInlineSnapshot(`
{
  "date": "Friday",
  "name": "science fair",
  "participants": [
    "Alice",
    "Bob",
  ],
}
`);
  });
});
