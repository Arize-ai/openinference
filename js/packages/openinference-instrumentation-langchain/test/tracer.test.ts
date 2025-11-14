import {
  OITracer,
  setAttributes,
  setSession,
} from "@arizeai/openinference-core";
import {
  MESSAGE_FUNCTION_CALL_NAME,
  OpenInferenceSpanKind,
  SemanticConventions,
} from "@arizeai/openinference-semantic-conventions";

import { trace } from "@opentelemetry/api";
import { context } from "@opentelemetry/api";
import { registerInstrumentations } from "@opentelemetry/instrumentation";
import {
  InMemorySpanExporter,
  SimpleSpanProcessor,
} from "@opentelemetry/sdk-trace-base";
import { NodeTracerProvider } from "@opentelemetry/sdk-trace-node";

import "dotenv/config";

import { isPatched, LangChainInstrumentation } from "../src";
import { LangChainTracer } from "../src/tracer";

import { completionsResponse, functionCallResponse } from "./fixtures";

import { createStuffDocumentsChain } from "@langchain/classic/chains/combine_documents";
import { createRetrievalChain } from "@langchain/classic/chains/retrieval";
import { RecursiveCharacterTextSplitter } from "@langchain/classic/text_splitter";
import { MemoryVectorStore } from "@langchain/classic/vectorstores/memory";
import * as CallbackManager from "@langchain/core/callbacks/manager";
import { ChatPromptTemplate } from "@langchain/core/prompts";
import { ChatOpenAI, OpenAIEmbeddings } from "@langchain/openai";
import { tool } from "langchain";
import { http, HttpResponse } from "msw";
import { setupServer } from "msw/node";

// Set up MSW server to mock API calls
const server = setupServer(
  http.post(
    "https://api.openai.com/v1/chat/completions",
    async ({ request }) => {
      const body = await request.text();
      const requestData = JSON.parse(body);

      // Check if this is a streaming request
      if (requestData.stream === true) {
        // Return streaming response
        const encoder = new TextEncoder();
        const stream = new ReadableStream({
          start(controller) {
            const chunks = [
              'data: {"id":"chatcmpl-test","object":"chat.completion.chunk","created":1234567890,"model":"gpt-3.5-turbo","choices":[{"index":0,"delta":{"role":"assistant","content":"This is "},"finish_reason":null}]}\n\n',
              'data: {"id":"chatcmpl-test","object":"chat.completion.chunk","created":1234567890,"model":"gpt-3.5-turbo","choices":[{"index":0,"delta":{"content":"a test stream."},"finish_reason":null}]}\n\n',
              'data: {"id":"chatcmpl-test","object":"chat.completion.chunk","created":1234567890,"model":"gpt-3.5-turbo","choices":[{"index":0,"delta":{},"finish_reason":"stop"}],"usage":{"prompt_tokens":13,"completion_tokens":6,"total_tokens":19}}\n\n',
              "data: [DONE]\n\n",
            ];

            chunks.forEach((chunk) => {
              controller.enqueue(encoder.encode(chunk));
            });
            controller.close();
          },
        });

        return new Response(stream, {
          headers: {
            "Content-Type": "text/event-stream",
            "Cache-Control": "no-cache",
            Connection: "keep-alive",
          },
        });
      }

      // Return regular completion response
      return HttpResponse.json(completionsResponse);
    },
  ),
  http.post("https://api.openai.com/v1/embeddings", () => {
    return HttpResponse.json({
      object: "list",
      data: [
        { embedding: [1, 2, 3], index: 0 },
        { embedding: [4, 5, 6], index: 1 },
        { embedding: [7, 8, 9], index: 2 },
      ],
    });
  }),
);

// Start server before all tests
beforeAll(() => server.listen({ onUnhandledRequest: "error" }));
// Reset handlers after each test
afterEach(() => server.resetHandlers());
// Clean up after all tests
afterAll(() => server.close());

const memoryExporter = new InMemorySpanExporter();

const {
  INPUT_VALUE,
  LLM_INPUT_MESSAGES,
  OUTPUT_VALUE,
  LLM_OUTPUT_MESSAGES,
  INPUT_MIME_TYPE,
  OUTPUT_MIME_TYPE,
  MESSAGE_ROLE,
  MESSAGE_CONTENT,
  DOCUMENT_CONTENT,
  DOCUMENT_METADATA,
  OPENINFERENCE_SPAN_KIND,
  LLM_MODEL_NAME,
  LLM_INVOCATION_PARAMETERS,
  LLM_TOKEN_COUNT_COMPLETION,
  LLM_TOKEN_COUNT_PROMPT,
  LLM_TOKEN_COUNT_TOTAL,
  TOOL_NAME,
  LLM_FUNCTION_CALL,
  PROMPT_TEMPLATE_TEMPLATE,
  PROMPT_TEMPLATE_VARIABLES,
  RETRIEVAL_DOCUMENTS,
  LLM_TOOLS,
  TOOL_JSON_SCHEMA,
} = SemanticConventions;

describe("LangChainInstrumentation", () => {
  const tracerProvider = new NodeTracerProvider({
    spanProcessors: [new SimpleSpanProcessor(memoryExporter)],
  });
  tracerProvider.register();
  const instrumentation = new LangChainInstrumentation();
  instrumentation.disable();

  const provider = new NodeTracerProvider();
  provider.getTracer("default");

  instrumentation.setTracerProvider(tracerProvider);

  const PROMPT_TEMPLATE = `Use the context below to answer the question.
  ----------------
  {context}
    
  Question:
  {input}
  `;
  const prompt = ChatPromptTemplate.fromTemplate(PROMPT_TEMPLATE);

  beforeAll(() => {
    // Use manual instrumentation as intended for LangChain
    instrumentation.manuallyInstrument(CallbackManager);
    instrumentation.enable();
  });
  afterAll(() => {
    instrumentation.disable();
  });
  beforeEach(() => {
    memoryExporter.reset();
    vi.clearAllMocks();
  });
  afterEach(() => {
    vi.clearAllMocks();
    vi.restoreAllMocks();
  });
  it("should patch the callback manager module", async () => {
    // Check global patched state - this is the reliable indicator
    expect(isPatched()).toBe(true);
  });

  it("should trace an llm call", async () => {
    const chatModel = new ChatOpenAI({
      apiKey: "test-api-key",
      modelName: "gpt-3.5-turbo",
    });
    await chatModel.invoke("hello, this is a test");

    const spans = memoryExporter.getFinishedSpans();
    const llmSpan = spans.find(
      (span) =>
        span.attributes[OPENINFERENCE_SPAN_KIND] === OpenInferenceSpanKind.LLM,
    );
    expect(llmSpan).toBeDefined();
    expect(llmSpan?.attributes).toMatchObject({
      [OPENINFERENCE_SPAN_KIND]: OpenInferenceSpanKind.LLM,
      [LLM_MODEL_NAME]: "gpt-3.5-turbo",
      [`${LLM_INPUT_MESSAGES}.0.${MESSAGE_ROLE}`]: "user",
      [`${LLM_INPUT_MESSAGES}.0.${MESSAGE_CONTENT}`]: "hello, this is a test",
      [`${LLM_OUTPUT_MESSAGES}.0.${MESSAGE_ROLE}`]: "assistant",
      [`${LLM_OUTPUT_MESSAGES}.0.${MESSAGE_CONTENT}`]: "This is a test.",
      [LLM_TOKEN_COUNT_COMPLETION]: 5,
      [LLM_TOKEN_COUNT_PROMPT]: 12,
      [LLM_TOKEN_COUNT_TOTAL]: 17,
      [INPUT_MIME_TYPE]: "application/json",
      [OUTPUT_MIME_TYPE]: "application/json",
    });
  });
  const testDocuments = [
    "dogs are cute",
    "rainbows are colorful",
    "water is wet",
  ];

  it("should properly nest spans", async () => {
    const chatModel = new ChatOpenAI({
      apiKey: "test-api-key",
      modelName: "gpt-3.5-turbo",
    });
    const textSplitter = new RecursiveCharacterTextSplitter({
      chunkSize: 1000,
    });
    const docs = await textSplitter.createDocuments(testDocuments);
    const vectorStore = await MemoryVectorStore.fromDocuments(
      docs,
      new OpenAIEmbeddings({
        apiKey: "test-api-key",
      }),
    );
    const combineDocsChain = await createStuffDocumentsChain({
      llm: chatModel,
      prompt,
    });
    const chain = await createRetrievalChain({
      combineDocsChain: combineDocsChain,
      retriever: vectorStore.asRetriever(),
    });

    await chain.invoke({
      input: "What are cats?",
    });

    const spans = memoryExporter.getFinishedSpans();
    const rootSpan = spans.find((span) => span.parentSpanId == null);
    const llmSpan = spans.find(
      (span) =>
        span.attributes[SemanticConventions.OPENINFERENCE_SPAN_KIND] ===
        OpenInferenceSpanKind.LLM,
    );
    const retrieverSpan = spans.find(
      (span) =>
        span.attributes[SemanticConventions.OPENINFERENCE_SPAN_KIND] ===
        OpenInferenceSpanKind.RETRIEVER,
    );

    const retrievalChainSpan = spans.find(
      (span) => span.name === "retrieval_chain",
    );

    const retrieveDocumentsSpan = spans.find(
      (span) => span.name === "retrieve_documents",
    );

    // Langchain creates a ton of generic spans that are deeply nested. This is a simple test to ensure we have the spans we care about and they are at least nested under something. It is not possible to test the exact nesting structure because it is too complex and generic.
    expect(rootSpan).toBe(retrievalChainSpan);
    expect(retrieverSpan).toBeDefined();
    expect(llmSpan).toBeDefined();

    expect(retrieverSpan?.parentSpanId).toBe(
      retrieveDocumentsSpan?.spanContext().spanId,
    );
    expect(llmSpan?.parentSpanId).toBeDefined();
  });

  it("should add attributes to llm spans", async () => {
    const chatModel = new ChatOpenAI({
      apiKey: "test-api-key",
      modelName: "gpt-3.5-turbo",
      temperature: 0,
    });

    await chatModel.invoke("hello, this is a test");

    const spans = memoryExporter.getFinishedSpans();
    const llmSpan = spans.find(
      (span) =>
        span.attributes[OPENINFERENCE_SPAN_KIND] === OpenInferenceSpanKind.LLM,
    );
    expect(llmSpan).toBeDefined();

    expect(llmSpan?.attributes).toMatchObject({
      [OPENINFERENCE_SPAN_KIND]: OpenInferenceSpanKind.LLM,
      [LLM_MODEL_NAME]: "gpt-3.5-turbo",
      [`${LLM_INPUT_MESSAGES}.0.${MESSAGE_ROLE}`]: "user",
      [`${LLM_INPUT_MESSAGES}.0.${MESSAGE_CONTENT}`]: "hello, this is a test",
      [`${LLM_OUTPUT_MESSAGES}.0.${MESSAGE_ROLE}`]: "assistant",
      [`${LLM_OUTPUT_MESSAGES}.0.${MESSAGE_CONTENT}`]: "This is a test.",
      [LLM_TOKEN_COUNT_COMPLETION]: 5,
      [LLM_TOKEN_COUNT_PROMPT]: 12,
      [LLM_TOKEN_COUNT_TOTAL]: 17,
      [INPUT_MIME_TYPE]: "application/json",
      [OUTPUT_MIME_TYPE]: "application/json",
    });

    // Test that invocation parameters contain expected fields
    const invocationParams = JSON.parse(
      String(llmSpan?.attributes[LLM_INVOCATION_PARAMETERS]),
    );
    expect(invocationParams.model).toBe("gpt-3.5-turbo");
    expect(invocationParams.temperature).toBe(0);
  });

  it("should add attributes to llm spans when streaming", async () => {
    const chatModel = new ChatOpenAI({
      apiKey: "test-api-key",
      modelName: "gpt-3.5-turbo",
      streaming: true,
    });

    await chatModel.invoke("hello, this is a test");

    const span = memoryExporter.getFinishedSpans()[0];
    expect(span).toBeDefined();

    // Test basic structure for streaming
    expect(span.attributes).toMatchObject({
      [OPENINFERENCE_SPAN_KIND]: OpenInferenceSpanKind.LLM,
      [LLM_MODEL_NAME]: "gpt-3.5-turbo",
      [`${LLM_INPUT_MESSAGES}.0.${MESSAGE_ROLE}`]: "user",
      [`${LLM_INPUT_MESSAGES}.0.${MESSAGE_CONTENT}`]: "hello, this is a test",
      [`${LLM_OUTPUT_MESSAGES}.0.${MESSAGE_ROLE}`]: "assistant",
      [`${LLM_OUTPUT_MESSAGES}.0.${MESSAGE_CONTENT}`]: "This is a test stream.",
      [LLM_TOKEN_COUNT_COMPLETION]: 6,
      [LLM_TOKEN_COUNT_PROMPT]: 13,
      [LLM_TOKEN_COUNT_TOTAL]: 19,
      [INPUT_MIME_TYPE]: "application/json",
      [OUTPUT_MIME_TYPE]: "application/json",
    });

    // Test that invocation parameters contain streaming fields
    const invocationParams = JSON.parse(
      String(span.attributes[LLM_INVOCATION_PARAMETERS]),
    );
    expect(invocationParams.model).toBe("gpt-3.5-turbo");
    expect(invocationParams.stream).toBe(true);
    expect(invocationParams.stream_options).toBeDefined();
  });

  it("should add documents to retriever spans", async () => {
    const chatModel = new ChatOpenAI({
      apiKey: "my-api-key",
      modelName: "gpt-3.5-turbo",
    });

    const textSplitter = new RecursiveCharacterTextSplitter({
      chunkSize: 1000,
    });
    const docs = await textSplitter.createDocuments(testDocuments);
    const vectorStore = await MemoryVectorStore.fromDocuments(
      docs,
      new OpenAIEmbeddings({
        apiKey: "my-api-key",
      }),
    );
    const combineDocsChain = await createStuffDocumentsChain({
      llm: chatModel,
      prompt,
    });
    const chain = await createRetrievalChain({
      combineDocsChain: combineDocsChain,
      retriever: vectorStore.asRetriever(),
    });

    await chain.invoke({
      input: "What are cats?",
    });

    const spans = memoryExporter.getFinishedSpans();
    const retrieverSpan = spans.find(
      (span) =>
        span.attributes[SemanticConventions.OPENINFERENCE_SPAN_KIND] ===
        OpenInferenceSpanKind.RETRIEVER,
    );

    expect(retrieverSpan).toBeDefined();

    expect(retrieverSpan?.attributes).toStrictEqual({
      [OPENINFERENCE_SPAN_KIND]: OpenInferenceSpanKind.RETRIEVER,
      [OUTPUT_MIME_TYPE]: "application/json",
      [OUTPUT_VALUE]:
        '{"documents":[{"pageContent":"dogs are cute","metadata":{"loc":{"lines":{"from":1,"to":1}}}},{"pageContent":"rainbows are colorful","metadata":{"loc":{"lines":{"from":1,"to":1}}}},{"pageContent":"water is wet","metadata":{"loc":{"lines":{"from":1,"to":1}}}}]}',
      [INPUT_MIME_TYPE]: "text/plain",
      [INPUT_VALUE]: "What are cats?",
      [`${RETRIEVAL_DOCUMENTS}.0.${DOCUMENT_CONTENT}`]: "dogs are cute",
      [`${RETRIEVAL_DOCUMENTS}.0.${DOCUMENT_METADATA}`]: JSON.stringify({
        loc: {
          lines: {
            from: 1,
            to: 1,
          },
        },
      }),
      [`${RETRIEVAL_DOCUMENTS}.1.${DOCUMENT_CONTENT}`]: "rainbows are colorful",
      [`${RETRIEVAL_DOCUMENTS}.1.${DOCUMENT_METADATA}`]: JSON.stringify({
        loc: {
          lines: {
            from: 1,
            to: 1,
          },
        },
      }),
      [`${RETRIEVAL_DOCUMENTS}.2.${DOCUMENT_CONTENT}`]: "water is wet",
      [`${RETRIEVAL_DOCUMENTS}.2.${DOCUMENT_METADATA}`]: JSON.stringify({
        loc: {
          lines: {
            from: 1,
            to: 1,
          },
        },
      }),
      metadata: "{}",
    });
  });

  it("should add a prompt template to a span if found ", async () => {
    const chatModel = new ChatOpenAI({
      apiKey: "my-api-key",
      modelName: "gpt-3.5-turbo",
    });
    const chain = prompt.pipe(chatModel);
    await chain.invoke({
      context: "This is a test.",
      input: "What is this?",
    });

    const spans = memoryExporter.getFinishedSpans();
    expect(spans).toBeDefined();

    const promptSpan = spans.find((span) => span.name === "ChatPromptTemplate");

    expect(promptSpan).toBeDefined();
    expect(promptSpan?.attributes).toStrictEqual({
      [OPENINFERENCE_SPAN_KIND]: OpenInferenceSpanKind.CHAIN,
      [PROMPT_TEMPLATE_TEMPLATE]: PROMPT_TEMPLATE,
      [PROMPT_TEMPLATE_VARIABLES]: JSON.stringify({
        context: "This is a test.",
        input: "What is this?",
      }),
      [INPUT_VALUE]: '{"context":"This is a test.","input":"What is this?"}',
      [INPUT_MIME_TYPE]: "application/json",
      [OUTPUT_VALUE]:
        '{"lc":1,"type":"constructor","id":["langchain_core","prompt_values","ChatPromptValue"],"kwargs":{"messages":[{"lc":1,"type":"constructor","id":["langchain_core","messages","HumanMessage"],"kwargs":{"content":"Use the context below to answer the question.\\n  ----------------\\n  This is a test.\\n    \\n  Question:\\n  What is this?\\n  ","additional_kwargs":{},"response_metadata":{}}}]}}',
      [OUTPUT_MIME_TYPE]: "application/json",
      metadata: "{}",
    });
    setTimeout(() => {}, 10000);
  });

  it("should add function calls to spans", async () => {
    // Override the default handler for this test
    server.use(
      http.post("https://api.openai.com/v1/chat/completions", () => {
        return HttpResponse.json(functionCallResponse);
      }),
    );

    const chatModel = new ChatOpenAI({
      apiKey: "my-api-key",
      modelName: "gpt-3.5-turbo",
      temperature: 1,
    });

    const weatherFunction = {
      name: "get_current_weather",
      description: "Get the current weather in a given location",
      parameters: {
        type: "object",
        properties: {
          location: {
            type: "string",
            description: "The city and state, e.g. San Francisco, CA",
          },
          unit: { type: "string", enum: ["celsius", "fahrenheit"] },
        },
        required: ["location"],
      },
    };

    await chatModel.invoke(
      "whats the weather like in seattle, wa in fahrenheit?",
      {
        functions: [weatherFunction],
      },
    );

    const spans = memoryExporter.getFinishedSpans();
    expect(spans).toBeDefined();

    const llmSpan = spans.find(
      (span) =>
        span.attributes[OPENINFERENCE_SPAN_KIND] === OpenInferenceSpanKind.LLM,
    );
    expect(llmSpan).toBeDefined();
    // We strip out the input and output values because they are unstable
    const {
      [INPUT_VALUE]: inputValue,
      [OUTPUT_VALUE]: outputValue,
      ...attributes
    } = llmSpan?.attributes || {};
    expect(attributes).toStrictEqual({
      [OPENINFERENCE_SPAN_KIND]: OpenInferenceSpanKind.LLM,
      [LLM_MODEL_NAME]: "gpt-3.5-turbo",
      [LLM_FUNCTION_CALL]:
        '{"name":"get_current_weather","arguments":"{\\"location\\":\\"Seattle, WA\\",\\"unit\\":\\"fahrenheit\\"}"}',
      [`${LLM_INPUT_MESSAGES}.0.${MESSAGE_ROLE}`]: "user",
      [`${LLM_INPUT_MESSAGES}.0.${MESSAGE_CONTENT}`]:
        "whats the weather like in seattle, wa in fahrenheit?",
      [`${LLM_OUTPUT_MESSAGES}.0.${MESSAGE_FUNCTION_CALL_NAME}`]:
        "get_current_weather",
      [`${LLM_OUTPUT_MESSAGES}.0.${MESSAGE_CONTENT}`]: "",
      [`${LLM_OUTPUT_MESSAGES}.0.${MESSAGE_ROLE}`]: "assistant",
      [LLM_TOKEN_COUNT_COMPLETION]: 22,
      [LLM_TOKEN_COUNT_PROMPT]: 88,
      [LLM_TOKEN_COUNT_TOTAL]: 110,
      [LLM_INVOCATION_PARAMETERS]:
        '{"model":"gpt-3.5-turbo","temperature":1,"stream":false,"functions":[{"name":"get_current_weather","description":"Get the current weather in a given location","parameters":{"type":"object","properties":{"location":{"type":"string","description":"The city and state, e.g. San Francisco, CA"},"unit":{"type":"string","enum":["celsius","fahrenheit"]}},"required":["location"]}}]}',
      [INPUT_MIME_TYPE]: "application/json",
      [OUTPUT_MIME_TYPE]: "application/json",
      metadata:
        '{"ls_provider":"openai","ls_model_name":"gpt-3.5-turbo","ls_model_type":"chat","ls_temperature":1}',
    });
    // Make sure the input and output values are set
    expect(inputValue).toBeDefined();
    expect(outputValue).toBeDefined();
  });

  it("should capture tool json schema in llm spans for bound tools", async () => {
    const chatModel = new ChatOpenAI({
      apiKey: "my-api-key",
      modelName: "gpt-4o-mini",
      temperature: 1,
    });

    const multiply = tool(
      async ({ a, b }: { a: number; b: number }): Promise<number> => {
        return a * b;
      },
      {
        name: "multiply",
        description: "Multiply two numbers",
      },
    );

    const modelWithTools = chatModel.bindTools([multiply]);
    await modelWithTools.invoke("What is 2 * 3?");

    const spans = memoryExporter.getFinishedSpans();
    expect(spans).toBeDefined();

    const llmSpan = spans.find(
      (span) =>
        span.attributes[OPENINFERENCE_SPAN_KIND] === OpenInferenceSpanKind.LLM,
    );
    expect(llmSpan).toBeDefined();
    const toolSchema = JSON.parse(
      String(llmSpan?.attributes[`${LLM_TOOLS}.0.${TOOL_JSON_SCHEMA}`]),
    );
    expect(toolSchema).toMatchObject({
      type: "function",
      function: {
        name: "multiply",
        description: "Multiply two numbers",
        parameters: {
          type: "object",
          properties: {
            input: {
              type: "string",
            },
          },
          additionalProperties: false,
          $schema: "http://json-schema.org/draft-07/schema#",
        },
      },
    });
  });

  it("should add tool information to tool spans", async () => {
    const simpleTool = tool(
      async () => Promise.resolve("this is a test tool"),
      {
        name: "test_tool",
        description:
          "call this to get the value of a test, input should be an empty string",
      },
    );

    await simpleTool.invoke("hello");

    const spans = memoryExporter.getFinishedSpans();
    expect(spans).toBeDefined();

    const toolSpan = spans.find(
      (span) =>
        span.attributes[OPENINFERENCE_SPAN_KIND] === OpenInferenceSpanKind.TOOL,
    );
    expect(toolSpan).toBeDefined();
    expect(toolSpan?.attributes).toMatchObject({
      [OPENINFERENCE_SPAN_KIND]: OpenInferenceSpanKind.TOOL,
      [TOOL_NAME]: "test_tool",
      [INPUT_VALUE]: '{"input":"hello"}',
      [INPUT_MIME_TYPE]: "application/json",
      [OUTPUT_VALUE]: "this is a test tool",
      [OUTPUT_MIME_TYPE]: "text/plain",
      metadata: "{}",
    });
  });

  it("should capture context attributes and add them to spans", async () => {
    await context.with(
      setSession(
        setAttributes(context.active(), {
          "test-attribute": "test-value",
        }),
        { sessionId: "session-id" },
      ),
      async () => {
        const chatModel = new ChatOpenAI({
          apiKey: "my-api-key",
          modelName: "gpt-3.5-turbo",
          temperature: 0,
        });
        await chatModel.invoke("hello, this is a test");
      },
    );

    const spans = memoryExporter.getFinishedSpans();
    expect(spans.length).toBe(1);
    const span = spans[0];
    // Check all attributes except for the input and output values
    const {
      ["input.value"]: inputValue,
      ["output.value"]: outputValue,
      ...attributes
    } = span.attributes;
    expect(attributes).toMatchInlineSnapshot(`
      {
        "input.mime_type": "application/json",
        "llm.input_messages.0.message.content": "hello, this is a test",
        "llm.input_messages.0.message.role": "user",
        "llm.invocation_parameters": "{"model":"gpt-3.5-turbo","temperature":0,"stream":false}",
        "llm.model_name": "gpt-3.5-turbo",
        "llm.output_messages.0.message.content": "This is a test.",
        "llm.output_messages.0.message.role": "assistant",
        "llm.token_count.completion": 5,
        "llm.token_count.prompt": 12,
        "llm.token_count.total": 17,
        "metadata": "{"ls_provider":"openai","ls_model_name":"gpt-3.5-turbo","ls_model_type":"chat","ls_temperature":0}",
        "openinference.span.kind": "LLM",
        "output.mime_type": "application/json",
        "session.id": "session-id",
        "test-attribute": "test-value",
      }
    `);
    expect(inputValue).toBeDefined();
    expect(outputValue).toBeDefined();
  });

  it("should extract session ID from run metadata with session_id", async () => {
    const chatModel = new ChatOpenAI({
      apiKey: "my-api-key",
      modelName: "gpt-3.5-turbo",
    });

    await chatModel.invoke("test message", {
      metadata: {
        session_id: "test-session-123",
      },
    });

    const spans = memoryExporter.getFinishedSpans();
    expect(spans[0].attributes[SemanticConventions.SESSION_ID]).toBe(
      "test-session-123",
    );
  });

  it("should extract session ID from run metadata with thread_id", async () => {
    const chatModel = new ChatOpenAI({
      apiKey: "my-api-key",
      modelName: "gpt-3.5-turbo",
    });

    await chatModel.invoke("test message", {
      metadata: {
        thread_id: "thread-456",
      },
    });

    const spans = memoryExporter.getFinishedSpans();
    expect(spans[0].attributes[SemanticConventions.SESSION_ID]).toBe(
      "thread-456",
    );
  });

  it("should extract session ID from run metadata with conversation_id", async () => {
    const chatModel = new ChatOpenAI({
      apiKey: "my-api-key",
      modelName: "gpt-3.5-turbo",
    });

    await chatModel.invoke("test message", {
      metadata: {
        conversation_id: "conv-789",
      },
    });

    const spans = memoryExporter.getFinishedSpans();
    expect(spans[0].attributes[SemanticConventions.SESSION_ID]).toBe(
      "conv-789",
    );
  });

  it("should prioritize session_id over thread_id and conversation_id", async () => {
    const chatModel = new ChatOpenAI({
      apiKey: "my-api-key",
      modelName: "gpt-3.5-turbo",
    });

    await chatModel.invoke("test message", {
      metadata: {
        session_id: "session-123",
        thread_id: "thread-456",
        conversation_id: "conv-789",
      },
    });

    const spans = memoryExporter.getFinishedSpans();
    expect(spans[0].attributes[SemanticConventions.SESSION_ID]).toBe(
      "session-123",
    );
  });

  it("should handle missing session identifiers in metadata", async () => {
    const chatModel = new ChatOpenAI({
      apiKey: "my-api-key",
      modelName: "gpt-3.5-turbo",
    });

    await chatModel.invoke("test message", {
      metadata: {
        other_field: "some-value",
      },
    });

    const spans = memoryExporter.getFinishedSpans();
    expect(spans[0].attributes[SemanticConventions.SESSION_ID]).toBeUndefined();
  });
});

describe("LangChainInstrumentation with TraceConfigOptions", () => {
  const tracerProvider = new NodeTracerProvider({
    spanProcessors: [new SimpleSpanProcessor(memoryExporter)],
  });
  tracerProvider.register();
  const instrumentation = new LangChainInstrumentation({
    traceConfig: {
      hideInputs: true,
    },
  });
  instrumentation.disable();
  const provider = new NodeTracerProvider();
  provider.getTracer("default");
  instrumentation.setTracerProvider(tracerProvider);

  beforeAll(() => {
    // Use manual instrumentation as intended for LangChain
    instrumentation.manuallyInstrument(CallbackManager);
    instrumentation.enable();
  });
  afterAll(() => {
    instrumentation.disable();
  });
  beforeEach(() => {
    memoryExporter.reset();
    vi.clearAllMocks();
  });
  afterEach(() => {
    vi.clearAllMocks();
    vi.restoreAllMocks();
  });
  it("should patch the callback manager module", async () => {
    // Check global patched state - this is the reliable indicator
    expect(isPatched()).toBe(true);
  });

  it("should respect trace config options", async () => {
    await context.with(
      setSession(
        setAttributes(context.active(), {
          "test-attribute": "test-value",
        }),
        { sessionId: "session-id" },
      ),
      async () => {
        const chatModel = new ChatOpenAI({
          apiKey: "my-api-key",
          modelName: "gpt-3.5-turbo",
          temperature: 0,
        });
        await chatModel.invoke("hello, this is a test");
      },
    );

    const spans = memoryExporter.getFinishedSpans();
    expect(spans.length).toBe(1);
    const span = spans[0];
    expect(span.attributes["input.value"]).toBe("__REDACTED__");
    const invocationParams = JSON.parse(
      String(span.attributes["llm.invocation_parameters"]),
    );
    expect(invocationParams.model).toBe("gpt-3.5-turbo");
    expect(invocationParams.temperature).toBe(0);
    expect(span.attributes["test-attribute"]).toBe("test-value");
    expect(span.attributes["llm.model_name"]).toBe("gpt-3.5-turbo");
    expect(span.attributes["llm.output_messages.0.message.content"]).toBe(
      "This is a test.",
    );
    expect(span.attributes["llm.output_messages.0.message.role"]).toBe(
      "assistant",
    );
    expect(span.attributes["llm.token_count.completion"]).toBe(5);
    expect(span.attributes["llm.token_count.prompt"]).toBe(12);
    expect(span.attributes["llm.token_count.total"]).toBe(17);
    expect(span.attributes["metadata"]).toBe(
      `{"ls_provider":"openai","ls_model_name":"gpt-3.5-turbo","ls_model_type":"chat","ls_temperature":0}`,
    );
    expect(span.attributes["openinference.span.kind"]).toBe("LLM");
    expect(span.attributes["output.mime_type"]).toBe("application/json");
    // Output value is unstable, so we don't check it
    expect(span.attributes["session.id"]).toBe("session-id");
  });
});

describe("LangChainInstrumentation with a custom tracer provider", () => {
  describe("LangChainInstrumentation with custom TracerProvider passed in", () => {
    const customMemoryExporter = new InMemorySpanExporter();
    const customTracerProvider = new NodeTracerProvider({
      spanProcessors: [new SimpleSpanProcessor(customMemoryExporter)],
    });

    // Instantiate instrumentation with the custom provider
    const instrumentation = new LangChainInstrumentation({
      tracerProvider: customTracerProvider,
    });
    instrumentation.disable();

    beforeAll(() => {
      // Use manual instrumentation as intended for LangChain
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      instrumentation.manuallyInstrument(CallbackManager as any);
      instrumentation.enable();
    });

    afterAll(() => {
      instrumentation.disable();
    });

    beforeEach(() => {
      memoryExporter.reset();
      customMemoryExporter.reset();
    });

    afterEach(() => {
      vi.clearAllMocks();
      vi.restoreAllMocks();
    });

    it("should use the provided tracer provider instead of the global one", async () => {
      const chatModel = new ChatOpenAI({
        apiKey: "my-api-key",
        modelName: "gpt-3.5-turbo",
      });

      await chatModel.invoke("test message", {
        metadata: {
          conversation_id: "conv-789",
        },
      });

      const spans = customMemoryExporter.getFinishedSpans();
      const globalSpans = memoryExporter.getFinishedSpans();
      expect(spans.length).toBe(1);
      expect(globalSpans.length).toBe(0);
    });
  });

  describe("LangChainInstrumentation with custom TracerProvider set", () => {
    const customMemoryExporter = new InMemorySpanExporter();
    const customTracerProvider = new NodeTracerProvider({
      spanProcessors: [new SimpleSpanProcessor(customMemoryExporter)],
    });

    // Instantiate instrumentation with the custom provider
    const instrumentation = new LangChainInstrumentation();
    instrumentation.disable();

    beforeAll(() => {
      // Set tracer provider BEFORE manual instrumentation to ensure correct tracer is used
      instrumentation.setTracerProvider(customTracerProvider);
      // Use manual instrumentation as intended for LangChain
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      instrumentation.manuallyInstrument(CallbackManager as any);
      instrumentation.enable();
    });

    afterAll(() => {
      instrumentation.disable();
    });

    beforeEach(() => {
      memoryExporter.reset();
      customMemoryExporter.reset();
      vi.clearAllMocks();
    });

    afterEach(() => {
      vi.clearAllMocks();
      vi.restoreAllMocks();
    });

    it("should use the provided tracer provider instead of the global one", async () => {
      const chatModel = new ChatOpenAI({
        apiKey: "my-api-key",
        modelName: "gpt-3.5-turbo",
      });
      await chatModel.invoke("test message");

      const spans = customMemoryExporter.getFinishedSpans();
      const globalSpans = memoryExporter.getFinishedSpans();
      expect(spans.length).toBe(1);
      expect(globalSpans.length).toBe(0);
    });
  });

  describe("LangChainInstrumentation with custom TracerProvider set via registerInstrumentations", () => {
    const customMemoryExporter = new InMemorySpanExporter();
    const customTracerProvider = new NodeTracerProvider({
      spanProcessors: [new SimpleSpanProcessor(customMemoryExporter)],
    });

    // Instantiate instrumentation with the custom provider
    const instrumentation = new LangChainInstrumentation();
    registerInstrumentations({
      instrumentations: [instrumentation],
      tracerProvider: customTracerProvider,
    });
    instrumentation.disable();

    beforeAll(() => {
      // For manual instrumentation, we need to explicitly set the tracer provider
      instrumentation.setTracerProvider(customTracerProvider);
      // Use manual instrumentation as intended for LangChain
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      instrumentation.manuallyInstrument(CallbackManager as any);
      instrumentation.enable();
    });

    afterAll(() => {
      instrumentation.disable();
    });

    beforeEach(() => {
      memoryExporter.reset();
      customMemoryExporter.reset();
      vi.clearAllMocks();
    });

    afterEach(() => {
      vi.clearAllMocks();
      vi.restoreAllMocks();
    });

    it("should use the provided tracer provider instead of the global one", async () => {
      const chatModel = new ChatOpenAI({
        apiKey: "my-api-key",
        modelName: "gpt-3.5-turbo",
      });
      await chatModel.invoke("test message");

      const spans = customMemoryExporter.getFinishedSpans();
      const globalSpans = memoryExporter.getFinishedSpans();
      expect(spans.length).toBe(1);
      expect(globalSpans.length).toBe(0);
    });
  });
});

describe("LangChainTracer", () => {
  const testSerialized = {
    lc: 1,
    type: "not_implemented" as const,
    id: [],
  };
  it("should delete runs after they are ended", async () => {
    const oiTracer = new OITracer({ tracer: trace.getTracer("default") });
    const langChainTracer = new LangChainTracer(oiTracer);
    for (let i = 0; i < 10; i++) {
      await langChainTracer.handleLLMStart(testSerialized, [], "runId");
      expect(Object.keys(langChainTracer["runs"]).length).toBe(1);

      await langChainTracer.handleRetrieverStart(testSerialized, "", "runId2");
      expect(Object.keys(langChainTracer["runs"]).length).toBe(2);

      await langChainTracer.handleLLMEnd({ generations: [] }, "runId");
      expect(Object.keys(langChainTracer["runs"]).length).toBe(1);

      await langChainTracer.handleRetrieverEnd([], "runId2");
      expect(Object.keys(langChainTracer["runs"]).length).toBe(0);
    }

    expect(langChainTracer["runs"]).toBeDefined();
    expect(Object.keys(langChainTracer["runs"]).length).toBe(0);
  });
});
