import {
  InMemorySpanExporter,
  SimpleSpanProcessor,
} from "@opentelemetry/sdk-trace-base";
import { NodeTracerProvider } from "@opentelemetry/sdk-trace-node";
import { LangChainInstrumentation } from "../src";
import * as CallbackManager from "@langchain/core/callbacks/manager";
import { ChatPromptTemplate } from "@langchain/core/prompts";
import { MemoryVectorStore } from "langchain/vectorstores/memory";
import { createStuffDocumentsChain } from "langchain/chains/combine_documents";
import { createRetrievalChain } from "langchain/chains/retrieval";
import { ChatOpenAI, OpenAIEmbeddings } from "@langchain/openai";
import { Stream } from "openai/streaming";
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";
import "dotenv/config";
import {
  MESSAGE_FUNCTION_CALL_NAME,
  METADATA,
  OpenInferenceSpanKind,
  SemanticConventions,
} from "@arizeai/openinference-semantic-conventions";
import { LangChainTracer } from "../src/tracer";
import { trace } from "@opentelemetry/api";
import { completionsResponse, functionCallResponse } from "./fixtures";
import { DynamicTool } from "@langchain/core/tools";
import {
  OITracer,
  setAttributes,
  setSession,
} from "@arizeai/openinference-core";
import { context } from "@opentelemetry/api";
import { tool } from "@langchain/core/tools";
import { registerInstrumentations } from "@opentelemetry/instrumentation";

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

jest.mock("@langchain/openai", () => {
  const originalModule = jest.requireActual("@langchain/openai");
  class MockChatOpenAI extends originalModule.ChatOpenAI {
    constructor(...args: Parameters<typeof originalModule.ChatOpenAI>) {
      super(...args);
      this.client = {
        chat: {
          completions: {
            create: jest.fn().mockResolvedValue(completionsResponse),
          },
        },
      };
    }
  }
  return {
    ...originalModule,
    ChatOpenAI: MockChatOpenAI,
    OpenAIEmbeddings: class extends originalModule.OpenAIEmbeddings {
      embedDocuments = async () => {
        return Promise.resolve([
          [1, 2, 3],
          [4, 5, 6],
          [7, 8, 9],
        ]);
      };
      embedQuery = async () => {
        return Promise.resolve([1, 2, 4]);
      };
    },
  };
});

const expectedSpanAttributes = {
  [OPENINFERENCE_SPAN_KIND]: OpenInferenceSpanKind.LLM,
  [INPUT_VALUE]: JSON.stringify({
    messages: [
      [
        {
          lc: 1,
          type: "constructor",
          id: ["langchain_core", "messages", "HumanMessage"],
          kwargs: {
            content: "hello, this is a test",
            additional_kwargs: {},
            response_metadata: {},
          },
        },
      ],
    ],
  }),
  [INPUT_MIME_TYPE]: "application/json",
  [OUTPUT_VALUE]:
    '{"generations":[[{"text":"This is a test.","message":{"lc":1,"type":"constructor","id":["langchain_core","messages","AIMessage"],"kwargs":{"content":"This is a test.","additional_kwargs":{},"response_metadata":{"tokenUsage":{"promptTokens":12,"completionTokens":5,"totalTokens":17},"finish_reason":"stop","model_name":"gpt-3.5-turbo-0613"},"id":"chatcmpl-8adq9JloOzNZ9TyuzrKyLpGXexh6p","tool_calls":[],"invalid_tool_calls":[],"usage_metadata":{"output_tokens":5,"input_tokens":12,"total_tokens":17,"input_token_details":{},"output_token_details":{}}}},"generationInfo":{"finish_reason":"stop"}}]],"llmOutput":{"tokenUsage":{"promptTokens":12,"completionTokens":5,"totalTokens":17}}}',
  [LLM_TOKEN_COUNT_COMPLETION]: 5,
  [LLM_TOKEN_COUNT_PROMPT]: 12,
  [LLM_TOKEN_COUNT_TOTAL]: 17,
  [OUTPUT_MIME_TYPE]: "application/json",
  [`${LLM_INPUT_MESSAGES}.0.${MESSAGE_ROLE}`]: "user",
  [`${LLM_INPUT_MESSAGES}.0.${MESSAGE_CONTENT}`]: "hello, this is a test",
  [`${LLM_OUTPUT_MESSAGES}.0.${MESSAGE_ROLE}`]: "assistant",
  [`${LLM_OUTPUT_MESSAGES}.0.${MESSAGE_CONTENT}`]: "This is a test.",
  [LLM_MODEL_NAME]: "gpt-3.5-turbo",
  [LLM_INVOCATION_PARAMETERS]:
    '{"model":"gpt-3.5-turbo","temperature":0,"top_p":1,"frequency_penalty":0,"presence_penalty":0,"n":1,"stream":true,"stream_options":{"include_usage":true}}',
  metadata:
    '{"ls_provider":"openai","ls_model_name":"gpt-3.5-turbo","ls_model_type":"chat","ls_temperature":0}',
};

describe("LangChainInstrumentation", () => {
  const tracerProvider = new NodeTracerProvider();
  tracerProvider.register();
  const instrumentation = new LangChainInstrumentation();
  instrumentation.disable();

  const provider = new NodeTracerProvider();
  provider.getTracer("default");

  instrumentation.setTracerProvider(tracerProvider);
  tracerProvider.addSpanProcessor(new SimpleSpanProcessor(memoryExporter));

  const PROMPT_TEMPLATE = `Use the context below to answer the question.
  ----------------
  {context}
    
  Question:
  {input}
  `;
  const prompt = ChatPromptTemplate.fromTemplate(PROMPT_TEMPLATE);

  // @ts-expect-error the moduleExports property is private. This is needed to make the test work with auto-mocking
  instrumentation._modules[0].moduleExports = CallbackManager;
  beforeAll(() => {
    instrumentation.enable();
  });
  afterAll(() => {
    instrumentation.disable();
  });
  beforeEach(() => {
    memoryExporter.reset();
  });
  afterEach(() => {
    jest.resetAllMocks();
    jest.clearAllMocks();
  });
  it("should patch the callback manager module", async () => {
    expect(
      (CallbackManager as { openInferencePatched?: boolean })
        .openInferencePatched,
    ).toBe(true);
  });

  const testDocuments = [
    "dogs are cute",
    "rainbows are colorful",
    "water is wet",
  ];

  it("should properly nest spans", async () => {
    const chatModel = new ChatOpenAI({
      openAIApiKey: "my-api-key",
      modelName: "gpt-3.5-turbo",
    });
    const textSplitter = new RecursiveCharacterTextSplitter({
      chunkSize: 1000,
    });
    const docs = await textSplitter.createDocuments(testDocuments);
    const vectorStore = await MemoryVectorStore.fromDocuments(
      docs,
      new OpenAIEmbeddings({
        openAIApiKey: "my-api-key",
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
      openAIApiKey: "my-api-key",
      modelName: "gpt-3.5-turbo",
      temperature: 0,
    });

    await chatModel.invoke("hello, this is a test");

    const span = memoryExporter.getFinishedSpans()[0];
    expect(span).toBeDefined();

    expect(span.attributes).toStrictEqual({
      ...expectedSpanAttributes,
      [LLM_INVOCATION_PARAMETERS]:
        '{"model":"gpt-3.5-turbo","temperature":0,"top_p":1,"frequency_penalty":0,"presence_penalty":0,"n":1,"stream":false}',
    });
  });

  it("should add attributes to llm spans when streaming", async () => {
    // Do this to update the mock to return a streaming response
    // eslint-disable-next-line @typescript-eslint/no-var-requires
    const { ChatOpenAI } = jest.requireMock("@langchain/openai");

    const chatModel = new ChatOpenAI({
      openAIApiKey: "my-api-key",
      modelName: "gpt-3.5-turbo",
      streaming: true,
    });

    chatModel.client.chat.completions.create.mockResolvedValue(
      new Stream(async function* iterator() {
        yield { choices: [{ delta: { content: "This is " } }] };
        yield { choices: [{ delta: { content: "a test stream." } }] };
        yield { choices: [{ delta: { finish_reason: "stop" } }] };
      }, new AbortController()),
    );

    await chatModel.invoke("hello, this is a test");

    const span = memoryExporter.getFinishedSpans()[0];
    expect(span).toBeDefined();

    const expectedStreamingAttributes = {
      ...expectedSpanAttributes,
      [`${LLM_OUTPUT_MESSAGES}.0.${MESSAGE_CONTENT}`]: "This is a test stream.",
      [LLM_INVOCATION_PARAMETERS]:
        '{"model":"gpt-3.5-turbo","temperature":1,"top_p":1,"frequency_penalty":0,"presence_penalty":0,"n":1,"stream":true,"stream_options":{"include_usage":true}}',
      [LLM_TOKEN_COUNT_PROMPT]: 13,
      [LLM_TOKEN_COUNT_COMPLETION]: 6,
      [LLM_TOKEN_COUNT_TOTAL]: 19,
      [OUTPUT_VALUE]:
        '{"generations":[[{"text":"This is a test stream.","generationInfo":{"prompt":0,"completion":0},"message":{"lc":1,"type":"constructor","id":["langchain_core","messages","ChatMessageChunk"],"kwargs":{"content":"This is a test stream.","additional_kwargs":{},"response_metadata":{"estimatedTokenUsage":{"promptTokens":13,"completionTokens":6,"totalTokens":19},"prompt":0,"completion":0,"usage":{}}}}}]],"llmOutput":{"estimatedTokenUsage":{"promptTokens":13,"completionTokens":6,"totalTokens":19}}}',
      [METADATA]:
        '{"ls_provider":"openai","ls_model_name":"gpt-3.5-turbo","ls_model_type":"chat","ls_temperature":1}',
    };
    delete expectedStreamingAttributes[
      `${LLM_OUTPUT_MESSAGES}.0.${MESSAGE_ROLE}`
    ];

    // Remove the id since it is randomly generated and inherited from the run
    const actualAttributes = { ...span.attributes };
    const output = JSON.parse(String(actualAttributes[OUTPUT_VALUE]));
    delete output.generations[0][0].message.kwargs.id;
    const newOutputValue = JSON.stringify(output);
    actualAttributes[OUTPUT_VALUE] = newOutputValue;

    expect(actualAttributes).toStrictEqual(expectedStreamingAttributes);
  });

  it("should add documents to retriever spans", async () => {
    const chatModel = new ChatOpenAI({
      openAIApiKey: "my-api-key",
      modelName: "gpt-3.5-turbo",
    });

    const textSplitter = new RecursiveCharacterTextSplitter({
      chunkSize: 1000,
    });
    const docs = await textSplitter.createDocuments(testDocuments);
    const vectorStore = await MemoryVectorStore.fromDocuments(
      docs,
      new OpenAIEmbeddings({
        openAIApiKey: "my-api-key",
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
      openAIApiKey: "my-api-key",
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
    // Do this to update the mock to return a function call response
    // eslint-disable-next-line @typescript-eslint/no-var-requires
    const { ChatOpenAI } = jest.requireMock("@langchain/openai");

    const chatModel = new ChatOpenAI({
      openAIApiKey: "my-api-key",
      modelName: "gpt-3.5-turbo",
      temperature: 1,
    });

    chatModel.client.chat.completions.create.mockResolvedValue(
      functionCallResponse,
    );

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
        '{"model":"gpt-3.5-turbo","temperature":1,"top_p":1,"frequency_penalty":0,"presence_penalty":0,"n":1,"stream":false,"functions":[{"name":"get_current_weather","description":"Get the current weather in a given location","parameters":{"type":"object","properties":{"location":{"type":"string","description":"The city and state, e.g. San Francisco, CA"},"unit":{"type":"string","enum":["celsius","fahrenheit"]}},"required":["location"]}}]}',
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
    // eslint-disable-next-line @typescript-eslint/no-var-requires
    const { ChatOpenAI } = jest.requireMock("@langchain/openai");

    const chatModel = new ChatOpenAI({
      openAIApiKey: "my-api-key",
      modelName: "gpt-4o-mini",
      temperature: 1,
    });

    const multiply = tool(
      ({ a, b }: { a: number; b: number }): number => {
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
    expect(llmSpan?.attributes[`${LLM_TOOLS}.0.${TOOL_JSON_SCHEMA}`]).toBe(
      '{"type":"function","function":{"name":"multiply","description":"Multiply two numbers","parameters":{"type":"object","properties":{"input":{"type":"string"}},"required":["input"],"additionalProperties":false,"$schema":"http://json-schema.org/draft-07/schema#"}}}',
    );
  });

  it("should add tool information to tool spans", async () => {
    const simpleTool = new DynamicTool({
      name: "test_tool",
      description:
        "call this to get the value of a test, input should be an empty string",
      func: async () => Promise.resolve("this is a test tool"),
    });

    await simpleTool.invoke("hello");

    const spans = memoryExporter.getFinishedSpans();
    expect(spans).toBeDefined();

    const toolSpan = spans.find(
      (span) =>
        span.attributes[OPENINFERENCE_SPAN_KIND] === OpenInferenceSpanKind.TOOL,
    );
    expect(toolSpan).toBeDefined();
    expect(toolSpan?.attributes).toStrictEqual({
      [OPENINFERENCE_SPAN_KIND]: OpenInferenceSpanKind.TOOL,
      [TOOL_NAME]: "test_tool",
      [INPUT_VALUE]: "hello",
      [INPUT_MIME_TYPE]: "text/plain",
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
          openAIApiKey: "my-api-key",
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
        "llm.invocation_parameters": "{"model":"gpt-3.5-turbo","temperature":0,"top_p":1,"frequency_penalty":0,"presence_penalty":0,"n":1,"stream":false}",
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
      openAIApiKey: "my-api-key",
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
      openAIApiKey: "my-api-key",
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
      openAIApiKey: "my-api-key",
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
      openAIApiKey: "my-api-key",
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
      openAIApiKey: "my-api-key",
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
  const tracerProvider = new NodeTracerProvider();
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
  tracerProvider.addSpanProcessor(new SimpleSpanProcessor(memoryExporter));

  // @ts-expect-error the moduleExports property is private. This is needed to make the test work with auto-mocking
  instrumentation._modules[0].moduleExports = CallbackManager;
  beforeAll(() => {
    instrumentation.enable();
  });
  afterAll(() => {
    instrumentation.disable();
  });
  beforeEach(() => {
    memoryExporter.reset();
  });
  afterEach(() => {
    jest.resetAllMocks();
    jest.clearAllMocks();
  });
  it("should patch the callback manager module", async () => {
    expect(
      (CallbackManager as { openInferencePatched?: boolean })
        .openInferencePatched,
    ).toBe(true);
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
          openAIApiKey: "my-api-key",
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
    expect(span.attributes["llm.invocation_parameters"]).toBe(
      `{"model":"gpt-3.5-turbo","temperature":0,"top_p":1,"frequency_penalty":0,"presence_penalty":0,"n":1,"stream":false}`,
    );
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
    const customTracerProvider = new NodeTracerProvider();
    const customMemoryExporter = new InMemorySpanExporter();

    // Note: We don't register this provider globally.
    customTracerProvider.addSpanProcessor(
      new SimpleSpanProcessor(customMemoryExporter),
    );

    // Instantiate instrumentation with the custom provider
    const instrumentation = new LangChainInstrumentation({
      tracerProvider: customTracerProvider,
    });
    instrumentation.disable();

    // Mock the module exports like in other tests
    // @ts-expect-error the moduleExports property is private. This is needed to make the test work with auto-mocking
    instrumentation._modules[0].moduleExports = CallbackManager;

    beforeAll(() => {
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
      jest.resetAllMocks();
      jest.clearAllMocks();
    });

    it("should use the provided tracer provider instead of the global one", async () => {
      const chatModel = new ChatOpenAI({
        openAIApiKey: "my-api-key",
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
    const customTracerProvider = new NodeTracerProvider();
    const customMemoryExporter = new InMemorySpanExporter();

    // Note: We don't register this provider globally.
    customTracerProvider.addSpanProcessor(
      new SimpleSpanProcessor(customMemoryExporter),
    );

    // Instantiate instrumentation with the custom provider
    const instrumentation = new LangChainInstrumentation();
    instrumentation.setTracerProvider(customTracerProvider);
    instrumentation.disable();

    // Mock the module exports like in other tests
    // @ts-expect-error the moduleExports property is private. This is needed to make the test work with auto-mocking
    instrumentation._modules[0].moduleExports = CallbackManager;

    beforeAll(() => {
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
      jest.resetAllMocks();
      jest.clearAllMocks();
    });

    it("should use the provided tracer provider instead of the global one", async () => {
      const chatModel = new ChatOpenAI({
        openAIApiKey: "my-api-key",
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
    const customTracerProvider = new NodeTracerProvider();
    const customMemoryExporter = new InMemorySpanExporter();

    // Note: We don't register this provider globally.
    customTracerProvider.addSpanProcessor(
      new SimpleSpanProcessor(customMemoryExporter),
    );

    // Instantiate instrumentation with the custom provider
    const instrumentation = new LangChainInstrumentation();
    registerInstrumentations({
      instrumentations: [instrumentation],
      tracerProvider: customTracerProvider,
    });
    instrumentation.disable();

    // Mock the module exports like in other tests
    // @ts-expect-error the moduleExports property is private. This is needed to make the test work with auto-mocking
    instrumentation._modules[0].moduleExports = CallbackManager;

    beforeAll(() => {
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
      jest.resetAllMocks();
      jest.clearAllMocks();
    });

    it("should use the provided tracer provider instead of the global one", async () => {
      const chatModel = new ChatOpenAI({
        openAIApiKey: "my-api-key",
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
