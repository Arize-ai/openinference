import {
  InMemorySpanExporter,
  SimpleSpanProcessor,
} from "@opentelemetry/sdk-trace-base";
import { NodeTracerProvider } from "@opentelemetry/sdk-trace-node";
import { LangChainInstrumentation } from "../src";
import * as CallbackManager from "@langchain/core/callbacks/manager";
import { ChatPromptTemplate } from "@langchain/core/prompts";
import { MemoryVectorStore } from "langchain/vectorstores/memory";
import { ChatOpenAI, OpenAIEmbeddings } from "@langchain/openai";
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";
import { loadQAStuffChain, RetrievalQAChain } from "langchain/chains";
import "dotenv/config";
import {
  MESSAGE_FUNCTION_CALL_NAME,
  OpenInferenceSpanKind,
  RetrievalAttributePostfixes,
  SemanticAttributePrefixes,
  SemanticConventions,
} from "@arizeai/openinference-semantic-conventions";
import { LangChainTracer } from "../src/tracer";
import { trace } from "@opentelemetry/api";
import {
  LLM_FUNCTION_CALL,
  PROMPT_TEMPLATE_TEMPLATE,
  PROMPT_TEMPLATE_VARIABLES,
  TOOL_NAME,
} from "../src/constants";

import { completionsResponse, functionCallResponse } from "./fixtures";
import { DynamicTool } from "@langchain/core/tools";
jest.useFakeTimers();

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
} = SemanticConventions;
const RETRIEVAL_DOCUMENTS = `${SemanticAttributePrefixes.retrieval}.${RetrievalAttributePostfixes.documents}`;

const tracerProvider = new NodeTracerProvider();
tracerProvider.register();

const instrumentation = new LangChainInstrumentation();
instrumentation.disable();

jest.mock("@langchain/openai", () => {
  const originalModule = jest.requireActual("@langchain/openai");
  class MockChatOpenAI extends originalModule.ChatOpenAI {
    constructor() {
      super();
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

describe("LangChainInstrumentation", () => {
  const memoryExporter = new InMemorySpanExporter();
  const provider = new NodeTracerProvider();
  provider.getTracer("default");

  instrumentation.setTracerProvider(tracerProvider);
  tracerProvider.addSpanProcessor(new SimpleSpanProcessor(memoryExporter));

  const PROMPT_TEMPLATE = `Use the context below to answer the question.
  ----------------
  {context}
  
  Question:
  {question}
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
    const chain = new RetrievalQAChain({
      combineDocumentsChain: loadQAStuffChain(chatModel, { prompt }),
      retriever: vectorStore.asRetriever(),
    });
    await chain.invoke({
      query: "What are cats?",
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

    const stuffDocSpan = spans.find(
      (span) => span.name === "StuffDocumentsChain",
    );
    const llmChainSpan = spans.find((span) => span.name === "LLMChain");

    expect(rootSpan).toBeDefined();
    expect(retrieverSpan).toBeDefined();
    expect(llmSpan).toBeDefined();
    expect(stuffDocSpan).toBeDefined();
    expect(llmChainSpan).toBeDefined();

    expect(retrieverSpan?.parentSpanId).toBe(rootSpan?.spanContext().spanId);
    expect(stuffDocSpan?.parentSpanId).toBe(rootSpan?.spanContext().spanId);
    expect(llmChainSpan?.parentSpanId).toBe(stuffDocSpan?.spanContext().spanId);
    expect(llmSpan?.parentSpanId).toBe(llmChainSpan?.spanContext().spanId);
  });

  it("should add attributes to llm spans", async () => {
    const chatModel = new ChatOpenAI({
      openAIApiKey: "my-api-key",
      modelName: "gpt-3.5-turbo",
    });

    await chatModel.invoke("hello, this is a test");

    const span = memoryExporter.getFinishedSpans()[0];
    expect(span).toBeDefined();

    expect(span.attributes).toStrictEqual({
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
      [OUTPUT_VALUE]: JSON.stringify({
        generations: [
          [
            {
              text: "This is a test.",
              message: {
                lc: 1,
                type: "constructor",
                id: ["langchain_core", "messages", "AIMessage"],
                kwargs: {
                  content: "This is a test.",
                  additional_kwargs: {},
                  response_metadata: {
                    tokenUsage: {
                      completionTokens: 5,
                      promptTokens: 12,
                      totalTokens: 17,
                    },
                    finish_reason: "stop",
                  },
                },
              },
              generationInfo: { finish_reason: "stop" },
            },
          ],
        ],
        llmOutput: {
          tokenUsage: {
            completionTokens: 5,
            promptTokens: 12,
            totalTokens: 17,
          },
        },
      }),
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
        '{"model":"gpt-3.5-turbo","temperature":1,"top_p":1,"frequency_penalty":0,"presence_penalty":0,"n":1,"stream":false}',
    });
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
    const chain = new RetrievalQAChain({
      combineDocumentsChain: loadQAStuffChain(chatModel, { prompt }),
      retriever: vectorStore.asRetriever(),
    });
    await chain.invoke({
      query: "What are cats?",
    });

    const spans = memoryExporter.getFinishedSpans();
    const retrieverSpan = spans.find(
      (span) =>
        span.attributes[SemanticConventions.OPENINFERENCE_SPAN_KIND] ===
        OpenInferenceSpanKind.RETRIEVER,
    );
    const stuffDocSpan = spans.find(
      (span) => span.name === "StuffDocumentsChain",
    );
    expect(retrieverSpan).toBeDefined();
    expect(stuffDocSpan).toBeDefined();
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
    });

    expect(stuffDocSpan?.attributes).toStrictEqual({
      [OPENINFERENCE_SPAN_KIND]: OpenInferenceSpanKind.CHAIN,
      [OUTPUT_MIME_TYPE]: "text/plain",
      [OUTPUT_VALUE]: "This is a test.",
      [INPUT_MIME_TYPE]: "application/json",
      [INPUT_VALUE]:
        '{"question":"What are cats?","input_documents":[{"pageContent":"dogs are cute","metadata":{"loc":{"lines":{"from":1,"to":1}}}},{"pageContent":"rainbows are colorful","metadata":{"loc":{"lines":{"from":1,"to":1}}}},{"pageContent":"water is wet","metadata":{"loc":{"lines":{"from":1,"to":1}}}}],"query":"What are cats?"}',
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
      question: "What is this?",
    });

    const spans = memoryExporter.getFinishedSpans();
    expect(spans).toBeDefined();

    const promptSpan = spans.find((span) => span.name === "ChatPromptTemplate");

    expect(promptSpan).toBeDefined();
    expect(promptSpan?.attributes).toStrictEqual({
      [OPENINFERENCE_SPAN_KIND]: "UNKNOWN",
      [PROMPT_TEMPLATE_TEMPLATE]: PROMPT_TEMPLATE,
      [PROMPT_TEMPLATE_VARIABLES]: JSON.stringify({
        context: "This is a test.",
        question: "What is this?",
      }),
      [INPUT_VALUE]: '{"context":"This is a test.","question":"What is this?"}',
      [INPUT_MIME_TYPE]: "application/json",
      [OUTPUT_VALUE]:
        '{"lc":1,"type":"constructor","id":["langchain_core","prompt_values","ChatPromptValue"],"kwargs":{"messages":[{"lc":1,"type":"constructor","id":["langchain_core","messages","HumanMessage"],"kwargs":{"content":"Use the context below to answer the question.\\n  ----------------\\n  This is a test.\\n  \\n  Question:\\n  What is this?\\n  ","additional_kwargs":{},"response_metadata":{}}}]}}',
      [OUTPUT_MIME_TYPE]: "application/json",
    });
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
    expect(llmSpan?.attributes).toStrictEqual({
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
      [INPUT_VALUE]:
        '{"messages":[[{"lc":1,"type":"constructor","id":["langchain_core","messages","HumanMessage"],"kwargs":{"content":"whats the weather like in seattle, wa in fahrenheit?","additional_kwargs":{},"response_metadata":{}}}]]}',
      [INPUT_MIME_TYPE]: "application/json",
      [OUTPUT_VALUE]:
        '{"generations":[[{"text":"","message":{"lc":1,"type":"constructor","id":["langchain_core","messages","AIMessage"],"kwargs":{"content":"","additional_kwargs":{"function_call":{"name":"get_current_weather","arguments":"{\\"location\\":\\"Seattle, WA\\",\\"unit\\":\\"fahrenheit\\"}"}},"response_metadata":{"tokenUsage":{"completionTokens":22,"promptTokens":88,"totalTokens":110},"finish_reason":"function_call"}}},"generationInfo":{"finish_reason":"function_call"}}]],"llmOutput":{"tokenUsage":{"completionTokens":22,"promptTokens":88,"totalTokens":110}}}',
      [OUTPUT_MIME_TYPE]: "application/json",
    });
  });

  it("should add tool information to tool spans", async () => {
    const simpleTool = new DynamicTool({
      name: "test_tool",
      description:
        "call this to get the value of a test, input should be an empty string",
      func: async () => Promise.resolve("this is a test tool"),
    });

    await simpleTool.call("");

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
      [INPUT_VALUE]: "",
      [INPUT_MIME_TYPE]: "text/plain",
      [OUTPUT_VALUE]: "this is a test tool",
      [OUTPUT_MIME_TYPE]: "text/plain",
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
    const langChainTracer = new LangChainTracer(trace.getTracer("default"));
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
