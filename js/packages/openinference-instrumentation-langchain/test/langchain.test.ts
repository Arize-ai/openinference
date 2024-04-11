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
  OpenInferenceSpanKind,
  SemanticConventions,
} from "@arizeai/openinference-semantic-conventions";
import { LangChainTracer } from "../src/tracer";
import { trace } from "@opentelemetry/api";
jest.useFakeTimers();

const tracerProvider = new NodeTracerProvider();
tracerProvider.register();

const instrumentation = new LangChainInstrumentation();
instrumentation.disable();

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

jest.mock("@langchain/openai", () => {
  const originalModule = jest.requireActual("@langchain/openai");
  return {
    ...originalModule,
    ChatOpenAI: class extends originalModule.ChatOpenAI {
      client = {
        chat: {
          completions: {
            create: async () => {
              return Promise.resolve(response);
            },
          },
        },
      };
    },
    OpenAIEmbeddings: class extends originalModule.OpenAIEmbeddings {
      embedDocuments = async () => {
        return Promise.resolve([[1, 2, 3]]);
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
    jest.clearAllMocks();
  });
  it("should patch the callback manager module", async () => {
    expect(
      (CallbackManager as { openInferencePatched?: boolean })
        .openInferencePatched,
    ).toBe(true);
  });

  it("should properly nest spans", async () => {
    const chatModel = new ChatOpenAI({
      openAIApiKey: "my-api-key",
      modelName: "gpt-3.5-turbo",
    });
    const PROMPT_TEMPLATE = `Use the context below to answer the question.
    ----------------
    {context}
    `;
    const prompt = ChatPromptTemplate.fromTemplate(PROMPT_TEMPLATE);
    const text = "This is a test";
    const textSplitter = new RecursiveCharacterTextSplitter({
      chunkSize: 1000,
    });
    const docs = await textSplitter.createDocuments([text]);
    const vectorStore = await MemoryVectorStore.fromDocuments(
      docs,
      new OpenAIEmbeddings({ openAIApiKey: "my-api-key" }),
    );
    const chain = new RetrievalQAChain({
      combineDocumentsChain: loadQAStuffChain(chatModel, { prompt }),
      retriever: vectorStore.asRetriever(),
    });
    await chain.invoke({
      query: "Is this a test?",
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

  it("should add input and output messages to spans", async () => {
    const chatModel = new ChatOpenAI({
      openAIApiKey: "my-api-key",
      modelName: "gpt-3.5-turbo",
    });

    await chatModel.invoke("hello, this is a test");

    const span = memoryExporter.getFinishedSpans()[0];
    expect(span).toBeDefined();

    expect(span.attributes).toStrictEqual({
      "openinference.span.kind": "llm",
      "input.value": JSON.stringify({
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
      "input.mime_type": "application/json",
      "output.value": JSON.stringify({
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
      "output.mime_type": "application/json",
      "llm.input_messages.0.message.role": "user",
      "llm.input_messages.0.message.content": "hello, this is a test",
      "llm.output_messages.0.message.role": "assistant",
      "llm.output_messages.0.message.content": "This is a test.",
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
