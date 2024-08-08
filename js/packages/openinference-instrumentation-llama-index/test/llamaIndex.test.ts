import * as llamaindex from "llamaindex";

import {
  InMemorySpanExporter,
  SimpleSpanProcessor,
} from "@opentelemetry/sdk-trace-base";
import { LlamaIndexInstrumentation, isPatched } from "../src/index";
import { NodeTracerProvider } from "@opentelemetry/sdk-trace-node";
import {
  SemanticConventions,
  OpenInferenceSpanKind,
  RETRIEVAL_DOCUMENTS,
} from "@arizeai/openinference-semantic-conventions";
import {
  Document,
  VectorStoreIndex,
  GeminiEmbedding,
  HuggingFaceEmbedding,
  MistralAIEmbedding,
  OllamaEmbedding,
  OpenAIEmbedding,
} from "llamaindex";
import { isEmbedding } from "../src/utils";
import { OpenAI } from "openai";

// Mocked return values
const DUMMY_RESPONSE = "lorem ipsum";
const FAKE_EMBEDDING = Array(1536).fill(0); // Mock out the embeddings response to size 1536 (ada-2)
const RESPONSE_MESSAGE = {
  content: DUMMY_RESPONSE,
  role: "assistant",
};

const EMBEDDING_RESPONSE = {
  object: "list",
  data: [
    { object: "embedding", index: 0, embedding: FAKE_EMBEDDING },
    { object: "embedding", index: 0, embedding: FAKE_EMBEDDING },
    { object: "embedding", index: 0, embedding: FAKE_EMBEDDING },
  ],
};

// Mock out the chat completions endpoint
const CHAT_RESPONSE = {
  id: "chatcmpl-8adq9JloOzNZ9TyuzrKyLpGXexh6p",
  object: "chat.completion",
  created: 1703743645,
  model: "gpt-3.5-turbo-0613",
  choices: [
    {
      index: 0,
      message: RESPONSE_MESSAGE,
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

const tracerProvider = new NodeTracerProvider();
tracerProvider.register();

const instrumentation = new LlamaIndexInstrumentation();
instrumentation.disable();

/*
 * Tests for embeddings patching
 */
describe("LlamaIndexInstrumentation - Embeddings", () => {
  const memoryExporter = new InMemorySpanExporter();
  const spanProcessor = new SimpleSpanProcessor(memoryExporter);
  instrumentation.setTracerProvider(tracerProvider);

  tracerProvider.addSpanProcessor(spanProcessor);
  // @ts-expect-error the moduleExports property is private. This is needed to make the test work with auto-mocking
  instrumentation._modules[0].moduleExports = llamaindex;

  beforeAll(() => {
    instrumentation.enable();
    process.env["OPENAI_API_KEY"] = "fake-api-key";
    process.env["MISTRAL_API_KEY"] = "fake-api-key";
    process.env["GOOGLE_API_KEY"] = "fake-api-key";
  });

  afterAll(() => {
    instrumentation.disable();
    delete process.env["OPENAI_API_KEY"];
    delete process.env["MISTRAL_API_KEY"];
    delete process.env["GOOGLE_API_KEY"];
  });
  beforeEach(() => {
    jest.spyOn(OpenAI.Embeddings.prototype, "create").mockImplementation(
      // @ts-expect-error the response type is not correct - this is just for testing
      async (): Promise<unknown> => {
        return EMBEDDING_RESPONSE;
      },
    );

    memoryExporter.reset();
  });
  afterEach(() => {
    jest.clearAllMocks();
    jest.restoreAllMocks();
    jest.resetModules();
  });

  it("is patched", () => {
    expect(isPatched()).toBe(true);
  });

  it("isEmbeddings should identify embedders correctly", async () => {
    expect(isEmbedding(new HuggingFaceEmbedding())).toEqual(true);
    expect(isEmbedding(new GeminiEmbedding())).toEqual(true);
    expect(isEmbedding(new MistralAIEmbedding())).toEqual(true);
    expect(isEmbedding(new OpenAIEmbedding())).toEqual(true);
    expect(isEmbedding(new OllamaEmbedding({ model: "test" }))).toEqual(true);

    expect(isEmbedding(new llamaindex.MistralAI())).toEqual(false);
    expect(isEmbedding(llamaindex.MistralAI)).toEqual(false);
    expect(isEmbedding(new llamaindex.Gemini())).toEqual(false);

    expect(isEmbedding(new llamaindex.TextNode())).toEqual(false);
    expect(isEmbedding(llamaindex.TextNode)).toEqual(false);
    expect(isEmbedding(new llamaindex.CorrectnessEvaluator())).toEqual(false);
  });

  it("should create a span for embeddings (query)", async () => {
    // Get embeddings
    const embedder = new OpenAIEmbedding();
    const embeddedVector = await embedder.getQueryEmbedding(
      "What did the author do in college?",
    );

    const spans = memoryExporter.getFinishedSpans();
    expect(spans.length).toBeGreaterThan(0);

    // Expect a span for the embedding
    const queryEmbeddingSpan = spans.find((span) =>
      span.name.includes("embedding"),
    );
    expect(queryEmbeddingSpan).toBeDefined();

    // Verify span attributes
    expect(
      queryEmbeddingSpan?.attributes[
        SemanticConventions.OPENINFERENCE_SPAN_KIND
      ],
    ).toEqual(OpenInferenceSpanKind.EMBEDDING);
    expect(
      queryEmbeddingSpan?.attributes[
        `${SemanticConventions.EMBEDDING_EMBEDDINGS}.0.${SemanticConventions.EMBEDDING_TEXT}`
      ],
    ).toEqual("What did the author do in college?");
    expect(
      queryEmbeddingSpan?.attributes[
        `${SemanticConventions.EMBEDDING_EMBEDDINGS}.0.${SemanticConventions.EMBEDDING_VECTOR}`
      ],
    ).toEqual(embeddedVector);
    expect(
      queryEmbeddingSpan?.attributes[SemanticConventions.EMBEDDING_MODEL_NAME],
    ).toEqual("text-embedding-ada-002");
  });
});

describe("LlamaIndexInstrumentation", () => {
  const memoryExporter = new InMemorySpanExporter();
  const spanProcessor = new SimpleSpanProcessor(memoryExporter);
  instrumentation.setTracerProvider(tracerProvider);

  tracerProvider.addSpanProcessor(spanProcessor);
  // @ts-expect-error the moduleExports property is private. This is needed to make the test work with auto-mocking
  instrumentation._modules[0].moduleExports = llamaindex;

  beforeAll(() => {
    instrumentation.enable();

    // Setup the OPENAI_API_KEY
    process.env["OPENAI_API_KEY"] = "fake-api-key";
  });
  afterAll(() => {
    instrumentation.disable();
    // Delete the OPENAI_API_KEY
    delete process.env["OPENAI_API_KEY"];
  });
  beforeEach(() => {
    memoryExporter.reset();

    jest.spyOn(OpenAI.Chat.Completions.prototype, "create").mockImplementation(
      // @ts-expect-error the response type is not correct - this is just for testing
      async (): Promise<unknown> => {
        return CHAT_RESPONSE;
      },
    );
    jest.spyOn(OpenAI.Embeddings.prototype, "create").mockImplementation(
      // @ts-expect-error the response type is not correct - this is just for testing
      async (): Promise<unknown> => {
        return EMBEDDING_RESPONSE;
      },
    );
  });
  afterEach(() => {
    jest.clearAllMocks();
    jest.restoreAllMocks();
    jest.resetModules();
  });

  it("is patched", () => {
    expect(isPatched()).toBe(true);
  });

  it("should create a span for query engines", async () => {
    // Create Document object with essay
    const document = new Document({ text: "lorem ipsum" });

    // Split text and create embeddings. Store them in a VectorStoreIndex
    const index = await VectorStoreIndex.fromDocuments([document]);

    // Query the index
    const queryEngine = index.asQueryEngine();
    const response = await queryEngine.query({
      query: "What did the author do in college?",
    });

    // Output response
    expect(response.response).toEqual(RESPONSE_MESSAGE.content);

    const spans = memoryExporter.getFinishedSpans();
    expect(spans.length).toBeGreaterThan(0);

    // Expect a span for the query engine
    const queryEngineSpan = spans.find((span) => span.name.includes("query"));
    expect(queryEngineSpan).toBeDefined();

    // Verify query span attributes
    expect(
      queryEngineSpan?.attributes[SemanticConventions.OPENINFERENCE_SPAN_KIND],
    ).toEqual(OpenInferenceSpanKind.CHAIN);
    expect(
      queryEngineSpan?.attributes[SemanticConventions.INPUT_VALUE],
    ).toEqual("What did the author do in college?");
    expect(
      queryEngineSpan?.attributes[SemanticConventions.OUTPUT_VALUE],
    ).toEqual(RESPONSE_MESSAGE.content);
  });

  it("should create a span for retrieve method", async () => {
    // Create Document objects with essays
    const documents = [
      new Document({ text: "lorem ipsum 1" }),
      new Document({ text: "lorem ipsum 2" }),
      new Document({ text: "lorem ipsum 3" }),
    ];

    // Split text and create embeddings. Store them in a VectorStoreIndex
    const index = await VectorStoreIndex.fromDocuments(documents);

    // Retrieve documents from the index
    const retriever = index.asRetriever();

    const response = await retriever.retrieve({
      query: "What did the author do in college?",
    });

    const spans = memoryExporter.getFinishedSpans();
    expect(spans.length).toBeGreaterThan(0);

    // Expect a span for the retrieve method
    const retrievalSpan = spans.find((span) => span.name.includes("retrieve"));
    expect(retrievalSpan).toBeDefined();

    // Verify query span attributes
    expect(
      retrievalSpan?.attributes[SemanticConventions.OPENINFERENCE_SPAN_KIND],
    ).toEqual(OpenInferenceSpanKind.RETRIEVER);
    expect(retrievalSpan?.attributes[SemanticConventions.INPUT_VALUE]).toEqual(
      "What did the author do in college?",
    );

    // Check document attributes
    response.forEach((document, index) => {
      const { node, score } = document;

      if (node instanceof llamaindex.TextNode) {
        const nodeId = node.id_;
        const nodeText = node.getContent();
        const nodeMetadata = node.metadata;

        expect(
          retrievalSpan?.attributes[
            `${RETRIEVAL_DOCUMENTS}.${index}.document.id`
          ],
        ).toEqual(nodeId);
        expect(
          retrievalSpan?.attributes[
            `${RETRIEVAL_DOCUMENTS}.${index}.document.score`
          ],
        ).toEqual(score);
        expect(
          retrievalSpan?.attributes[
            `${RETRIEVAL_DOCUMENTS}.${index}.document.content`
          ],
        ).toEqual(nodeText);
        expect(
          retrievalSpan?.attributes[
            `${RETRIEVAL_DOCUMENTS}.${index}.document.metadata`
          ],
        ).toEqual(JSON.stringify(nodeMetadata));
      }
    });
  });
});
