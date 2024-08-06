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

const { Document, VectorStoreIndex, OpenAIEmbedding } = llamaindex;

const DUMMY_RESPONSE = "lorem ipsum";
// Mock out the embeddings response to size 1536 (ada-2)
const FAKE_EMBEDDING = Array(1536).fill(0);

const tracerProvider = new NodeTracerProvider();
tracerProvider.register();

const instrumentation = new LlamaIndexInstrumentation();
instrumentation.disable();
describe("llamaIndex", () => {
  it("llamaindex should be available", () => {
    expect(true).toBe(true);
  });
});

describe("LlamaIndexInstrumentation", () => {
  const memoryExporter = new InMemorySpanExporter();
  const spanProcessor = new SimpleSpanProcessor(memoryExporter);
  instrumentation.setTracerProvider(tracerProvider);

  tracerProvider.addSpanProcessor(spanProcessor);
  // @ts-expect-error the moduleExports property is private. This is needed to make the test work with auto-mocking
  instrumentation._modules[0].moduleExports = llamaindex;

  let openAISpy: jest.SpyInstance;
  let openAITextEmbedSpy: jest.SpyInstance;
  let openAIQueryEmbedSpy: jest.SpyInstance;

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

    // Use OpenAI and mock out the calls
    const response: llamaindex.ChatResponse<llamaindex.ToolCallLLMMessageOptions> =
      {
        message: {
          content: DUMMY_RESPONSE,
          role: "assistant",
        },
        raw: null,
      };
    // Mock out the chat completions endpoint
    openAISpy = jest
      .spyOn(llamaindex.OpenAI.prototype, "chat")
      .mockImplementation(() => {
        return Promise.resolve(response);
      });

    // Mock out the embeddings endpoint
    openAITextEmbedSpy = jest
      .spyOn(llamaindex.OpenAIEmbedding.prototype, "getTextEmbeddings")
      .mockImplementation(() => {
        return Promise.resolve([
          FAKE_EMBEDDING,
          FAKE_EMBEDDING,
          FAKE_EMBEDDING,
        ]);
      });

    openAIQueryEmbedSpy = jest
      .spyOn(llamaindex.OpenAIEmbedding.prototype, "getQueryEmbedding")
      .mockImplementation(() => {
        return Promise.resolve(FAKE_EMBEDDING);
      });
  });
  afterEach(() => {
    jest.clearAllMocks();
    openAISpy.mockRestore();
    openAIQueryEmbedSpy.mockRestore();
    openAITextEmbedSpy.mockRestore();
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

    // Verify that the OpenAI chat method was called once during synthesis
    expect(openAISpy).toHaveBeenCalledTimes(1);
    expect(openAIQueryEmbedSpy).toHaveBeenCalledTimes(1);
    expect(openAITextEmbedSpy).toHaveBeenCalledTimes(1);

    // Output response
    expect(response.response).toEqual(DUMMY_RESPONSE);

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
    ).toEqual(DUMMY_RESPONSE);
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

    // OpenAI Chat method should not be called
    expect(openAISpy).toHaveBeenCalledTimes(0);
    expect(openAIQueryEmbedSpy).toHaveBeenCalledTimes(1);
    expect(openAITextEmbedSpy).toHaveBeenCalledTimes(1);

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

/*
 * Tests for embeddings only
 */
describe("LlamaIndexInstrumentation - Embeddings", () => {
  const memoryExporter = new InMemorySpanExporter();
  const spanProcessor = new SimpleSpanProcessor(memoryExporter);
  instrumentation.setTracerProvider(tracerProvider);

  tracerProvider.addSpanProcessor(spanProcessor);
  // @ts-expect-error the moduleExports property is private. This is needed to make the test work with auto-mocking
  instrumentation._modules[0].moduleExports = llamaindex;

  let openAIQueryEmbedSpy: jest.SpyInstance;
  let openAITextEmbedSpy: jest.SpyInstance;
  let openAISpy: jest.SpyInstance;

  jest.mock("llamaindex", () => {
    const originalModule = jest.requireActual("llamaindex");
    return {
      ...originalModule,
      OpenAIEmbedding: jest.fn().mockImplementation(() => ({
        getQueryEmbedding:
          originalModule.OpenAIEmbedding.prototype.getQueryEmbedding,
      })),
    };
  });

  beforeAll(() => {
    instrumentation.enable();
  });
  afterAll(() => {
    instrumentation.disable();
  });
  beforeEach(() => {
    memoryExporter.reset();

    // Use OpenAI and mock out the calls
    const response: llamaindex.ChatResponse<llamaindex.ToolCallLLMMessageOptions> =
      {
        message: {
          content: DUMMY_RESPONSE,
          role: "assistant",
        },
        raw: null,
      };

    // Mock out the chat completions endpoint
    openAISpy = jest
      .spyOn(llamaindex.OpenAI.prototype, "chat")
      .mockImplementation(() => {
        return Promise.resolve(response);
      });

    // Mock out the embeddings endpoint
    openAITextEmbedSpy = jest
      .spyOn(llamaindex.OpenAIEmbedding.prototype, "getTextEmbeddings")
      .mockImplementation(() => {
        return Promise.resolve([
          FAKE_EMBEDDING,
          FAKE_EMBEDDING,
          FAKE_EMBEDDING,
        ]);
      });

    openAIQueryEmbedSpy = jest
      .spyOn(llamaindex.OpenAIEmbedding.prototype, "getQueryEmbedding")
      .mockImplementation(() => {
        return Promise.resolve(FAKE_EMBEDDING);
      });
  });
  afterEach(() => {
    jest.clearAllMocks();
    openAISpy.mockRestore();
    openAITextEmbedSpy.mockRestore();
    openAIQueryEmbedSpy.mockRestore();
  });

  it("is patched", () => {
    expect(isPatched()).toBe(true);
  });

  it("should create a span for embeddings (query)", async () => {
    // Get embeddings
    const embedder = new llamaindex.OpenAIEmbedding();
    const embeddedVector = await embedder.getQueryEmbedding(
      "What did the author do in college?",
    );

    // Verify calls
    expect(openAISpy).toHaveBeenCalledTimes(0);
    expect(openAIQueryEmbedSpy).toHaveBeenCalledTimes(1);

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
    ).toEqual(OpenInferenceSpanKind.CHAIN);
    expect(
      queryEmbeddingSpan?.attributes[SemanticConventions.EMBEDDING_TEXT],
    ).toEqual("What did the author do in college?");
    expect(
      queryEmbeddingSpan?.attributes[SemanticConventions.OUTPUT_VALUE],
    ).toEqual(embeddedVector);
    expect(
      queryEmbeddingSpan?.attributes[SemanticConventions.EMBEDDING_MODEL_NAME],
    ).toEqual("text-embedding-ada-002");
  });
});
