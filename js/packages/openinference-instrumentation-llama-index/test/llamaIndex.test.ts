import {
  InMemorySpanExporter,
  SimpleSpanProcessor,
} from "@opentelemetry/sdk-trace-base";
import { LlamaIndexInstrumentation, isPatched } from "../src/index";
import { NodeTracerProvider } from "@opentelemetry/sdk-trace-node";
import * as llamaindex from "llamaindex";
import exp from "constants";

const { Document, VectorStoreIndex } = llamaindex;

const DUMMY_RESPONSE = "lorem ipsum";

const tracerProvider = new NodeTracerProvider();
tracerProvider.register();

const instrumentation = new LlamaIndexInstrumentation();
instrumentation.disable();
describe("llamaIndex", () => {
  it("should pass", () => {
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
  let openAIEmbedSpy: jest.SpyInstance;
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

    // Mock out the embeddings response to size 1536 (ada-2)
    const embeddingsResponse: number[][] = [Array(1536).fill(0)];
    // Mock out the embeddings endpoint
    openAIEmbedSpy = jest
      .spyOn(llamaindex.OpenAIEmbedding.prototype, "getTextEmbeddings")
      .mockImplementation(() => {
        return Promise.resolve(embeddingsResponse);
      });
  });
  afterEach(() => {
    jest.clearAllMocks();
    openAISpy.mockRestore();
    openAIEmbedSpy.mockRestore();
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
    expect(openAIEmbedSpy).toHaveBeenCalledTimes(1);

    // Output response
    expect(response.response).toEqual(DUMMY_RESPONSE);

    const spans = memoryExporter.getFinishedSpans();
    expect(spans.length).toBeGreaterThan(0);

    // Expect a span for the query engine
    const queryEngineSpan = spans.find((span) => span.name.includes("Query"));
    expect(queryEngineSpan).toBeDefined();
  });
});
