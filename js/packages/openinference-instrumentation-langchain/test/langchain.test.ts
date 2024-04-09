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
import { OTLPTraceExporter } from "@opentelemetry/exporter-trace-otlp-proto";

const tracerProvider = new NodeTracerProvider();
tracerProvider.register();

const instrumentation = new LangChainInstrumentation();
instrumentation.disable();

/**
 * Dummy test for scaffolding package, remove when real tests are added
 */
describe("langchain", () => {
  const memoryExporter = new InMemorySpanExporter();
  const provider = new NodeTracerProvider();
  provider.getTracer("default");

  instrumentation.setTracerProvider(tracerProvider);
  tracerProvider.addSpanProcessor(new SimpleSpanProcessor(memoryExporter));
  tracerProvider.addSpanProcessor(
    new SimpleSpanProcessor(
      new OTLPTraceExporter({
        url: "http://localhost:6006/v1/traces",
      }),
    ),
  );

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
  it("should pass", async () => {
    const chatModel = new ChatOpenAI({
      openAIApiKey: process.env.OPENAI_API_KEY,
    });

    await chatModel.invoke("test");

    const spans = memoryExporter.getFinishedSpans();
    expect(spans.length).toBe(1);
  });

  it("should properly nest spans", async () => {
    const chatModel = new ChatOpenAI({
      modelName: "gpt-3.5-turbo",
    });
    const PROMPT_TEMPLATE = `Use the context below to answer the question.
    ----------------
    {context}`;
    const prompt = ChatPromptTemplate.fromTemplate(PROMPT_TEMPLATE);
    const text = "This is a test";
    const textSplitter = new RecursiveCharacterTextSplitter({
      chunkSize: 1000,
    });
    const docs = await textSplitter.createDocuments([text]);
    const vectorStore = await MemoryVectorStore.fromDocuments(
      docs,
      new OpenAIEmbeddings(),
    );
    const chain = new RetrievalQAChain({
      combineDocumentsChain: loadQAStuffChain(chatModel, { prompt }),
      retriever: vectorStore.asRetriever(),
      returnSourceDocuments: true,
    });
    const response = await chain.invoke({
      query: "Is this a test?",
    });
    await chain.invoke({
      query: "I said is this a test?",
    });
    const spans = memoryExporter.getFinishedSpans();
    console.log("test--response", response);
    // console.log("test--spans", spans);
  });
});
