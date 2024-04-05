import {
  InMemorySpanExporter,
  SimpleSpanProcessor,
} from "@opentelemetry/sdk-trace-base";
import { NodeTracerProvider } from "@opentelemetry/sdk-trace-node";
import { LangChainInstrumentation } from "../src";
import * as CallbackManager from "@langchain/core/callbacks/manager";
import "dotenv/config";

const tracerProvider = new NodeTracerProvider();
tracerProvider.register();

const instrumentation = new LangChainInstrumentation();
instrumentation.disable();

import { ChatOpenAI } from "@langchain/openai";

/**
 * Dummy test for scaffolding package, remove when real tests are added
 */
describe("langchain", () => {
  const memoryExporter = new InMemorySpanExporter();
  const provider = new NodeTracerProvider();
  provider.getTracer("default");

  instrumentation.setTracerProvider(tracerProvider);
  tracerProvider.addSpanProcessor(new SimpleSpanProcessor(memoryExporter));

  // @ts-expect-error the moduleExports property is private. This is needed to make the test work with auto-mockind
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
});
