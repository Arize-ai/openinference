import {
  InMemorySpanExporter,
  SimpleSpanProcessor,
} from "@opentelemetry/sdk-trace-base";
import { LlamaIndexInstrumentation, isPatched } from "../src/index";
import { NodeTracerProvider } from "@opentelemetry/sdk-trace-node";
import * as llamaindex from "llamaindex";

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
  it("is patched", () => {
    expect(isPatched()).toBe(true);
  });
});
