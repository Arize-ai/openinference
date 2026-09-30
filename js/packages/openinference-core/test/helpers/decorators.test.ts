import { trace } from "@opentelemetry/api";
import { resourceFromAttributes } from "@opentelemetry/resources";
import { NodeTracerProvider } from "@opentelemetry/sdk-trace-node";
import { InMemorySpanExporter, SimpleSpanProcessor } from "@opentelemetry/sdk-trace-node";
import { afterEach, beforeEach, describe, expect, it } from "vitest";

import { SemanticConventions } from "@arizeai/openinference-semantic-conventions";

import {
  defaultProcessInput,
  defaultProcessOutput,
  getLLMAttributes,
  observe,
} from "../../src/helpers";

let spanExporter: InMemorySpanExporter;
let tracerProvider: NodeTracerProvider;

describe("observe", () => {
  beforeEach(() => {
    spanExporter = new InMemorySpanExporter();
    tracerProvider = new NodeTracerProvider({
      resource: resourceFromAttributes({
        "service.name": "test-service",
      }),
      spanProcessors: [new SimpleSpanProcessor(spanExporter)],
    });

    tracerProvider.register();
  });

  afterEach(async () => {
    spanExporter.reset();
    await tracerProvider.shutdown();
    trace.disable();
  });

  // The package tsconfig enables legacy experimental decorators while observe
  // targets the stage-3 decorator signature, so invoke the decorator directly
  // with a minimal ClassMethodDecoratorContext instead of @observe syntax.
  const decorate = <Fn extends (...args: never[]) => unknown>(
    options: Parameters<typeof observe>[0],
    method: Fn,
    name: string,
  ) => observe(options)(method, { name } as ClassMethodDecoratorContext);

  it("should support model name attributes composed via getLLMAttributes", async () => {
    const tracer = tracerProvider.getTracer("test");
    const complete = async (request: { model: string; prompt: string }) => ({
      model: `${request.model}-0613`,
      text: `answer to ${request.prompt}`,
    });
    const decorated = decorate(
      {
        kind: "LLM",
        processInput: (request: { model: string }) => ({
          ...defaultProcessInput(request),
          ...getLLMAttributes({ requestModelName: request.model }),
        }),
        processOutput: (response: { model: string }) => ({
          ...defaultProcessOutput(response),
          ...getLLMAttributes({ responseModelName: response.model }),
        }),
        tracer,
      },
      complete,
      "complete",
    );

    const result = await decorated({ model: "gpt-4", prompt: "hi" });

    expect(result.text).toBe("answer to hi");

    const spans = spanExporter.getFinishedSpans();
    expect(spans).toHaveLength(1);

    const span = spans[0];
    expect(span.name).toBe("complete");
    expect(span.attributes[SemanticConventions.LLM_REQUEST_MODEL_NAME]).toBe("gpt-4");
    expect(span.attributes[SemanticConventions.LLM_RESPONSE_MODEL_NAME]).toBe("gpt-4-0613");
    expect(span.attributes[SemanticConventions.LLM_MODEL_NAME]).toBe("gpt-4-0613");
    expect(span.attributes[SemanticConventions.INPUT_VALUE]).toBe(
      JSON.stringify({ model: "gpt-4", prompt: "hi" }),
    );
    expect(span.attributes[SemanticConventions.OUTPUT_VALUE]).toBe(JSON.stringify(result));
  });
});
