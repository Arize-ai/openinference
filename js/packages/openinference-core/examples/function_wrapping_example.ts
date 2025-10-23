import {
  getInputAttributes,
  getRetrieverAttributes,
  traceAgent,
  withSpan,
} from "../src";

import {
  NodeTracerProvider,
  SimpleSpanProcessor,
  ConsoleSpanExporter,
} from "@opentelemetry/sdk-trace-node";
import { resourceFromAttributes } from "@opentelemetry/resources";
import { diag, DiagConsoleLogger, DiagLogLevel } from "@opentelemetry/api";
import { SEMRESATTRS_PROJECT_NAME } from "@arizeai/openinference-semantic-conventions";
import { OTLPTraceExporter } from "@opentelemetry/exporter-trace-otlp-proto";

// For troubleshooting, set the log level to DiagLogLevel.DEBUG
diag.setLogger(new DiagConsoleLogger(), DiagLogLevel.DEBUG);

const provider = new NodeTracerProvider({
  resource: resourceFromAttributes({
    [SEMRESATTRS_PROJECT_NAME]: "function-wrapping-example",
    "service.name": "function-wrapping-example",
  }),
  spanProcessors: [
    new SimpleSpanProcessor(new ConsoleSpanExporter()),
    new SimpleSpanProcessor(
      new OTLPTraceExporter({
        url: "http://localhost:6006/v1/traces",
      }),
    ),
  ],
});
provider.register();

const retriever = withSpan(
  (_query: string) => {
    return ["The capital of France is Paris."];
  },
  {
    kind: "RETRIEVER",
    name: "retriever",
    processInput: (query) => getInputAttributes(query),
    processOutput: (documents) => {
      return {
        ...getRetrieverAttributes({
          documents: documents.map((document) => ({
            content: document,
          })),
        }),
      };
    },
  },
);

// simple RAG agent
const agent = traceAgent(
  async (question: string) => {
    const documents = await retriever(question);
    return `Let me help you answer that: ${question} ${documents.join("\n")}`;
  },
  {
    name: "agent",
  },
);
async function main() {
  await agent("What is the capital of France?");
}

main();
