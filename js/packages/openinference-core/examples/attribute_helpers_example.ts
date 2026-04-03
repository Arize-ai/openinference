/**
 * Typesafe example backing the code snippets in docs/attribute-helpers.md
 *
 * Exercises the complex type shapes: Message (with multimodal contents and
 * tool calls), TokenCount (with promptDetails), Embedding, Document, Tool,
 * and all attribute helper functions.
 */

import { trace } from "@opentelemetry/api";
import {
  MimeType,
  SEMRESATTRS_PROJECT_NAME,
} from "@arizeai/openinference-semantic-conventions";
import { resourceFromAttributes } from "@opentelemetry/resources";
import {
  ConsoleSpanExporter,
  NodeTracerProvider,
  SimpleSpanProcessor,
} from "@opentelemetry/sdk-trace-node";

import {
  defaultProcessInput,
  defaultProcessOutput,
  getDocumentAttributes,
  getEmbeddingAttributes,
  getInputAttributes,
  getLLMAttributes,
  getMetadataAttributes,
  getOutputAttributes,
  getRetrieverAttributes,
  getToolAttributes,
  withSpan,
} from "../src";

// -- Provider setup -----------------------------------------------------------

const provider = new NodeTracerProvider({
  resource: resourceFromAttributes({
    [SEMRESATTRS_PROJECT_NAME]: "attribute-helpers-example",
  }),
  spanProcessors: [new SimpleSpanProcessor(new ConsoleSpanExporter())],
});
provider.register();

// -- getLLMAttributes: complete example (docs/attribute-helpers.md) -----------

function llmAttributesDemo() {
  const span = trace.getActiveSpan();
  span?.setAttributes(
    getLLMAttributes({
      provider: "openai",
      modelName: "gpt-4o",
      invocationParameters: {
        temperature: 0.7,
        max_tokens: 1000,
      },
      inputMessages: [
        { role: "system", content: "You are a helpful assistant." },
        { role: "user", content: "What is OpenInference?" },
      ],
      outputMessages: [
        {
          role: "assistant",
          content:
            "OpenInference is an open standard for tracing LLM applications.",
          toolCalls: [
            {
              id: "call_1",
              function: {
                name: "search",
                arguments: { query: "OpenInference docs" },
              },
            },
          ],
        },
      ],
      tokenCount: {
        prompt: 42,
        completion: 128,
        total: 170,
        promptDetails: { audio: 5, cacheRead: 10, cacheWrite: 8 },
      },
      tools: [
        {
          jsonSchema: {
            type: "function",
            function: {
              name: "search",
              parameters: {
                type: "object",
                properties: { query: { type: "string" } },
              },
            },
          },
        },
      ],
    }),
  );
}

// -- Multimodal messages (docs/attribute-helpers.md) --------------------------

function multimodalDemo() {
  const attrs = getLLMAttributes({
    inputMessages: [
      {
        role: "user",
        contents: [
          { type: "text", text: "What's in this image?" },
          { type: "image", image: { url: "data:image/png;base64,..." } },
        ],
      },
    ],
  });
  console.log("multimodal attrs:", attrs);
}

// -- getEmbeddingAttributes ---------------------------------------------------

function embeddingDemo() {
  const attrs = getEmbeddingAttributes({
    modelName: "text-embedding-3-small",
    embeddings: [
      { text: "hello world", vector: [0.1, 0.2, 0.3] },
      { text: "goodbye", vector: [0.4, 0.5, 0.6] },
    ],
  });
  console.log("embedding attrs:", attrs);
}

// -- getRetrieverAttributes ---------------------------------------------------

function retrieverDemo() {
  const attrs = getRetrieverAttributes({
    documents: [
      {
        content: "Machine learning is a subset of AI.",
        id: "doc_001",
        score: 0.95,
        metadata: { source: "wikipedia", category: "tech" },
      },
      {
        content: "Deep learning uses neural networks.",
        id: "doc_002",
        score: 0.87,
      },
    ],
  });
  console.log("retriever attrs:", attrs);
}

// -- getDocumentAttributes (single doc with custom prefix) --------------------

function documentAttributesDemo() {
  const attrs = getDocumentAttributes(
    { content: "Sample doc", id: "doc-1", score: 0.9 },
    0,
    "reranker.input_documents",
  );
  console.log("document attrs:", attrs);
}

// -- getToolAttributes --------------------------------------------------------

function toolAttributesDemo() {
  const attrs = getToolAttributes({
    name: "weather_lookup",
    description: "Get current weather for a city",
    parameters: {
      type: "object",
      properties: {
        city: { type: "string" },
        units: { type: "string", enum: ["celsius", "fahrenheit"] },
      },
    },
  });
  console.log("tool attrs:", attrs);
}

// -- getMetadataAttributes ----------------------------------------------------

function metadataDemo() {
  const attrs = getMetadataAttributes({
    version: "1.0.0",
    environment: "production",
    experimentId: "exp-42",
  });
  console.log("metadata attrs:", attrs);
}

// -- getInputAttributes / getOutputAttributes ---------------------------------

function inputOutputDemo() {
  // String input
  const stringInput = getInputAttributes("What is OpenInference?");
  console.log("string input:", stringInput);

  // Structured input
  const structuredInput = getInputAttributes({
    value: '{"query": "search term"}',
    mimeType: MimeType.JSON,
  });
  console.log("structured input:", structuredInput);

  // String output
  const stringOutput = getOutputAttributes("Here is the answer.");
  console.log("string output:", stringOutput);

  // Null output (returns empty attributes)
  const nullOutput = getOutputAttributes(null);
  console.log("null output:", nullOutput);
}

// -- defaultProcessInput / defaultProcessOutput -------------------------------

function defaultProcessDemo() {
  // Single string
  const singleString = defaultProcessInput("hello");
  console.log("single string:", singleString);

  // Single object
  const singleObject = defaultProcessInput({ key: "value" });
  console.log("single object:", singleObject);

  // Multiple arguments
  const multiArgs = defaultProcessInput("arg1", 42);
  console.log("multi args:", multiArgs);

  // Output
  const output = defaultProcessOutput({ status: "success" });
  console.log("output:", output);
}

// -- Using attribute helpers with withSpan ------------------------------------

async function withSpanIntegrationDemo() {
  const search = withSpan(
    async (query: string) => {
      // Simulate retrieval
      return [
        { content: "result 1", id: "1", score: 0.9 },
        { content: "result 2", id: "2", score: 0.8 },
      ];
    },
    {
      name: "search",
      kind: "RETRIEVER",
      processInput: (query) => getInputAttributes(query),
      processOutput: (docs) => getRetrieverAttributes({ documents: docs }),
    },
  );

  await search("what is openinference?");
}

// -- Run all demos ------------------------------------------------------------

async function main() {
  llmAttributesDemo();
  multimodalDemo();
  embeddingDemo();
  retrieverDemo();
  documentAttributesDemo();
  toolAttributesDemo();
  metadataDemo();
  inputOutputDemo();
  defaultProcessDemo();
  await withSpanIntegrationDemo();
}

main();
