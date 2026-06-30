import { SEMRESATTRS_PROJECT_NAME } from "@arizeai/openinference-semantic-conventions";

import {
  diag,
  DiagConsoleLogger,
  DiagLogLevel,
  trace,
} from "@opentelemetry/api";
import { OTLPTraceExporter } from "@opentelemetry/exporter-trace-otlp-proto";
import { resourceFromAttributes } from "@opentelemetry/resources";
import {
  NodeTracerProvider,
  SimpleSpanProcessor,
} from "@opentelemetry/sdk-trace-node";

import {
  getEmbeddingAttributes,
  getLLMAttributes,
  getMetadataAttributes,
  getRetrieverAttributes,
  getToolAttributes,
  observe,
  traceAgent,
} from "../src";
// For troubleshooting, set the log level to DiagLogLevel.DEBUG
diag.setLogger(new DiagConsoleLogger(), DiagLogLevel.DEBUG);

// Create OTLP exporter with error handling

const provider = new NodeTracerProvider({
  resource: resourceFromAttributes({
    [SEMRESATTRS_PROJECT_NAME]: "ai-agent-example",
    "service.name": "ai-agent-example",
  }),
  spanProcessors: [
    new SimpleSpanProcessor(
      new OTLPTraceExporter({
        url: "http://localhost:6006/v1/traces",
      }),
    ),
  ],
});

provider.register();

/**
 * A simple AI agent that demonstrates various tracing scenarios
 * including LLM interactions, embeddings, retrieval, and tool usage.
 */
class AIAgent {
  private name: string;
  private modelName: string;
  private provider: string;

  constructor(
    name: string,
    modelName: string = "gpt-4",
    provider: string = "openai",
  ) {
    this.name = name;
    this.modelName = modelName;
    this.provider = provider;
  }

  /**
   * Generate embeddings for text input
   */
  @observe({
    kind: "EMBEDDING",
  })
  async generateEmbeddings(
    texts: string[],
  ): Promise<Array<{ text: string; vector: number[] }>> {
    // Simulate embedding generation
    const embeddings = texts.map((text) => ({
      text,
      vector: Array.from({ length: 1536 }, () => Math.random() - 0.5), // Simulate 1536-dim vector
    }));
    const span = trace.getActiveSpan();
    // Add embedding-specific attributes (demonstrates usage)
    span?.setAttributes(
      getEmbeddingAttributes({
        modelName: "text-embedding-ada-002",
        embeddings,
      }),
    );

    return embeddings;
  }

  /**
   * Retrieve relevant documents based on query
   */
  @observe({
    kind: "RETRIEVER",
  })
  async retrieveDocuments(
    query: string,
    topK: number = 5,
  ): Promise<
    Array<{
      content: string;
      id: string;
      score: number;
      metadata?: Record<string, unknown>;
    }>
  > {
    // Simulate document retrieval
    const documents = [
      {
        content:
          "Machine learning is a subset of artificial intelligence that focuses on algorithms.",
        id: "doc_001",
        score: 0.95,
        metadata: { source: "wikipedia", category: "technology" },
      },
      {
        content:
          "Deep learning uses neural networks with multiple layers to process data.",
        id: "doc_002",
        score: 0.87,
        metadata: { source: "research_paper", category: "ai" },
      },
      {
        content:
          "Natural language processing enables computers to understand human language.",
        id: "doc_003",
        score: 0.82,
        metadata: { source: "textbook", category: "nlp" },
      },
    ].slice(0, topK);
    const span = trace.getActiveSpan();
    span?.setAttributes(
      getRetrieverAttributes({
        documents,
      }),
    );
    return documents;
  }

  /**
   * Process user query with LLM
   */
  @observe({
    kind: "LLM",
  })
  async processQuery(
    query: string,
    context?: string[],
  ): Promise<{
    response: string;
    tokenCount: { prompt: number; completion: number; total: number };
  }> {
    // Simulate LLM processing
    const inputMessages = [
      { role: "system", content: "You are a helpful AI assistant." },
      { role: "user", content: query },
    ];

    if (context && context.length > 0) {
      inputMessages.push({
        role: "system",
        content: `Context: ${context.join(" ")}`,
      });
    }

    const response = `Based on your query "${query}", here's a comprehensive response that demonstrates AI capabilities.`;

    const tokenCount = {
      prompt: Math.floor(Math.random() * 100) + 50,
      completion: Math.floor(Math.random() * 200) + 100,
      total: 0,
    };
    tokenCount.total = tokenCount.prompt + tokenCount.completion;

    // Add LLM-specific attributes (demonstrates usage)
    const span = trace.getActiveSpan();
    span?.setAttributes(
      getLLMAttributes({
        provider: this.provider,
        modelName: this.modelName,
        inputMessages,
        outputMessages: [{ role: "assistant", content: response }],
        tokenCount,
        invocationParameters: {
          temperature: 0.7,
          max_tokens: 1000,
          top_p: 0.9,
        },
      }),
    );

    return { response, tokenCount };
  }

  /**
   * Use a tool to perform a specific action
   */
  @observe({
    kind: "TOOL",
  })
  async useTool(
    toolName: string,
    parameters: Record<string, unknown>,
  ): Promise<unknown> {
    // Add tool-specific attributes (demonstrates usage)
    const span = trace.getActiveSpan();
    span?.setAttributes(
      getToolAttributes({
        name: toolName,
        description: `Tool for ${toolName} operations`,
        parameters,
      }),
    );

    // Simulate different tool behaviors
    switch (toolName) {
      case "calculator":
        return { result: (parameters.a as number) + (parameters.b as number) };
      case "weather":
        return {
          temperature: 72,
          condition: "sunny",
          location: parameters.location,
        };
      case "search":
        return { results: [`Search results for: ${parameters.query}`] };
      default:
        return { error: "Unknown tool" };
    }
  }

  /**
   * Complete AI workflow combining multiple operations
   */
  @observe()
  async processWorkflow(
    userQuery: string,
  ): Promise<{ answer: string; sources: string[] }> {
    // Step 1: Generate embeddings for the query
    await this.generateEmbeddings([userQuery]);

    // Step 2: Retrieve relevant documents
    const documents = await this.retrieveDocuments(userQuery, 3);

    // Step 3: Process with LLM using retrieved context
    const context = documents.map((doc) => doc.content);
    const llmResult = await this.processQuery(userQuery, context);

    // Step 4: Use calculator tool if needed
    if (userQuery.includes("calculate") || userQuery.includes("math")) {
      await this.useTool("calculator", { a: 10, b: 20 });
    }

    return {
      answer: llmResult.response,
      sources: documents.map((doc) => doc.id),
    };
  }
}

// Example usage
async function runExample() {
  const agent = new AIAgent("Sophia", "gpt-4", "openai");

  try {
    // Run a complete workflow
    const result = await agent.processWorkflow(
      "What is machine learning and how does it relate to deep learning?",
    );

    // Test individual components
    const embeddings = await agent.generateEmbeddings(["test text"]);
    const docs = await agent.retrieveDocuments("artificial intelligence", 2);
    const toolResult = await agent.useTool("weather", {
      location: "San Francisco",
    });

    return {
      workflowResult: result,
      embeddingsCount: embeddings.length,
      documentsCount: docs.length,
      toolResult,
    };
  } catch (error) {
    throw new Error(`Error in workflow: ${error}`);
  }
}

traceAgent(runExample, {
  name: "my-agent",
  attributes: getMetadataAttributes({
    version: "1.0.0",
    environment: "production",
  }),
})();
