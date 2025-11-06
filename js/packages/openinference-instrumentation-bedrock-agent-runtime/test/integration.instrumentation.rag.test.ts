import {
  InMemorySpanExporter,
  SimpleSpanProcessor,
} from "@opentelemetry/sdk-trace-base";
import { NodeTracerProvider } from "@opentelemetry/sdk-trace-node";

import { BedrockAgentInstrumentation } from "../src";

import { createPolly } from "./utils/polly.config";
import { setModuleExportsForInstrumentation } from "./utils/test-utils";

import {
  BedrockAgentRuntimeClient,
  RetrieveAndGenerateCommand,
  RetrieveAndGenerateCommandInput,
  RetrieveCommand,
  RetrieveCommandInput,
} from "@aws-sdk/client-bedrock-agent-runtime";
import * as bedrockAgentRuntime from "@aws-sdk/client-bedrock-agent-runtime";
import { Polly } from "@pollyjs/core";

describe("BedrockAgent RAG Instrumentation - attributes and API recording", () => {
  let instrumentation: BedrockAgentInstrumentation;
  let provider: NodeTracerProvider;
  let memoryExporter: InMemorySpanExporter;

  const knowledgeBaseId = "SSGLURQ9A5";
  const modelArn = "anthropic.claude-3-haiku-20240307-v1:0";
  const s3Uri = "s3://bedrock-az-kb/knowledge_bases/VLDBJ96.pdf";
  const s3InputText = "What is Telos?";
  const knowledgeBaseInputText = "What is Task Decomposition?";

  const cassettePrefix = "bedrock-agent-rag";

  let polly: Polly;

  beforeAll(() => {
    // Setup instrumentation and tracer provider (following OpenAI pattern)
    instrumentation = new BedrockAgentInstrumentation();
    instrumentation.disable();
    memoryExporter = new InMemorySpanExporter();
    provider = new NodeTracerProvider();
    provider.addSpanProcessor(new SimpleSpanProcessor(memoryExporter));
    provider.register();
    instrumentation.setTracerProvider(provider);
    // Manually set module exports for testing (following OpenAI pattern)
    setModuleExportsForInstrumentation(instrumentation, bedrockAgentRuntime);
    // Enable instrumentation ONCE
    instrumentation.enable();
  });

  beforeEach(() => {
    memoryExporter.reset();
    polly = createPolly(cassettePrefix);
  });

  afterEach(async () => {
    await polly.stop();
  });

  // Global cleanup
  afterAll(async () => {
    instrumentation.disable();
    await provider.shutdown();
  });

  it("should record rag attributes and API response in span", async () => {
    const client = new BedrockAgentRuntimeClient({
      region: "ap-south-1",
      credentials: {
        accessKeyId: "test",
        secretAccessKey: "test",
      },
    });
    const params: RetrieveAndGenerateCommandInput = {
      input: {
        text: knowledgeBaseInputText,
      },
      retrieveAndGenerateConfiguration: {
        knowledgeBaseConfiguration: {
          knowledgeBaseId: knowledgeBaseId,
          modelArn: modelArn,
        },
        type: "KNOWLEDGE_BASE",
      },
    };
    const command = new RetrieveAndGenerateCommand(params);
    const response = await client.send(command);
    expect(response).toBeDefined();
    expect(typeof response).toBe("object");
    const spans = memoryExporter.getFinishedSpans();
    expect(spans.length).toBe(1);
    const attrs = spans[0].attributes;
    expect(attrs).toBeDefined();
    expect(attrs["input.mime_type"]).toBe("text/plain");
    expect(attrs["input.value"]).toMatch(/^What is Task/);

    const invocation = attrs["llm.invocation_parameters"];
    expect(invocation).toContain('"retrieveAndGenerateConfiguration"');
    expect(invocation).toContain("SSGLURQ9A5");

    expect(attrs["llm.model_name"]).toBe(
      "anthropic.claude-3-haiku-20240307-v1:0",
    );
    expect(attrs["openinference.span.kind"]).toBe("RETRIEVER");
    expect(attrs["output.mime_type"]).toBe("text/plain");

    const outputVal = attrs["output.value"];
    expect(outputVal).toMatch(/^Task Decomposition is a technique/);
    expect(outputVal).toContain("Chain of Thought");
    expect(outputVal).toContain("Tree of Thoughts");

    for (let i = 0; i < 2; i++) {
      const prefix = `retrieval.documents.${i}.document`;
      const content = attrs[`${prefix}.content`];
      expect(content).toContain("Task Decomposition");

      const metadata = attrs[`${prefix}.metadata`];
      expect(metadata).toContain('"customDocumentLocation":{"id":"2222"}');
      expect(metadata).toContain(
        '"x-amz-bedrock-kb-data-source-id":"VYV3J5D9O6"',
      );
    }
  });
  it("should record rag external attributes and API response in span", async () => {
    const client = new BedrockAgentRuntimeClient({
      region: "ap-south-1",
      credentials: {
        accessKeyId: "test",
        secretAccessKey: "test",
      },
    });
    const params: RetrieveAndGenerateCommandInput = {
      input: {
        text: s3InputText,
      },
      retrieveAndGenerateConfiguration: {
        type: "EXTERNAL_SOURCES",
        externalSourcesConfiguration: {
          sources: [
            {
              s3Location: {
                uri: s3Uri,
              },
              sourceType: "S3",
            },
          ],
          modelArn: modelArn,
        },
      },
    };
    const command = new RetrieveAndGenerateCommand(params);
    const response = await client.send(command);
    expect(response).toBeDefined();
    expect(typeof response).toBe("object");
    const spans = memoryExporter.getFinishedSpans();
    expect(spans.length).toBe(1);
    const attrs = { ...spans[0].attributes };
    expect(attrs["input.mime_type"]).toBe("text/plain");
    expect(attrs["input.value"]).toBe("What is Telos?");
    expect(attrs["llm.model_name"]).toBe(
      "anthropic.claude-3-haiku-20240307-v1:0",
    );
    expect(attrs["openinference.span.kind"]).toBe("RETRIEVER");
    expect(attrs["output.mime_type"]).toBe("text/plain");

    const output = attrs["output.value"];
    expect(output).toContain("Telos is a knowledge representation language");
    expect(output).toContain("Telos treats attributes as first-class citizens");
    expect(output).toContain(
      "Telos propositions are organized along three dimensions",
    );
    expect(output).toContain("history time and belief time");
    expect(output).toContain(
      "assertion language for expressing deductive rules",
    );

    // Validate retrieval documents
    for (let i = 0; i < 9; i++) {
      const contentKey = `retrieval.documents.${i}.document.content`;
      const metadataKey = `retrieval.documents.${i}.document.metadata`;
      if (contentKey in attrs) {
        const content = attrs[contentKey];
        expect(content).not.toBeNull();
      }
      if (metadataKey in attrs) {
        const metadata = attrs[metadataKey];
        expect(metadata).toContain(
          "s3://bedrock-az-kb/knowledge_bases/VLDBJ96.pdf",
        );
      }
    }

    // Validate invocation parameters
    const invocation = attrs["llm.invocation_parameters"];
    expect(invocation).toContain("sourceType");
    expect(invocation).toContain("S3");
    expect(invocation).toContain("anthropic.claude-3-haiku-20240307-v1:0");
    const expectedKeys = [
      "input.mime_type",
      "input.value",
      "llm.model_name",
      "openinference.span.kind",
      "output.mime_type",
      "output.value",
      "llm.invocation_parameters",
      "llm.provider",
      ...[0, 1, 2, 3, 4, 5, 6, 7, 8].map(
        (i) => `retrieval.documents.${i}.document.content`,
      ),
      ...[0, 1, 2, 3, 4, 5, 6, 7, 8].map(
        (i) => `retrieval.documents.${i}.document.metadata`,
      ),
      ...[0, 1, 2, 3, 4, 5, 6, 7, 8].map(
        (i) => `retrieval.documents.${i}.document.score`,
      ),
    ];
    expect(Object.keys(attrs).sort()).toEqual(expectedKeys.sort());
  });

  it("should record retrieve attributes and API response in span", async () => {
    const client = new BedrockAgentRuntimeClient({
      region: "ap-south-1",
      credentials: {
        accessKeyId: "test",
        secretAccessKey: "test",
      },
    });
    const params: RetrieveCommandInput = {
      retrievalQuery: {
        text: knowledgeBaseInputText,
      },
      knowledgeBaseId: knowledgeBaseId,
    };
    const command = new RetrieveCommand(params);
    const response = await client.send(command);
    expect(response).toBeDefined();
    expect(typeof response).toBe("object");
    const spans = memoryExporter.getFinishedSpans();
    expect(spans.length).toBe(1);
    const attrs = { ...spans[0].attributes };
    expect(attrs["input.mime_type"]).toBe("text/plain");
    expect(attrs["input.value"]).toBe("What is Task Decomposition?");
    expect(attrs["openinference.span.kind"]).toBe("RETRIEVER");
    expect(attrs["llm.invocation_parameters"]).toContain("SSGLURQ9A5");

    for (let i = 0; i < 5; i++) {
      const contentKey = `retrieval.documents.${i}.document.content`;
      const metadataKey = `retrieval.documents.${i}.document.metadata`;
      if (contentKey in attrs) {
        const content = attrs[contentKey];
        expect(content).not.toBeNull();
      }
      if (metadataKey in attrs) {
        const metadata = attrs[metadataKey];
        expect(metadata).toContain('"customDocumentLocation":{"id":"2222"}');
      }
    }

    // Validate invocation parameters
    const invocation = attrs["llm.invocation_parameters"];
    expect(invocation).toContain('"knowledgeBaseId":"SSGLURQ9A5"');
    // Final assertion: no unexpected attributes remain
    const expectedKeys = [
      "input.mime_type",
      "input.value",
      "openinference.span.kind",
      "llm.invocation_parameters",
      "llm.provider",
      ...[0, 1, 2, 3, 4].map(
        (i) => `retrieval.documents.${i}.document.content`,
      ),
      ...[0, 1, 2, 3, 4].map(
        (i) => `retrieval.documents.${i}.document.metadata`,
      ),
      ...[0, 1, 2, 3, 4].map((i) => `retrieval.documents.${i}.document.score`),
    ];
    expect(Object.keys(attrs).sort()).toEqual(expectedKeys.sort());
  });
});
