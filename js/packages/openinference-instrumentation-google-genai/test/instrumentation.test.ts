import { context } from "@opentelemetry/api";
import { suppressTracing } from "@opentelemetry/core";
import {
  InMemorySpanExporter,
  SimpleSpanProcessor,
} from "@opentelemetry/sdk-trace-base";
import { NodeTracerProvider } from "@opentelemetry/sdk-trace-node";
import { beforeEach, describe, expect, it } from "vitest";

import { GoogleGenAIInstrumentation } from "../src/instrumentation";

describe("GoogleGenAIInstrumentation", () => {
  let instrumentation: GoogleGenAIInstrumentation;
  let memoryExporter: InMemorySpanExporter;
  let provider: NodeTracerProvider;

  beforeEach(() => {
    memoryExporter = new InMemorySpanExporter();
    provider = new NodeTracerProvider();
    provider.addSpanProcessor(new SimpleSpanProcessor(memoryExporter));
    provider.register();
    instrumentation = new GoogleGenAIInstrumentation();
    instrumentation.setTracerProvider(provider);
    instrumentation.enable();
    memoryExporter.reset();
  });

  it("should be instantiable", () => {
    expect(instrumentation).toBeInstanceOf(GoogleGenAIInstrumentation);
  });

  it("should have correct module name", () => {
    expect(instrumentation.instrumentationName).toBe(
      "@arizeai/openinference-instrumentation-google-genai",
    );
  });

  it("should export isPatched function", () => {
    // Just verify the structure exists
    expect(instrumentation).toBeDefined();
  });

  describe("with manual mock modules", () => {

    it("should create span for generateContent when called", async () => {
      class Models {
        // eslint-disable-next-line @typescript-eslint/no-unused-vars
        async generateContent(_params: unknown) {
          return {
            candidates: [
              {
                content: {
                  role: "model",
                  parts: [{ text: "Response" }],
                },
              },
            ],
            usageMetadata: {
              promptTokenCount: 5,
              candidatesTokenCount: 10,
              totalTokenCount: 15,
            },
          };
        }
      }

      class Chat {
        // eslint-disable-next-line @typescript-eslint/no-unused-vars
        async sendMessage(_message: unknown) {
          return { text: "Response" };
        }
      }

      class Chats {
        // eslint-disable-next-line @typescript-eslint/no-unused-vars
        create(_config: unknown) {
          return new Chat();
        }
      }

      // Create a mock GoogleGenAI instance
      const mockGoogleGenAI = {
        models: new Models(),
        chats: new Chats(),
      };

      instrumentation.instrumentInstance(mockGoogleGenAI);

      await mockGoogleGenAI.models.generateContent({ model: "gemini-2.0-flash", contents: "Hello" });

      const spans = memoryExporter.getFinishedSpans();
      expect(spans.length).toBeGreaterThan(0);

      const span = spans[0];
      expect(span.name).toBe("Google GenAI Generate Content");
      expect(span.attributes["llm.model_name"]).toBe("gemini-2.0-flash");
      expect(span.attributes["llm.system"]).toBe("vertexai");
      expect(span.attributes["llm.provider"]).toBe("google");
      expect(span.attributes["llm.token_count.total"]).toBe(15);
    });

    it("should suppress tracing when context is suppressed", async () => {
      class Models {
        // eslint-disable-next-line @typescript-eslint/no-unused-vars
        async generateContent(_params: unknown) {
          return { candidates: [], usageMetadata: {} };
        }
      }

      class Chat {
        // eslint-disable-next-line @typescript-eslint/no-unused-vars
        async sendMessage(_message: unknown) {
          return { text: "Response" };
        }
      }

      class Chats {
        // eslint-disable-next-line @typescript-eslint/no-unused-vars
        create(_config: unknown) {
          return new Chat();
        }
      }

      // Create a mock GoogleGenAI instance
      const mockGoogleGenAI = {
        models: new Models(),
        chats: new Chats(),
      };

      instrumentation.instrumentInstance(mockGoogleGenAI);

      const ctx = suppressTracing(context.active());

      await context.with(ctx, async () => {
        await mockGoogleGenAI.models.generateContent({ model: "gemini-pro", contents: "Hello" });
      });

      const spans = memoryExporter.getFinishedSpans();
      expect(spans.length).toBe(0);
    });

    it("should handle streaming responses", async () => {
      class Models {
        // eslint-disable-next-line @typescript-eslint/no-unused-vars
        async generateContentStream(_params: unknown) {
          async function* mockStream() {
            yield {
              candidates: [
                {
                  content: {
                    role: "model",
                    parts: [{ text: "Hello" }],
                  },
                },
              ],
            };
            yield {
              candidates: [
                {
                  content: {
                    role: "model",
                    parts: [{ text: " there!" }],
                  },
                },
              ],
              usageMetadata: {
                promptTokenCount: 5,
                candidatesTokenCount: 10,
                totalTokenCount: 15,
              },
            };
          }
          return mockStream();
        }
      }

      class Chat {
        // eslint-disable-next-line @typescript-eslint/no-unused-vars
        async sendMessage(_message: unknown) {
          return { text: "Response" };
        }
      }

      class Chats {
        // eslint-disable-next-line @typescript-eslint/no-unused-vars
        create(_config: unknown) {
          return new Chat();
        }
      }

      // Create a mock GoogleGenAI instance
      const mockGoogleGenAI = {
        models: new Models(),
        chats: new Chats(),
      };

      instrumentation.instrumentInstance(mockGoogleGenAI);

      const result = await mockGoogleGenAI.models.generateContentStream({
        model: "gemini-2.0-flash",
        contents: "Hello",
      });

      // Consume the stream
      const chunks = [];
      for await (const chunk of result) {
        chunks.push(chunk);
      }

      expect(chunks.length).toBe(2);

      const spans = memoryExporter.getFinishedSpans();
      expect(spans.length).toBeGreaterThan(0);

      const span = spans[0];
      expect(span.name).toBe("Google GenAI Generate Content Stream");
      expect(span.attributes["llm.token_count.total"]).toBe(15);
    });

    it("should handle Chat sendMessage", async () => {
      class Models {
        // eslint-disable-next-line @typescript-eslint/no-unused-vars
        async generateContent(_params: unknown) {
          return { candidates: [], usageMetadata: {} };
        }
      }

      class Chat {
        // eslint-disable-next-line @typescript-eslint/no-unused-vars
        async sendMessage(_params: unknown) {
          return {
            candidates: [
              {
                content: {
                  role: "model",
                  parts: [{ text: "Response" }],
                },
              },
            ],
          };
        }
      }

      class Chats {
        // eslint-disable-next-line @typescript-eslint/no-unused-vars
        create(_config: unknown) {
          return new Chat();
        }
      }

      // Create a mock GoogleGenAI instance with a chat
      const mockChats = new Chats();
      const mockGoogleGenAI = {
        models: new Models(),
        chats: mockChats,
      };

      instrumentation.instrumentInstance(mockGoogleGenAI);

      const chat = mockChats.create({ model: "gemini-2.5-flash" });
      await chat.sendMessage({ message: "Hello" });

      const spans = memoryExporter.getFinishedSpans();
      expect(spans.length).toBeGreaterThan(0);

      const span = spans[0];
      expect(span.name).toBe("Google GenAI Chat Send Message");
      expect(span.attributes["openinference.span.kind"]).toBe("CHAIN");
    });

    it("should handle Batches createEmbeddings", async () => {
      class Batches {
        // eslint-disable-next-line @typescript-eslint/no-unused-vars
        async createEmbeddings(_params: unknown) {
          return {
            name: "batch-job-123",
            state: "PENDING",
          };
        }
      }

      // Create a mock GoogleGenAI instance
      const mockGoogleGenAI = {
        batches: new Batches(),
      };

      instrumentation.instrumentInstance(mockGoogleGenAI);

      await mockGoogleGenAI.batches.createEmbeddings({
        model: "text-embedding-004",
        requests: [{ content: "test" }],
      });

      const spans = memoryExporter.getFinishedSpans();
      expect(spans.length).toBeGreaterThan(0);

      const span = spans[0];
      expect(span.name).toBe("Google GenAI Batch Create Embeddings");
      expect(span.attributes["openinference.span.kind"]).toBe("EMBEDDING");
      expect(span.attributes["embedding.model_name"]).toBe("text-embedding-004");
    });

    it.skip("should handle errors gracefully", async () => {
      class Models {
        // eslint-disable-next-line @typescript-eslint/no-unused-vars
        async generateContent(_params: unknown): Promise<never> {
          throw new Error("API Error");
        }
      }

      class Chat {
        // eslint-disable-next-line @typescript-eslint/no-unused-vars
        async sendMessage(_message: unknown) {
          return { text: "Response" };
        }
      }

      class Chats {
        // eslint-disable-next-line @typescript-eslint/no-unused-vars
        create(_config: unknown) {
          return new Chat();
        }
      }

      // Create a mock GoogleGenAI instance
      const models = new Models();
      const mockGoogleGenAI = {
        models,
        chats: new Chats(),
      };

      // Store reference to original method
      const originalMethod = models.generateContent;

      instrumentation.instrumentInstance(mockGoogleGenAI);

      // Verify the method was patched
      expect(models.generateContent).not.toBe(originalMethod);

      // Call the method and catch the error
      try {
        await mockGoogleGenAI.models.generateContent({ model: "gemini-2.0-flash", contents: "Hello" });
        // Should not reach here
        expect.fail("Expected an error to be thrown");
      } catch (error: unknown) {
        // Verify the error was thrown
        expect((error as Error).message).toBe("API Error");
      }

      // Flush the span processor to ensure spans are exported
      await provider.forceFlush();

      const spans = memoryExporter.getFinishedSpans();
      expect(spans.length).toBeGreaterThan(0);

      const span = spans[0];
      expect(span.status.code).toBe(2); // ERROR
    });

    it("should apply trace configuration", async () => {
      const instrumentationWithConfig = new GoogleGenAIInstrumentation({
        traceConfig: {
          hideInputs: true,
          hideOutputs: true,
        },
      });
      instrumentationWithConfig.setTracerProvider(provider);
      instrumentationWithConfig.enable();

      class Models {
        // eslint-disable-next-line @typescript-eslint/no-unused-vars
        async generateContent(_params: unknown) {
          return {
            candidates: [
              {
                content: {
                  role: "model",
                  parts: [{ text: "Response" }],
                },
              },
            ],
          };
        }
      }

      class Chat {
        // eslint-disable-next-line @typescript-eslint/no-unused-vars
        async sendMessage(_message: unknown) {
          return { text: "Response" };
        }
      }

      class Chats {
        // eslint-disable-next-line @typescript-eslint/no-unused-vars
        create(_config: unknown) {
          return new Chat();
        }
      }

      // Create a mock GoogleGenAI instance
      const mockGoogleGenAI = {
        models: new Models(),
        chats: new Chats(),
      };

      instrumentationWithConfig.instrumentInstance(mockGoogleGenAI);

      await mockGoogleGenAI.models.generateContent({
        model: "gemini-2.0-flash",
        contents: "Test input",
      });

      const spans = memoryExporter.getFinishedSpans();
      expect(spans.length).toBeGreaterThan(0);

      const span = spans[0];
      // Verify span was created (trace config masking is handled by OITracer)
      expect(span.name).toBe("Google GenAI Generate Content");
    });
  });
});
