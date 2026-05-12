import type { GoogleGenAI } from "@google/genai";
import { context } from "@opentelemetry/api";
import { suppressTracing } from "@opentelemetry/core";
import { InMemorySpanExporter, SimpleSpanProcessor } from "@opentelemetry/sdk-trace-base";
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

  describe("with manual mock modules", () => {
    it("should create span for generateContent when called", async () => {
      class Models {
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

      const mockGoogleGenAI = {
        models: new Models(),
      } as unknown as GoogleGenAI;

      instrumentation.instrumentInstance(mockGoogleGenAI);

      await mockGoogleGenAI.models.generateContent({
        model: "gemini-2.0-flash",
        contents: "Hello",
      });

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
        async generateContent(_params: unknown) {
          return { candidates: [], usageMetadata: {} };
        }
      }

      const mockGoogleGenAI = {
        models: new Models(),
      } as unknown as GoogleGenAI;

      instrumentation.instrumentInstance(mockGoogleGenAI);

      const ctx = suppressTracing(context.active());

      await context.with(ctx, async () => {
        await mockGoogleGenAI.models.generateContent({
          model: "gemini-pro",
          contents: "Hello",
        });
      });

      const spans = memoryExporter.getFinishedSpans();
      expect(spans.length).toBe(0);
    });

    it("should handle streaming responses", async () => {
      class Models {
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

      const mockGoogleGenAI = {
        models: new Models(),
      } as unknown as GoogleGenAI;

      instrumentation.instrumentInstance(mockGoogleGenAI);

      const result = await mockGoogleGenAI.models.generateContentStream({
        model: "gemini-2.0-flash",
        contents: "Hello",
      });

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
      class Chat {
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
        create(_config: unknown) {
          return new Chat();
        }
      }

      const mockChats = new Chats();
      const mockGoogleGenAI = {
        chats: mockChats,
      } as unknown as GoogleGenAI;

      instrumentation.instrumentInstance(mockGoogleGenAI);

      const chat = mockChats.create({ model: "gemini-2.5-flash" });
      await chat.sendMessage({ message: "Hello" });

      const spans = memoryExporter.getFinishedSpans();
      expect(spans.length).toBeGreaterThan(0);

      const span = spans[0];
      expect(span.name).toBe("Google GenAI Chat Send Message");
      expect(span.attributes["openinference.span.kind"]).toBe("LLM");
    });

    it("should handle Batches createEmbeddings", async () => {
      class Batches {
        async createEmbeddings(_params: unknown) {
          return {
            name: "batch-job-123",
            state: "PENDING",
          };
        }
      }

      const mockGoogleGenAI = {
        batches: new Batches(),
      } as unknown as GoogleGenAI;

      instrumentation.instrumentInstance(mockGoogleGenAI);

      await (
        mockGoogleGenAI.batches as unknown as {
          createEmbeddings: (params: unknown) => Promise<unknown>;
        }
      ).createEmbeddings({
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

    it("should record errors and end the span", async () => {
      class Models {
        async generateContent(_params: unknown): Promise<never> {
          throw new Error("API Error");
        }
      }

      const mockGoogleGenAI = {
        models: new Models(),
      } as unknown as GoogleGenAI;

      instrumentation.instrumentInstance(mockGoogleGenAI);

      await expect(
        mockGoogleGenAI.models.generateContent({
          model: "gemini-2.0-flash",
          contents: "Hello",
        }),
      ).rejects.toThrow("API Error");

      await provider.forceFlush();

      const spans = memoryExporter.getFinishedSpans();
      expect(spans.length).toBeGreaterThan(0);
      expect(spans[0].status.code).toBe(2); // ERROR
    });

    it("should split multi-turn function call/response inputs into separate messages", async () => {
      class Models {
        async generateContent(_params: unknown) {
          return {
            candidates: [
              {
                content: {
                  role: "model",
                  parts: [{ text: "Sunny" }],
                },
              },
            ],
          };
        }
      }

      const mockGoogleGenAI = {
        models: new Models(),
      } as unknown as GoogleGenAI;

      instrumentation.instrumentInstance(mockGoogleGenAI);

      await mockGoogleGenAI.models.generateContent({
        model: "gemini-2.5-flash",
        contents: [
          {
            role: "user",
            parts: [{ text: "What's the weather in SF?" }],
          },
          {
            role: "model",
            parts: [
              {
                functionCall: { name: "get_weather", args: { location: "SF" } },
              },
            ],
          },
          {
            role: "user",
            parts: [
              {
                functionResponse: {
                  name: "get_weather",
                  response: { output: "sunny" },
                },
              },
            ],
          },
        ],
      });

      const spans = memoryExporter.getFinishedSpans();
      const span = spans[0];

      expect(span.attributes["llm.input_messages.0.message.role"]).toBe("user");
      expect(span.attributes["llm.input_messages.0.message.content"]).toBe(
        "What's the weather in SF?",
      );

      expect(span.attributes["llm.input_messages.1.message.role"]).toBe("model");
      expect(
        span.attributes["llm.input_messages.1.message.tool_calls.0.tool_call.function.name"],
      ).toBe("get_weather");
      expect(
        span.attributes["llm.input_messages.1.message.tool_calls.0.tool_call.function.arguments"],
      ).toBe(JSON.stringify({ location: "SF" }));

      expect(span.attributes["llm.input_messages.2.message.role"]).toBe("tool");
      expect(span.attributes["llm.input_messages.2.message.name"]).toBe("get_weather");
      expect(span.attributes["llm.input_messages.2.message.content"]).toBe(
        JSON.stringify({ output: "sunny" }),
      );
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

      const mockGoogleGenAI = {
        models: new Models(),
      } as unknown as GoogleGenAI;

      instrumentationWithConfig.instrumentInstance(mockGoogleGenAI);

      await mockGoogleGenAI.models.generateContent({
        model: "gemini-2.0-flash",
        contents: "Test input",
      });

      const spans = memoryExporter.getFinishedSpans();
      expect(spans.length).toBeGreaterThan(0);

      const span = spans[0];
      expect(span.name).toBe("Google GenAI Generate Content");
    });
  });
});
