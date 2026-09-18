import { createReadStream, mkdtempSync, rmdirSync, unlinkSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { InMemorySpanExporter, SimpleSpanProcessor } from "@opentelemetry/sdk-trace-base";
import { NodeTracerProvider } from "@opentelemetry/sdk-trace-node";
import OpenAI from "openai";
import type { ImageGenStreamEvent, ImagesResponse } from "openai/resources/images";
import { Stream } from "openai/streaming";
import { vi } from "vitest";

import { generateTraceConfig } from "@arizeai/openinference-core";

import { OpenAIInstrumentation } from "../src";
import { getInputImageAttributes } from "../src/imageAttributes";

describe("OpenAIInstrumentation - Images", () => {
  const memoryExporter = new InMemorySpanExporter();
  const tracerProvider = new NodeTracerProvider({
    spanProcessors: [new SimpleSpanProcessor(memoryExporter)],
  });
  const instrumentation = new OpenAIInstrumentation({ tracerProvider });
  let openai: OpenAI;

  beforeAll(() => {
    instrumentation.disable();
    // @ts-expect-error moduleExports is private; tests set it for manual module mocking.
    instrumentation._modules[0].moduleExports = OpenAI;
    instrumentation.enable();
    openai = new OpenAI({ apiKey: "fake-api-key" });
  });

  afterAll(() => instrumentation.disable());
  beforeEach(() => memoryExporter.reset());
  afterEach(() => vi.restoreAllMocks());

  it("captures URL and base64 image generation outputs", async () => {
    const response = {
      created: 1,
      output_format: "webp",
      data: [{ url: "https://example.com/generated.png" }, { b64_json: "aW1hZ2U=" }],
    } satisfies ImagesResponse;
    vi.spyOn(openai, "post").mockImplementation(
      // @ts-expect-error return only needs to model the parsed SDK response.
      async () => response,
    );

    await openai.images.generate({ prompt: "a lighthouse", output_format: "jpeg" });

    const span = memoryExporter.getFinishedSpans()[0];
    expect(span.name).toBe("OpenAI Images");
    expect(span.attributes["output.images.0.image.url"]).toBe("https://example.com/generated.png");
    expect(span.attributes["output.images.1.image.url"]).toBe("data:image/webp;base64,aW1hZ2U=");
  });

  it("ends the span when an image request is rejected", async () => {
    // @ts-expect-error test rejection does not need to model APIPromise internals.
    vi.spyOn(openai, "post").mockImplementation(async () => {
      throw new Error("request failed");
    });

    await expect(openai.images.generate({ prompt: "a lighthouse" })).rejects.toThrow(
      "request failed",
    );

    const span = memoryExporter.getFinishedSpans()[0];
    expect(span.status.code).toBe(2);
    expect(span.status.message).toBe("request failed");
  });

  it("captures reusable edit image and mask uploads without serializing their bytes", async () => {
    const response = {
      created: 1,
      data: [{ b64_json: "ZWRpdGVk" }],
    } satisfies ImagesResponse;
    vi.spyOn(openai, "post").mockImplementation(
      // @ts-expect-error return only needs to model the parsed SDK response.
      async () => response,
    );
    const source = new File([new Uint8Array([0x52, 0x49, 0x46, 0x46])], "source.webp", {
      type: "image/webp",
    });
    const mask = new File(
      [new Uint8Array([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a])],
      "mask.png",
      { type: "image/png" },
    );

    await openai.images.edit({ image: source, mask, prompt: "remove the background" });

    const span = memoryExporter.getFinishedSpans()[0];
    expect(span.attributes["input.images.0.image.url"]).toBe(
      `data:image/webp;base64,${Buffer.from(await source.arrayBuffer()).toString("base64")}`,
    );
    expect(span.attributes["input.images.1.image.url"]).toBe(
      `data:image/png;base64,${Buffer.from(await mask.arrayBuffer()).toString("base64")}`,
    );
    expect(JSON.parse(span.attributes["input.value"] as string)).toEqual({
      prompt: "remove the background",
    });
    expect(span.attributes["output.images.0.image.url"]).toBe("data:image/png;base64,ZWRpdGVk");
  });

  it("does not read hidden uploads", async () => {
    const image = new File([new Uint8Array(128)], "source.png", { type: "image/png" });
    const arrayBufferSpy = vi.spyOn(image, "arrayBuffer");

    for (const traceConfig of [{ hideInputs: true }, { hideInputImages: true }]) {
      await expect(
        getInputImageAttributes([image], generateTraceConfig(traceConfig)),
      ).resolves.toEqual({});
    }
    expect(arrayBufferSpy).not.toHaveBeenCalled();
  });

  it("captures a Node file stream without consuming the upload", async () => {
    const response = {
      created: 1,
      data: [{ url: "https://example.com/variation.png" }],
    } satisfies ImagesResponse;
    vi.spyOn(openai, "post").mockImplementation(
      // @ts-expect-error return only needs to model the parsed SDK response.
      async () => response,
    );
    const directory = mkdtempSync(join(tmpdir(), "openinference-openai-images-"));
    const imagePath = join(directory, "source.png");
    const imageBytes = new Uint8Array([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]);
    writeFileSync(imagePath, imageBytes);

    try {
      await openai.images.createVariation({ image: createReadStream(imagePath) });
    } finally {
      unlinkSync(imagePath);
      rmdirSync(directory);
    }

    const span = memoryExporter.getFinishedSpans()[0];
    expect(span.attributes["input.images.0.image.url"]).toBe(
      `data:image/png;base64,${Buffer.from(imageBytes).toString("base64")}`,
    );
  });

  it("captures the completed image from a streaming generation", async () => {
    const completedEvent = {
      type: "image_generation.completed",
      b64_json: "c3RyZWFtZWQ=",
      background: "opaque",
      created_at: 1,
      output_format: "jpeg",
      quality: "medium",
      size: "1024x1024",
      usage: {
        input_tokens: 1,
        input_tokens_details: { image_tokens: 0, text_tokens: 1 },
        output_tokens: 1,
        total_tokens: 2,
      },
    } satisfies ImageGenStreamEvent;
    vi.spyOn(openai, "post").mockImplementation(
      () =>
        Promise.resolve(
          new Stream<ImageGenStreamEvent>(
            () =>
              (async function* () {
                yield completedEvent;
              })(),
            new AbortController(),
          ),
        ) as never,
    );

    const stream = await openai.images.generate({
      prompt: "a lighthouse",
      stream: true,
    });
    for await (const _event of stream) {
      // Consume the caller's half of the tee'd stream.
    }

    await vi.waitFor(() => expect(memoryExporter.getFinishedSpans()).toHaveLength(1));
    const span = memoryExporter.getFinishedSpans()[0];
    expect(span.attributes["output.images.0.image.url"]).toBe(
      "data:image/jpeg;base64,c3RyZWFtZWQ=",
    );
  });
});
