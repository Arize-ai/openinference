import { once } from "node:events";
import {
  createReadStream,
  mkdtempSync,
  promises as fileSystem,
  rmdirSync,
  unlinkSync,
  writeFileSync,
} from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { context, diag } from "@opentelemetry/api";
import { suppressTracing } from "@opentelemetry/core";
import { isWrapped } from "@opentelemetry/instrumentation";
import {
  AlwaysOffSampler,
  InMemorySpanExporter,
  type Sampler,
  SimpleSpanProcessor,
} from "@opentelemetry/sdk-trace-base";
import { NodeTracerProvider } from "@opentelemetry/sdk-trace-node";
import OpenAI, { APIPromise } from "openai";
import type { ImageGenStreamEvent, ImagesResponse } from "openai/resources/images";
import { Stream } from "openai/streaming";
import { vi } from "vitest";

import {
  generateTraceConfig,
  REDACTED_VALUE,
  setSession,
  type TraceConfigOptions,
} from "@arizeai/openinference-core";

import { OpenAIInstrumentation } from "../src";
import { getInputImageAttributes } from "../src/imageAttributes";
import type * as ImageAttributes from "../src/imageAttributes";

const uploadReads = vi.hoisted((): Promise<unknown>[] => []);

// Records each upload read, so mocked requests can finish after the reads they started.
vi.mock("../src/imageAttributes", async (importOriginal) => {
  const original = await importOriginal<typeof ImageAttributes>();
  return {
    ...original,
    getInputImageAttributes: (
      params: Parameters<typeof original.getInputImageAttributes>[0],
    ): ReturnType<typeof original.getInputImageAttributes> => {
      const uploadRead = original.getInputImageAttributes(params);
      uploadReads.push(uploadRead);
      return uploadRead;
    },
  };
});

const waitForUploadRead = () => Promise.all(uploadReads);

beforeEach(() => {
  uploadReads.length = 0;
});

describe("OpenAIInstrumentation - Images", () => {
  const memoryExporter = new InMemorySpanExporter();
  const tracerProvider = new NodeTracerProvider({
    spanProcessors: [new SimpleSpanProcessor(memoryExporter)],
  });
  tracerProvider.register();
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

    await openai.images.generate({
      model: "gpt-image-1",
      prompt: "a lighthouse",
      output_format: "jpeg",
    });

    const span = memoryExporter.getFinishedSpans()[0];
    expect(span.name).toBe("OpenAI Images");
    expect(span.attributes["openinference.span.kind"]).toBe("LLM");
    expect(span.attributes["input.mime_type"]).toBe("application/json");
    expect(span.attributes["output.mime_type"]).toBe("application/json");
    expect(span.attributes["llm.system"]).toBe("openai");
    expect(span.attributes["llm.model_name"]).toBe("gpt-image-1");
    expect(span.attributes["llm.provider"]).toBe("openai");
    expect(span.attributes["output.images.0.image.url"]).toBe("https://example.com/generated.png");
    expect(span.attributes["output.images.1.image.url"]).toBe("data:image/webp;base64,aW1hZ2U=");
  });

  it("keeps a base64 image that is already a data URL", async () => {
    vi.spyOn(openai, "post").mockImplementation(
      // @ts-expect-error return only needs to model the parsed SDK response.
      async () => ({ created: 1, data: [{ b64_json: "data:image/gif;base64,aW1hZ2U=" }] }),
    );

    await openai.images.generate({ prompt: "a lighthouse" });

    const span = memoryExporter.getFinishedSpans()[0];
    expect(span.attributes["output.images.0.image.url"]).toBe("data:image/gif;base64,aW1hZ2U=");
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

  it.each<[string, (image: File) => Promise<unknown>]>([
    ["edit", (image) => openai.images.edit({ image, prompt: "remove the background" })],
    ["createVariation", (image) => openai.images.createVariation({ image })],
  ])("records input images of a rejected %s request", async (_method, sendRequest) => {
    // @ts-expect-error test rejection does not need to model APIPromise internals.
    vi.spyOn(openai, "post").mockImplementation(async () => {
      await waitForUploadRead();
      throw new Error("request failed");
    });
    const image = new File([new Uint8Array([0x89, 0x50, 0x4e, 0x47])], "source.png", {
      type: "image/png",
    });

    await expect(sendRequest(image)).rejects.toThrow("request failed");

    const span = memoryExporter.getFinishedSpans()[0];
    expect(span.status.code).toBe(2);
    expect(span.attributes["input.images.0.image.url"]).toBe(
      `data:image/png;base64,${Buffer.from(await image.arrayBuffer()).toString("base64")}`,
    );
  });

  it.each<[string, () => unknown]>([
    ["generate", () => openai.images.generate({ prompt: "a lighthouse" })],
    [
      "edit",
      () => openai.images.edit({ image: new File([], "a.png"), prompt: "remove the background" }),
    ],
    ["createVariation", () => openai.images.createVariation({ image: new File([], "a.png") })],
  ])("ends the span when %s throws before sending the request", (_method, sendRequest) => {
    vi.spyOn(openai, "post").mockImplementation(() => {
      throw new Error("invalid request");
    });

    expect(sendRequest).toThrow("invalid request");

    const spans = memoryExporter.getFinishedSpans();
    expect(spans).toHaveLength(1);
    expect(spans[0].status.code).toBe(2);
    expect(spans[0].status.message).toBe("invalid request");
  });

  describe("over a real APIPromise", () => {
    // Assigned directly rather than via vi.spyOn: the spy awaits the returned
    // APIPromise, which would parse the one-shot response body itself.
    const mockPost = (responsePromise: Promise<Response>) => {
      openai.post = (() =>
        new APIPromise(
          openai,
          responsePromise.then((response) => ({
            response,
            options: { method: "post", path: "/images/generations" },
            controller: new AbortController(),
            requestLogID: "mock",
            retryOfRequestLogID: undefined,
            startTime: Date.now(),
          })) as never,
        )) as never;
    };
    const jsonResponse = () =>
      new Response(JSON.stringify({ created: 1, data: [{ url: "https://example.com/a.png" }] }), {
        headers: { "content-type": "application/json" },
      });
    afterEach(() => {
      // Drop the own-property override so the prototype's post is used again.
      Reflect.deleteProperty(openai, "post");
    });
    const requests: Array<[string, () => APIPromise<ImagesResponse>]> = [
      ["generate", () => openai.images.generate({ prompt: "a lighthouse" })],
      [
        "edit",
        () => openai.images.edit({ image: new File([], "a.png"), prompt: "remove the background" }),
      ],
      ["createVariation", () => openai.images.createVariation({ image: new File([], "a.png") })],
    ];

    it("records outputs and keeps the prompt out of invocation parameters", async () => {
      mockPost(Promise.resolve(jsonResponse()));

      const result = await openai.images.generate({ prompt: "a lighthouse", size: "1024x1024" });

      expect(result.data?.[0].url).toBe("https://example.com/a.png");
      const span = memoryExporter.getFinishedSpans()[0];
      expect(span.attributes["output.images.0.image.url"]).toBe("https://example.com/a.png");
      expect(JSON.parse(span.attributes["llm.invocation_parameters"] as string)).toEqual({
        size: "1024x1024",
      });
    });

    it.each(requests)(
      "does not consume the body for %s callers using asResponse()",
      async (_method, sendRequest) => {
        mockPost(Promise.resolve(jsonResponse()));

        const response = await sendRequest().asResponse();
        await new Promise((resolve) => setTimeout(resolve, 0));

        await expect(response.json()).resolves.toMatchObject({
          data: [{ url: "https://example.com/a.png" }],
        });
        const spans = memoryExporter.getFinishedSpans();
        expect(spans).toHaveLength(1);
        expect(spans[0].status.code).toBe(1);
        expect(spans[0].attributes["output.images.0.image.url"]).toBeUndefined();
      },
    );

    it("records input images for callers using asResponse()", async () => {
      mockPost(
        Promise.resolve()
          .then(waitForUploadRead)
          .then(() => jsonResponse()),
      );
      const image = new File([new Uint8Array([0x89, 0x50, 0x4e, 0x47])], "source.png", {
        type: "image/png",
      });

      await openai.images.edit({ image, prompt: "remove the background" }).asResponse();
      await new Promise((resolve) => setTimeout(resolve, 0));

      const span = memoryExporter.getFinishedSpans()[0];
      expect(span.status.code).toBe(1);
      expect(span.attributes["input.images.0.image.url"]).toBe(
        `data:image/png;base64,${Buffer.from(await image.arrayBuffer()).toString("base64")}`,
      );
    });

    it("does not touch the ended span when a caller awaits after asResponse()", async () => {
      mockPost(Promise.resolve(jsonResponse()));
      const warnSpy = vi.spyOn(diag, "warn");
      const errorSpy = vi.spyOn(diag, "error");

      const request = openai.images.generate({ prompt: "a lighthouse" });
      await request.asResponse();
      await new Promise((resolve) => setTimeout(resolve, 0));
      const result = await request;

      expect(result.data?.[0].url).toBe("https://example.com/a.png");
      expect(memoryExporter.getFinishedSpans()).toHaveLength(1);
      expect(warnSpy).not.toHaveBeenCalled();
      expect(errorSpy).not.toHaveBeenCalled();
    });

    it("records the result once for callers using withResponse()", async () => {
      mockPost(Promise.resolve(jsonResponse()));

      const { data, response } = await openai.images
        .generate({ prompt: "a lighthouse" })
        .withResponse();
      await new Promise((resolve) => setTimeout(resolve, 0));

      expect(data.data?.[0].url).toBe("https://example.com/a.png");
      expect(response.status).toBe(200);
      const spans = memoryExporter.getFinishedSpans();
      expect(spans).toHaveLength(1);
      expect(spans[0].attributes["output.images.0.image.url"]).toBe("https://example.com/a.png");
    });

    it.each(requests)(
      "ends the %s span with an error when the body fails to parse",
      async (_method, sendRequest) => {
        mockPost(
          Promise.resolve(
            new Response("{not json", { headers: { "content-type": "application/json" } }),
          ),
        );

        await expect(sendRequest()).rejects.toThrow(SyntaxError);

        const spans = memoryExporter.getFinishedSpans();
        expect(spans).toHaveLength(1);
        expect(spans[0].status.code).toBe(2);
        expect(spans[0].events.map((event) => event.name)).toEqual(["exception"]);
      },
    );

    it("ends the span when the HTTP request rejects", async () => {
      mockPost(Promise.reject(new Error("request failed")));

      await expect(openai.images.generate({ prompt: "a lighthouse" })).rejects.toThrow(
        "request failed",
      );

      const span = memoryExporter.getFinishedSpans()[0];
      expect(span.status.code).toBe(2);
      expect(span.status.message).toBe("request failed");
      expect(span.events.map((event) => event.name)).toEqual(["exception"]);
    });
  });

  it.each([
    ["data that is not an array", '{"created":1,"data":{"url":"u"}}', 200],
    ["a null image", '{"created":1,"data":[null]}', 200],
    ["no body", null, 204],
  ])("does not fail the request for a response with %s", async (_name, body, status) => {
    const client = new OpenAI({
      apiKey: "fake-api-key",
      fetch: async () =>
        new Response(body, { status, headers: { "content-type": "application/json" } }),
    });
    const debugSpy = vi.spyOn(diag, "debug");

    await client.images.generate({ prompt: "a lighthouse" });

    const spans = memoryExporter.getFinishedSpans();
    expect(spans).toHaveLength(1);
    expect(spans[0].status.code).toBe(1);
    expect(spans[0].attributes["input.value"]).toBeDefined();
    expect(debugSpy).toHaveBeenCalledWith(
      "Failed to record OpenAI images response",
      expect.any(TypeError),
    );
  });

  it("records a body that is not a JSON object as it is", async () => {
    const client = new OpenAI({
      apiKey: "fake-api-key",
      fetch: async () =>
        new Response("<html>gateway</html>", { headers: { "content-type": "text/html" } }),
    });

    await client.images.generate({ prompt: "a lighthouse" });

    const span = memoryExporter.getFinishedSpans()[0];
    expect(span.status.code).toBe(1);
    expect(span.attributes["output.value"]).toBe('"<html>gateway</html>"');
  });

  it.each<[string, () => Promise<unknown>]>([
    ["generate", () => openai.images.generate({ prompt: "a lighthouse", output_format: "webp" })],
    [
      "edit",
      () =>
        openai.images.edit({
          image: new File([], "a.png"),
          prompt: "remove the background",
          output_format: "webp",
        }),
    ],
  ])("uses the format requested from %s when the response has none", async (_method, send) => {
    vi.spyOn(openai, "post").mockImplementation(
      // @ts-expect-error return only needs to model the parsed SDK response.
      async () => ({ created: 1, data: [{ b64_json: "aW1hZ2U=" }] }),
    );

    await send();

    const span = memoryExporter.getFinishedSpans()[0];
    expect(span.attributes["output.images.0.image.url"]).toBe("data:image/webp;base64,aW1hZ2U=");
  });

  it("keeps variation uploads out of input.value", async () => {
    vi.spyOn(openai, "post").mockImplementation(
      // @ts-expect-error return only needs to model the parsed SDK response.
      async () => ({ created: 1, data: [] }),
    );
    const image = new File([new Uint8Array(8)], "source.png", { type: "image/png" });

    await openai.images.createVariation({ image, n: 2 });

    const span = memoryExporter.getFinishedSpans()[0];
    expect(JSON.parse(span.attributes["input.value"] as string)).toEqual({ n: 2 });
  });

  it.each([
    ["a declared image type", [1, 2, 3], "a.bin", "image/gif", "image/gif"],
    [
      "the bytes, over a declared non-image type",
      [0xff, 0xd8, 0xff],
      "a.txt",
      "text/plain",
      "image/jpeg",
    ],
    ["the bytes, with no declared type", [0xff, 0xd8, 0xff], "a.bin", "", "image/jpeg"],
    [
      "PNG bytes, over the filename",
      [0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a],
      "a.webp",
      "",
      "image/png",
    ],
    ["GIF bytes", [...Buffer.from("GIF89a")], "a.bin", "", "image/gif"],
    ["WebP bytes", [...Buffer.from("RIFF\0\0\0\0WEBP")], "a.bin", "", "image/webp"],
    ["the filename, when the bytes are unknown", [1, 2, 3], "a.webp", "", "image/webp"],
    ["a .jpg filename", [1, 2, 3], "a.jpg", "", "image/jpeg"],
    ["PNG, when nothing else is known", [1, 2, 3], "a.bin", "", "image/png"],
  ])("detects an upload's media type from %s", async (_name, bytes, filename, type, expected) => {
    const image = new File([new Uint8Array(bytes)], filename, { type });

    await expect(getInputImageAttributes({ uploads: [image] })).resolves.toEqual({
      "input.images.0.image.url": `data:${expected};base64,${Buffer.from(bytes).toString("base64")}`,
    });
  });

  it("captures reusable edit image and mask uploads without serializing their bytes", async () => {
    const response = {
      created: 1,
      data: [{ b64_json: "ZWRpdGVk" }],
    } satisfies ImagesResponse;
    vi.spyOn(openai, "post").mockImplementation(
      // @ts-expect-error return only needs to model the parsed SDK response.
      async () => {
        await waitForUploadRead();
        return response;
      },
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

  it("captures every image of an edit given an array, in order", async () => {
    vi.spyOn(openai, "post").mockImplementation(
      // @ts-expect-error return only needs to model the parsed SDK response.
      async () => {
        await waitForUploadRead();
        return { created: 1, data: [] };
      },
    );
    const firstBytes = [0x89, 0x50, 0x4e, 0x47];
    const secondBytes = [0xff, 0xd8, 0xff];

    await openai.images.edit({
      image: [
        new File([new Uint8Array(firstBytes)], "first.png", { type: "image/png" }),
        new File([new Uint8Array(secondBytes)], "second.jpg", { type: "image/jpeg" }),
      ],
      prompt: "combine these",
    });

    const span = memoryExporter.getFinishedSpans()[0];
    expect(span.attributes["input.images.0.image.url"]).toBe(
      `data:image/png;base64,${Buffer.from(firstBytes).toString("base64")}`,
    );
    expect(span.attributes["input.images.1.image.url"]).toBe(
      `data:image/jpeg;base64,${Buffer.from(secondBytes).toString("base64")}`,
    );
  });

  it("does not wait for an upload read to finish", async () => {
    const response = { created: 1, data: [] } satisfies ImagesResponse;
    vi.spyOn(openai, "post").mockImplementation(
      // @ts-expect-error return only needs to model the parsed SDK response.
      async () => response,
    );
    const image = new File([new Uint8Array(8)], "source.png", { type: "image/png" });
    vi.spyOn(image, "arrayBuffer").mockReturnValue(new Promise(() => undefined));

    await expect(openai.images.createVariation({ image })).resolves.toEqual(response);

    const span = memoryExporter.getFinishedSpans()[0];
    expect(span.status.code).toBe(1);
    expect(span.attributes["input.images.0.image.url"]).toBeUndefined();
  });

  it("logs uploads that fail to read and leaves them off the span", async () => {
    const image = new File([new Uint8Array(8)], "source.png", { type: "image/png" });
    const readError = new Error("read failed");
    vi.spyOn(image, "arrayBuffer").mockRejectedValue(readError);
    const debugSpy = vi.spyOn(diag, "debug");

    await expect(getInputImageAttributes({ uploads: [image] })).resolves.toEqual({});
    expect(debugSpy).toHaveBeenCalledWith("Failed to capture OpenAI image upload", readError);
  });

  it("does not read hidden uploads", async () => {
    const image = new File([new Uint8Array(128)], "source.png", { type: "image/png" });
    const arrayBufferSpy = vi.spyOn(image, "arrayBuffer");

    for (const traceConfig of [{ hideInputs: true }, { hideInputImages: true }]) {
      await expect(
        getInputImageAttributes({
          uploads: [image],
          traceConfigOptions: generateTraceConfig(traceConfig),
        }),
      ).resolves.toEqual({});
    }
    expect(arrayBufferSpy).not.toHaveBeenCalled();
  });

  it("does not read uploads when tracing is suppressed", async () => {
    const response = { created: 1, data: [] } satisfies ImagesResponse;
    vi.spyOn(openai, "post").mockImplementation(
      // @ts-expect-error return only needs to model the parsed SDK response.
      async () => response,
    );
    const image = new File([new Uint8Array(128)], "source.png", { type: "image/png" });
    const arrayBufferSpy = vi.spyOn(image, "arrayBuffer");

    await context.with(suppressTracing(context.active()), () =>
      openai.images.createVariation({ image }),
    );

    expect(arrayBufferSpy).not.toHaveBeenCalled();
    expect(memoryExporter.getFinishedSpans()).toHaveLength(0);
  });

  it("redacts uploads longer than base64ImageMaxLength without reading them", async () => {
    const image = new File([new Uint8Array(128)], "source.png", { type: "image/png" });
    const arrayBufferSpy = vi.spyOn(image, "arrayBuffer");
    const upload = new Response(new Uint8Array(128), {
      headers: { "content-length": "128" },
    });
    const cloneSpy = vi.spyOn(upload, "clone");

    await expect(
      getInputImageAttributes({
        uploads: [image, upload],
        traceConfigOptions: generateTraceConfig({ base64ImageMaxLength: 64 }),
      }),
    ).resolves.toEqual({
      "input.images.0.image.url": REDACTED_VALUE,
      "input.images.1.image.url": REDACTED_VALUE,
    });
    expect(arrayBufferSpy).not.toHaveBeenCalled();
    expect(cloneSpy).not.toHaveBeenCalled();
  });

  describe("with a Response that has no Content-Length", () => {
    const chunkLength = 16;
    const totalChunks = 20;
    /**
     * Creates a response whose body counts how many chunks have been read from it.
     */
    const createChunkedResponse = () => {
      let pulledChunks = 0;
      const cancelSource = vi.fn();
      const body = new ReadableStream<Uint8Array>({
        cancel: cancelSource,
        pull(controller) {
          if (pulledChunks === totalChunks) {
            controller.close();
            return;
          }
          pulledChunks++;
          controller.enqueue(new Uint8Array(chunkLength));
        },
      });
      return {
        upload: new Response(body, { headers: { "content-type": "image/png" } }),
        getPulledChunks: () => pulledChunks,
        cancelSource,
      };
    };

    it("stops reading once the body is too long and leaves it readable", async () => {
      const { upload, getPulledChunks } = createChunkedResponse();

      await expect(
        getInputImageAttributes({
          uploads: [upload],
          traceConfigOptions: generateTraceConfig({ base64ImageMaxLength: 64 }),
        }),
      ).resolves.toEqual({ "input.images.0.image.url": REDACTED_VALUE });

      expect(getPulledChunks()).toBeLessThan(totalChunks);
      const sentBytes = await upload.arrayBuffer();
      expect(sentBytes.byteLength).toBe(chunkLength * totalChunks);
    });

    it("gives up its copy of a body that is too long", async () => {
      const { upload, cancelSource } = createChunkedResponse();

      await getInputImageAttributes({
        uploads: [upload],
        traceConfigOptions: generateTraceConfig({ base64ImageMaxLength: 64 }),
      });
      // The source is only cancelled once both copies of the body are.
      await upload.body?.cancel();

      expect(cancelSource).toHaveBeenCalledOnce();
    });

    it("redacts a long upload and still sends all of it", async () => {
      const uploadLength = 5_000_000;
      const sentBodies: Buffer[] = [];
      const client = new OpenAI({
        apiKey: "fake-api-key",
        fetch: async (_url, init) => {
          // The SDK first probes fetch with a request that has no body to check FormData support.
          if (init?.body) {
            sentBodies.push(Buffer.from(await new Response(init.body).arrayBuffer()));
          }
          await waitForUploadRead();
          return new Response(JSON.stringify({ created: 1, data: [] }), {
            headers: { "content-type": "application/json" },
          });
        },
      });
      const upload = new Response(new Uint8Array(uploadLength), {
        headers: { "content-type": "image/png" },
      });

      await client.images.createVariation({ image: upload });

      expect(sentBodies).toHaveLength(1);
      expect(sentBodies[0].length).toBeGreaterThan(uploadLength);
      const span = memoryExporter.getFinishedSpans()[0];
      expect(span.attributes["input.images.0.image.url"]).toBe(REDACTED_VALUE);
    });

    it("captures a body within the limit and leaves it readable", async () => {
      const { upload } = createChunkedResponse();
      const expectedBase64 = Buffer.alloc(chunkLength * totalChunks).toString("base64");

      await expect(getInputImageAttributes({ uploads: [upload] })).resolves.toEqual({
        "input.images.0.image.url": `data:image/png;base64,${expectedBase64}`,
      });

      const sentBytes = await upload.arrayBuffer();
      expect(sentBytes.byteLength).toBe(chunkLength * totalChunks);
    });
  });

  it("redacts data URLs that only exceed base64ImageMaxLength with their prefix", async () => {
    const image = new File([new Uint8Array(3)], "source.png", { type: "image/png" });

    await expect(
      getInputImageAttributes({
        uploads: [image],
        traceConfigOptions: generateTraceConfig({ base64ImageMaxLength: 4 }),
      }),
    ).resolves.toEqual({ "input.images.0.image.url": REDACTED_VALUE });
  });

  describe("with a Node file stream", () => {
    const imageBytes = new Uint8Array([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]);
    let directory: string;
    let imagePath: string;

    beforeEach(() => {
      directory = mkdtempSync(join(tmpdir(), "openinference-openai-images-"));
      imagePath = join(directory, "source.png");
      writeFileSync(imagePath, imageBytes);
    });
    afterEach(() => {
      unlinkSync(imagePath);
      rmdirSync(directory);
    });

    it("captures the upload and leaves it for the SDK to send", async () => {
      const sentBodies: Buffer[] = [];
      const client = new OpenAI({
        apiKey: "fake-api-key",
        fetch: async (_url, init) => {
          // Some SDK versions first probe fetch with a request that has no body.
          if (init?.body) {
            sentBodies.push(Buffer.from(await new Response(init.body).arrayBuffer()));
          }
          await waitForUploadRead();
          return new Response(JSON.stringify({ created: 1, data: [] }), {
            headers: { "content-type": "application/json" },
          });
        },
      });

      await client.images.createVariation({ image: createReadStream(imagePath) });

      expect(sentBodies).toHaveLength(1);
      expect(sentBodies[0].includes(Buffer.from(imageBytes))).toBe(true);
      const span = memoryExporter.getFinishedSpans()[0];
      expect(span.attributes["input.images.0.image.url"]).toBe(
        `data:image/png;base64,${Buffer.from(imageBytes).toString("base64")}`,
      );
    });

    it("redacts long files without reading them", async () => {
      const stream = createReadStream(imagePath);
      await once(stream, "open");
      const readFileSpy = vi.spyOn(fileSystem, "readFile");

      await expect(
        getInputImageAttributes({
          uploads: [stream],
          traceConfigOptions: generateTraceConfig({ base64ImageMaxLength: 4 }),
        }),
      ).resolves.toEqual({ "input.images.0.image.url": REDACTED_VALUE });
      expect(readFileSpy).not.toHaveBeenCalled();
      stream.destroy();
    });

    it.each([
      ["start", { start: 4 }],
      ["end", { end: 3 }],
    ])("skips streams with a %s offset", async (_name, range) => {
      const stream = createReadStream(imagePath, range);
      await once(stream, "open");

      await expect(getInputImageAttributes({ uploads: [stream] })).resolves.toEqual({});
      stream.destroy();
    });

    it("skips objects that only look like file streams", async () => {
      await expect(
        getInputImageAttributes({
          uploads: [{ path: imagePath, start: undefined, end: Infinity }],
        }),
      ).resolves.toEqual({});
    });
  });

  it.skipIf(process.platform === "win32")("skips streams that are not regular files", async () => {
    const stream = createReadStream("/dev/null");

    await expect(getInputImageAttributes({ uploads: [stream] })).resolves.toEqual({});
    stream.destroy();
  });

  it("keeps base64 images out of output.value", async () => {
    const response = {
      created: 1,
      data: [{ b64_json: "aW1hZ2U=", revised_prompt: "a tall lighthouse" }],
    } satisfies ImagesResponse;
    vi.spyOn(openai, "post").mockImplementation(
      // @ts-expect-error return only needs to model the parsed SDK response.
      async () => response,
    );

    const result = await openai.images.generate({ prompt: "a lighthouse" });

    expect(result.data?.[0].b64_json).toBe("aW1hZ2U=");
    const { attributes } = memoryExporter.getFinishedSpans()[0];
    expect(JSON.parse(attributes["output.value"] as string)).toEqual({
      created: 1,
      data: [{ revised_prompt: "a tall lighthouse" }],
    });
    expect(attributes["output.images.0.image.url"]).toBe("data:image/png;base64,aW1hZ2U=");
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
    const outputValue = JSON.parse(span.attributes["output.value"] as string);
    expect(outputValue.type).toBe("image_generation.completed");
    expect(outputValue.b64_json).toBeUndefined();
  });

  describe("with a streamed response", () => {
    /**
     * Creates a client whose requests get the given server-sent events, once uploads are read.
     */
    const createStreamClient = (events: ReadonlyArray<{ event: string; data: unknown }>) =>
      new OpenAI({
        apiKey: "fake-api-key",
        fetch: async () => {
          await waitForUploadRead();
          const body = events
            .map(({ event, data }) => `event: ${event}\ndata: ${JSON.stringify(data)}\n\n`)
            .join("");
          return new Response(body, { headers: { "content-type": "text/event-stream" } });
        },
      });
    const consumeStream = async (stream: AsyncIterable<unknown>) => {
      for await (const _event of stream) {
        // Consume the caller's half of the tee'd stream.
      }
    };
    const getFinishedSpan = async () => {
      await vi.waitFor(() => expect(memoryExporter.getFinishedSpans()).toHaveLength(1));
      return memoryExporter.getFinishedSpans()[0];
    };

    it("ends the span with an error when no image completes", async () => {
      const client = createStreamClient([
        {
          event: "image_generation.partial_image",
          data: { type: "image_generation.partial_image", b64_json: "cGFydGlhbA==" },
        },
      ]);

      await consumeStream(await client.images.generate({ prompt: "a lighthouse", stream: true }));

      const span = await getFinishedSpan();
      expect(span.status.code).toBe(2);
      expect(span.attributes["output.images.0.image.url"]).toBeUndefined();
    });

    it("records an error that arrives during the stream", async () => {
      const client = createStreamClient([{ event: "error", data: { error: { message: "boom" } } }]);

      const stream = await client.images.generate({ prompt: "a lighthouse", stream: true });
      await expect(consumeStream(stream)).rejects.toThrow("boom");

      const span = await getFinishedSpan();
      expect(span.status.code).toBe(2);
      expect(span.status.message).toContain("boom");
      expect(span.events.map((event) => event.name)).toEqual(["exception"]);
    });

    it("captures the input and output images of a streamed edit", async () => {
      const client = createStreamClient([
        {
          event: "image_edit.completed",
          data: { type: "image_edit.completed", b64_json: "ZWRpdGVk", output_format: "png" },
        },
      ]);
      const imageBytes = new Uint8Array([0x89, 0x50, 0x4e, 0x47]);
      const image = new File([imageBytes], "source.png", { type: "image/png" });

      await consumeStream(
        await client.images.edit({ image, prompt: "remove the background", stream: true }),
      );

      const span = await getFinishedSpan();
      expect(span.status.code).toBe(1);
      expect(span.attributes["input.images.0.image.url"]).toBe(
        `data:image/png;base64,${Buffer.from(imageBytes).toString("base64")}`,
      );
      expect(span.attributes["output.images.0.image.url"]).toBe("data:image/png;base64,ZWRpdGVk");
    });
  });
});

describe("OpenAIInstrumentation - Images configuration", () => {
  const response = {
    created: 1,
    data: [{ b64_json: "aW1hZ2U=" }],
  } satisfies ImagesResponse;
  const imageBytes = new Uint8Array([0x89, 0x50, 0x4e, 0x47]);
  const imageDataURL = `data:image/png;base64,${Buffer.from(imageBytes).toString("base64")}`;

  /**
   * Runs an images.edit call through an instrumentor set up with the given options.
   *
   * @param params.traceConfig - The trace config given to the instrumentor
   * @param params.sampler - The sampler given to the tracer provider
   * @param params.runRequest - Wraps the request, such as to set context attributes
   * @returns The finished spans
   */
  async function getEditSpans({
    traceConfig,
    sampler,
    runRequest = (request) => request(),
  }: {
    traceConfig?: TraceConfigOptions;
    sampler?: Sampler;
    runRequest?: <T>(request: () => Promise<T>) => Promise<T>;
  }) {
    const memoryExporter = new InMemorySpanExporter();
    const tracerProvider = new NodeTracerProvider({
      sampler,
      spanProcessors: [new SimpleSpanProcessor(memoryExporter)],
    });
    const instrumentation = new OpenAIInstrumentation({ tracerProvider, traceConfig });
    instrumentation.disable();
    // @ts-expect-error moduleExports is private; tests set it for manual module mocking.
    instrumentation._modules[0].moduleExports = OpenAI;
    instrumentation.enable();
    try {
      const openai = new OpenAI({ apiKey: "fake-api-key" });
      vi.spyOn(openai, "post").mockImplementation(
        // @ts-expect-error return only needs to model the parsed SDK response.
        async () => {
          await waitForUploadRead();
          return response;
        },
      );
      const image = new File([imageBytes], "source.png", { type: "image/png" });
      const arrayBufferSpy = vi.spyOn(image, "arrayBuffer");
      await runRequest(() => openai.images.edit({ image, prompt: "secret prompt" }));
      return { spans: memoryExporter.getFinishedSpans(), arrayBufferSpy };
    } finally {
      instrumentation.disable();
    }
  }

  afterEach(() => vi.restoreAllMocks());

  it.each(["generate", "edit", "createVariation"] as const)(
    "restores images.%s when disabled",
    (method) => {
      const instrumentation = new OpenAIInstrumentation({
        tracerProvider: new NodeTracerProvider(),
      });
      instrumentation.disable();
      // @ts-expect-error moduleExports is private; tests set it for manual module mocking.
      instrumentation._modules[0].moduleExports = OpenAI;
      instrumentation.enable();
      expect(isWrapped(OpenAI.Images.prototype[method])).toBe(true);

      instrumentation.disable();

      expect(isWrapped(OpenAI.Images.prototype[method])).toBe(false);
    },
  );

  it("does not record or read input images with hideInputImages", async () => {
    const { spans, arrayBufferSpy } = await getEditSpans({
      traceConfig: { hideInputImages: true },
    });

    expect(spans[0].attributes["input.images.0.image.url"]).toBeUndefined();
    expect(spans[0].attributes["output.images.0.image.url"]).toBeDefined();
    expect(arrayBufferSpy).not.toHaveBeenCalled();
  });

  it("does not record the prompt or input images with hideInputs", async () => {
    const { spans, arrayBufferSpy } = await getEditSpans({ traceConfig: { hideInputs: true } });

    expect(spans[0].attributes["input.value"]).toBe(REDACTED_VALUE);
    expect(spans[0].attributes["input.images.0.image.url"]).toBeUndefined();
    expect(spans[0].attributes["llm.invocation_parameters"]).toBe("{}");
    expect(arrayBufferSpy).not.toHaveBeenCalled();
  });

  it("does not record output images with hideOutputs", async () => {
    const { spans } = await getEditSpans({ traceConfig: { hideOutputs: true } });

    expect(spans[0].attributes["output.value"]).toBe(REDACTED_VALUE);
    expect(spans[0].attributes["output.images.0.image.url"]).toBeUndefined();
    expect(spans[0].attributes["input.images.0.image.url"]).toBe(imageDataURL);
  });

  it("applies base64ImageMaxLength from the instrumentor's trace config", async () => {
    const { spans, arrayBufferSpy } = await getEditSpans({
      traceConfig: { base64ImageMaxLength: 4 },
    });

    expect(spans[0].attributes["input.images.0.image.url"]).toBe(REDACTED_VALUE);
    expect(spans[0].attributes["output.images.0.image.url"]).toBe(REDACTED_VALUE);
    expect(arrayBufferSpy).not.toHaveBeenCalled();
  });

  it("does not read uploads for spans that are sampled out", async () => {
    const { spans, arrayBufferSpy } = await getEditSpans({ sampler: new AlwaysOffSampler() });

    expect(spans).toHaveLength(0);
    expect(arrayBufferSpy).not.toHaveBeenCalled();
  });

  it("propagates context attributes", async () => {
    const { spans } = await getEditSpans({
      runRequest: (request) =>
        context.with(setSession(context.active(), { sessionId: "session-id" }), request),
    });

    expect(spans[0].attributes["session.id"]).toBe("session-id");
  });
});
