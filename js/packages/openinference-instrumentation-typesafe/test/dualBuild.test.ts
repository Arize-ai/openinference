// Prototype methods are compared by identity, never called without a receiver.
/* oxlint-disable typescript/unbound-method */
import { execFileSync } from "node:child_process";
import { createRequire } from "node:module";
import { join } from "node:path";

import { isWrapped } from "@opentelemetry/instrumentation";
import { InMemorySpanExporter, SimpleSpanProcessor } from "@opentelemetry/sdk-trace-base";
import { NodeTracerProvider } from "@opentelemetry/sdk-trace-node";
import type * as TypeSafe from "@typesafe-ai/sdk";
import { describe, expect, it } from "vitest";

import { TypeSafeInstrumentation } from "../src";

const requireCjs = createRequire(join(process.cwd(), "package.json"));

describe("module loading", () => {
  it("instruments distinct CJS/ESM prototypes once each and restores both", async () => {
    const cjs = requireCjs("@typesafe-ai/sdk") as typeof TypeSafe;
    const esm = await import("@typesafe-ai/sdk");
    expect(cjs.TypeSafeClient).not.toBe(esm.TypeSafeClient);
    const exporter = new InMemorySpanExporter();
    const provider = new NodeTracerProvider({
      spanProcessors: [new SimpleSpanProcessor(exporter)],
    });
    const instrumentation = new TypeSafeInstrumentation({ tracerProvider: provider });
    try {
      for (const sdk of [cjs, esm]) {
        instrumentation.manuallyInstrument(sdk);
        const patched = sdk.TypeSafeClient.prototype.systemOne;
        instrumentation.manuallyInstrument(sdk);
        expect(sdk.TypeSafeClient.prototype.systemOne).toBe(patched);
        expect(isWrapped(patched)).toBe(true);
        const client = new sdk.TypeSafeClient({
          apiKey: "test",
          fetch: async () =>
            Response.json({
              model: "jev",
              answers: { ok: { type: "noul", noul: 0.9 } },
              usage: { input_tokens: 1, output_tokens: 1 },
            }),
        });
        const promise = client.systemOne({ state: "test", questions: { ok: sdk.noul("OK?") } });
        expect(promise).toBeInstanceOf(sdk.APIPromise);
        await promise.withResponse();
      }
      expect(exporter.getFinishedSpans()).toHaveLength(2);
      instrumentation.disable();
      expect(isWrapped(cjs.TypeSafeClient.prototype.systemOne)).toBe(false);
      expect(isWrapped(esm.TypeSafeClient.prototype.systemOne)).toBe(false);
      instrumentation.enable();
      expect(isWrapped(cjs.TypeSafeClient.prototype.systemOne)).toBe(true);
      expect(isWrapped(esm.TypeSafeClient.prototype.systemOne)).toBe(true);
    } finally {
      instrumentation.disable();
      await provider.shutdown();
    }
  });

  // Separate processes exercise Node's actual module loaders and published
  // artifacts rather than relying on Vite's transformed imports.
  it.each(["commonjs", "module"])("loads the built package and SDK with Node (%s)", (mode) => {
    const imports =
      mode === "commonjs"
        ? `
      const assert = require('node:assert/strict');
      const { TypeSafeInstrumentation } = require('./dist/src/index.js');
      const { InMemorySpanExporter, SimpleSpanProcessor } = require('@opentelemetry/sdk-trace-base');
      const { NodeTracerProvider } = require('@opentelemetry/sdk-trace-node');
    `
        : `
      import assert from 'node:assert/strict';
      import { TypeSafeInstrumentation } from './dist/esm/index.js';
      import { InMemorySpanExporter, SimpleSpanProcessor } from '@opentelemetry/sdk-trace-base';
      import { NodeTracerProvider } from '@opentelemetry/sdk-trace-node';
    `;
    const load =
      mode === "commonjs"
        ? `
      const sdk = require('@typesafe-ai/sdk');
    `
        : `
      const sdk = await import('@typesafe-ai/sdk');
      instrumentation.manuallyInstrument(sdk);
    `;
    const script = `${imports}
      (async () => {
        const exporter = new InMemorySpanExporter();
        const provider = new NodeTracerProvider({ spanProcessors: [new SimpleSpanProcessor(exporter)] });
        const instrumentation = new TypeSafeInstrumentation({ tracerProvider: provider });
        ${load}
        const client = new sdk.TypeSafeClient({ apiKey: 'test', fetch: async () => Response.json({ model: 'jev', answers: {}, usage: { input_tokens: 1, output_tokens: 2 } }) });
        const request = { state: 'test', questions: { ok: sdk.noul('OK?') } };
        const promise = client.systemOne(request);
        assert(promise instanceof sdk.APIPromise);
        await promise.withResponse();
        assert.equal(exporter.getFinishedSpans().length, 1);
        assert.equal(exporter.getFinishedSpans()[0].attributes['llm.token_count.total'], 3);
        instrumentation.disable();
        await client.systemOne(request);
        assert.equal(exporter.getFinishedSpans().length, 1);
        instrumentation.enable();
        await client.systemOne(request);
        assert.equal(exporter.getFinishedSpans().length, 2);
        instrumentation.disable();
        await provider.shutdown();
      })().catch((error) => { console.error(error); process.exitCode = 1; });
    `;
    expect(() =>
      execFileSync(process.execPath, [`--input-type=${mode}`, "--eval", script], {
        cwd: process.cwd(),
        timeout: 15000,
        stdio: "pipe",
      }),
    ).not.toThrow();
  });
});
