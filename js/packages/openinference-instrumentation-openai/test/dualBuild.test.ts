import { createRequire } from "node:module";
import { join } from "node:path";

import { isWrapped } from "@opentelemetry/instrumentation";
import { NodeTracerProvider } from "@opentelemetry/sdk-trace-node";
import type * as openai from "openai";
import { describe, expect, it } from "vitest";

import { OpenAIInstrumentation, isPatched } from "../src/instrumentation";

// Not import.meta.url: this file is also compiled under the package's
// CommonJS tsconfig, which rejects import.meta. vitest runs with the
// package directory as cwd, so its package.json is a valid resolution base.
const requireCjs = createRequire(join(process.cwd(), "package.json"));

type ManualModule = Parameters<OpenAIInstrumentation["manuallyInstrument"]>[0];

type OpenAIClass = typeof openai.OpenAI & {
  Chat: { Completions: { prototype: { create: unknown } } };
};

// openai is a dual package: its CJS (`require`) and ESM (`import`) builds are
// separate module objects with separate class prototypes. Patching one build
// does not instrument consumers of the other, so both must be patchable in
// the same process — previously the module-global _isOpenInferencePatched
// flag made whichever build was patched first block the other forever
// (#3557).
describe("dual-build patching", () => {
  it("patches the CJS and ESM builds independently, each at most once", async () => {
    const instrumentation = new OpenAIInstrumentation({
      tracerProvider: new NodeTracerProvider(),
    });

    const cjs = requireCjs("openai") as { OpenAI: OpenAIClass };
    const esm = (await import("openai")) as unknown as { OpenAI: OpenAIClass };

    // The premise: two distinct builds. If this ever fails, the SDK stopped
    // shipping dual builds and this whole test is moot.
    expect(esm.OpenAI).not.toBe(cjs.OpenAI);

    instrumentation.manuallyInstrument(cjs as unknown as ManualModule);
    expect(isWrapped(cjs.OpenAI.Chat.Completions.prototype.create)).toBe(true);

    // Before the WeakSet guard this second call was a silent no-op: the
    // global flag was already set by the CJS patch above.
    instrumentation.manuallyInstrument(esm as unknown as ManualModule);
    expect(isWrapped(esm.OpenAI.Chat.Completions.prototype.create)).toBe(true);

    expect(isPatched()).toBe(true);
  });

  it("does not double-wrap a build on a repeated patch", async () => {
    const instrumentation = new OpenAIInstrumentation({
      tracerProvider: new NodeTracerProvider(),
    });

    const cjs = requireCjs("openai") as { OpenAI: OpenAIClass };
    const esm = (await import("openai")) as unknown as { OpenAI: OpenAIClass };

    instrumentation.manuallyInstrument(cjs as unknown as ManualModule);
    instrumentation.manuallyInstrument(esm as unknown as ManualModule);

    // The CJS build is re-guarded by the openInferencePatched property; the
    // ESM namespace rejects that property write, so its re-guard is the
    // WeakSet alone. Both must hold: a second wrap would emit two spans per
    // call.
    const cjsPatched = cjs.OpenAI.Chat.Completions.prototype.create;
    instrumentation.manuallyInstrument(cjs as unknown as ManualModule);
    expect(cjs.OpenAI.Chat.Completions.prototype.create).toBe(cjsPatched);

    const esmPatched = esm.OpenAI.Chat.Completions.prototype.create;
    instrumentation.manuallyInstrument(esm as unknown as ManualModule);
    expect(esm.OpenAI.Chat.Completions.prototype.create).toBe(esmPatched);
  });
});
