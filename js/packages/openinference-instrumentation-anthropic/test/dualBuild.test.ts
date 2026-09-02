import { createRequire } from "node:module";
import { join } from "node:path";

import type Anthropic from "@anthropic-ai/sdk";
import { isWrapped } from "@opentelemetry/instrumentation";
import { NodeTracerProvider } from "@opentelemetry/sdk-trace-node";
import { describe, expect, it } from "vitest";

import { AnthropicInstrumentation, isPatched } from "../src/instrumentation";

// Not import.meta.url: this file is also compiled under the package's
// CommonJS tsconfig, which rejects import.meta. vitest runs with the
// package directory as cwd, so its package.json is a valid resolution base.
const requireCjs = createRequire(join(process.cwd(), "package.json"));

type ManualModule = Parameters<AnthropicInstrumentation["manuallyInstrument"]>[0];

type AnthropicClass = typeof Anthropic & {
  Messages: { prototype: { create: unknown } };
};

// @anthropic-ai/sdk is a dual package: its CJS (`require`) and ESM (`import`)
// builds are separate module objects with separate class prototypes. Patching
// one build does not instrument consumers of the other, so both must be
// patchable in the same process — previously the module-global
// _isOpenInferencePatched flag made whichever build was patched first block
// the other forever (#3557).
describe("dual-build patching", () => {
  it("patches the CJS and ESM builds independently, each at most once", async () => {
    const instrumentation = new AnthropicInstrumentation({
      tracerProvider: new NodeTracerProvider(),
    });

    const cjs = requireCjs("@anthropic-ai/sdk") as { default?: AnthropicClass } & AnthropicClass;
    const esm = (await import("@anthropic-ai/sdk")) as unknown as {
      default: AnthropicClass;
    };
    const CjsAnthropic = cjs.default ?? cjs;
    const EsmAnthropic = esm.default;

    // The premise: two distinct builds. If this ever fails, the SDK stopped
    // shipping dual builds and this whole test is moot.
    expect(EsmAnthropic).not.toBe(CjsAnthropic);

    instrumentation.manuallyInstrument(cjs as unknown as ManualModule);
    expect(isWrapped(CjsAnthropic.Messages.prototype.create)).toBe(true);

    // Before the WeakSet guard this second call was a silent no-op: the
    // global flag was already set by the CJS patch above.
    instrumentation.manuallyInstrument(esm as unknown as ManualModule);
    expect(isWrapped(EsmAnthropic.Messages.prototype.create)).toBe(true);

    expect(isPatched()).toBe(true);

    instrumentation.disable();
    expect(isWrapped(CjsAnthropic.Messages.prototype.create)).toBe(false);
    expect(isWrapped(EsmAnthropic.Messages.prototype.create)).toBe(false);
    expect(isPatched()).toBe(false);
  });

  it("does not double-wrap a build on a repeated patch", async () => {
    const instrumentation = new AnthropicInstrumentation({
      tracerProvider: new NodeTracerProvider(),
    });

    const cjs = requireCjs("@anthropic-ai/sdk") as { default?: AnthropicClass } & AnthropicClass;
    const esm = (await import("@anthropic-ai/sdk")) as unknown as {
      default: AnthropicClass;
    };
    const CjsAnthropic = cjs.default ?? cjs;
    const EsmAnthropic = esm.default;

    instrumentation.manuallyInstrument(cjs as unknown as ManualModule);
    instrumentation.manuallyInstrument(esm as unknown as ManualModule);

    // The CJS build is re-guarded by the openInferencePatched property; the
    // ESM namespace rejects that property write, so its re-guard is the
    // WeakSet alone. Both must hold: a second wrap would emit two spans per
    // call.
    const cjsPatched = CjsAnthropic.Messages.prototype.create;
    instrumentation.manuallyInstrument(cjs as unknown as ManualModule);
    expect(CjsAnthropic.Messages.prototype.create).toBe(cjsPatched);

    const esmPatched = EsmAnthropic.Messages.prototype.create;
    instrumentation.manuallyInstrument(esm as unknown as ManualModule);
    expect(EsmAnthropic.Messages.prototype.create).toBe(esmPatched);

    instrumentation.disable();
  });
});
