import type {
  Options as SDKOptions,
  SDKMessage,
  SDKResultMessage,
  SDKSession,
  SDKSessionOptions,
  SDKUserMessage,
} from "@anthropic-ai/claude-agent-sdk";
import type { Tracer, TracerProvider } from "@opentelemetry/api";
import { diag } from "@opentelemetry/api";
import type {
  InstrumentationConfig,
  InstrumentationModuleDefinition,
} from "@opentelemetry/instrumentation";
import {
  InstrumentationBase,
  InstrumentationNodeModuleDefinition,
} from "@opentelemetry/instrumentation";

import type { TraceConfigOptions } from "@arizeai/openinference-core";
import { OITracer } from "@arizeai/openinference-core";

import { wrapQuery } from "./v1QueryWrapper";
import { wrapCreateSession, wrapPrompt, wrapResumeSession } from "./v2Wrappers";
// oxlint-disable-next-line typescript/prefer-ts-expect-error
// @ts-ignore - No version file until build
import { VERSION } from "./version";

const MODULE_NAME = "@anthropic-ai/claude-agent-sdk";

const INSTRUMENTATION_NAME = "@arizeai/openinference-instrumentation-claude-agent-sdk";

/**
 * Flag to check if the module has been patched.
 * Fallback in case the module is made immutable (e.g. Deno, webpack, etc.)
 */
let _isOpenInferencePatched = false;

/**
 * Check if instrumentation is enabled / disabled.
 */
export function isPatched() {
  return _isOpenInferencePatched;
}

/**
 * Reset the patch state. Intended for testing only.
 */
export function _resetPatchState() {
  _isOpenInferencePatched = false;
}

/**
 * Typed interface for the Claude Agent SDK module exports.
 * Uses SDK types directly for compile-time safety.
 */
interface ClaudeAgentSDKModule {
  query?: (params: {
    prompt: string | AsyncIterable<SDKUserMessage>;
    options?: SDKOptions;
  }) => AsyncIterable<SDKMessage>;
  unstable_v2_createSession?: (options: SDKSessionOptions) => SDKSession;
  unstable_v2_resumeSession?: (sessionId: string, options: SDKSessionOptions) => SDKSession;
  unstable_v2_prompt?: (message: string, options: SDKSessionOptions) => Promise<SDKResultMessage>;
  openInferencePatched?: boolean;
}

/**
 * Returns true if a property on the given object can be assigned to.
 * ESM namespace objects have getter-only, non-configurable descriptors.
 */
function isPropertyWritable(obj: object, prop: string): boolean {
  const desc = Object.getOwnPropertyDescriptor(obj, prop);
  if (!desc) return true; // no descriptor → assignment will create it
  if (desc.get && !desc.set && !desc.configurable) return false;
  return desc.writable !== false;
}

/**
 * OpenInference instrumentation for the Claude Agent SDK.
 *
 * Produces AGENT spans for query()/prompt() invocations and TOOL spans
 * for each tool call via hook injection.
 *
 * @example
 * ```typescript
 * import { ClaudeAgentSDKInstrumentation } from "@arizeai/openinference-instrumentation-claude-agent-sdk";
 *
 * const instrumentation = new ClaudeAgentSDKInstrumentation();
 * instrumentation.setTracerProvider(provider);
 * ```
 */
export class ClaudeAgentSDKInstrumentation extends InstrumentationBase<ClaudeAgentSDKModule> {
  private oiTracer: OITracer;
  private tracerProvider?: TracerProvider;
  private traceConfig?: TraceConfigOptions;
  private _originals: Partial<
    Pick<
      ClaudeAgentSDKModule,
      "query" | "unstable_v2_prompt" | "unstable_v2_createSession" | "unstable_v2_resumeSession"
    >
  > = {};
  private _patchedModule?: ClaudeAgentSDKModule;

  constructor({
    instrumentationConfig,
    traceConfig,
    tracerProvider,
  }: {
    instrumentationConfig?: InstrumentationConfig;
    traceConfig?: TraceConfigOptions;
    tracerProvider?: TracerProvider;
  } = {}) {
    super(INSTRUMENTATION_NAME, VERSION, Object.assign({}, instrumentationConfig));
    this.tracerProvider = tracerProvider;
    this.traceConfig = traceConfig;
    this.oiTracer = new OITracer({
      tracer: this.tracerProvider?.getTracer(INSTRUMENTATION_NAME, VERSION) ?? this.tracer,
      traceConfig,
    });
  }

  protected init(): InstrumentationModuleDefinition<ClaudeAgentSDKModule> {
    const module = new InstrumentationNodeModuleDefinition<ClaudeAgentSDKModule>(
      MODULE_NAME,
      [">=0.2.0"],
      this.patch.bind(this),
      this.unpatch.bind(this),
    );
    return module;
  }

  /**
   * Manually instruments the Claude Agent SDK module.
   * Needed when the module is not loaded via require (commonjs).
   */
  manuallyInstrument(module: ClaudeAgentSDKModule) {
    diag.debug(`Manually instrumenting ${MODULE_NAME}`);
    this.patch(module);
  }

  get tracer(): Tracer {
    if (this.tracerProvider) {
      return this.tracerProvider.getTracer(this.instrumentationName, this.instrumentationVersion);
    }
    return super.tracer;
  }

  setTracerProvider(tracerProvider: TracerProvider): void {
    super.setTracerProvider(tracerProvider);
    this.tracerProvider = tracerProvider;
    this.oiTracer = new OITracer({
      tracer: this.tracer,
      traceConfig: this.traceConfig,
    });
  }

  /**
   * Patches the Claude Agent SDK module exports.
   *
   * ESM modules export read-only getters that cannot be reassigned directly.
   * We first try in-place mutation (works for CJS / plain objects / tests),
   * and fall back to returning a new module object when the property is
   * non-writable (ESM).  `require-in-the-middle` uses the return value as
   * the replacement module.
   */
  private patch(module: ClaudeAgentSDKModule, moduleVersion?: string): ClaudeAgentSDKModule {
    diag.debug(`Applying patch for ${MODULE_NAME}@${moduleVersion}`);

    if (module?.openInferencePatched || _isOpenInferencePatched) {
      return module;
    }

    // Handle ES module default export structure
    const sdkModule =
      (module as ClaudeAgentSDKModule & { default?: ClaudeAgentSDKModule }).default || module;

    // Check if the module is mutable (CJS / plain objects) or frozen (ESM).
    const isMutable =
      isPropertyWritable(sdkModule, "query") ||
      !Object.getOwnPropertyDescriptor(sdkModule, "query");

    // Target is either the original module (mutable) or a shallow copy (frozen ESM).
    const target: ClaudeAgentSDKModule = isMutable
      ? sdkModule
      : ({ ...sdkModule } as ClaudeAgentSDKModule);

    // Store originals for unpatch restoration
    this._originals = {
      query: sdkModule.query,
      unstable_v2_prompt: sdkModule.unstable_v2_prompt,
      unstable_v2_createSession: sdkModule.unstable_v2_createSession,
      unstable_v2_resumeSession: sdkModule.unstable_v2_resumeSession,
    };
    this._patchedModule = target;

    // Patch V1: query()
    if (typeof sdkModule.query === "function") {
      target.query = wrapQuery({ original: sdkModule.query, oiTracer: this.oiTracer });
    } else {
      diag.debug(`Cannot find query export in ${MODULE_NAME}@${moduleVersion}`);
    }

    // Patch V2: unstable_v2_prompt()
    if (typeof sdkModule.unstable_v2_prompt === "function") {
      target.unstable_v2_prompt = wrapPrompt({
        original: sdkModule.unstable_v2_prompt,
        oiTracer: this.oiTracer,
      });
    }

    // Patch V2: unstable_v2_createSession()
    if (typeof sdkModule.unstable_v2_createSession === "function") {
      target.unstable_v2_createSession = wrapCreateSession({
        original: sdkModule.unstable_v2_createSession,
        oiTracer: this.oiTracer,
      });
    }

    // Patch V2: unstable_v2_resumeSession()
    if (typeof sdkModule.unstable_v2_resumeSession === "function") {
      target.unstable_v2_resumeSession = wrapResumeSession({
        original: sdkModule.unstable_v2_resumeSession,
        oiTracer: this.oiTracer,
      });
    }

    _isOpenInferencePatched = true;
    try {
      target.openInferencePatched = true;
    } catch {
      diag.debug(`Failed to set ${MODULE_NAME} patched flag on the module`);
    }

    return target;
  }

  /**
   * Un-patches the Claude Agent SDK module, restoring original functions.
   */
  private unpatch(moduleExports: ClaudeAgentSDKModule, moduleVersion?: string) {
    diag.debug(`Removing patch for ${MODULE_NAME}@${moduleVersion}`);

    // Restore original functions on the patched target
    const target = this._patchedModule ?? moduleExports;
    if (this._originals.query) {
      target.query = this._originals.query;
    }
    if (this._originals.unstable_v2_prompt) {
      target.unstable_v2_prompt = this._originals.unstable_v2_prompt;
    }
    if (this._originals.unstable_v2_createSession) {
      target.unstable_v2_createSession = this._originals.unstable_v2_createSession;
    }
    if (this._originals.unstable_v2_resumeSession) {
      target.unstable_v2_resumeSession = this._originals.unstable_v2_resumeSession;
    }
    this._originals = {};
    this._patchedModule = undefined;

    _isOpenInferencePatched = false;
    try {
      moduleExports.openInferencePatched = false;
    } catch (e) {
      diag.warn(`Failed to unset ${MODULE_NAME} patched flag on the module`, e);
    }
  }
}
