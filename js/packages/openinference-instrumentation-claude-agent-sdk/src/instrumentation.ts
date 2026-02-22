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
 * Structural type for the Claude Agent SDK module exports.
 * We use this instead of importing SDK types at runtime.
 */
interface ClaudeAgentSDKModule {
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  query?: (...args: any[]) => any;
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  unstable_v2_createSession?: (...args: any[]) => any;
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  unstable_v2_resumeSession?: (...args: any[]) => any;
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  unstable_v2_prompt?: (...args: any[]) => any;
  openInferencePatched?: boolean;
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
   */
  private patch(module: ClaudeAgentSDKModule, moduleVersion?: string): ClaudeAgentSDKModule {
    diag.debug(`Applying patch for ${MODULE_NAME}@${moduleVersion}`);

    if (module?.openInferencePatched || _isOpenInferencePatched) {
      return module;
    }

    // Handle ES module default export structure
    const sdkModule =
      (module as ClaudeAgentSDKModule & { default?: ClaudeAgentSDKModule }).default || module;

    // Patch V1: query()
    if (typeof sdkModule.query === "function") {
      sdkModule.query = wrapQuery(sdkModule.query, this.oiTracer);
    } else {
      diag.debug(`Cannot find query export in ${MODULE_NAME}@${moduleVersion}`);
    }

    // Patch V2: unstable_v2_prompt()
    if (typeof sdkModule.unstable_v2_prompt === "function") {
      sdkModule.unstable_v2_prompt = wrapPrompt(sdkModule.unstable_v2_prompt, this.oiTracer);
    }

    // Patch V2: unstable_v2_createSession()
    if (typeof sdkModule.unstable_v2_createSession === "function") {
      sdkModule.unstable_v2_createSession = wrapCreateSession(
        sdkModule.unstable_v2_createSession,
        this.oiTracer,
      );
    }

    // Patch V2: unstable_v2_resumeSession()
    if (typeof sdkModule.unstable_v2_resumeSession === "function") {
      sdkModule.unstable_v2_resumeSession = wrapResumeSession(
        sdkModule.unstable_v2_resumeSession,
        this.oiTracer,
      );
    }

    _isOpenInferencePatched = true;
    try {
      module.openInferencePatched = true;
    } catch (e) {
      diag.debug(`Failed to set ${MODULE_NAME} patched flag on the module`, e);
    }

    return module;
  }

  /**
   * Un-patches the Claude Agent SDK module.
   */
  private unpatch(moduleExports: ClaudeAgentSDKModule, moduleVersion?: string) {
    diag.debug(`Removing patch for ${MODULE_NAME}@${moduleVersion}`);

    _isOpenInferencePatched = false;
    try {
      moduleExports.openInferencePatched = false;
    } catch (e) {
      diag.warn(`Failed to unset ${MODULE_NAME} patched flag on the module`, e);
    }
  }
}
