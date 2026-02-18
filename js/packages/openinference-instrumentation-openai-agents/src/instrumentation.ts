import { TraceConfigOptions } from "@arizeai/openinference-core";

import { trace, Tracer, TracerProvider } from "@opentelemetry/api";

import {
  OpenInferenceTracingProcessor,
  OpenInferenceTracingProcessorConfig,
} from "./processor";
import { VERSION } from "./version";

const INSTRUMENTATION_NAME =
  "@arizeai/openinference-instrumentation-openai-agents";

/**
 * Type for the SDK module that can be passed to instrument()
 * This allows users to pass their statically imported SDK module to avoid
 * ESM module resolution issues.
 */
export interface OpenAIAgentsSDK {
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  addTraceProcessor: (processor: any) => void;
  startTraceExportLoop?: () => void;
}

export interface OpenAIAgentsInstrumentationConfig
  extends OpenInferenceTracingProcessorConfig {
  /**
   * An optional custom tracer provider to be used for tracing.
   * If not provided, the global tracer provider will be used.
   */
  tracerProvider?: TracerProvider;
}

/**
 * Instrumentation for OpenAI Agents SDK that creates OpenInference compliant spans
 *
 * Unlike other instrumentations that patch module prototypes, this instrumentation
 * registers a custom TracingProcessor with the OpenAI Agents SDK's native tracing
 * infrastructure.
 *
 * IMPORTANT: Due to ESM module resolution, you MUST pass the SDK module from your
 * own static import to ensure the processor is registered with the correct instance.
 *
 * @example
 * ```typescript
 * import { OpenAIAgentsInstrumentation } from "@arizeai/openinference-instrumentation-openai-agents";
 * import * as agentsSdk from "@openai/agents";
 *
 * const instrumentation = new OpenAIAgentsInstrumentation();
 * instrumentation.instrument(agentsSdk);
 *
 * // Use the OpenAI Agents SDK as normal
 * const agent = new agentsSdk.Agent({ ... });
 * await agentsSdk.run(agent, input);
 *
 * // When done
 * instrumentation.uninstrument();
 * ```
 */
export class OpenAIAgentsInstrumentation {
  private processor: OpenInferenceTracingProcessor | null = null;
  private tracerProvider?: TracerProvider;
  private traceConfig?: TraceConfigOptions;
  private _enabled = false;

  constructor(config: OpenAIAgentsInstrumentationConfig = {}) {
    this.tracerProvider = config.tracerProvider;
    this.traceConfig = config.traceConfig;
  }

  /**
   * Get the tracer for this instrumentation
   */
  get tracer(): Tracer {
    if (this.tracerProvider) {
      return this.tracerProvider.getTracer(INSTRUMENTATION_NAME, VERSION);
    }
    return trace.getTracer(INSTRUMENTATION_NAME, VERSION);
  }

  /**
   * Check if instrumentation is enabled
   */
  isEnabled(): boolean {
    return this._enabled;
  }

  /**
   * Get the processor instance (useful for direct registration)
   */
  getProcessor(): OpenInferenceTracingProcessor | null {
    return this.processor;
  }

  /**
   * Create and return the processor without registering it.
   * Useful when you want to register the processor yourself.
   *
   * @example
   * ```typescript
   * import { addTraceProcessor } from "@openai/agents";
   * import { OpenAIAgentsInstrumentation } from "@arizeai/openinference-instrumentation-openai-agents";
   *
   * const instrumentation = new OpenAIAgentsInstrumentation({ tracerProvider: provider });
   * const processor = instrumentation.createProcessor();
   * addTraceProcessor(processor);
   * ```
   */
  createProcessor(): OpenInferenceTracingProcessor {
    this.processor = new OpenInferenceTracingProcessor({
      tracer: this.tracer,
      traceConfig: this.traceConfig,
    });
    this._enabled = true;
    return this.processor;
  }

  /**
   * Instrument the OpenAI Agents SDK by registering the OpenInference tracing processor
   *
   * IMPORTANT: You MUST pass the SDK module from your own static import to ensure
   * the processor is registered with the correct module instance. This is required
   * due to ESM module resolution behavior where dynamic imports may resolve to
   * different module instances.
   *
   * @param sdk - The @openai/agents SDK module from your static import
   *
   * @example
   * ```typescript
   * import * as agentsSdk from "@openai/agents";
   * import { OpenAIAgentsInstrumentation } from "@arizeai/openinference-instrumentation-openai-agents";
   *
   * const instrumentation = new OpenAIAgentsInstrumentation({ tracerProvider: provider });
   * instrumentation.instrument(agentsSdk);
   * ```
   */
  instrument(sdk: OpenAIAgentsSDK): void {
    if (this._enabled) {
      return;
    }

    if (!sdk || typeof sdk.addTraceProcessor !== "function") {
      throw new Error(
        "Invalid SDK module. Please pass the @openai/agents module from your static import. " +
          'Example: import * as agentsSdk from "@openai/agents"; instrumentation.instrument(agentsSdk);',
      );
    }

    this.processor = new OpenInferenceTracingProcessor({
      tracer: this.tracer,
      traceConfig: this.traceConfig,
    });

    // Use the SDK's addTraceProcessor from the user's import
    sdk.addTraceProcessor(this.processor);

    // Start the export loop if available
    if (typeof sdk.startTraceExportLoop === "function") {
      sdk.startTraceExportLoop();
    }

    this._enabled = true;
  }

  /**
   * Remove the OpenInference tracing processor from the SDK
   *
   * Note: The OpenAI Agents SDK's TraceProvider does not have a direct method
   * to remove individual processors. This method shuts down our processor
   * to stop it from processing new spans.
   */
  async uninstrument(): Promise<void> {
    if (!this._enabled || !this.processor) {
      return;
    }

    await this.processor.shutdown();
    this.processor = null;
    this._enabled = false;
  }

  /**
   * Set a new tracer provider
   */
  setTracerProvider(tracerProvider: TracerProvider): void {
    this.tracerProvider = tracerProvider;
    if (this.processor) {
      this.processor.setTracer(this.tracer);
    }
  }

  /**
   * Force flush any pending spans
   */
  async forceFlush(): Promise<void> {
    if (this.processor) {
      await this.processor.forceFlush();
    }
  }
}
