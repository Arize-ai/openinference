import type * as llamaindex from "llamaindex";

import {
  InstrumentationBase,
  InstrumentationConfig,
  InstrumentationModuleDefinition,
  InstrumentationNodeModuleDefinition,
} from "@opentelemetry/instrumentation";
import { diag } from "@opentelemetry/api";
import {
  patchQueryMethod,
  patchRetrieveMethod,
  patchQueryEmbeddingMethod,
} from "./utils";
import { BaseEmbedding } from "llamaindex";
import { VERSION } from "./version";

const MODULE_NAME = "llamaindex";

/**
 * Flag to check if the LlamaIndex module has been patched
 * Note: This is a fallback in case the module is made immutable (e.x. Deno, webpack, etc.)
 */
let _isOpenInferencePatched = false;

/**
 * function to check if instrumentation is enabled / disabled
 */
export function isPatched() {
  return _isOpenInferencePatched;
}

export class LlamaIndexInstrumentation extends InstrumentationBase<
  typeof llamaindex
> {
  constructor(config?: InstrumentationConfig) {
    super(
      "@arizeai/openinference-instrumentation-llama-index",
      VERSION,
      Object.assign({}, config),
    );
  }

  public manuallyInstrument(module: typeof llamaindex) {
    diag.debug(`Manually instrumenting ${MODULE_NAME}`);
    this.patch(module);
  }

  protected init(): InstrumentationModuleDefinition<typeof llamaindex> {
    const module = new InstrumentationNodeModuleDefinition<typeof llamaindex>(
      "llamaindex",
      [">=0.1.0"],
      this.patch.bind(this),
      this.unpatch.bind(this),
    );
    return module;
  }

  private isEmbedding(value: unknown): value is BaseEmbedding {
    return value != null && value instanceof BaseEmbedding;
  }

  private patch(moduleExports: typeof llamaindex, moduleVersion?: string) {
    this._diag.debug(`Applying patch for ${MODULE_NAME}@${moduleVersion}`);
    if (_isOpenInferencePatched) {
      return moduleExports;
    }

    // TODO: Support streaming
    this._wrap(
      moduleExports.RetrieverQueryEngine.prototype,
      "query",
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      (original): any => {
        return patchQueryMethod(original, moduleExports, this.tracer);
      },
    );

    this._wrap(
      moduleExports.VectorIndexRetriever.prototype,
      "retrieve",
      (original) => {
        return patchRetrieveMethod(original, moduleExports, this.tracer);
      },
    );

    for (const value of Object.values(moduleExports)) {
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      const prototype = (value as any).prototype;
      if (this.isEmbedding(prototype)) {
        this._wrap(prototype, "getQueryEmbedding", (original) => {
          return patchQueryEmbeddingMethod(original, this.tracer);
        });
      }
    }

    _isOpenInferencePatched = true;
    return moduleExports;
  }

  private unpatch(moduleExports: typeof llamaindex, moduleVersion?: string) {
    this._diag.debug(`Un-patching ${MODULE_NAME}@${moduleVersion}`);
    this._unwrap(moduleExports.RetrieverQueryEngine.prototype, "query");
    this._unwrap(moduleExports.VectorIndexRetriever.prototype, "retrieve");

    _isOpenInferencePatched = false;
  }
}
