import type * as llamaindex from "llamaindex";

import {
  InstrumentationBase,
  InstrumentationConfig,
  InstrumentationModuleDefinition,
  InstrumentationNodeModuleDefinition,
} from "@opentelemetry/instrumentation";
import { diag } from "@opentelemetry/api";
import {
  isEmbedding,
  isRetriever,
  patchQueryEngineQueryMethod,
  patchRetrieveMethod,
  patchQueryEmbeddingMethod,
} from "./utils";
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

  private patch(moduleExports: typeof llamaindex, moduleVersion?: string) {
    this._diag.debug(`Applying patch for ${MODULE_NAME}@${moduleVersion}`);
    if (_isOpenInferencePatched) {
      return moduleExports;
    }

    // TODO: Support streaming
    // TODO: Generalize to QueryEngine interface (RetrieverQueryEngine, RouterQueryEngine)
    this._wrap(
      moduleExports.RetrieverQueryEngine.prototype,
      "query",
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      (original): any => {
        return patchQueryEngineQueryMethod(original, this.tracer);
      },
    );

    for (const value of Object.values(moduleExports)) {
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      const prototype = (value as any).prototype;

      if (isRetriever(prototype)) {
        this._wrap(prototype, "retrieve", (original) => {
          return patchRetrieveMethod(original, this.tracer);
        });
      }

      if (isEmbedding(prototype)) {
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

    for (const value of Object.values(moduleExports)) {
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      const prototype = (value as any).prototype;

      if (isRetriever(prototype)) {
        this._unwrap(prototype, "retrieve");
      }

      if (isEmbedding(prototype)) {
        this._unwrap(prototype, "getQueryEmbedding");
      }
    }

    _isOpenInferencePatched = false;
  }
}
