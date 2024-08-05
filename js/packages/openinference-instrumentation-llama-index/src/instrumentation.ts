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
import { VERSION } from "./version";
import { BaseEmbedding } from "llamaindex";

const MODULE_NAME = "llamaindex";

type EmbeddingTypes =
  | typeof llamaindex.HuggingFaceEmbedding
  | typeof llamaindex.GeminiEmbedding
  | typeof llamaindex.MistralAIEmbedding
  | typeof llamaindex.MultiModalEmbedding
  | typeof llamaindex.OpenAIEmbedding
  | typeof llamaindex.Ollama;

// type LLMTypes =
//   | typeof ToolCallLLM
//   | typeof llamaindex.HuggingFaceInferenceAPI
//   | typeof llamaindex.HuggingFaceLLM
//   | typeof llamaindex.MistralAI
//   | typeof llamaindex.Portkey
//   | typeof llamaindex.ReplicateLLM;

function isClassConstructor(
  value: unknown,
): value is new (...args: any[]) => any {
  return typeof value === "function" && value.prototype;
}

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

  private isEmbedding(value: unknown): value is EmbeddingTypes {
    if (isClassConstructor(value)) {
      if (value.prototype instanceof BaseEmbedding) {
        return true;
      }
    }
    return false;
  }

  // private isLLM(llm: unknown): llm is LLMTypes {
  //   if (isClassConstructor(llm)) {
  //     if (llm.prototype instanceof BaseLLM) {
  //       return true;
  //     }
  //   }
  //   return false;
  // }

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

    for (const [_, value] of Object.entries(moduleExports)) {
      if (isClassConstructor(value)) {
        if (this.isEmbedding(value)) {
          this._wrap(value.prototype, "getQueryEmbedding", (original) => {
            return patchQueryEmbeddingMethod(original, this.tracer);
          });

          // this._wrap(value.prototype, "getTextEmbedding", (original) => {
          //   return patchTextEmbedding(original, this.tracer, key);
          // });

          // if (value.prototype.getTextEmbeddings != null) {
          //   this._wrap(value.prototype, "getTextEmbeddings", (original) => {
          //     return patchTextEmbeddings(original, this.tracer, key);
          //   });
          // }
        }
        // else if (this.isLLM(value)) {
        //   // eslint-disable-next-line @typescript-eslint/no-explicit-any
        //   this._wrap(value.prototype, "chat", (original): any => {
        //     return patchLLMChat(original, this.tracer);
        //   });
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
