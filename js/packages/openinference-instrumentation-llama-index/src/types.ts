import * as llamaindex from "llamaindex";
import { BaseRetriever } from "llamaindex";
import { BaseLLM } from "llamaindex/dist/type/llm/base";

// eslint-disable-next-line @typescript-eslint/no-explicit-any
export type GenericFunction = (...args: any[]) => any;

export type SafeFunction<T extends GenericFunction> = (
  ...args: Parameters<T>
) => ReturnType<T> | null;

export type ObjectWithModel = { model: string };
export type ObjectWithID = { id: string };

export type QueryEngineQueryMethod =
  typeof llamaindex.RetrieverQueryEngine.prototype.query;

export type RetrieverRetrieveMethod = BaseRetriever["retrieve"];

export type QueryEmbeddingMethod =
  typeof llamaindex.BaseEmbedding.prototype.getQueryEmbedding;

export type TextEmbeddingsMethod =
  typeof llamaindex.BaseEmbedding.prototype.getTextEmbeddings;

export type LLMChatMethodType = BaseLLM["chat"];
