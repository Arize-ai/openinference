import * as llamaindex from "llamaindex";
import { BaseRetriever } from "llamaindex";

// eslint-disable-next-line @typescript-eslint/no-explicit-any
export type GenericFunction = (...args: any[]) => any;

export type SafeFunction<T extends GenericFunction> = (
  ...args: Parameters<T>
) => ReturnType<T> | null;

export type ObjectWithModel = { model: string };

export type RetrieverQueryEngineQueryMethodType =
  typeof llamaindex.RetrieverQueryEngine.prototype.query;

export type RetrieverRetrieveMethodType = BaseRetriever["retrieve"];

export type QueryEmbeddingMethodType =
  typeof llamaindex.BaseEmbedding.prototype.getQueryEmbedding;

export type LLMChatMethodType = llamaindex.LLM["chat"];
export type LLMObject = { metadata: llamaindex.LLMMetadata };
