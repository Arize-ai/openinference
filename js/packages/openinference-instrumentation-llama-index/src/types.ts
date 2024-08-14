import * as llamaindex from "llamaindex";
import { BaseEmbedding } from "@llamaindex/core/dist/embeddings";
import { LLM, LLMMetadata } from "@llamaindex/core/dist/llms";

// eslint-disable-next-line @typescript-eslint/no-explicit-any
export type GenericFunction = (...args: any[]) => any;

export type SafeFunction<T extends GenericFunction> = (
  ...args: Parameters<T>
) => ReturnType<T> | null;

export type ObjectWithModel = { model: string };
export type ObjectWithMetadata = { metadata: LLMMetadata };

export type RetrieverQueryEngineQueryMethodType =
  typeof llamaindex.RetrieverQueryEngine.prototype.query;

export type RetrieverRetrieveMethodType = llamaindex.BaseRetriever["retrieve"];

export type QueryEmbeddingMethodType =
  typeof BaseEmbedding.prototype.getQueryEmbedding;

export type LLMChatMethodType = LLM["chat"];
export type LLMCompleteMethodType = LLM["complete"];
