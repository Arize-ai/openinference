import type Anthropic from "@anthropic-ai/sdk";
import type { Stream } from "@anthropic-ai/sdk/streaming";
import { diag } from "@opentelemetry/api";

import { LLMProvider } from "@arizeai/openinference-semantic-conventions";

import { MODULE_NAME } from "./constants";
import type { Message, RawMessageStreamEvent } from "./types";

type AnthropicModuleWithOptionalBeta = Omit<typeof Anthropic, "Beta"> & {
  Beta?: { Messages?: typeof Anthropic.Beta.Messages };
};

/**
 * Resolves the Anthropic namespace and its optional `Beta.Messages` from a
 * module export, unwrapping the ES-module default. `patch` and `unpatch` must
 * agree on the object they (un)wrap, so they share this one resolution.
 */
export function resolveAnthropicModule(moduleExports: typeof Anthropic) {
  const anthropicModule =
    (moduleExports as typeof Anthropic & { default?: typeof Anthropic }).default || moduleExports;
  return {
    anthropicModule,
    betaMessages: (anthropicModule as AnthropicModuleWithOptionalBeta).Beta?.Messages,
  };
}

/**
 * Narrows a patched method to the type of the SDK method it replaces. A patch
 * takes the same arguments and returns what the original returns (through
 * `traceAnthropicAPIPromise` or the original call itself), which TypeScript
 * cannot check against the SDK's overloads.
 */
export function isPatchOf<Method extends (...args: never[]) => unknown>(
  _original: Method,
  patched: unknown,
): patched is Method {
  return typeof patched === "function";
}

/**
 * Identifies the cloud serving a request from the client the patched resource
 * belongs to: the Bedrock (including Mantle), Vertex and Foundry SDK clients
 * reuse this SDK's `Messages` resource, so the same patch traces them.
 * Detected by class name (AWS / Google Cloud named clients included), and by
 * the fields the Bedrock and Vertex clients carry for when a bundler renames
 * classes.
 */
export function getAnthropicProvider(resource: unknown): LLMProvider {
  try {
    const client: unknown =
      resource != null && typeof resource === "object" ? Reflect.get(resource, "_client") : null;
    if (client == null || typeof client !== "object") {
      return LLMProvider.ANTHROPIC;
    }
    const name = client.constructor?.name ?? "";
    if (/Bedrock|AWS/.test(name) || "awsRegion" in client) {
      return LLMProvider.AWS;
    }
    if (/Vertex|GoogleCloud/.test(name) || "projectId" in client) {
      return LLMProvider.GOOGLE;
    }
    if (/Foundry|Azure/.test(name)) {
      return LLMProvider.AZURE;
    }
  } catch (error) {
    diag.debug(`Failed to identify the ${MODULE_NAME} client's provider`, error);
  }
  return LLMProvider.ANTHROPIC;
}

/**
 * type-guard that checks if the response is an Anthropic message response
 */
export function isAnthropicMessageResponse(response: unknown): response is Message {
  return (
    response != null && typeof response === "object" && "content" in response && "role" in response
  );
}

/**
 * type-guard that checks if the response is an Anthropic stream
 */
export function isAnthropicStream(response: unknown): response is Stream<RawMessageStreamEvent> {
  return response != null && typeof response === "object" && "tee" in response;
}
