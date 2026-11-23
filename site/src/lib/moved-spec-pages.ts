import { routeOf } from "./site";

/**
 * Spec pages that moved when the spec was split into section directories,
 * keyed by their old page (`spec/<old>.html`). Each redirects to `to`, a new
 * page id. Sections that moved to a different page, or were renamed, are in
 * `anchors`: an old anchor to its new `id#anchor`. An anchor not listed is
 * kept as is on the `to` page.
 */
export interface MovedSpecPage {
  to: string;
  anchors?: Record<string, string>;
}

export const movedSpecPages: Record<string, MovedSpecPage> = {
  annotations: { to: "feedback/annotations" },
  configuration: {
    to: "configuration/trace-config",
    anchors: {
      "redacted-content": "configuration/data-masking#redacted-content",
      "external-blob-upload-experimental": "configuration/blob-upload",
    },
  },
  decision_spans: {
    to: "spans/decision",
    anchors: {
      "required-attributes": "spans/decision#attributes",
      "common-attributes": "spans/decision#attributes",
    },
  },
  embedding_spans: {
    to: "spans/embedding",
    anchors: {
      "required-attributes": "spans/embedding#attributes",
      "common-attributes": "spans/embedding#attributes",
    },
  },
  llm_spans: {
    to: "spans/llm",
    anchors: {
      "required-attributes": "spans/llm#attributes",
      "common-attributes": "spans/llm#attributes",
      "context-attributes": "attributes/context",
      "attribute-flattening": "attributes/conventions#flattening",
      "tool-role-messages": "content/messages#tool-role-messages",
      "reasoning-content": "content/messages#reasoning-content",
      "openai-responses": "content/messages#openai-responses",
      "anthropic-messages": "content/messages#anthropic-messages",
      "gemini-function-calling": "content/messages#gemini-function-calling",
    },
  },
  multimodal_attributes: {
    to: "content/multimodal",
    anchors: {
      "external-storage-for-large-media": "configuration/blob-upload",
      "privacy-considerations": "configuration/data-masking",
      "hiding-images": "configuration/data-masking#messages",
      "base64-image-truncation": "configuration/data-masking#base64-size-limit",
      "hiding-text-content": "configuration/data-masking#messages",
      "example-multimodal-message": "content/messages#content-items",
      "fallback-for-simple-messages": "content/messages#text-messages",
    },
  },
  semantic_conventions: {
    to: "attributes/semantic-conventions",
    anchors: {
      "span-kinds": "general/span-kinds",
      "token-count-details": "spans/llm#token-counts",
      "system-and-model-identification": "spans/llm#model-identification",
      "attribute-naming-conventions": "attributes/conventions",
      "indexed-attribute-prefixes": "attributes/conventions#flattening",
      "common-flattened-attribute-patterns": "attributes/conventions#list-attributes",
      "annotations-and-evaluations": "feedback/annotations",
      "llm-inputoutput-messages": "content/messages",
      "completions-api-legacy-text-completion": "content/messages#completions-api",
      "message-content-arrays-multimodal": "content/multimodal#message-content-arrays",
      "span-kind-independent-images": "content/multimodal#span-kind-independent-images",
      "tool-calls-in-output-messages": "content/tool-calling#tool-calls-in-messages",
      "available-tools": "content/tool-calling#tool-definitions",
      python: "attributes/conventions#examples",
      "javascripttypescript-es6": "attributes/conventions#examples",
      "go-example-helper-functions": "attributes/conventions#examples",
    },
  },
  tool_calling: { to: "content/tool-calling" },
  traces: {
    to: "general/traces",
    anchors: {
      tracer: "general/traces#instrumentation-components",
      "trace-exporters": "general/traces#instrumentation-components",
      chain: "general/span-kinds#chain",
      retriever: "general/span-kinds#retriever",
      reranker: "general/span-kinds#reranker",
      llm: "general/span-kinds#llm",
      embedding: "general/span-kinds#embedding",
      tool: "general/span-kinds#tool",
      agent: "general/span-kinds#agent",
      guardrail: "general/span-kinds#guardrail",
      evaluator: "general/span-kinds#evaluator",
      prompt: "general/span-kinds#prompt",
    },
  },
};

/** `spans/llm#token-counts` → `/spec/spans/llm.html#token-counts`. */
const hrefOf = (target: string) => {
  const [id, anchor] = target.split("#");
  return `${routeOf(`spec/${id}.md`)}${anchor ? `#${anchor}` : ""}`;
};

/** Each moved page's redirect: its new href, and its moved anchors' new hrefs. */
export function movedSpecRedirects() {
  return Object.entries(movedSpecPages).map(([moved, page]) => ({
    moved,
    to: hrefOf(page.to),
    anchors: Object.fromEntries(
      Object.entries(page.anchors ?? {}).map(([anchor, target]) => [anchor, hrefOf(target)]),
    ),
  }));
}
