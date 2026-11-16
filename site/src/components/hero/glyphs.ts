/**
 * Span-kind glyphs for the trace hero: the shapes of Phoenix's SpanKindIcon,
 * redrawn as bare strokes on a 16px grid in the kind's colour, so the hero
 * and Phoenix read as one system. Each is the inner markup of an
 * `<svg viewBox="0 0 16 16">` drawn in currentColor; `.fill` parts are filled.
 * The geometry is constructed, not traced: every shape sits on the grid's
 * centre and keeps a 2px margin.
 */
import type { Kind } from "./model";

export const GLYPHS: Record<Kind, string> = {
  // A head with two eyes over shoulders: an agent.
  AGENT:
    '<rect x="4.5" y="2.25" width="7" height="5.5" rx="1.75"/>' +
    '<circle class="fill" cx="6.75" cy="5" r=".8"/><circle class="fill" cx="9.25" cy="5" r=".8"/>' +
    '<path d="M2.75 14c0-2.9 2.35-4.4 5.25-4.4s5.25 1.5 5.25 4.4"/>',
  // A hexagon around a core: a model call.
  LLM: '<path d="M8 2l5.2 3v6L8 14l-5.2-3V5z"/><circle class="fill" cx="8" cy="8" r="1.4"/>',
  // An open-jawed wrench on the diagonal: a tool.
  TOOL: '<path d="M13.23 4.4l-1.71 1.71-1.63-1.63 1.71-1.71A3.6 3.6 0 0 0 6.75 7.55l-3.89 3.89a1.2 1.2 0 0 0 1.7 1.7l3.89-3.89A3.6 3.6 0 0 0 13.23 4.4z"/>',
  // A store, two bands deep: a retrieval.
  RETRIEVER:
    '<ellipse cx="8" cy="4" rx="5" ry="1.9"/>' +
    '<path d="M3 4v8c0 1.05 2.24 1.9 5 1.9s5-.85 5-1.9V4"/><path d="M3 8c0 1.05 2.24 1.9 5 1.9s5-.85 5-1.9"/>',
  // A column vector in brackets, its values as dots: an embedding.
  EMBEDDING:
    '<path d="M5.5 2.5h-2v11h2"/><path d="M10.5 2.5h2v11h-2"/>' +
    '<circle class="fill" cx="8" cy="5" r="1.15"/><circle class="fill" cx="8" cy="8" r="1.15"/><circle class="fill" cx="8" cy="11" r="1.15"/>',
  // One up, one down, corner to corner: a reranking.
  RERANKER: '<path d="M2 7.25l3-5.2 3 5.2z"/><path d="M8 8.75h6l-3 5.2z"/>',
  // A shield with a check: a guardrail.
  GUARDRAIL:
    '<path d="M8 1.9l5 1.85V7.6c0 3.15-2.1 5.3-5 6.5-2.9-1.2-5-3.35-5-6.5V3.75z"/><path d="M5.6 7.95l1.7 1.7 3.1-3.2"/>',
  // A globe: a plain OpenTelemetry HTTP client span, not an OpenInference kind.
  HTTP: '<circle cx="8" cy="8" r="5.9"/><ellipse cx="8" cy="8" rx="2.5" ry="5.9"/><path d="M2.1 8h11.8"/>',
};

/** The CSS class that carries a kind's colour. */
export const kindClass = (kind: Kind) => `kind-${kind.toLowerCase()}`;
