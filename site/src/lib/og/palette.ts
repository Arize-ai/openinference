/**
 * The social cards' colours: the light theme's values from tokens.css (a card
 * is always paper) and the span kinds' colours (`--kind-*`, light).
 */
import type { Kind } from "../../components/hero/model";
import { palette } from "../mark.mjs";

export const COLOR = {
  bg: palette.paper,
  fg: palette.ink,
  soft: "#4b4745",
  dim: "#6f6b66",
  rule: "#d8d5ce",
  /** A step past --dim, for the shared start of a package name: large type needs less contrast to recede. */
  faint: "#aaa59e",
  accent: "#ff5a1f",
  white: "#ffffff",
};

export const KIND: Record<Kind, string> = {
  AGENT: COLOR.soft,
  LLM: "#ee8700",
  TOOL: "#e7b700",
  RETRIEVER: "#049690",
  EMBEDDING: "#647cf6",
  RERANKER: "#019c23",
  GUARDRAIL: "#d23fd6",
  HTTP: COLOR.dim,
};
