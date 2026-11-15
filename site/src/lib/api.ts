/**
 * What the Python and JavaScript API references share: the shape of a
 * documented object's docs (written by `scripts/python-api.py` and
 * `scripts/js-api.mjs` alike), names that link, and rendering docs to HTML.
 * Each language's own model and routes are in `python-api.ts` and
 * `js-api.ts`.
 */
import { createMarkdownProcessor, type MarkdownRenderer } from "@astrojs/markdown-remark";

import rehypeTableScroll from "../plugins/rehype-table-scroll.mjs";
import { shikiConfig } from "./shiki.mjs";

// --- Expressions --------------------------------------------------------------

/**
 * A run of text in a type, an annotation or a default, or a name that may
 * link: by its Python path (`path`), to an object in the JS reference
 * (`package` and `anchor`), or to external docs (`url`).
 */
export type Part =
  | string
  | { text: string; path?: string; url?: string; package?: string; anchor?: string };
export type NamePart = Exclude<Part, string>;
export type Expression = Part[];

/** A name's link, or undefined when it isn't documented anywhere. */
export type Resolve = (part: NamePart) => string | undefined;

/** An expression as plain text, e.g. for a signature's length. */
export const textOf = (expression: Expression | null | undefined) =>
  (expression ?? []).map((part) => (typeof part === "string" ? part : part.text)).join("");

export const escape = (text: string) =>
  text.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;");

/** An expression as HTML: names `resolve` finds a page for link to it. */
export function expressionHtml(expression: Expression | null | undefined, resolve: Resolve): string {
  return (expression ?? [])
    .map((part) => {
      if (typeof part === "string") return escape(part);
      const href = resolve(part);
      const title = part.path ?? part.url ?? part.text;
      return href
        ? `<a href="${escape(href)}" title="${escape(title)}">${escape(part.text)}</a>`
        : escape(part.text);
    })
    .join("");
}

// --- Docstrings ---------------------------------------------------------------

export interface DocstringItem {
  name: string | null;
  annotation: Expression | null;
  default: Expression | null;
  description: string;
}

/**
 * A docstring (Python) or doc comment (TypeScript) as sections: prose,
 * tables (parameters, returns, raises), examples and notes.
 */
export type DocstringSection =
  | { kind: "text"; value: string }
  | { kind: "admonition"; title: string; value: string }
  | { kind: "deprecated"; version: string; value: string }
  | { kind: "examples"; items: { kind: "text" | "examples"; value: string }[] }
  | { kind: string; title: string | null; items: DocstringItem[] };

/** The docstring's first paragraph, as plain text. */
export function summaryOf(
  object: { docstring?: DocstringSection[] },
  plain: (text: string) => string = (text) => text,
): string {
  const text = object.docstring?.find((section) => section.kind === "text");
  if (!text || !("value" in text)) return "";
  const paragraph = text.value.split(/\n\s*\n/)[0] ?? "";
  return plain(paragraph)
    .replace(/\[([^\]]*)\]\([^)]*\)/g, "$1")
    .replace(/`([^`]+)`/g, "$1")
    .replace(/\s+/g, " ")
    .trim();
}

export const fence = (code: string, lang: string) => `\`\`\`${lang}\n${code}\n\`\`\``;

/**
 * An example, or a titled admonition that is one (`Example:`, `Usage:`). A
 * Python docstring writes it as indented code with no fence, so fence it,
 * unless it already is.
 */
export function exampleToMarkdown(value: string, kind: "text" | "examples" = "text"): string {
  if (value.includes("```")) return value;
  return fence(value, kind === "examples" ? "pycon" : "python");
}

export const isCodeTitle = (title: string) => /example|usage|equivalent/i.test(title);

// --- Markdown -----------------------------------------------------------------

let renderer: Promise<MarkdownRenderer> | undefined;

/** Markdown to HTML, with the site's code theme and table scroll boxes. */
export async function renderMarkdown(markdown: string): Promise<string> {
  if (!markdown.trim()) return "";
  renderer ??= createMarkdownProcessor({
    shikiConfig,
    rehypePlugins: [rehypeTableScroll],
  });
  const result = await (await renderer).render(markdown);
  return result.code;
}

/** At most `length` characters, cut at a word, with an ellipsis if cut. */
export const clip = (text: string, length: number) =>
  text.length <= length ? text : `${text.slice(0, length).replace(/\s+\S*$/, "")}…`;

// --- Pages --------------------------------------------------------------------

/** A documented object, for an overview's search. */
export interface SearchEntry {
  /** Its name, e.g. `using_session`. */
  name: string;
  /** Its full name, e.g. `openinference.instrumentation.using_session`. */
  path: string;
  kind: string;
  href: string;
  /** The package's short name. */
  package: string;
  summary: string;
}

/** A labelled value in a package page's head: its version, what it instruments. */
export interface Fact {
  label: string;
  value: string | null | undefined;
  code?: boolean;
}

/** A heading on a page, for DocsLayout's "On this page". */
export interface Heading {
  depth: number;
  slug: string;
  text: string;
}
