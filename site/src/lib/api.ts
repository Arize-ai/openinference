/**
 * What the Python and JavaScript API references share: the shape of a
 * documented object's docs (written by `scripts/python-api.py` and
 * `scripts/js-api.mjs` alike), names that link, and rendering docs to HTML.
 * Each language's own model and routes are in `python-api.ts` and
 * `js-api.ts`.
 */
import { createMarkdownProcessor, type MarkdownRenderer } from "@astrojs/markdown-remark";
import { createHighlighter, type Highlighter, type ThemedToken, type ThemeRegistrationAny } from "shiki";

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

// --- Signatures ---------------------------------------------------------------

export type SignatureLanguage = "python" | "typescript";

const themes = shikiConfig.themes as { light: ThemeRegistrationAny; dark: ThemeRegistrationAny };
let highlighter: Promise<Highlighter> | undefined;

const unescape = (text: string) =>
  text.replace(/&quot;/g, '"').replace(/&lt;/g, "<").replace(/&gt;/g, ">").replace(/&amp;/g, "&");

/** An element in a signature's HTML (a link, a name), by the text it wraps. */
interface Wrapper {
  open: string;
  close: string;
  start: number;
  end: number;
}

/**
 * A signature's HTML as its text and the elements around parts of it. The
 * signature builders write only `<a>` and `<span>`, and escape text with
 * `escape`.
 */
function parseSignature(html: string): { text: string; wrappers: Wrapper[] } {
  let text = "";
  const wrappers: Wrapper[] = [];
  const open: Wrapper[] = [];
  for (const match of html.matchAll(/<(\/?)(a|span)\b[^>]*>|[^<]+|</g)) {
    const [whole, closing, tag] = match;
    if (whole === "<") throw new Error(`Unexpected markup in a signature: ${html}`);
    if (!tag) {
      text += unescape(whole);
    } else if (closing) {
      const wrapper = open.pop();
      if (wrapper) wrapper.end = text.length;
    } else {
      const wrapper = { open: whole, close: `</${tag}>`, start: text.length, end: -1 };
      open.push(wrapper);
      wrappers.push(wrapper);
    }
  }
  return { text, wrappers };
}

const styleOf = (token: ThemedToken) =>
  Object.entries(token.htmlStyle ?? (token.color ? { color: token.color } : {}))
    .map(([property, value]) => `${property}:${value}`)
    .join(";");

interface Token {
  content: string;
  style: string;
}

/**
 * `code`'s tokens, read as if `context` came before it: the grammar needs a
 * declaration around a bare method to colour it as one. The context's own
 * tokens are dropped.
 */
async function tokenize(code: string, lang: SignatureLanguage, context = ""): Promise<Token[]> {
  highlighter ??= createHighlighter({ themes: [themes.light, themes.dark], langs: ["python", "typescript"] });
  const lines = (await highlighter).codeToTokens(context + code, {
    lang,
    themes: { light: themes.light, dark: themes.dark },
  }).tokens;
  let skip = context.length;
  const tokens: Token[] = [];
  lines.forEach((line, index) => {
    const all = [...(index ? [{ content: "\n", style: "" }] : []), ...line.map((t) => ({ content: t.content, style: styleOf(t) }))];
    for (const token of all) {
      if (skip >= token.content.length) {
        skip -= token.content.length;
        continue;
      }
      tokens.push({ content: token.content.slice(skip), style: token.style });
      skip = 0;
    }
  });
  return tokens;
}

/**
 * A signature's tokens. TypeScript members aren't declarations on their own:
 * a method (`onStart(span: Span): void`) reads as a call, so it's read inside
 * a class; a constructor (`new OITracer(…): OITracer`) as a function, after
 * its `new`.
 */
async function tokensOf(text: string, lang: SignatureLanguage): Promise<Token[]> {
  if (lang !== "typescript") return tokenize(text, lang);
  const constructor = /^new (?=[A-Za-z_$])/.exec(text);
  if (constructor) {
    return [...(await tokenize("new ", lang)), ...(await tokenize(text.slice(constructor[0].length), lang, "function "))];
  }
  const declaration = /^(?:function|class|interface|enum|type|const|let|var|namespace|declare|abstract|async)\b/;
  return tokenize(text, lang, declaration.test(text) ? "" : "class _ {\n");
}

/**
 * A signature's HTML, highlighted like the site's code blocks (the same
 * Shiki themes) with its links kept. Each token is a span with the light
 * colour inline and the dark one as --shiki-dark; a token a link starts or
 * ends inside is split there, so the markup stays nested.
 */
export async function highlightSignature(html: string, lang: SignatureLanguage): Promise<string> {
  const { text, wrappers } = parseSignature(html);
  const tokens = await tokensOf(text, lang);
  // Shiki keeps every character; if it ever doesn't, the plain signature is
  // still right.
  if (tokens.map((token) => token.content).join("") !== text) return html;

  // Outer elements first: those that start earlier, then those that end later.
  const pending = [...wrappers].sort((a, b) => a.start - b.start || b.end - a.end);
  const stack: Wrapper[] = [];
  let out = "";
  let position = 0;
  const boundary = () => {
    while (stack.length && stack[stack.length - 1]!.end === position) out += stack.pop()!.close;
    while (pending.length && pending[0]!.start === position) {
      const wrapper = pending.shift()!;
      out += wrapper.open;
      if (wrapper.end === position) out += wrapper.close;
      else stack.push(wrapper);
    }
  };
  for (const token of tokens) {
    let rest = token.content;
    while (rest) {
      boundary();
      const next = Math.min(
        position + rest.length,
        ...stack.map((wrapper) => wrapper.end),
        ...pending.map((wrapper) => wrapper.start).filter((start) => start > position),
      );
      const piece = rest.slice(0, next - position);
      out += token.style ? `<span style="${token.style}">${escape(piece)}</span>` : escape(piece);
      rest = rest.slice(piece.length);
      position = next;
    }
  }
  boundary();
  return out + stack.reverse().map((wrapper) => wrapper.close).join("");
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
