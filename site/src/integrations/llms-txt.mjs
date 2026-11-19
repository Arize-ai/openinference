// @ts-check
/**
 * Markdown for agents, following https://llmstxt.org.
 *
 * - `llms.txt`, the index, is drawn by `src/pages/llms.txt.ts` from the same
 *   data as the pages.
 * - After the build, every page with an element marked `data-markdown` (the
 *   article of the README and its translations, the mission, the spec, the
 *   API references and the news) gets a markdown copy at its URL plus `.md`
 *   (`spec/spans/llm.html` → `spec/spans/llm.html.md`, `spec/` →
 *   `spec/index.html.md`), converted from that element's HTML, so it says
 *   exactly what the page says. Its links are absolute, and point at the
 *   markdown copy of any page that has one. What only works in a browser
 *   (the API search, and its heading) is marked `data-markdown-ignore`.
 * - Each of those pages links to its copy (`rel="alternate"`) and to
 *   `llms.txt` (`rel="describedby"`) from its <head>.
 * - `llms-full.txt` is every page `llms.txt` lists in `FULL_SECTIONS`, in
 *   full and in its order: the README and the spec. The API references
 *   would make it over a megabyte, so an agent reads those a page at a time.
 */
import { readdir, readFile, writeFile } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";

import { toText } from "hast-util-to-text";
import rehypeParse from "rehype-parse";
import rehypeRemark from "rehype-remark";
import remarkGfm from "remark-gfm";
import remarkStringify from "remark-stringify";
import { unified } from "unified";

/**
 * The route of a page's markdown copy, by the page's route:
 * `/spec/llm.html` → `/spec/llm.html.md`,
 * `/spec/` → `/spec/index.html.md`.
 * @param {string} route
 */
export function markdownRouteOf(route) {
  const [pathname] = route.split("#");
  return `${pathname.endsWith("/") ? `${pathname}index.html` : pathname}.md`;
}

/**
 * The sections of `llms.txt` that `llms-full.txt` includes, by their
 * headings.
 */
export const FULL_SECTIONS = ["Project", "Specification"];

/**
 * Elements with nothing to say in markdown: scripts, controls, icons, and
 * the page's own navigation (a news post's footer and pager).
 */
const DROPPED = new Set([
  "button",
  "footer",
  "form",
  "input",
  "label",
  "nav",
  "noscript",
  "script",
  "select",
  "source",
  "style",
  "svg",
  "template",
  "textarea",
]);

/** Badge images (a package's version) read as their alt text. */
const BADGE_HOSTS = new Set(["img.shields.io", "badge.fury.io"]);

/**
 * @typedef {import("hast").Root} Root
 * @typedef {import("hast").Element} Element
 * @typedef {import("hast").ElementContent} Content
 * @typedef {import("hast").RootContent} RootContent
 */

/**
 * @typedef {object} Site
 * @property {URL} root the site's root URL, with its base:
 *   `https://openinf.dev/`
 * @property {Set<string>} mirrored the pages with a markdown copy, by their
 *   path in the build: `spec/spans/llm.html`
 */

/** @param {Element} node */
const classesOf = (node) => {
  const value = node.properties?.className;
  return Array.isArray(value) ? value.map(String) : [];
};

/**
 * @param {string} tagName
 * @param {Content[]} children
 * @param {Element["properties"]} [properties]
 * @returns {Element}
 */
const element = (tagName, children, properties = {}) => ({
  type: "element",
  tagName,
  properties,
  children,
});

/** @param {string} value @returns {Content} */
const text = (value) => ({ type: "text", value });

/**
 * The first element in a tree that `test` accepts.
 * @param {Root | Element} node
 * @param {(node: Element) => boolean} test
 * @returns {Element | undefined}
 */
function find(node, test) {
  for (const child of node.children) {
    if (child.type !== "element") continue;
    if (test(child)) return child;
    const found = find(child, test);
    if (found) return found;
  }
  return undefined;
}

/**
 * A link from a page as an absolute URL: to the markdown copy of the page it
 * points at, if that page has one, else as it is.
 * @param {{ href: string, page: string, site: Site }} options
 */
function resolveHref({ href, page, site }) {
  let url;
  try {
    url = new URL(href, new URL(page, site.root));
  } catch {
    return href;
  }
  if (url.origin !== site.root.origin || !url.pathname.startsWith(site.root.pathname)) return url.href;
  const relative = decodeURI(url.pathname.slice(site.root.pathname.length));
  const file = relative === "" || relative.endsWith("/") ? `${relative}index.html` : relative;
  if (!site.mirrored.has(file)) return url.href;
  return new URL(`${markdownRouteOf(url.pathname)}${url.hash}`, site.root).href;
}

/**
 * A code block as a fenced one: its text, and its language from Shiki's
 * `data-language` or, for an API signature, the reference's language.
 * @param {{ node: Element, page: string }} options
 */
function codeBlock({ node, page }) {
  const signatureLanguage = page.startsWith("python/") ? "python" : "typescript";
  const language = node.properties?.dataLanguage ?? (classesOf(node).includes("api-signature") ? signatureLanguage : undefined);
  const code = element("code", [text(toText(node, { whitespace: "pre" }).replace(/\n+$/, ""))], {
    className: language ? [`language-${language}`] : [],
  });
  return element("pre", [code]);
}

/**
 * A description list (an API page's facts) as a list: `**Version**: 0.1.73`.
 * @param {{ node: Element, page: string, site: Site }} options
 * @returns {Element}
 */
function factList({ node, page, site }) {
  /** @type {Element[]} */
  const terms = [];
  /** @param {Element} parent */
  const collect = (parent) => {
    for (const child of parent.children) {
      if (child.type !== "element") continue;
      if (child.tagName === "dt" || child.tagName === "dd") terms.push(child);
      else collect(child);
    }
  };
  collect(node);
  /** @type {Element[]} */
  const items = [];
  terms.forEach((term, index) => {
    if (term.tagName !== "dt") return;
    const definition = terms[index + 1]?.tagName === "dd" ? terms[index + 1] : undefined;
    const value = definition ? prepareChildren({ node: definition, page, site }) : [];
    items.push(element("li", [element("strong", [text(toText(term).trim())]), text(": "), ...value]));
  });
  return element("ul", items);
}

/**
 * An API object's header (its heading, its kind and its source link) as the
 * heading, then a line under it: `_class_ · [Source](…)`.
 * @param {{ node: Element, page: string, site: Site }} options
 * @returns {Content[]}
 */
function objectHead({ node, page, site }) {
  const heading = find(node, (child) => /^h[1-6]$/.test(child.tagName));
  const kind = find(node, (child) => classesOf(child).includes("kind"));
  const source = find(node, (child) => child.tagName === "a" && classesOf(child).includes("source"));
  /** @type {Content[]} */
  const meta = [];
  if (kind) meta.push(element("em", [text(toText(kind).trim())]));
  if (source) {
    if (meta.length) meta.push(text(" · "));
    meta.push(...prepare({ node: source, page, site }));
  }
  return [
    ...(heading ? prepare({ node: heading, page, site }) : []),
    ...(meta.length ? [element("p", meta)] : []),
  ];
}

/**
 * @param {{ node: Element, page: string, site: Site }} options
 * @returns {Content[]}
 */
function prepareChildren({ node, page, site }) {
  return node.children.flatMap((child) => prepare({ node: child, page, site }));
}

/**
 * A node of an article, readied for markdown: what doesn't read as text is
 * dropped, code blocks are plain, links are absolute, and the API pages'
 * labels and headers are spelled out. Returns its replacement: nothing, the
 * node, or other nodes.
 * @param {{ node: RootContent, page: string, site: Site }} options
 * @returns {Content[]}
 */
function prepare({ node, page, site }) {
  if (node.type === "text") return [text(node.value.replace(/\s*↗/g, ""))];
  if (node.type !== "element") return [];
  const { tagName, properties = {} } = node;
  const classes = classesOf(node);
  if (DROPPED.has(tagName) || properties.hidden || properties.ariaHidden === "true" || "dataMarkdownIgnore" in properties) {
    return [];
  }
  if (tagName === "pre") return [codeBlock({ node, page })];
  if (tagName === "dl") return [factList({ node, page, site })];
  if (classes.includes("api-object-head")) return objectHead({ node, page, site });
  if (tagName === "img") {
    const src = String(properties.src ?? "");
    const alt = String(properties.alt ?? "");
    if (!src || src.startsWith("data:")) return [];
    const url = new URL(src, new URL(page, site.root));
    if (BADGE_HOSTS.has(url.hostname)) return alt ? [text(alt)] : [];
    return [element("img", [], { src: url.href, alt })];
  }
  if (/^h[1-6]$/.test(tagName)) {
    // A heading's link to itself is for the page, not the text.
    const unwrapped = node.children.flatMap((child) =>
      child.type === "element" && child.tagName === "a" && String(child.properties?.href ?? "").startsWith("#")
        ? child.children
        : [child],
    );
    return [element(tagName, prepareChildren({ node: element(tagName, unwrapped), page, site }))];
  }
  if (tagName === "header") {
    // A page's header (an API package's, a news post's) puts a label over
    // its heading; markdown starts with the heading.
    const blocks = prepareChildren({ node, page, site });
    const isTitle = (/** @type {Content} */ child) => child.type === "element" && child.tagName === "h1";
    return [...blocks.filter(isTitle), ...blocks.filter((child) => !isTitle(child))];
  }
  const children = prepareChildren({ node, page, site });
  // An API package's links (PyPI, npm, README, source) sit side by side.
  if (classes.includes("api-links")) {
    return [element("p", children.flatMap((child, index) => (index > 0 ? [text(" · "), child] : [child])))];
  }
  if (tagName === "a") {
    const href = typeof properties.href === "string" ? properties.href : undefined;
    return href ? [element("a", children, { href: resolveHref({ href, page, site }) })] : children;
  }
  // A label over a block (`Parameters`, `Examples`) or a page (`Python · Core`).
  if (tagName === "p" && classes.includes("eyebrow")) return [element("p", [element("strong", children)])];
  if (tagName === "aside") return [element("blockquote", children)];
  return [{ ...node, children }];
}

const toMarkdown = unified()
  .use(rehypeRemark)
  // Unpadded table cells: padding them to line up would double the size of
  // the README's package tables.
  .use(remarkGfm, { tablePipeAlign: false })
  .use(remarkStringify, { bullet: "-", emphasis: "_", fences: true, listItemIndent: "one", rule: "-" });

/**
 * A page's article as markdown. If the article has no heading of its own
 * (the landing page draws its title and introduction as the hero), it is
 * headed by the page's title and description.
 * @param {{ article: Element, title: string, description: string | undefined, page: string, site: Site }} options
 */
async function articleMarkdown({ article, title, description, page, site }) {
  const children = prepareChildren({ node: article, page, site });
  const hasTitle = find(element("div", children), (child) => child.tagName === "h1");
  const heading = [
    element("h1", [text(title)]),
    ...(description ? [element("blockquote", [element("p", [text(description)])])] : []),
  ];
  /** @type {Root} */
  const tree = { type: "root", children: [...(hasTitle ? [] : heading), ...children] };
  const markdown = toMarkdown.stringify(/** @type {any} */ (await toMarkdown.run(tree)));
  // Markdown may escape `&` in a URL (`?a=1\&b=2`); an agent copying the
  // link should get the URL.
  return markdown.replace(/\]\(([^)\s]+)\)/g, (link) => link.replace(/\\&/g, "&")).trim() + "\n";
}

/**
 * What `llms.txt` says: its title and summary, and the pages it links to the
 * markdown copies of, by section, in order, by their path in the build.
 * @param {{ index: string, site: Site }} options
 */
function readIndex({ index, site }) {
  /** @type {string[]} */
  const head = [];
  /** @type {Map<string, string[]>} */
  const sections = new Map();
  /** @type {string[] | undefined} */
  let pages;
  for (const line of index.split("\n")) {
    const heading = line.match(/^##\s+(.+)$/);
    if (heading) {
      pages = [];
      sections.set(heading[1].trim(), pages);
      continue;
    }
    if (!pages) {
      if (/^(#|>)\s/.test(line)) head.push(line);
      continue;
    }
    for (const [, href] of line.matchAll(/\]\(([^)\s]+)\)/g)) {
      if (!href.startsWith(site.root.href) || !href.endsWith(".md")) continue;
      pages.push(decodeURI(href.slice(site.root.href.length, -".md".length)));
    }
  }
  return { head, sections };
}

/**
 * @param {URL} dir
 * @returns {Promise<string[]>} every HTML file in the build, by its path in it
 */
async function htmlFiles(dir) {
  const root = fileURLToPath(dir);
  const entries = await readdir(root, { recursive: true, withFileTypes: true });
  return entries
    .filter((entry) => entry.isFile() && entry.name.endsWith(".html"))
    .map((entry) => path.relative(root, path.join(entry.parentPath, entry.name)).split(path.sep).join("/"))
    .sort();
}

const escapeAttribute = (/** @type {string} */ value) => value.replace(/&/g, "&amp;").replace(/"/g, "&quot;");

/** @returns {import("astro").AstroIntegration} */
export default function llmsTxt() {
  /** @type {URL | undefined} */
  let root;
  return {
    name: "llms-txt",
    hooks: {
      "astro:config:done": ({ config }) => {
        if (!config.site) throw new Error("llms-txt needs `site` set, for the absolute links it writes.");
        root = new URL(config.base.replace(/\/?$/, "/"), config.site);
      },
      "astro:build:done": async ({ dir, logger }) => {
        if (!root) return;
        const parser = unified().use(rehypeParse);
        /** @type {Map<string, { html: string, article: Element, title: string, description: string | undefined }>} */
        const articles = new Map();
        for (const page of await htmlFiles(dir)) {
          const html = await readFile(new URL(page, dir), "utf8");
          const tree = parser.parse(html);
          const article = find(tree, (node) => "dataMarkdown" in (node.properties ?? {}));
          if (!article) continue;
          const titleElement = find(tree, (node) => node.tagName === "title");
          const title = titleElement ? toText(titleElement).replace(/\s+·\s+OpenInference$/, "") : page;
          const meta = find(tree, (node) => node.tagName === "meta" && node.properties?.name === "description");
          const description = meta?.properties?.content ? String(meta.properties.content) : undefined;
          articles.set(page, { html, article, title, description });
        }

        /** @type {Site} */
        const site = { root, mirrored: new Set(articles.keys()) };
        const indexRoute = `${root.pathname}llms.txt`;
        /** @type {Map<string, string>} */
        const markdown = new Map();
        for (const [page, { html, article, title, description }] of articles) {
          const body = await articleMarkdown({ article, title, description, page, site });
          markdown.set(page, body);
          await writeFile(new URL(`${page}.md`, dir), body);
          const links = [
            `<link rel="alternate" type="text/markdown" href="${escapeAttribute(markdownRouteOf(`${root.pathname}${page}`))}">`,
            `<link rel="describedby" href="${escapeAttribute(indexRoute)}">`,
          ].join("");
          await writeFile(new URL(page, dir), html.replace("</head>", `${links}</head>`));
        }
        logger.info(`Wrote markdown copies of ${markdown.size} pages`);

        const index = await readFile(new URL("llms.txt", dir), "utf8").catch(() => undefined);
        if (index === undefined) {
          logger.warn("No llms.txt in the build, so no llms-full.txt");
          return;
        }
        const { head, sections } = readIndex({ index, site });
        const missing = [...sections.values()].flat().filter((page) => !markdown.has(page));
        if (missing.length) throw new Error(`llms.txt links to pages with no markdown copy: ${missing.join(", ")}`);
        const pages = [...new Set(FULL_SECTIONS.flatMap((section) => sections.get(section) ?? []))];
        const full = [
          head.join("\n\n"),
          `The ${FULL_SECTIONS.join(" and ")} pages of ${new URL("llms.txt", root).href}, in full. Each starts with its URL.`,
          ...pages.map((page) => {
            const url = new URL(page.replace(/(^|\/)index\.html$/, "$1"), root).href;
            return `---\n\nSource: ${url}\n\n${markdown.get(page)}`;
          }),
        ].join("\n\n");
        await writeFile(new URL("llms-full.txt", dir), full.trim() + "\n");
        logger.info(`Wrote llms-full.txt: ${pages.length} pages, ${Math.round(full.length / 1024)} KB`);
      },
    },
  };
}
