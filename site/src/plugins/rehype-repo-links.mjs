// @ts-check
/**
 * Rewrites relative links in repository markdown so they work on the
 * published site.
 *
 * The markdown rendered by this site (the root README, `spec/`, and
 * `docs/i18n/`) is written for GitHub, so it links to other files with
 * repository-relative paths such as `./llm_spans.md` or
 * `../../python/openinference-instrumentation`. This plugin maps links to
 * files the site publishes onto their site routes and sends everything else
 * to the file on GitHub.
 */
import { existsSync, statSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const REPO_ROOT = fileURLToPath(new URL("../../../", import.meta.url));
const GITHUB_URL = "https://github.com/Arize-ai/openinference";
const RAW_URL = "https://raw.githubusercontent.com/Arize-ai/openinference";
const BRANCH = "main";

/** Matches hrefs that are absolute, protocol-relative, or in-page anchors. */
const NON_RELATIVE = /^(?:[a-z][a-z0-9+.-]*:|\/|#)/i;

/**
 * Maps a repository-relative path to its route on the site, or returns
 * undefined if the site does not publish that path.
 * @param {string} repoPath path relative to the repository root, using `/`
 * @param {string} base the site's base path, with a trailing slash
 */
function siteRoute(repoPath, base) {
  const p = repoPath.replace(/\/$/, "");
  if (p === "" || p === "README.md") return base;
  if (p === "spec" || p === "spec/README.md") return `${base}spec/`;
  let match = p.match(/^spec\/([^/]+)\.md$/);
  if (match) return `${base}spec/${match[1]}.html`;
  match = p.match(/^docs\/i18n\/(README\.[^/]+)\.md$/);
  if (match) return `${base}docs/i18n/${match[1]}.html`;
  return undefined;
}

/**
 * @param {string} repoPath
 * @param {"href" | "src"} kind
 */
function githubUrl(repoPath, kind) {
  if (kind === "src") return `${RAW_URL}/${BRANCH}/${repoPath}`;
  const abs = path.join(REPO_ROOT, repoPath);
  const isDir = existsSync(abs) && statSync(abs).isDirectory();
  return `${GITHUB_URL}/${isDir ? "tree" : "blob"}/${BRANCH}/${repoPath}`;
}

/**
 * @param {string} url the link as written in the markdown
 * @param {string} sourceFile absolute path of the markdown file
 * @param {"href" | "src"} kind
 * @param {string} base
 */
export function rewriteRepoLink(url, sourceFile, kind, base) {
  if (!url || NON_RELATIVE.test(url)) return url;
  const [pathPart, hash = ""] = url.split("#", 2);
  const resolved = path.relative(
    REPO_ROOT,
    path.resolve(path.dirname(sourceFile), decodeURI(pathPart)),
  );
  // Leave links that escape the repository untouched.
  if (resolved.startsWith("..")) return url;
  const repoPath = resolved.split(path.sep).join("/");
  const anchor = hash ? `#${hash}` : "";
  if (kind === "href") {
    const route = siteRoute(repoPath, base);
    if (route) return `${route}${anchor}`;
  }
  return `${githubUrl(encodeURI(repoPath), kind)}${anchor}`;
}

/**
 * @param {{ base: string }} options
 */
export default function rehypeRepoLinks({ base }) {
  const normalizedBase = base.endsWith("/") ? base : `${base}/`;
  /**
   * @param {any} tree
   * @param {{ path?: string, history?: string[] }} file
   */
  return (tree, file) => {
    const sourceFile = file.path ?? file.history?.[0];
    if (!sourceFile) return;
    /** @param {any} node */
    const visit = (node) => {
      // Inline HTML in markdown (e.g. the README's language switcher) is kept
      // as a raw string rather than parsed into elements.
      if (node.type === "raw" && typeof node.value === "string") {
        node.value = node.value.replace(
          /\b(href|src)=(["'])(.*?)\2/g,
          (
            /** @type {string} */ _match,
            /** @type {"href" | "src"} */ attr,
            /** @type {string} */ quote,
            /** @type {string} */ url,
          ) =>
            `${attr}=${quote}${rewriteRepoLink(url, sourceFile, attr, normalizedBase)}${quote}`,
        );
      }
      if (node.type === "element") {
        const props = node.properties ?? {};
        if (node.tagName === "a" && typeof props.href === "string") {
          props.href = rewriteRepoLink(
            props.href,
            sourceFile,
            "href",
            normalizedBase,
          );
        }
        if (node.tagName === "img" && typeof props.src === "string") {
          props.src = rewriteRepoLink(
            props.src,
            sourceFile,
            "src",
            normalizedBase,
          );
        }
      }
      node.children?.forEach(visit);
    };
    visit(tree);
  };
}
