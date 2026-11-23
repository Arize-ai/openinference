import { getCollection } from "astro:content";

import { GITHUB_URL, siteRoute } from "../plugins/rehype-repo-links.mjs";

/** The site's base path, with a trailing slash (`/`). */
export const base = import.meta.env.BASE_URL.replace(/\/?$/, "/");

/** The route of a markdown file the site publishes, by its repository path. */
export const routeOf = (repoPath: string): string => {
  const route = siteRoute(repoPath, base);
  if (!route) throw new Error(`The site does not publish ${repoPath}`);
  return route;
};

export const site = {
  name: "OpenInference",
  /** The landing hero's heading. */
  tagline: "Telemetry for the AI Frontier",
  /** The sentence under it. */
  lede: "Trace critical agent steps with OpenTelemetry.",
  /** The mission in one paragraph: the landing page's statement, under the hero. */
  statement:
    "OpenInference strives to trace the frontier. We make agents observable now, not someday, and bring what we learn to OpenTelemetry.",
  /** The site's meta description. */
  description: "Telemetry for the AI frontier, built on OpenTelemetry",
  repo: GITHUB_URL,
  community:
    "https://join.slack.com/t/arize-ai/shared_invite/zt-1px8dcmlf-fmThhDFD_V_48oU7ALan4Q",
  phoenix: "https://github.com/Arize-ai/phoenix",
  ax: "https://arize.com/products/ax?utm_source=docs&utm_medium=web&utm_content=openinference",
  arize: "https://arize.com",
};

/**
 * Pages drawn by the site rather than from markdown or the API data: their
 * titles and descriptions, shared by the page's <head> and its social card.
 */
export const pages = {
  mission: {
    title: "Mission",
    description:
      "Why OpenInference exists: we instrument agents as soon as they matter, and bring what works back to OpenTelemetry.",
  },
  js: {
    title: "JavaScript API",
    description:
      "API reference for the OpenInference JavaScript packages: the core tracing utilities, the semantic conventions, every instrumentation, and the integrations.",
  },
  python: {
    title: "Python API",
    description:
      "API reference for the OpenInference Python packages: the core instrumentation utilities, the semantic conventions, and every instrumentor.",
  },
  brand: {
    title: "Brand guidelines",
    description: "The OpenInference mark and wordmark: files, clear space, colour and typography.",
  },
  news: {
    title: "News",
    description:
      "What's new in OpenInference: new instrumentation, changes to the spec, and the releases worth knowing about.",
  },
  schemas: {
    title: "Schemas",
    description:
      "The OpenInference semantic conventions as a machine-readable registry: every published version, with its manifest and resolved registry.",
  },
  credits: {
    title: "Credits",
    description: "Work by others that this site is built with, and the licences it is used under.",
  },
};

/** The header nav. Keep labels short enough to fit beside the logo at 390px. */
export const nav: { href: string; label: string; external?: boolean }[] = [
  { href: routeOf("spec/README.md"), label: "Spec" },
  { href: `${base}js/`, label: "JavaScript" },
  { href: `${base}python/`, label: "Python" },
  { href: `${base}news/`, label: "News" },
  { href: site.repo, label: "GitHub", external: true },
];

interface Language {
  /** BCP 47 code, e.g. `pt-BR`. */
  code: string;
  /** The language's name in that language, e.g. `Português (Brasil)`. */
  label: string;
  href: string;
}

/** A translated README's language code, from its id (`README.pt-BR` → `pt-BR`). */
export const languageOf = (id: string) => id.replace(/^README\./, "");

const RTL_LANGUAGES = new Set(["ar", "he", "ur"]);
export const directionOf = (code: string) => (RTL_LANGUAGES.has(code) ? "rtl" : "ltr");

let languages: Promise<Language[]> | undefined;

/**
 * English first, then the translations in `docs/i18n/` by name. Every
 * landing page shows the list, so it is built once per build.
 */
export function getLanguages(): Promise<Language[]> {
  languages ??= (async () => {
    const entries = await getCollection("i18n");
    const name = (code: string) => {
      const label = new Intl.DisplayNames([code], { type: "language" }).of(code) ?? code;
      return label.charAt(0).toLocaleUpperCase(code) + label.slice(1);
    };
    const translations = entries
      .map((entry) => {
        const code = languageOf(entry.id);
        return { code, label: name(code), href: routeOf(`docs/i18n/${entry.id}.md`) };
      })
      .sort((a, b) => a.label.localeCompare(b.label, "en"));
    return [{ code: "en", label: "English", href: base }, ...translations];
  })();
  return languages;
}
