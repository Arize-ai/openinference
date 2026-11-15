import { getCollection } from "astro:content";

import { GITHUB_URL, siteRoute } from "../plugins/rehype-repo-links.mjs";

/** The site's base path, with a trailing slash (`/openinference/`). */
export const base = import.meta.env.BASE_URL.replace(/\/?$/, "/");

/** The route of a markdown file the site publishes, by its repository path. */
export const routeOf = (repoPath: string): string => {
  const route = siteRoute(repoPath, base);
  if (!route) throw new Error(`The site does not publish ${repoPath}`);
  return route;
};

const TAGLINE = "Open conventions and instrumentation for observing AI agents with OpenTelemetry.";

export const site = {
  name: "OpenInference",
  /** The landing hero's heading, and the site's meta description. */
  tagline: TAGLINE,
  description: TAGLINE,
  repo: GITHUB_URL,
  community:
    "https://join.slack.com/t/arize-ai/shared_invite/zt-1px8dcmlf-fmThhDFD_V_48oU7ALan4Q",
  phoenix: "https://github.com/Arize-ai/phoenix",
  ax: "https://arize.com/products/ax?utm_source=docs&utm_medium=web&utm_content=openinference",
  arize: "https://arize.com",
};

/** The header nav. Keep labels short enough to fit beside the logo at 390px. */
export const nav: { href: string; label: string; external?: boolean }[] = [
  { href: routeOf("spec/README.md"), label: "Spec" },
  { href: `${base}js/`, label: "JavaScript" },
  { href: `${base}python/`, label: "Python" },
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
