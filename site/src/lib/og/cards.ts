/**
 * The social cards: one 1200 × 630 image per page, for the Open Graph and
 * Twitter tags `BaseLayout` writes. See DESIGN.md ("Social cards").
 *
 * Each card is drawn from the same data as its page (the spec's markdown,
 * the API references' JSON, the news posts' frontmatter), so a page and its card can't disagree. A page
 * with no card of its own (a README translation) shares the landing card.
 */
import { packages as jsPackages } from "../js-api";
import { packages as pythonPackages } from "../python-api";
import { formatDate, getNews } from "../news";
import { base, pages, site } from "../site";
import { getSpecSections, ledeOf, titleOf } from "../spec";

export interface Card {
  /** The card's path under `og/`, without `.png`: the page's route, `index` for a directory. */
  id: string;
  /** The landing card draws the tagline beside the trace; every other card is a page title. */
  variant: "landing" | "page";
  /** Mono caps over the title: where the page sits (`Specification · Spans`). */
  eyebrow: string;
  title: string;
  /** The start of the title every sibling shares (`@arizeai/openinference-`), set back in the dim colour. */
  titlePrefix?: string;
  description: string;
  /** The page's address, without the scheme, for the card's foot. */
  url: string;
}

/** The card id for a page's pathname: `/openinference/spec/spans/llm.html` → `spec/spans/llm`. */
export function cardIdOf(pathname: string): string {
  const route = pathname.startsWith(base) ? pathname.slice(base.length) : pathname.replace(/^\//, "");
  return route.replace(/\.html$/, "").replace(/(^|\/)$/, "$1index");
}

const host = (route: string) => `${new URL(base, import.meta.env.SITE).host}${base}${route}`.replace(/\/$/, "");

/** What package names share, longest first: the card sets it back so the rest reads first. */
const PACKAGE_PREFIXES = [
  "@arizeai/openinference-instrumentation-",
  "@arizeai/openinference-",
  "openinference-instrumentation-",
  "openinference-",
];
const prefixOf = (name: string) => PACKAGE_PREFIXES.find((prefix) => name.startsWith(prefix) && name.length > prefix.length);

const LANGUAGES = { js: "JavaScript", python: "Python" } as const;
const GROUPS = { core: "Core", instrumentation: "Instrumentation", integration: "Integration" } as const;

let cards: Promise<Card[]> | undefined;

/** Every card, built once per build. */
export function getCards(): Promise<Card[]> {
  cards ??= (async () => {
    const page = (card: Omit<Card, "variant" | "url">): Card => ({
      ...card,
      variant: "page",
      url: host(card.id.replace(/(^|\/)index$/, "$1").replace(/^(.+[^/])$/, "$1.html")),
    });

    const landing: Card = {
      id: "index",
      variant: "landing",
      eyebrow: "OpenInference",
      title: site.tagline,
      description: site.lede,
      url: host(""),
    };

    const spec = (await getSpecSections()).flatMap((section) =>
      section.pages.map((entry) =>
        page({
          id: entry.id === "README" ? "spec/index" : `spec/${entry.id}`,
          eyebrow: section.label ? `Specification · ${section.label}` : "Specification",
          title: titleOf(entry),
          description: ledeOf(entry.body) ?? site.description,
        }),
      ),
    );

    const api = (["js", "python"] as const).flatMap((language) => {
      const list = language === "js" ? jsPackages : pythonPackages;
      return [
        page({
          id: `${language}/index`,
          eyebrow: `${LANGUAGES[language]} · API reference`,
          ...pages[language],
        }),
        ...list.map((pkg) =>
          page({
            id: `${language}/${"slug" in pkg ? pkg.slug : pkg.name}`,
            eyebrow: `${LANGUAGES[language]} · ${GROUPS[pkg.group]}`,
            title: pkg.name,
            titlePrefix: prefixOf(pkg.name),
            description: pkg.description,
          }),
        ),
      ];
    });

    const news = [
      page({ id: "news/index", eyebrow: "News", ...pages.news }),
      ...(await getNews()).map((post) =>
        page({
          id: `news/${post.id}`,
          eyebrow: `News · ${formatDate(post.data.pubDate)}`,
          title: post.data.title,
          description: post.data.description,
        }),
      ),
    ];

    return [
      landing,
      ...spec,
      ...api,
      ...news,
      page({ id: "mission", eyebrow: "Mission", ...pages.mission }),
      page({ id: "brand", eyebrow: "Brand", ...pages.brand }),
      page({ id: "credits", eyebrow: "Credits", ...pages.credits }),
    ];
  })();
  return cards;
}

/** The card for a page, by its pathname; the landing card for a page without one. */
export async function cardFor(pathname: string): Promise<Card> {
  const all = await getCards();
  const id = cardIdOf(pathname);
  return all.find((card) => card.id === id) ?? all[0];
}

/** Where a card's image is published. */
export const imagePathOf = (card: Card) => `${base}og/${card.id}.png`;

export const CARD_SIZE = { width: 1200, height: 630 } as const;
