/**
 * News: the posts in `docs/news/`, newest first, and the links a post
 * makes. See docs/news/README.md for how to write one.
 */
import { getCollection, type CollectionEntry } from "astro:content";

import { base, site } from "./site";

export type NewsPost = CollectionEntry<"news">;

/** The news timeline. */
export const newsIndex = `${base}news/`;

/** The RSS feed of the posts. */
export const newsFeed = `${base}news/rss.xml`;

/** Release notes are written on GitHub, by release-please, per package. */
export const releasesUrl = `${site.repo}/releases`;

let posts: Promise<NewsPost[]> | undefined;

/** Published posts, newest first. Drafts show in `pnpm dev` only. */
export function getNews(): Promise<NewsPost[]> {
  posts ??= getCollection("news", (post) => !(import.meta.env.PROD && post.data.draft)).then((entries) =>
    entries.sort((a, b) => b.data.pubDate.valueOf() - a.data.pubDate.valueOf() || a.id.localeCompare(b.id)),
  );
  return posts;
}

/** How long a post stays on the landing page, from its date to the build. */
const LATEST_DAYS = 60;

/** The newest post, if it's recent enough to lead the landing page. */
export async function getLatestNews(): Promise<NewsPost | undefined> {
  const [latest] = await getNews();
  const age = latest && (Date.now() - latest.data.pubDate.valueOf()) / 86_400_000;
  return age !== undefined && age <= LATEST_DAYS ? latest : undefined;
}

export const newsUrl = (post: NewsPost | { id: string }) => `${base}news/${post.id}.html`;

/** A release's notes on GitHub, by its tag. */
export const releaseUrl = (tag: string) => `${releasesUrl}/tag/${encodeURI(tag)}`;

export const authorUrl = (handle: string) => `https://github.com/${handle}`;

/** `Nov 18, 2026`, in UTC so a date in the frontmatter never shifts a day. */
export const formatDate = (date: Date) =>
  date.toLocaleDateString("en-US", { year: "numeric", month: "short", day: "numeric", timeZone: "UTC" });

/** `Nov 18`, on the timeline, under its year. */
export const formatDay = (date: Date) =>
  date.toLocaleDateString("en-US", { month: "short", day: "numeric", timeZone: "UTC" });

/** `2026-11-18`, for `<time datetime>`. */
export const isoDate = (date: Date) => date.toISOString().slice(0, 10);

/** The posts by year, newest year first, for the timeline. */
export function byYear(entries: NewsPost[]): [number, NewsPost[]][] {
  const years = new Map<number, NewsPost[]>();
  for (const post of entries) {
    const year = post.data.pubDate.getUTCFullYear();
    years.set(year, [...(years.get(year) ?? []), post]);
  }
  return [...years.entries()];
}
