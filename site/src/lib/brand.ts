/**
 * The OpenInference mark and wordmark, for the brand page.
 *
 * The artwork in `src/assets/brand/` is the canonical drawing, as outlines,
 * so it needs no font: the mark (six panels around a bolt) and the wordmark
 * (Urbanist SemiBold, outlined). They are used apart: there is no lockup.
 * `scripts/build-brand.mjs` draws the mark from the rules in `mark.mjs`; this
 * module reads the artwork and quotes the same geometry, so the page and the
 * files agree.
 * Units: the mark is 1000 tall, centred on the bolt, y down.
 */
import markSvg from "../assets/brand/openinference-mark.svg?raw";
import markUrl from "../assets/brand/openinference-mark.svg?url";
import wordmarkSvg from "../assets/brand/openinference-wordmark.svg?raw";
import wordmarkUrl from "../assets/brand/openinference-wordmark.svg?url";
import wordmarkPaperUrl from "../assets/brand/openinference-wordmark-paper.svg?url";
import iconUrl from "../assets/brand/openinference-icon.svg?url";
import { CAP_UNITS, markGeometry, palette } from "./mark.mjs";

export interface ViewBox {
  x: number;
  y: number;
  w: number;
  h: number;
}

export interface Artwork {
  /** File name offered on download. */
  file: string;
  url: string;
  viewBox: ViewBox;
  /** The SVG's children, without the root element or its title. */
  body: string;
}

function artwork({ file, url, raw }: { file: string; url: string; raw: string }): Artwork {
  const values = raw
    .match(/viewBox="([^"]+)"/)?.[1]
    .trim()
    .split(/\s+/)
    .map(Number);
  if (!values || values.length !== 4 || values.some((n) => !Number.isFinite(n))) {
    throw new Error(`${file}: missing or malformed viewBox`);
  }
  const body = raw
    .replace(/^[\s\S]*?<svg[^>]*>/, "")
    .replace(/<\/svg>\s*$/, "")
    .replace(/<title>[\s\S]*?<\/title>/, "");
  const [x, y, w, h] = values;
  return { file, url, viewBox: { x, y, w, h }, body };
}

export const art = {
  mark: artwork({ file: "openinference-mark.svg", url: markUrl, raw: markSvg }),
  wordmark: artwork({ file: "openinference-wordmark.svg", url: wordmarkUrl, raw: wordmarkSvg }),
} as const;

export type ArtworkName = keyof typeof art;

/** The wordmark in paper, for dark grounds where the file can't be recoloured. */
export const wordmarkPaper = { file: "openinference-wordmark-paper.svg", url: wordmarkPaperUrl };

/** The app icon and avatar file, shown as an image (it has its own ground). */
export const icon = { file: "openinference-icon.svg", url: iconUrl };

export const colors = {
  ...palette,
  signal: "#FF5A1F",
  black: "#000000",
} as const;

const geometry = markGeometry();

if (Math.abs(art.mark.viewBox.h - geometry.height) > 0.01 || Math.abs(art.mark.viewBox.w - geometry.width) > 0.01) {
  throw new Error("The brand artwork is out of date with src/lib/mark.mjs: run `pnpm brand`.");
}

/** Clear space on every side: one bolt diameter round the mark, one cap height round the wordmark. */
export const clearSpace = { mark: geometry.clearSpace, wordmark: CAP_UNITS };
