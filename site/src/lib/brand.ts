/**
 * The OpenInference mark and lockup, for the brand page.
 *
 * The artwork in `src/assets/brand/` is the canonical drawing, as outlines,
 * so it needs no font: the mark (six panels around a bolt), the wordmark
 * (Urbanist SemiBold, outlined) and the lockup. `scripts/build-brand.mjs`
 * draws the mark and lockup from the rules in `mark.mjs`; this module reads
 * the artwork and quotes the same geometry, so the page and the files agree.
 * Units: the mark is 1000 tall, centred on the bolt, y down.
 */
import lockupSvg from "../assets/brand/openinference-lockup.svg?raw";
import lockupUrl from "../assets/brand/openinference-lockup.svg?url";
import markSvg from "../assets/brand/openinference-mark.svg?raw";
import markUrl from "../assets/brand/openinference-mark.svg?url";
import wordmarkSvg from "../assets/brand/openinference-wordmark.svg?raw";
import wordmarkUrl from "../assets/brand/openinference-wordmark.svg?url";
import iconUrl from "../assets/brand/openinference-icon.svg?url";
import { CAP_SHARE, hexagon, markGeometry, palette, rules } from "./mark.mjs";

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
  lockup: artwork({ file: "openinference-lockup.svg", url: lockupUrl, raw: lockupSvg }),
  wordmark: artwork({ file: "openinference-wordmark.svg", url: wordmarkUrl, raw: wordmarkSvg }),
} as const;

export type ArtworkName = keyof typeof art;

/** The app icon and avatar file, shown as an image (it has its own ground). */
export const icon = { file: "openinference-icon.svg", url: iconUrl };

export const colors = {
  ...palette,
  signal: "#FF5A1F",
  black: "#000000",
} as const;

const geometry = markGeometry();
/** The six panels, one path each, clockwise from the top right. */
export const panelPaths = geometry.panels;
/** The six panels, as one path (each panel is one closed subpath). */
export const panelsPath = panelPaths.join("");

if (Math.abs(art.mark.viewBox.h - geometry.height) > 0.01 || Math.abs(art.mark.viewBox.w - geometry.width) > 0.01) {
  throw new Error("The brand artwork is out of date with src/lib/mark.mjs: run `pnpm brand`.");
}

/** The mark's measurements, from the rules in `mark.mjs`. */
export const mark = {
  height: geometry.height,
  width: geometry.width,
  side: geometry.side,
  seam: geometry.seam,
  corner: geometry.corner,
  /** Socket apothem: centre to the middle of a socket flat. */
  socketApothem: geometry.socket,
  /** Socket circumradius: centre to a socket point. */
  socketPoint: geometry.socketPoint,
  /** Outer edge to inner edge. */
  depth: geometry.depth,
  /** Bolt radius. */
  bolt: geometry.bolt,
  /** Bolt to socket flat: the gap round the bolt at its narrowest. */
  gap: geometry.gap,
};

export { hexagon, rules };

/** Clear space on every side of the mark or lockup: one bolt diameter. */
export const clearSpace = geometry.clearSpace;

/* Lockup: the mark, one bolt diameter, then the wordmark, caps on the bolt. */

const markBox = art.mark.viewBox;
const lockupBox = art.lockup.viewBox;

export const lockup = {
  /** Mark's right edge to the wordmark: one bolt diameter. */
  gap: clearSpace,
  markRight: markBox.x + markBox.w,
  wordmarkLeft: markBox.x + markBox.w + clearSpace,
  /** Cap height, flat to flat, centred on the bolt. */
  capHeight: CAP_SHARE * geometry.height,
  /** How far the p's descender hangs below the mark. */
  descender: lockupBox.y + lockupBox.h - (markBox.y + markBox.h),
  /** Lockup width over the mark's height. */
  aspect: lockupBox.w / markBox.h,
};

export const minimumSize = { mark: 16, lockup: 24 };
