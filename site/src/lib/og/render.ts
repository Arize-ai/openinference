/**
 * Draws a social card (see cards.ts) as a PNG: satori lays out the type in
 * the site's own faces, as outlines, and sharp rasterises it. The landing
 * card's art is drawn by dial.ts: the Keyway with the trace hero's run wound
 * round its bolt. See DESIGN.md ("Social cards").
 */
import { readFileSync } from "node:fs";
import { createRequire } from "node:module";

import satori from "satori";
import sharp from "sharp";

import markSvg from "../../assets/brand/openinference-mark.svg?raw";
import wordmarkSvg from "../../assets/brand/openinference-wordmark.svg?raw";
import { CARD_SIZE, type Card } from "./cards";
import { drawLandingArt, type Label } from "./dial";
import { COLOR } from "./palette";

/* The card's geometry, in pixels: the margin, the wordmark, and the mark. */
const { width: W, height: H } = CARD_SIZE;
const PAD_X = 72;
const PAD_Y = 64;
const WORDMARK_HEIGHT = 40;
/** The mark, top right, across the card from the wordmark: never beside it. */
const MARK_HEIGHT = 56;
/** The foot: its hairline, the gap under it, and the address. */
const FOOT_HEIGHT = 46;

const fontsDir = (pkg: string) => {
  const require = createRequire(import.meta.url);
  return (file: string) => readFileSync(require.resolve(`${pkg}/files/${file}`));
};
const urbanist = fontsDir("@fontsource/urbanist");
const geistMono = fontsDir("@fontsource/geist-mono");
const FONTS = [
  { name: "Urbanist", data: urbanist("urbanist-latin-500-normal.woff"), weight: 500 as const, style: "normal" as const },
  { name: "Urbanist", data: urbanist("urbanist-latin-600-normal.woff"), weight: 600 as const, style: "normal" as const },
  { name: "Geist Mono", data: geistMono("geist-mono-latin-500-normal.woff"), weight: 500 as const, style: "normal" as const },
  { name: "Geist Mono", data: geistMono("geist-mono-latin-600-normal.woff"), weight: 600 as const, style: "normal" as const },
];

/* ---------- A tiny element builder for satori. ---------- */

type Style = Record<string, string | number>;
interface Node {
  type: string;
  props: { style?: Style; children?: Child | Child[]; [key: string]: unknown };
}
type Child = Node | string;
const h = (type: string, props: Node["props"], ...children: Child[]): Node => ({
  type,
  props: { ...props, children: children.length === 1 ? children[0] : children },
});
const div = (style: Style, ...children: Child[]) => h("div", { style: { display: "flex", ...style } }, ...children);
const svgUrl = (svg: string) => `data:image/svg+xml;base64,${Buffer.from(svg).toString("base64")}`;

/** A piece of brand artwork in ink, `height` tall. */
function artwork(svg: string, height: number): Node {
  const [, , vw, vh] = svg.match(/viewBox="([^"]+)"/)![1].split(/\s+/).map(Number);
  return h("img", { src: svgUrl(svg), width: Math.round((height * vw) / vh), height });
}
const wordmark = (height: number) => artwork(wordmarkSvg, height);

/**
 * The card's head: the wordmark top left and the mark top right, on one
 * centre line, used apart as the brand asks (no lockup).
 */
const head = (width: number) =>
  div(
    { position: "absolute", left: PAD_X, top: PAD_Y, width, height: MARK_HEIGHT, alignItems: "center", justifyContent: "space-between" },
    wordmark(WORDMARK_HEIGHT),
    artwork(markSvg, MARK_HEIGHT),
  );

/** Mono caps, tracked: the site's `.eyebrow`. */
const eyebrow = (text: string, style: Style = {}) =>
  div(
    {
      fontFamily: "Geist Mono",
      fontWeight: 500,
      fontSize: 17,
      letterSpacing: 1.6,
      textTransform: "uppercase",
      color: COLOR.dim,
      ...style,
    },
    text,
  );

/** The card's foot: a hairline, then the page's address. */
const foot = (card: Card, width: number) =>
  div(
    { flexDirection: "column", width },
    div({ height: 1, width, backgroundColor: COLOR.rule }),
    div({ marginTop: 22, alignItems: "center", justifyContent: "space-between", width }, eyebrow(card.url, { letterSpacing: 0.6, textTransform: "none" })),
  );

/* ---------- Page cards. ---------- */

/**
 * The largest title size whose lines fit: a greedy wrap on an estimate of
 * Urbanist SemiBold's widths, breaking after spaces, hyphens and slashes as
 * the layout does.
 */
function titleSize(title: string, width: number): number {
  const EM_PER_CHAR = 0.52;
  const pieces = title.match(/[^\s/-]+[\s/-]*/g) ?? [title];
  const lines = (size: number) => {
    let count = 1;
    let used = 0;
    for (const piece of pieces) {
      const w = piece.trimEnd().length * EM_PER_CHAR * size;
      if (used > 0 && used + w > width) {
        count += 1;
        used = 0;
      }
      used += piece.length * EM_PER_CHAR * size;
    }
    return count;
  };
  return [92, 84, 76, 68, 60, 54].find((size) => lines(size) <= 2) ?? 48;
}

function pageCard(card: Card): Node {
  const inner = W - 2 * PAD_X;
  const size = titleSize(card.title, inner);
  return div(
    { width: W, height: H, backgroundColor: COLOR.bg, fontFamily: "Urbanist", position: "relative" },
    head(inner),
    // The title block, centred between the wordmark and the foot.
    div(
      {
        position: "absolute",
        left: PAD_X,
        top: PAD_Y + MARK_HEIGHT,
        bottom: FOOT_HEIGHT + PAD_Y - 8,
        width: inner,
        flexDirection: "column",
        justifyContent: "center",
      },
      eyebrow(card.eyebrow),
      div(
        {
          marginTop: 22,
          fontSize: size,
          fontWeight: 600,
          lineHeight: 1.04,
          letterSpacing: -0.04 * size,
          color: COLOR.fg,
          maxWidth: inner,
          display: "block",
        },
        ...(card.titlePrefix
          ? [h("span", { style: { color: COLOR.faint } }, card.titlePrefix), h("span", {}, card.title.slice(card.titlePrefix.length))]
          : [card.title]),
      ),
      div(
        {
          marginTop: 26,
          fontSize: 30,
          fontWeight: 500,
          lineHeight: 1.36,
          color: COLOR.soft,
          maxWidth: 980,
          display: "block",
          lineClamp: 2,
        },
        card.description,
      ),
    ),
    div({ position: "absolute", left: PAD_X, bottom: PAD_Y - 8 }, foot(card, inner)),
  );
}

/* ---------- The landing card. ---------- */

/**
 * Where a line of Geist Mono's baseline sits in its box, as a fraction of the
 * size, at line-height 1: satori places boxes, the art places baselines.
 */
const MONO_BASELINE = 0.86;

/** A label from the art (dial.ts), laid out by satori on its baseline. */
function label({ x, y, text, color, size, anchor, weight = 500, tracking = 0 }: Label): Node {
  const box = 240;
  const left = anchor === "start" ? x : anchor === "middle" ? x - box / 2 : x - box;
  return div(
    {
      position: "absolute",
      left,
      top: y - MONO_BASELINE * size,
      width: box,
      height: size,
      justifyContent: anchor === "start" ? "flex-start" : anchor === "middle" ? "center" : "flex-end",
      fontFamily: "Geist Mono",
      fontWeight: weight,
      fontSize: size,
      lineHeight: 1,
      letterSpacing: tracking,
      color,
      whiteSpace: "nowrap",
    },
    text,
  );
}

/** The landing card: the Keyway, large, with the run wound round its bolt (dial.ts). */
function landingCard(card: Card): Node {
  const { width, height } = card.size ?? CARD_SIZE;
  const art = drawLandingArt({ width, height }, wordmarkSvg);
  return div(
    { width, height, backgroundColor: COLOR.bg, position: "relative" },
    h("img", { src: svgUrl(art.svg), width, height, style: { position: "absolute", left: 0, top: 0 } }),
    ...art.labels.map(label),
  );
}

/** The card as a PNG. */
export async function renderCard(card: Card): Promise<Buffer> {
  const size = card.size ?? CARD_SIZE;
  const tree = card.variant === "landing" ? landingCard(card) : pageCard(card);
  const svg = await satori(tree as Parameters<typeof satori>[0], { ...size, fonts: FONTS });
  return sharp(Buffer.from(svg), { density: 144 })
    .resize(size.width, size.height)
    .png({ compressionLevel: 9 })
    .toBuffer();
}
