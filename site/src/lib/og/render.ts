/**
 * Draws a social card (see cards.ts) as a PNG: satori lays out the type in
 * the site's own faces, as outlines, and sharp rasterises it. The landing
 * card's trace is the trace hero's run (components/hero/model.ts), finished,
 * drawn as the hero draws it: glyphs and a span tree beside solid bars in
 * each kind's colour, the failed request hatched in
 * the accent. See DESIGN.md ("Social cards").
 */
import { readFileSync } from "node:fs";
import { createRequire } from "node:module";

import satori from "satori";
import sharp from "sharp";

import markSvg from "../../assets/brand/openinference-mark.svg?raw";
import wordmarkSvg from "../../assets/brand/openinference-wordmark.svg?raw";
import { GLYPHS } from "../../components/hero/glyphs";
import { SPANS, T, type Kind } from "../../components/hero/model";
import { palette } from "../mark.mjs";
import { CARD_SIZE, type Card } from "./cards";

/** The light theme's values from tokens.css: the card is always paper. */
const COLOR = {
  bg: palette.paper,
  fg: palette.ink,
  soft: "#4b4745",
  dim: "#6f6b66",
  rule: "#d8d5ce",
  /** A step past --dim, for the shared start of a package name: large type needs less contrast to recede. */
  faint: "#aaa59e",
  accent: "#ff5a1f",
};
const KIND: Record<Kind, string> = {
  AGENT: COLOR.soft,
  LLM: "#ee8700",
  TOOL: "#e7b700",
  RETRIEVER: "#049690",
  EMBEDDING: "#647cf6",
  RERANKER: "#019c23",
  GUARDRAIL: "#d23fd6",
  HTTP: COLOR.dim,
};

/* The card's geometry, in pixels: the margin, the wordmark, and the trace. */
const { width: W, height: H } = CARD_SIZE;
const PAD_X = 72;
const PAD_Y = 64;
const WORDMARK_HEIGHT = 40;
/** The mark, top right, across the card from the wordmark: never beside it. */
const MARK_HEIGHT = 56;
/** The foot: its hairline, the gap under it, and the address. */
const FOOT_HEIGHT = 46;
const TRACE = {
  /** The trace's left edge and width, in the card's right half. */
  x: 664,
  width: W - PAD_X - 664,
  /** The tree's width: glyph, indent and name; the timeline takes the rest. */
  tree: 196,
  indent: 14,
  glyph: 14,
  /** Row pitch, and the bar's height within it. */
  row: 21,
  bar: 10,
  /** The time axis over the rows. */
  axis: 26,
};
const TRACE_HEIGHT = TRACE.axis + SPANS.length * TRACE.row;
/** The trace ends on the foot's hairline, leaving the mark its clear space above. */
const TRACE_Y = H - (PAD_Y - 8) - FOOT_HEIGHT - TRACE_HEIGHT;

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

/** The trace's geometry, drawn as SVG: gridlines, tree lines, glyphs and bars. */
function traceSvg(): string {
  const { width, tree, indent, glyph, row, bar, axis } = TRACE;
  const timeline = { x: tree, width: width - tree };
  const xOf = (ms: number) => timeline.x + (ms / T) * timeline.width;
  const rowY = (i: number) => axis + i * row;
  const glyphX = (depth: number) => depth * indent;
  const parts: string[] = [];

  // Dotted gridlines, one a second, and the axis rule.
  for (let s = 0; s * 1000 <= T; s += 1) {
    const x = xOf(s * 1000);
    parts.push(
      `<line x1="${x}" y1="${axis - 6}" x2="${x}" y2="${rowY(SPANS.length)}" stroke="${COLOR.rule}" stroke-width="1" stroke-dasharray="1 3"/>`,
    );
  }
  parts.push(`<line x1="0" y1="${axis - 6}" x2="${width}" y2="${axis - 6}" stroke="${COLOR.rule}" stroke-width="1"/>`);

  // Tree lines: from under the parent's glyph down to the child's, then across.
  SPANS.forEach((span, i) => {
    if (span.depth === 0) return;
    let parent = i - 1;
    while (SPANS[parent].depth >= span.depth) parent -= 1;
    const x = glyphX(SPANS[parent].depth) + glyph / 2;
    const top = rowY(parent) + row / 2 + glyph / 2 + 2;
    const mid = rowY(i) + row / 2;
    parts.push(
      `<path d="M${x} ${top}V${mid}H${glyphX(span.depth) - 3}" fill="none" stroke="${COLOR.rule}" stroke-width="1"/>`,
    );
  });

  // The hatch for the failed request: the accent on the diagonal.
  parts.push(
    `<defs><pattern id="hatch" width="4" height="4" patternUnits="userSpaceOnUse" patternTransform="rotate(45)">` +
      `<rect width="4" height="4" fill="${COLOR.accent}" fill-opacity=".18"/><line x1="0" y1="0" x2="0" y2="4" stroke="${COLOR.accent}" stroke-width="2"/></pattern></defs>`,
  );

  SPANS.forEach((span, i) => {
    const color = KIND[span.kind];
    const y = rowY(i) + row / 2;
    // The kind's glyph, stroked in its colour; `.fill` parts filled.
    const body = GLYPHS[span.kind].replace(/class="fill"/g, `fill="${color}" stroke="none"`);
    parts.push(
      `<g transform="translate(${glyphX(span.depth)} ${y - glyph / 2}) scale(${glyph / 16})" fill="none" stroke="${color}" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round">${body}</g>`,
    );
    // The bar: solid, every span having ended; the failed request hatched.
    const x0 = xOf(span.start);
    const x1 = Math.max(xOf(span.end), x0 + 2);
    const top = y - bar / 2;
    if (span.error) {
      parts.push(
        `<rect x="${x0 + 0.5}" y="${top + 0.5}" width="${x1 - x0 - 1}" height="${bar - 1}" fill="url(#hatch)" stroke="${COLOR.accent}" stroke-width="1"/>`,
      );
    } else {
      parts.push(`<rect x="${x0}" y="${top}" width="${x1 - x0}" height="${bar}" fill="${color}"/>`);
    }
  });

  return `<svg xmlns="http://www.w3.org/2000/svg" width="${width}" height="${TRACE_HEIGHT}" viewBox="0 0 ${width} ${TRACE_HEIGHT}">${parts.join("")}</svg>`;
}

/** The trace's type, laid out by satori over the SVG: the ticks and the span names. */
function traceLabels(): Node[] {
  const { x, width, tree, indent, glyph, row, axis } = TRACE;
  const timeline = { x: tree, width: width - tree };
  const label = (style: Style, text: string) =>
    div({ position: "absolute", fontFamily: "Geist Mono", fontWeight: 500, fontSize: 13, ...style }, text);
  const ticks = [0, 2, 4, 6].map((s) =>
    label(
      { left: x + timeline.x + ((s * 1000) / T) * timeline.width + 5, top: TRACE_Y, fontSize: 11, color: COLOR.dim },
      `${s}s`,
    ),
  );
  const names = SPANS.map((span, i) =>
    label(
      {
        left: x + span.depth * indent + glyph + 7,
        top: TRACE_Y + axis + i * row + row / 2 - 9,
        height: 18,
        alignItems: "center",
        color: span.error ? COLOR.accent : COLOR.soft,
      },
      span.note ? `${span.name} ${span.note}` : span.name,
    ),
  );
  return [...ticks, ...names];
}

function landingCard(card: Card): Node {
  const column = TRACE.x - PAD_X - 56;
  const at = card.title.indexOf(" for ");
  const [lineOne, lineTwo] = at < 0 ? [card.title, ""] : [card.title.slice(0, at), card.title.slice(at + 1)];
  const size = 54;
  return div(
    { width: W, height: H, backgroundColor: COLOR.bg, fontFamily: "Urbanist", position: "relative" },
    head(W - 2 * PAD_X),
    div(
      {
        position: "absolute",
        left: PAD_X,
        top: PAD_Y,
        bottom: PAD_Y - 8,
        width: column,
        flexDirection: "column",
        justifyContent: "space-between",
      },
      // Room for the head, which spans the card.
      div({ height: MARK_HEIGHT }),
      div(
        { flexDirection: "column" },
        div(
          { flexDirection: "column", fontSize: size, fontWeight: 600, lineHeight: 1.06, letterSpacing: -0.04 * size, color: COLOR.fg },
          div({}, lineOne),
          div({}, lineTwo),
        ),
        div(
          { marginTop: 24, fontSize: 26, fontWeight: 500, lineHeight: 1.4, color: COLOR.soft, maxWidth: column },
          card.description,
        ),
      ),
      foot(card, column),
    ),
    h("img", {
      src: svgUrl(traceSvg()),
      width: TRACE.width,
      height: TRACE_HEIGHT,
      style: { position: "absolute", left: TRACE.x, top: TRACE_Y },
    }),
    ...traceLabels(),
  );
}

/** The card as a PNG. */
export async function renderCard(card: Card): Promise<Buffer> {
  const tree = card.variant === "landing" ? landingCard(card) : pageCard(card);
  const svg = await satori(tree as Parameters<typeof satori>[0], { ...CARD_SIZE, fonts: FONTS });
  return sharp(Buffer.from(svg), { density: 144 })
    .resize(CARD_SIZE.width, CARD_SIZE.height)
    .png({ compressionLevel: 9 })
    .toBuffer();
}
