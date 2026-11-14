// @ts-check
/**
 * Draws the brand artwork from the mark's rules (src/lib/mark.mjs):
 *
 *   src/assets/brand/openinference-mark.svg     the mark
 *   src/assets/brand/openinference-lockup.svg        the mark and wordmark, in ink
 *   src/assets/brand/openinference-lockup-paper.svg  the same in paper, for dark grounds
 *                                                    (the README banner's dark mode)
 *   src/assets/brand/openinference-icon.svg          app icon and avatar
 *   public/favicon.svg                               the mark, ink or paper by theme
 *   public/favicon-32.png                            the icon, for browsers without SVG favicons
 *   public/apple-touch-icon.png                      the icon at 180 px, for iOS home screens
 *
 * The wordmark (src/assets/brand/openinference-wordmark.svg) is drawn
 * artwork, Urbanist SemiBold outlined, in its own font units with caps from
 * y 0 to 1000; this script places it but never redraws it.
 *
 * Run it after changing the rules: `pnpm brand`.
 */
import { readFileSync, writeFileSync } from "node:fs";
import { fileURLToPath } from "node:url";

import sharp from "sharp";

import { CAP_SHARE, markGeometry, num } from "../src/lib/mark.mjs";

const root = (/** @type {string} */ path) => fileURLToPath(new URL(`../${path}`, import.meta.url));
const INK = "#231F20";
const PAPER = "#F6F5F2";
/** The wordmark's cap height in its font units. */
const CAP_UNITS = 1000;

const mark = markGeometry();
const body = `<path d="${mark.panels.join("")}"/><circle r="${num(mark.bolt)}"/>`;
const box = { x: -mark.width / 2, y: -mark.height / 2, w: mark.width, h: mark.height };
const viewBox = (/** @type {{ x: number, y: number, w: number, h: number }} */ b) =>
  [b.x, b.y, b.w, b.h].map(num).join(" ");
const svg = ({ box: b, inner, style = "" }) =>
  `<svg xmlns="http://www.w3.org/2000/svg" viewBox="${viewBox(b)}"><title>OpenInference</title>${style}<g fill="${INK}">${inner}</g></svg>\n`;

/* The wordmark: one bolt diameter right of the mark, caps centred on the bolt. */
const wordmarkSvg = readFileSync(root("src/assets/brand/openinference-wordmark.svg"), "utf8");
const wordmarkPath = wordmarkSvg.match(/<path d="([^"]+)"/)?.[1];
const wordmarkBox = wordmarkSvg.match(/viewBox="([^"]+)"/)?.[1].split(/\s+/).map(Number);
if (!wordmarkPath || !wordmarkBox || wordmarkBox.length !== 4) {
  throw new Error("openinference-wordmark.svg: expected one path and a viewBox");
}
const [wx, wy, ww, wh] = wordmarkBox;
const scale = (CAP_SHARE * mark.height) / CAP_UNITS;
const left = box.x + box.w + 2 * mark.bolt;
const top = -(CAP_SHARE * mark.height) / 2;
const lockupRight = left + (wx + ww) * scale;
const lockupBottom = Math.max(box.y + box.h, top + (wy + wh) * scale);
const lockupBox = { x: box.x, y: box.y, w: lockupRight - box.x, h: lockupBottom - box.y };
const wordmark = `<path transform="translate(${num(left)} ${num(top)}) scale(${num(scale * 10000) / 10000})" d="${wordmarkPath}"/>`;

/* The favicon: a square, the mark centred with a little room so the
   antialiased points aren't clipped. */
const faviconSide = Math.max(box.w, box.h) * 1.04;
const faviconBox = { x: -faviconSide / 2, y: -faviconSide / 2, w: faviconSide, h: faviconSide };
const faviconStyle = `<style>g{fill:${INK}}@media (prefers-color-scheme:dark){g{fill:${PAPER}}}</style>`;

/* The app icon and avatar: paper mark on an ink square, one bolt diameter
   clear on every side, so it also survives a circular crop. */
const iconSide = mark.height + 4 * mark.bolt;
const iconBox = { x: -iconSide / 2, y: -iconSide / 2, w: iconSide, h: iconSide };
const icon =
  `<svg xmlns="http://www.w3.org/2000/svg" viewBox="${viewBox(iconBox)}"><title>OpenInference</title>` +
  `<rect x="${num(iconBox.x)}" y="${num(iconBox.y)}" width="${num(iconSide)}" height="${num(iconSide)}" fill="${INK}"/>` +
  `<g fill="${PAPER}">${body}</g></svg>\n`;

writeFileSync(root("src/assets/brand/openinference-mark.svg"), svg({ box, inner: body }));
writeFileSync(root("src/assets/brand/openinference-icon.svg"), icon);
const lockupSvg = svg({ box: lockupBox, inner: body + wordmark });
writeFileSync(root("src/assets/brand/openinference-lockup.svg"), lockupSvg);
writeFileSync(
  root("src/assets/brand/openinference-lockup-paper.svg"),
  lockupSvg.replace(`<g fill="${INK}">`, `<g fill="${PAPER}">`),
);
writeFileSync(
  root("public/favicon.svg"),
  svg({ box: faviconBox, inner: body, style: faviconStyle }).replace(`<g fill="${INK}">`, "<g>"),
);

/* Raster fallbacks, from the icon: it carries its own ground, so it reads on
   light and dark browser chrome alike, and iOS needs an opaque square. */
const raster = (/** @type {number} */ px, /** @type {string} */ path) =>
  sharp(Buffer.from(icon), { density: (72 * px) / iconSide * 4 })
    .resize(px, px)
    .flatten({ background: INK })
    .png({ compressionLevel: 9 })
    .toFile(root(path));
await raster(32, "public/favicon-32.png");
await raster(180, "public/apple-touch-icon.png");

console.log(
  `mark ${num(mark.width)} × ${num(mark.height)}; bolt r ${num(mark.bolt)}, socket ${num(mark.socket)}, ` +
    `panels ${num(mark.depth)} deep; lockup ${num(lockupBox.w)} × ${num(lockupBox.h)}`,
);
