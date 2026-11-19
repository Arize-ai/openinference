/**
 * The landing card's art, rooted in the trace hero: its run, wound once round
 * the Keyway's bolt, the replay's hand at its end (or paused anywhere, as the
 * hero pauses). See DESIGN.md ("Social cards").
 *
 * Everything on the dial is the hero's own vocabulary, turned through 300°:
 *  - one orbit per level of the span tree, a hairline tick down to the parent
 *    where each span starts (the hero's tree lines);
 *  - spans that have ended are solid bars in their kind's colour; spans still
 *    running are dashed outlines from their start to the playhead; spans not
 *    yet started are not drawn;
 *  - dotted gridlines a second apart, labelled 0s, 2s, 4s…;
 *  - the failed request hatched in the accent;
 *  - the playhead, here a knob's hand pivoting on the bolt, with the hero's
 *    ink time tag at its tip and one readout: the span it has just reached
 *    (at the root, just "agent").
 * The run's dead zone sits over the keyway, as a knob's travel leaves one at
 * the bottom. The mark bleeds off the card; the dial lives in its window.
 *
 * Pure geometry: SVG shapes plus labels placed by baseline, so the build sets
 * the type with satori (render.ts) and a browser can draw the same thing.
 */
import { GLYPHS } from "../../components/hero/glyphs";
import { SPANS, T, type Kind } from "../../components/hero/model";
import { markGeometry } from "../mark.mjs";
import { COLOR, KIND } from "./palette";

export interface Label { x: number; y: number; text: string; color: string; size: number; anchor: "start" | "middle" | "end"; weight?: 500 | 600; tracking?: number }
export interface ArtOptions {
  width: number;
  height: number;
  /** Where the replay's hand stands, ms into the run; the whole run by default. */
  at?: number;
  /** Draw the spans still to come as faint ghosts. */
  ghosts?: boolean;
  wordmarkCap?: number;
  watermark?: number;
  markHeight?: number;
  boltX?: number;
  gap?: number;
  sweep?: number;
}
export const ART_DEFAULTS = { at: 8200, ghosts: false, wordmarkCap: 50, watermark: 0.07, markHeight: 900, boltX: 915, gap: 30, sweep: 300 };

const geometry = markGeometry();
const r2 = (n: number) => Math.round(n * 100) / 100;

export function drawLandingArt(options: ArtOptions, wordmarkSvg: string) {
  const o = { ...ART_DEFAULTS, ...options };
  const { width: W, height: H } = o;
  const k = W / 1280;
  const F = { x0: Math.round(104 * k), top: Math.round((64 * H) / 640), base: H - Math.round((64 * H) / 640), centre: H / 2 };
  const svg: string[] = [];
  const labels: Label[] = [];
  const anchors: Record<string, [number, number]> = {};
  const now = Math.min(Math.max(o.at, 0), T);

  /* The mark: large, bleeding off the card; its bolt on the centre line. */
  const s = (o.markHeight * k) / geometry.height;
  const bx = o.boltX * k, by = F.centre;
  svg.push(`<g transform="translate(${r2(bx)} ${r2(by - geometry.drop * s)}) scale(${r2(s * 1e4) / 1e4})" opacity="${o.watermark}"><path d="${geometry.path}" fill="${COLOR.fg}"/></g>`);
  anchors.centre = [bx, by];

  /* The wordmark: caps centred on the centre line, flush to the margin. */
  {
    const paths = [...wordmarkSvg.matchAll(/ d="([^"]+)"/g)].map((m) => m[1]);
    const cap = o.wordmarkCap * k, capTop = F.centre - cap / 2;
    svg.push(`<g transform="translate(${F.x0} ${r2(capTop)}) scale(${r2((cap / 1000) * 1e5) / 1e5})" fill="${COLOR.fg}">${paths.map((d) => `<path d="${d}"/>`).join("")}</g>`);
    anchors.wordmark = [F.x0, capTop];
  }

  /* ---------- The dial ---------- */
  const ang = (ms: number) => 90 + (360 - o.sweep) / 2 + (ms / T) * o.sweep;
  const rad = (a: number) => (a * Math.PI) / 180;
  const pol = (r: number, a: number): [number, number] => [r2(bx + r * Math.cos(rad(a))), r2(by + r * Math.sin(rad(a)))];
  const sector = (r0: number, r1: number, a0: number, a1: number) => {
    const L = a1 - a0 > 180 ? 1 : 0, [p0, p1, p2, p3] = [pol(r1, a0), pol(r1, a1), pol(r0, a1), pol(r0, a0)];
    return `M${p0}A${r2(r1)} ${r2(r1)} 0 ${L} 1 ${p1}L${p2}A${r2(r0)} ${r2(r0)} 0 ${L} 0 ${p3}Z`;
  };
  const line = (r0: number, r1: number, a: number, attrs: string) => { const [p, t] = [pol(r0, a), pol(r1, a)]; return `<line x1="${p[0]}" y1="${p[1]}" x2="${t[0]}" y2="${t[1]}" ${attrs}/>`; };

  const boltR = geometry.bolt * s;
  /** The orbits, `gap` apart; the bars on them, the hero's bar height. */
  const orbit = (d: number) => boltR + (22 + d * o.gap) * k;
  const BAR = [6, 10, 10, 10].map((b) => b * k);
  const inner = orbit(0) - 10 * k, outer = orbit(3) + 12 * k;

  /* Gridlines a second apart, dotted, through every orbit; labelled on even seconds, as the hero's axis is, except where the hand's tag stands. */
  for (let sec = 0; sec * 1000 <= T; sec++) {
    svg.push(line(inner, outer, ang(sec * 1000), `stroke="${COLOR.fg}" stroke-opacity=".3" stroke-width="1" stroke-dasharray="1 3"`));
    if (sec % 2 === 0 && Math.abs(ang(sec * 1000) - ang(now)) > 12) {
      const [x, y] = pol(outer + 16 * k, ang(sec * 1000));
      labels.push({ x, y: r2(y + 4 * k), text: `${sec}s`, color: COLOR.dim, size: r2(12 * k), anchor: "middle", weight: 500 });
    }
  }
  /* Where the tag and readout go: at the hand's tip while they fit inside the mark's window, else in the
     dead zone over the keyway, like a knob's value display, so they never sit on the mark. */
  const ah = ang(now);
  const TAG_H = 22 * k;
  const tagText = `${(now / 1000).toFixed(2)}s`;
  const tagW = (tagText.length * 7.6 + 14) * k;
  const under = SPANS.map((sp, i) => ({ sp, i })).filter(({ sp }) => sp.start <= now && now <= sp.end + 1).sort((a, b) => b.sp.depth - a.sp.depth)[0];
  const readText = !under ? "" : under.sp.depth === 0 ? "agent" : `${under.sp.name}${under.sp.note ? ` ${under.sp.note}` : ""}  ${under.sp.end <= now ? `${Math.round(under.sp.end - under.sp.start)}ms` : `${Math.round(now - under.sp.start)}ms…`}`;
  const readW = !under ? 0 : (readText.length * 7.8 + (under.sp.depth === 0 ? 10 : 34)) * k;
  /** The window: the nut, a hexagon round the bolt with a point at the top; inside when within every flat. */
  const flat = geometry.nut * s - 10 * k;
  const inWindow = (x: number, y: number) => [0, 60, 120, 180, 240, 300].every((a) => (x - bx) * Math.cos(rad(a)) + (y - by) * Math.sin(rad(a)) <= flat);
  const tipAt = outer + 30 * k;
  const [tx0, ty0] = pol(tipAt, ah);
  const right = Math.cos(rad(ah)) >= -0.1 && tx0 + tagW + readW < W - 40 * k;
  const below = Math.sin(rad(ah)) > 0.35;
  const tagX = right ? tx0 - 4 * k : tx0 - tagW + 4 * k;
  const tagY = below ? ty0 + 2 * k : ty0 - TAG_H - 2 * k;
  const rowX0 = right ? tagX : tagX - readW, rowX1 = right ? tagX + tagW + readW : tagX + tagW;
  const fits = [[rowX0, tagY], [rowX1, tagY], [rowX0, tagY + TAG_H], [rowX1, tagY + TAG_H]].every(([x, y]) => inWindow(x, y));

  /* The hand, under the bars: a knob's pointer on the bolt, out to the tag, or just past the dial. */
  const tip = fits ? tipAt : outer + 8 * k;
  svg.push(line(0, tip, ah, `stroke="${COLOR.fg}" stroke-width="${r2(1.25 * k)}"`));

  svg.push(`<defs><pattern id="hatch" width="4" height="4" patternUnits="userSpaceOnUse" patternTransform="rotate(${r2(ang(SPANS[SPANS.findIndex((x) => x.error)].start) + 45)})"><rect width="4" height="4" fill="${COLOR.accent}" fill-opacity=".18"/><line x1="0" y1="0" x2="0" y2="4" stroke="${COLOR.accent}" stroke-width="2"/></pattern></defs>`);
  /* Tree ticks, then the spans: ended solid, running dashed to the playhead, to come not drawn (or ghosted). */
  const ticks: string[] = [], bars: string[] = [];
  SPANS.forEach((sp) => {
    const started = sp.start <= now, ended = sp.end <= now;
    if (!started && !o.ghosts) return;
    const d = sp.depth, [r0, r1] = [orbit(d) - BAR[d] / 2, orbit(d) + BAR[d] / 2];
    const a0 = ang(sp.start) + 0.3;
    const a1 = Math.max(ang(ended || !started ? sp.end : now), a0 + 0.9) - 0.3;
    const color = sp.error ? COLOR.accent : d === 0 ? COLOR.soft : KIND[sp.kind as Kind];
    if (d > 0 && started) ticks.push(line(orbit(d - 1) + BAR[d - 1] / 2 + 1, r0 - 1, a0, `stroke="${COLOR.fg}" stroke-opacity=".3" stroke-width="1"`));
    if (!started) bars.push(`<path d="${sector(r0, r1, a0, a1)}" fill="${color}" fill-opacity=".08"/>`);
    else if (!ended) bars.push(`<path d="${sector(r0 + 0.5, r1 - 0.5, a0, a1)}" fill="none" stroke="${color}" stroke-width="1.25" stroke-dasharray="3 2.5"/>`);
    else if (sp.error) bars.push(`<path d="${sector(r0 + 0.5, r1 - 0.5, a0, a1)}" fill="url(#hatch)" stroke="${COLOR.accent}" stroke-width="1"/>`);
    else bars.push(`<path d="${sector(r0, r1, a0, a1)}" fill="${color}"/>`);
  });
  svg.push(ticks.join(""), bars.join(""));
  anchors.orbit = pol(orbit(1), ang(700));

  /* The pivot, and the tag at the hand's tip. */
  svg.push(
    `<circle cx="${r2(bx)}" cy="${r2(by)}" r="${r2(9 * k)}" fill="${COLOR.fg}"/><circle cx="${r2(bx)}" cy="${r2(by)}" r="${r2(3 * k)}" fill="${COLOR.bg}"/>`,
  );
  {
    /* The tag: at the hand's tip, or centred in the dead zone with the readout beside it. */
    const gap = 10 * k;
    const x = fits ? tagX : bx - (tagW + gap + readW) / 2;
    const y = fits ? tagY : by + outer * 0.82 - TAG_H / 2;
    const toRight = fits ? right : true;
    svg.push(`<rect x="${r2(x)}" y="${r2(y)}" width="${r2(tagW)}" height="${r2(TAG_H)}" fill="${COLOR.fg}"/>`);
    labels.push({ x: r2(x + tagW / 2), y: r2(y + TAG_H / 2 + 4.5 * k), text: tagText, color: COLOR.bg, size: r2(13 * k), anchor: "middle", weight: 500 });
    anchors.tag = [x + tagW / 2, y + TAG_H / 2];

    /* The readout: the span the hand has just reached; at the root, just what it is, the agent. */
    if (under) {
      const { sp } = under;
      const gy = y + TAG_H / 2;
      let tx = toRight ? x + tagW + gap : x - gap;
      if (sp.depth > 0) {
        const gs = 14 * k;
        const gc = sp.error ? COLOR.accent : KIND[sp.kind as Kind];
        const gxx = toRight ? tx : tx - gs;
        svg.push(`<g transform="translate(${r2(gxx)} ${r2(gy - gs / 2)}) scale(${r2(gs / 16)})" fill="none" stroke="${gc}" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round">${GLYPHS[sp.kind as Kind].replace(/class="fill"/g, `fill="${gc}" stroke="none"`)}</g>`);
        tx = toRight ? gxx + gs + 8 * k : gxx - 8 * k;
      }
      labels.push({ x: r2(tx), y: r2(gy + 4.5 * k), text: readText, color: sp.error ? COLOR.accent : COLOR.fg, size: r2(13 * k), anchor: toRight ? "start" : "end", weight: 500 });
      anchors.readout = [toRight ? tx + 30 * k : tx - 30 * k, gy];
    }
  }

  return {
    svg: `<svg xmlns="http://www.w3.org/2000/svg" width="${W}" height="${H}" viewBox="0 0 ${W} ${H}"><rect width="${W}" height="${H}" fill="${COLOR.bg}"/>${svg.join("")}</svg>`,
    labels,
    anchors,
  };
}
