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
  /* The hand, under the bars: a knob's pointer on the bolt, out past the dial to the hero's time tag. */
  const ah = ang(now);
  const tip = outer + 30 * k;
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
    const [tx, ty] = pol(tip, ah);
    const text = `${(now / 1000).toFixed(2)}s`;
    const w = (text.length * 7.6 + 14) * k, h = 22 * k;
    /* The tag sits on the tip's outer side; the readout runs away from the dial, toward whichever side has room. */
    const under0 = SPANS.filter((sp) => sp.start <= now && now <= sp.end + 1).sort((a, b) => b.depth - a.depth)[0];
    const readW = under0 ? (under0.depth === 0 ? 5 * 7.8 + 14 : (`${under0.name}${under0.note ? ` ${under0.note}` : ""}  0000ms`.length * 7.8 + 34)) * k : 0;
    const right = Math.cos(rad(ah)) >= -0.1 && tx + w + readW < W - 40 * k;
    const below = Math.sin(rad(ah)) > 0.35;
    const x = right ? tx - 4 * k : tx - w + 4 * k, y = below ? ty + 2 * k : ty - h - 2 * k;
    svg.push(`<rect x="${r2(x)}" y="${r2(y)}" width="${r2(w)}" height="${r2(h)}" fill="${COLOR.fg}"/>`);
    labels.push({ x: r2(x + w / 2), y: r2(y + h / 2 + 4.5 * k), text, color: COLOR.bg, size: r2(13 * k), anchor: "middle", weight: 500 });
    anchors.tag = [x + w / 2, y + h / 2];

    /* The readout: the span the hand has just reached, the deepest under it, or the failure as it ends. */
    const under = SPANS.map((sp, i) => ({ sp, i })).filter(({ sp }) => sp.start <= now && now <= sp.end + 1).sort((a, b) => b.sp.depth - a.sp.depth)[0];
    if (under && under.sp.depth === 0) {
      /* At the root, the readout is just what it is: the agent. */
      labels.push({ x: r2(right ? x + w + 10 * k : x - 10 * k), y: r2(y + h / 2 + 4.5 * k), text: "agent", color: COLOR.fg, size: r2(13 * k), anchor: right ? "start" : "end", weight: 500 });
      anchors.readout = [right ? x + w + 30 * k : x - 30 * k, y + h / 2];
    } else if (under) {
      const { sp } = under;
      const c = sp.error ? COLOR.accent : COLOR.fg;
      const gx = right ? x + w + 12 * k : x - 12 * k - 14 * k;
      const gs = 14 * k;
      const gy = y + h / 2;
      const gxx = right ? gx : x - 12 * k - gs;
      svg.push(`<g transform="translate(${r2(gxx)} ${r2(gy - gs / 2)}) scale(${r2(gs / 16)})" fill="none" stroke="${sp.error ? COLOR.accent : KIND[sp.kind as Kind]}" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round">${GLYPHS[sp.kind as Kind].replace(/class="fill"/g, `fill="${sp.error ? COLOR.accent : KIND[sp.kind as Kind]}" stroke="none"`)}</g>`);
      const name = `${sp.name}${sp.note ? ` ${sp.note}` : ""}`;
      const dur = sp.end <= now ? `${Math.round(sp.end - sp.start)}ms` : `${Math.round(now - sp.start)}ms…`;
      labels.push({ x: r2(right ? gxx + gs + 8 * k : gxx - 8 * k), y: r2(gy + 4.5 * k), text: `${name}  ${dur}`, color: c, size: r2(13 * k), anchor: right ? "start" : "end", weight: 500 });
      anchors.readout = [right ? gxx + gs + 60 * k : gxx - 60 * k, gy];
    }
  }

  return {
    svg: `<svg xmlns="http://www.w3.org/2000/svg" width="${W}" height="${H}" viewBox="0 0 ${W} ${H}"><rect width="${W}" height="${H}" fill="${COLOR.bg}"/>${svg.join("")}</svg>`,
    labels,
    anchors,
  };
}
