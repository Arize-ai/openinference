/**
 * The landing card's art: the wordmark on the card's centre line, and on the
 * same line the Keyway, large and faint, with the trace hero's run wound once
 * round its bolt. See DESIGN.md ("Social cards").
 *
 * The dial is the trace tree as orbits, drawn in hairlines. The root span is
 * the inner orbit; each level of the tree is the next orbit out, and each
 * span is an arc on its level's orbit in its kind's colour, with a short
 * radial tick down to its parent: the tree's elbow, wound round. The run
 * sweeps 300°, starting and ending either side of the keyway at the bottom,
 * the way a knob's travel leaves a dead zone, under a scale a tick every
 * 100 ms. One marker per kind leads out past the scale to its glyph; the
 * failed request's leads to its duration, in the accent.
 *
 * Pure geometry: it returns the artwork as SVG (shapes and outlined brand
 * artwork only) and its type as labels, so the build can set the type in
 * the site's faces with satori (render.ts) and a browser can draw the same
 * thing.
 */
import { GLYPHS } from "../../components/hero/glyphs";
import { SPANS, T, type Kind } from "../../components/hero/model";
import { markGeometry } from "../mark.mjs";
import { COLOR, KIND } from "./palette";

/** A line of type, placed by its baseline. */
export interface Label {
  x: number;
  y: number;
  text: string;
  color: string;
  size: number;
  anchor: "start" | "middle" | "end";
  weight?: 500 | 600;
  /** Letter spacing, in px. */
  tracking?: number;
}

export interface ArtOptions {
  width: number;
  height: number;
  /** The wordmark's cap height, px at 1280 × 640. */
  wordmarkCap?: number;
  /** The watermark's opacity. */
  watermark?: number;
  /** Space between the tree's levels, px at 1280 × 640. */
  gap?: number;
  /** The run's sweep, in degrees; the rest is the dead zone over the keyway. */
  sweep?: number;
  /** Number every second, or every other. */
  numerals?: "all" | "even";
  /** Mute the LLM colour so the accent means only the failure. */
  calm?: boolean;
}

export const ART_DEFAULTS: Required<Omit<ArtOptions, "width" | "height">> = {
  wordmarkCap: 50,
  watermark: 0.09,
  gap: 32,
  sweep: 300,
  numerals: "all",
  calm: false,
};

/** The frame: rulers, content edges, and the centre line everything sits on. */
export interface Frame {
  rl: number;
  rr: number;
  x0: number;
  x1: number;
  top: number;
  base: number;
  centre: number;
}

export interface Art {
  svg: string;
  labels: Label[];
  frame: Frame;
  /** Named points, for a lab's guides and notes. */
  anchors: Record<string, [number, number]>;
}

const geometry = markGeometry();
const r2 = (n: number) => Math.round(n * 100) / 100;
const mix = (hex: string, t: number, to: string = COLOR.white) => {
  const n = (h: string) => [1, 3, 5].map((i) => parseInt(h.slice(i, i + 2), 16));
  const [a, b] = [n(hex), n(to)];
  return `#${a.map((v, i) => Math.round(v + (b[i] - v) * t).toString(16).padStart(2, "0")).join("")}`;
};

export function frameOf(width: number, height: number): Frame {
  const kx = width / 1280;
  const top = Math.round((64 * height) / 640);
  const rl = Math.round(48 * kx);
  const x0 = Math.round(104 * kx);
  return { rl, rr: width - rl, x0, x1: width - x0, top, base: height - top, centre: height / 2 };
}

/** The kinds that get a marker, each on the span that best spreads them round the dial; the hub is the agent. */
const MARKED: string[] = ["input_guard", "CreateEmbeddings", "vector_search", "rerank", "execute_code", "Response@15"];

/** The card's artwork and type, for a card `width` × `height`. */
export function drawLandingArt(options: ArtOptions, wordmarkSvg: string): Art {
  const o = { ...ART_DEFAULTS, ...options };
  const { width: W, height: H } = o;
  const F = frameOf(W, H);
  const svg: string[] = [];
  const labels: Label[] = [];
  const anchors: Record<string, [number, number]> = {};
  const kindColor = (kind: Kind) => (o.calm && kind === "LLM" ? mix(KIND.LLM, 0.5) : KIND[kind]);

  /* ---------- The mark: as tall as the frame, right flat on x1, its bolt on the centre line. ---------- */
  const s = (F.base - F.top) / geometry.height;
  /** Lengths below are px on the 1280 × 640 card; q carries them to this card's scale. */
  const q = s / (512 / geometry.height);
  const bx = F.x1 - (geometry.width / 2) * s;
  const by = F.centre;
  const inkY = by - geometry.drop * s;
  svg.push(`<g transform="translate(${r2(bx)} ${r2(inkY)}) scale(${r2(s * 1e4) / 1e4})" opacity="${o.watermark}"><path d="${geometry.path}" fill="${COLOR.fg}"/></g>`);
  anchors.markBox = [bx - (geometry.width / 2) * s, inkY - (geometry.height / 2) * s];
  anchors.centre = [bx, by];

  /* ---------- The page markings' rulers, crossed on the centre line. ---------- */
  {
    const p: string[] = [];
    const ink = mix(COLOR.fg, 0.72, COLOR.bg);
    for (const [x, dir] of [[F.rl, -1], [F.rr, 1]] as const) {
      p.push(`<rect x="${x}" y="0" width="1" height="${H}"/>`);
      for (let y = 0; y <= H; y += 16) {
        const len = y % 320 === 0 ? 12 : y % 80 === 0 ? 7 : 3;
        p.push(`<rect x="${dir < 0 ? x - len : x + 1}" y="${y}" width="${len}" height="1"/>`);
        if (y % 80 === 0 && y > 0 && y < H - 8)
          labels.push({ x: x + dir * 14, y: y + 3, text: String(y).padStart(4, "0"), color: ink, size: 9, anchor: dir < 0 ? "end" : "start", tracking: 0.5 });
      }
      p.push(`<path d="M${x - 6.5} ${F.centre + 0.5}h15M${x + 0.5} ${F.centre - 7}v15" stroke="${COLOR.fg}" stroke-width="1.25" fill="none"/>`);
    }
    svg.push(
      `<defs><linearGradient id="fade" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="#fff" stop-opacity="0"/><stop offset=".12" stop-color="#fff"/><stop offset=".88" stop-color="#fff"/><stop offset="1" stop-color="#fff" stop-opacity="0"/></linearGradient><mask id="rulers"><rect width="${W}" height="${H}" fill="url(#fade)"/></mask></defs>`,
      `<g mask="url(#rulers)" fill="${COLOR.fg}" opacity=".28">${p.join("")}</g>`,
    );
  }

  /* ---------- The wordmark: its caps centred on the centre line, flush to x0. ---------- */
  {
    const [, , vw] = wordmarkSvg.match(/viewBox="([^"]+)"/)![1].split(/\s+/).map(Number);
    const paths = [...wordmarkSvg.matchAll(/ d="([^"]+)"/g)].map((m) => m[1]);
    const cap = o.wordmarkCap * q;
    const k = cap / 1000;
    const capTop = F.centre - cap / 2;
    svg.push(`<g transform="translate(${F.x0} ${r2(capTop)}) scale(${r2(k * 1e5) / 1e5})" fill="${COLOR.fg}">${paths.map((d) => `<path d="${d}"/>`).join("")}</g>`);
    anchors.wordmark = [F.x0, capTop];
    anchors.wordmarkEnd = [F.x0 + vw * k, capTop + cap];
  }

  /* ---------- The dial. ---------- */
  const ang = (ms: number) => 90 + (360 - o.sweep) / 2 + (ms / T) * o.sweep;
  const rad = (a: number) => (a * Math.PI) / 180;
  const pol = (r: number, a: number): [number, number] => [r2(bx + r * Math.cos(rad(a))), r2(by + r * Math.sin(rad(a)))];
  const line = (r0: number, r1: number, a: number, attrs: string) => {
    const [p, t] = [pol(r0, a), pol(r1, a)];
    return `<line x1="${p[0]}" y1="${p[1]}" x2="${t[0]}" y2="${t[1]}" ${attrs}/>`;
  };
  const arc = (r: number, a0: number, a1: number) => `M${pol(r, a0)}A${r2(r)} ${r2(r)} 0 ${a1 - a0 > 180 ? 1 : 0} 1 ${pol(r, a1)}`;

  /* The orbits: one per level of the tree, `gap` apart, each span an arc on its level's orbit. */
  const boltR = geometry.bolt * s;
  const orbit = (d: number) => boltR + (14 + d * o.gap) * q;
  const weight = [2.5, 5, 4, 3.5];
  const [start, end] = [ang(0), ang(T)];
  svg.push(`<g fill="none" stroke="${COLOR.fg}" stroke-opacity=".16" stroke-width="1">${[1, 2, 3].map((d) => `<path d="${arc(orbit(d), start, end)}"/>`).join("")}</g>`);
  const span = (i: number) => {
    const a0 = ang(SPANS[i].start);
    return [a0, Math.max(ang(SPANS[i].end), a0 + 1.2)] as const;
  };
  const elbows: string[] = [];
  const arcs = SPANS.map((sp, i) => {
    const [a0, a1] = span(i);
    const color = sp.error ? COLOR.accent : sp.depth === 0 ? COLOR.fg : sp.kind === "HTTP" ? "#aaa59e" : kindColor(sp.kind);
    if (sp.depth > 0) elbows.push(line(orbit(sp.depth - 1) + 3 * q, orbit(sp.depth) - weight[sp.depth] / 2, a0 + 0.35, `stroke="${COLOR.fg}" stroke-opacity=".4" stroke-width="1"`));
    return `<path d="${arc(orbit(sp.depth), a0 + 0.35, a1 - 0.35)}" fill="none" stroke="${color}" stroke-width="${r2(weight[sp.depth] * q)}"/>`;
  });
  svg.push(elbows.join(""), arcs.join(""));
  anchors.orbit = pol(orbit(2), ang(1700));

  /* The hub: a crosshair through the bolt, and the agent's glyph. */
  const glyph = (kind: Kind, [x, y]: [number, number], size: number, color: string) =>
    `<g transform="translate(${r2(x - size / 2)} ${r2(y - size / 2)}) scale(${r2(size / 16)})" fill="none" stroke="${color}" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round">${GLYPHS[kind].replace(/class="fill"/g, `fill="${color}" stroke="none"`)}</g>`;
  {
    const c = orbit(0) - 10 * q;
    svg.push(
      `<g stroke="${COLOR.fg}" stroke-opacity=".35" stroke-width="1"><line x1="${r2(bx - c)}" y1="${by}" x2="${r2(bx - c * 0.45)}" y2="${by}"/><line x1="${r2(bx + c * 0.45)}" y1="${by}" x2="${r2(bx + c)}" y2="${by}"/><line x1="${r2(bx)}" y1="${r2(by - c)}" x2="${r2(bx)}" y2="${r2(by - c * 0.45)}"/><line x1="${r2(bx)}" y1="${r2(by + c * 0.45)}" x2="${r2(bx)}" y2="${r2(by + c)}"/></g>`,
      glyph("AGENT", [bx, by], 18 * q, COLOR.fg),
    );
  }

  /* The scale: a hairline, 100 ms ticks, 500 ms longer, each second longest and numbered; a stop at the run's end. */
  const rs = orbit(3) + 22 * q;
  const rn = rs + 20 * q;
  const scale: string[] = [`<path d="${arc(rs, start, end)}" fill="none" stroke="${COLOR.fg}" stroke-opacity=".5" stroke-width="1"/>`];
  for (let ms = 0; ms <= Math.floor(T / 1000) * 1000; ms += 100) {
    const sec = ms % 1000 === 0;
    const half = ms % 500 === 0;
    scale.push(line(rs, rs + (sec ? 9 : half ? 5.5 : 3) * q, ang(ms), `stroke="${COLOR.fg}" stroke-width="${sec ? 1.5 : 1}" stroke-opacity="${sec ? 1 : 0.5}"`));
    if (sec && (o.numerals === "all" || (ms / 1000) % 2 === 0)) {
      const [x, y] = pol(rn, ang(ms));
      labels.push({ x, y: r2(y + 3.5 * q), text: String(ms / 1000), color: ms === 0 ? COLOR.fg : COLOR.dim, size: r2(10.5 * q), anchor: "middle", weight: 500 });
    }
  }
  scale.push(line(rs - 5 * q, rs + 13 * q, end, `stroke="${COLOR.fg}" stroke-width="2" stroke-linecap="round"`));
  svg.push(scale.join(""));
  anchors.scale = pol(rn, ang(6000));

  /* Markers: a ring on the arc, a leader out past the scale, the kind's glyph at its end. */
  const rl = rn + 32 * q;
  const marker = (i: number, color: string) => {
    const [a0, a1] = span(i);
    const a = (a0 + a1) / 2;
    const r = orbit(SPANS[i].depth);
    const [mx, my] = pol(r, a);
    svg.push(
      line(r + 5 * q, rl, a, `stroke="${color}" stroke-opacity="${color === COLOR.accent ? 1 : 0.55}" stroke-width="1"`),
      `<circle cx="${mx}" cy="${my}" r="${r2(4 * q)}" fill="${COLOR.white}" stroke="${color}" stroke-width="1.5"/>`,
    );
    return { a, tip: pol(rl, a) };
  };
  for (const name of MARKED) {
    const [n, nth] = name.split("@");
    const i = nth ? Number(nth) : SPANS.findIndex((sp) => sp.name === n);
    const kind = SPANS[i].kind;
    const { a } = marker(i, kindColor(kind));
    svg.push(glyph(kind, pol(rl + 11 * q, a), 16 * q, kindColor(kind)));
  }
  anchors.marker = pol(rl + 11 * q, span(SPANS.findIndex((sp) => sp.name === "vector_search"))[0]);

  /* The failure: its marker and leader in the accent, its duration at the end. */
  {
    const i = SPANS.findIndex((sp) => sp.error);
    const { a } = marker(i, COLOR.accent);
    const [lx, ly] = pol(rl + 12 * q, a);
    labels.push({ x: lx, y: r2(ly + 4 * q), text: `${Math.round(SPANS[i].end - SPANS[i].start)} ms`, color: COLOR.accent, size: r2(12 * q), anchor: Math.cos(rad(a)) > 0.2 ? "start" : Math.cos(rad(a)) < -0.2 ? "end" : "middle", weight: 600, tracking: 0.3 });
    anchors.failure = [lx, ly - 4];
  }

  return {
    svg: `<svg xmlns="http://www.w3.org/2000/svg" width="${W}" height="${H}" viewBox="0 0 ${W} ${H}"><rect width="${W}" height="${H}" fill="${COLOR.bg}"/>${svg.join("")}</svg>`,
    labels,
    frame: F,
    anchors,
  };
}
