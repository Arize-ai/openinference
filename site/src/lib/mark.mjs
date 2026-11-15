// @ts-check
/**
 * The OpenInference mark, from first principles.
 *
 * Six panels in a hexagonal ring, cut apart by six seams through the points,
 * around a round bolt. Four rules fix every dimension, on a mark 1000 units
 * tall (centred on the bolt, y down):
 *
 *   1. The seam, the room between panels, is 96 (9.6% of the height).
 *   2. Every corner rounds at 28.
 *   3. The gap round the bolt, at its narrowest (the middle of each panel's
 *      inner edge), is 1.5 seams.
 *   4. The bolt is 1.2 times as wide as a panel is deep.
 *
 * The hexagon isn't a rule: it is solved so that, once the seams cut its
 * points and the shoulders are rounded, the mark is exactly 1000 tall.
 *
 * `scripts/build-brand.mjs` draws the artwork in `src/assets/brand/` from
 * this module, and `src/lib/brand.ts` quotes it, so the two can't drift.
 */

export const rules = {
  height: 1000,
  seam: 96,
  corner: 28,
  /** Gap round the bolt ÷ seam. */
  gapPerSeam: 1.5,
  /** Bolt width ÷ panel depth. */
  boltPerDepth: 1.2,
};

/** The wordmark artwork's cap height, in its own font units (caps from y 0 to 1000). */
export const CAP_UNITS = 1000;

/** The brand colours the artwork is drawn in; `src/styles/tokens.css` repeats them as `--color-*`. */
export const palette = { ink: "#231F20", paper: "#F6F5F2" };

const TAN30 = Math.tan(Math.PI / 6);
const COS30 = Math.cos(Math.PI / 6);

/** @typedef {[number, number]} Point */
/** @type {(a: Point, b: Point) => Point} */
const add = (a, b) => [a[0] + b[0], a[1] + b[1]];
/** @type {(a: Point, b: Point) => Point} */
const sub = (a, b) => [a[0] - b[0], a[1] - b[1]];
/** @type {(a: Point, k: number) => Point} */
const mul = (a, k) => [a[0] * k, a[1] * k];
/** @type {(a: Point) => number} */
const len = (a) => Math.hypot(a[0], a[1]);
/** @type {(a: Point) => Point} */
const unit = (a) => mul(a, 1 / len(a));
/** @type {(p: Point, deg: number) => Point} */
const rotate = (p, deg) => {
  const r = (deg * Math.PI) / 180;
  return [p[0] * Math.cos(r) - p[1] * Math.sin(r), p[0] * Math.sin(r) + p[1] * Math.cos(r)];
};
/** @typedef {{ p: Point, d: Point }} Line */
/** @type {(l1: Line, l2: Line) => Point} */
const intersect = (l1, l2) => {
  const den = l1.d[0] * l2.d[1] - l1.d[1] * l2.d[0];
  const t = ((l2.p[0] - l1.p[0]) * l2.d[1] - (l2.p[1] - l1.p[1]) * l2.d[0]) / den;
  return add(l1.p, mul(l1.d, t));
};
/** A number for path data: two decimals, no trailing zeros. */
export const num = (/** @type {number} */ n) => String(Math.round(n * 100) / 100);

/** A pointy-top hexagon with the given circumradius, from the top point clockwise. */
export const hexagon = (/** @type {number} */ radius) =>
  Array.from({ length: 6 }, (_, i) => rotate([0, -radius], i * 60));

/** A closed polygon with every corner rounded at `radius`, as path data. */
function roundedPolygon({ points, radius }) {
  /** @type {Point[]} */
  const pts = points;
  return (
    pts
      .map((v, i) => {
        const prev = pts[(i - 1 + pts.length) % pts.length];
        const next = pts[(i + 1) % pts.length];
        const a = unit(sub(prev, v));
        const b = unit(sub(next, v));
        const theta = Math.acos(Math.max(-1, Math.min(1, a[0] * b[0] + a[1] * b[1])));
        const t = radius / Math.tan(theta / 2);
        const from = add(v, mul(a, t));
        const to = add(v, mul(b, t));
        const turn = sub(v, prev)[0] * sub(next, v)[1] - sub(v, prev)[1] * sub(next, v)[0];
        return `${i === 0 ? "M" : "L"}${num(from[0])} ${num(from[1])}A${num(radius)} ${num(radius)} 0 0 ${turn > 0 ? 1 : 0} ${num(to[0])} ${num(to[1])}`;
      })
      .join("") + "Z"
  );
}

/**
 * Every dimension of the mark, and its artwork, from the rules.
 * @param {typeof rules} [r]
 */
export function markGeometry(r = rules) {
  const { height, seam, corner, gapPerSeam, boltPerDepth } = r;
  // What the seams and the shoulder rounding take off each point.
  const overshoot = (seam / 2) * TAN30 + corner * (Math.sqrt(3) - 1);
  const side = height / 2 + overshoot;
  /** Centre to the hexagon's flat sides: half the mark's width. */
  const flats = side * COS30;
  const gap = gapPerSeam * seam;
  // bolt width = boltPerDepth × (flats − socket), with socket = bolt + gap.
  const bolt = (boltPerDepth * (flats - gap)) / (2 + boltPerDepth);
  const socket = bolt + gap;
  const socketPoint = socket / COS30;

  // One panel (the top right), between the seams through the top and
  // upper-right points; the others are it, turned.
  /** @type {Line} */
  const outer = { p: [0, -side], d: [flats, side / 2] };
  /** @type {Line} */
  const inner = { p: [0, -socketPoint], d: [socket, socketPoint / 2] };
  /** @type {Line} */
  const seamTop = { p: [seam / 2, 0], d: [0, 1] };
  /** @type {Line} */
  const seamRight = { p: mul([-0.5, -COS30], seam / 2), d: [COS30, -0.5] };
  const panel = [
    intersect(outer, seamTop),
    intersect(outer, seamRight),
    intersect(inner, seamRight),
    intersect(inner, seamTop),
  ];
  const panels = Array.from({ length: 6 }, (_, k) =>
    roundedPolygon({ points: panel.map((p) => rotate(p, k * 60)), radius: corner }),
  );
  return {
    height,
    width: 2 * flats,
    side,
    seam,
    corner,
    gap,
    bolt,
    socket,
    socketPoint,
    /** Outer edge to inner edge. */
    depth: flats - socket,
    /** Clear space on every side of the mark: one bolt diameter. */
    clearSpace: 2 * bolt,
    panels,
  };
}
