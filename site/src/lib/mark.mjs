// @ts-check
/**
 * The OpenInference mark, Keyway, from first principles.
 *
 * Six sides around a nut and bolt, with one keyway cut at the bottom.
 * From a distance it is the O of OpenInference; closer, the bolt over
 * its slot is the lowercase i; as hardware, it is a keyed socket, machined to
 * assemble one way only.
 *
 * Units are Labs units (Labs side = 1000), y down, centred on the mark's
 * ink. Six rules fix it:
 *
 *   height     757.77  the Labs height: the mark's ink, crown to shoulders
 *   wall        135.5  the ring's thickness, outer flat to window flat
 *   bolt         70.5  the bolt's radius
 *   corner        123  the five outer corners' radius
 *   nutCorner    70.5  the window's corners, and the keyway's mouth
 *   shoulder       25  where the keyway meets the silhouette (the Labs outer corner)
 *
 * The keyway is as wide as the bolt: its walls run straight down from the
 * bolt's sides, so the bolt sits over its slot edge on edge, the dot of the i.
 *
 * The hexagon isn't a rule: it is solved so that, once its five corners are
 * rounded and the keyway is cut through its bottom point, the ink is exactly
 * `height` tall. The window (the nut) is the hexagon brought in by `wall`.
 *
 * `scripts/build-brand.mjs` draws the artwork in `src/assets/brand/` from
 * this module, and `src/lib/brand.ts` quotes it, so the two can't drift.
 */

export const rules = {
  height: 757.77,
  wall: 135.5,
  bolt: 70.5,
  corner: 123,
  nutCorner: 70.5,
  shoulder: 25,
};

/** The wordmark artwork's cap height, in its own font units (caps from y 0 to 1000). */
export const CAP_UNITS = 1000;

/** The smallest the mark is drawn, in CSS pixels tall; the favicon is sized to it. */
export const MINIMUM_SIZE = 16;

/** The brand colours the artwork is drawn in; `src/styles/tokens.css` repeats them as `--color-*`. */
export const palette = { ink: "#231F20", paper: "#F6F5F2" };

const SQRT3 = Math.sqrt(3);
const TAN30 = 1 / SQRT3;
const COS30 = SQRT3 / 2;

/** @typedef {[number, number]} Point */
/** @type {(a: Point, b: Point) => Point} */
const add = (a, b) => [a[0] + b[0], a[1] + b[1]];
/** @type {(a: Point, b: Point) => Point} */
const sub = (a, b) => [a[0] - b[0], a[1] - b[1]];
/** @type {(a: Point, k: number) => Point} */
const mul = (a, k) => [a[0] * k, a[1] * k];
/** @type {(a: Point) => Point} */
const unit = (a) => mul(a, 1 / Math.hypot(a[0], a[1]));
/** @type {(p: Point, deg: number) => Point} */
const rotate = (p, deg) => {
  const r = (deg * Math.PI) / 180;
  return [p[0] * Math.cos(r) - p[1] * Math.sin(r), p[0] * Math.sin(r) + p[1] * Math.cos(r)];
};
/**
 * A number for path data: two decimals, no trailing zeros. Floating-point
 * noise is dropped first and halves round toward zero, so mirrored points
 * stay mirrored.
 */
export const num = (/** @type {number} */ n) => {
  const micro = Math.round(Math.abs(n) * 1e6);
  return String((Math.sign(n) * Math.ceil(micro / 1e4 - 0.5)) / 100 || 0);
};

/**
 * A pointy-top hexagon, from the top point clockwise.
 * @param {{ radius: number, centre?: Point }} options
 * @returns {Point[]}
 */
export const hexagon = ({ radius, centre = [0, 0] }) =>
  Array.from({ length: 6 }, (_, i) => add(centre, rotate([0, -radius], i * 60)));

/** A closed polygon with sharp corners, as path data. */
const polygonPath = (/** @type {Point[]} */ points) =>
  `M${points.map(([x, y]) => `${num(x)} ${num(y)}`).join("L")}Z`;

/**
 * A closed polygon with each corner rounded at its own radius, as path data.
 * Convex and concave corners alike: each arc turns the way the outline does.
 * @param {{ corners: { at: Point, radius: number }[] }} options
 */
function roundedPolygon({ corners }) {
  return (
    corners
      .map(({ at, radius }, i) => {
        const prev = corners[(i - 1 + corners.length) % corners.length].at;
        const next = corners[(i + 1) % corners.length].at;
        const toPrev = unit(sub(prev, at));
        const toNext = unit(sub(next, at));
        const angle = Math.acos(Math.max(-1, Math.min(1, toPrev[0] * toNext[0] + toPrev[1] * toNext[1])));
        const tangent = radius / Math.tan(angle / 2);
        const from = add(at, mul(toPrev, tangent));
        const to = add(at, mul(toNext, tangent));
        const inward = sub(at, prev);
        const outward = sub(next, at);
        const turn = inward[0] * outward[1] - inward[1] * outward[0];
        return `${i === 0 ? "M" : "L"}${num(from[0])} ${num(from[1])}A${num(radius)} ${num(radius)} 0 0 ${turn > 0 ? 1 : 0} ${num(to[0])} ${num(to[1])}`;
      })
      .join("") + "Z"
  );
}

/**
 * A circle as path data, so the mark can be one path.
 * @param {{ centre: Point, radius: number }} options
 */
const circlePath = ({ centre: [x, y], radius: r }) =>
  `M${num(x - r)} ${num(y)}a${num(r)} ${num(r)} 0 1 1 ${num(2 * r)} 0a${num(r)} ${num(r)} 0 1 1 ${num(-2 * r)} 0Z`;

/**
 * Every dimension of the mark, and its artwork, from the rules.
 * @param {typeof rules} [r]
 */
export function markGeometry(r = rules) {
  const { height, wall, bolt, corner, nutCorner, shoulder } = r;
  /** The keyway is the bolt's diameter, so their edges line up. */
  const cut = 2 * bolt;
  const halfCut = cut / 2;

  // What the corner rounding takes off the top point, and what the keyway
  // and its rounded shoulders take off the bottom one.
  const overshoot = {
    top: corner / COS30 - corner,
    bottom: halfCut * TAN30 + shoulder * (SQRT3 - 1),
  };
  /** The hexagon's side (and circumradius). */
  const side = (height + overshoot.top + overshoot.bottom) / 2;
  /** How far the hexagon's centre, and the nut and bolt on it, sit below the ink's. */
  const drop = (overshoot.bottom - overshoot.top) / 2;
  /** Centre to the hexagon's flat sides: half the mark's width. */
  const flats = side * COS30;
  /** The window: the hexagon brought in by the ring's thickness. */
  const nut = flats - wall;
  /** Centre to the nut's points. */
  const nutPoint = nut / COS30;
  /** Where the keyway's walls meet the silhouette, and open into the window. */
  const shoulderY = drop + side - halfCut * TAN30;
  const mouthY = drop + nutPoint - halfCut * TAN30;

  const outline = hexagon({ radius: side, centre: [0, drop] });
  const nutOutline = hexagon({ radius: nutPoint, centre: [0, drop] });
  const [top, upperRight, lowerRight, , lowerLeft, upperLeft] = outline;
  const [nutTop, nutUpperRight, nutLowerRight, , nutLowerLeft, nutUpperLeft] = nutOutline;

  /* One outline, anticlockwise from the right shoulder: round the silhouette,
     down the keyway's left wall, round the window, and back down its right wall. */
  const ring = roundedPolygon({
    corners: [
      { at: [halfCut, shoulderY], radius: shoulder },
      { at: lowerRight, radius: corner },
      { at: upperRight, radius: corner },
      { at: top, radius: corner },
      { at: upperLeft, radius: corner },
      { at: lowerLeft, radius: corner },
      { at: [-halfCut, shoulderY], radius: shoulder },
      { at: [-halfCut, mouthY], radius: nutCorner },
      { at: nutLowerLeft, radius: nutCorner },
      { at: nutUpperLeft, radius: nutCorner },
      { at: nutTop, radius: nutCorner },
      { at: nutUpperRight, radius: nutCorner },
      { at: nutLowerRight, radius: nutCorner },
      { at: [halfCut, mouthY], radius: nutCorner },
    ],
  });
  const boltPath = circlePath({ centre: [0, drop], radius: bolt });

  return {
    height,
    width: 2 * flats,
    side,
    drop,
    overshoot,
    corner,
    wall,
    nut,
    nutCorner,
    bolt,
    cut,
    shoulder,
    /** Paper between the bolt and the window's flats. */
    clearance: nut - bolt,
    /** Clear space on every side of the mark: one bolt diameter. */
    clearSpace: 2 * bolt,
    /** The whole mark, as one path. */
    path: ring + boltPath,
    /** The unrounded hexagon and nut the mark is cut from, for construction drawings. */
    construction: { hexagon: polygonPath(outline), nut: polygonPath(nutOutline) },
  };
}
