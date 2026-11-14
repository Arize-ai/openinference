/**
 * The hero's background art: a photo printed as a 45° halftone, under the
 * lockup (HeroLockup.astro). Ported from the Arize Labs site
 * (github.com/arize-labs/website, src/components/heroes/hero-photo.ts),
 * without its glass mark; the photo is placed by a focal point held behind
 * the lockup rather than by its top edge. It drifts with the pointer under
 * a fixed screen. Raw WebGL2; still under reduced motion.
 */

/**
 * The lockup's layout within the band, shared with HeroLockup.astro: the
 * band fades out from 55% down, so the lockup sits above its centre.
 */
export const LOCKUP_Y = 0.44;

/**
 * A 45° screen's dots repeat every cell/√2 px along each axis. Snapping that
 * to whole pixels renders every dot the same, with no beat (streaks).
 */
const snapCell = (cell: number) => Math.max(2, Math.round(cell / Math.SQRT2)) * Math.SQRT2;

/** The band in device pixels, origin bottom-left, and its halftone cell. */
function bandFrame(band: { clientWidth: number; clientHeight: number }) {
  const w = band.clientWidth, h = band.clientHeight;
  const dpr = Math.min(devicePixelRatio || 1, 2);
  return { w, h, dpr, W: Math.round(w * dpr), H: Math.round(h * dpr), cell: snapCell((w < 640 ? 3.4 : 4.4) * dpr) };
}

/** Where the photo sits in the band. */
export interface PhotoLayout {
  /** Point of the photo kept at the band's horizontal centre, 0–1 from the left. */
  focusX: number;
  /** The photo's height as a fraction of the band's height. */
  fitHeight: number;
  /**
   * The photo's minimum width as a fraction of the band's: above 1 leaves
   * room to drift; below 1 the photo is centred and its edges fade out.
   */
  minWidth: number;
  /** Point of the photo kept behind the lockup's centre, 0–1 from the top. */
  focusY: number;
  /**
   * How heavily shadow prints on the paper ground, 0–1 (0.42 on the Arize
   * Labs site). Lower it for a photo that is mostly shadow.
   */
  paperInk: number;
}

const INTRO_MS = 1800;
// Drift as a fraction of the photo shown: pointer parallax, and a slow wander.
const PARALLAX = [0.012, 0.008];
const WANDER = [0.004, 0.003];

const VERTEX = /* glsl */ `#version 300 es
in vec2 aPos;
void main() { gl_Position = vec4(aPos, 0.0, 1.0); }
`;

const FRAGMENT = /* glsl */ `#version 300 es
precision highp float;
uniform sampler2D uPhoto;
uniform vec2 uRes;
uniform vec2 uImgSize;
uniform float uFocusX;
uniform float uFitHeight;
uniform float uMinWidth;
uniform float uFocusY;
uniform float uPaperInk;
uniform float uLockupY;
uniform vec2 uShift;
uniform float uDpr;
uniform float uCell;
uniform float uIntro;
uniform float uPaper;
uniform vec3 uBg;
uniform vec3 uInk;
out vec4 outColor;
const vec3 CREAM = vec3(0.937, 0.914, 0.863);

float hash(vec2 p) {
  vec3 q = fract(vec3(p.xyx) * 0.1031);
  q += dot(q, q.yzx + 33.33);
  return fract((q.x + q.y) * q.z);
}

// On ink, whiteness rather than luminance: cloud prints, blue sea drops out.
float keyAt(vec2 uv, float lod) {
  vec3 c = textureLod(uPhoto, uv, lod).rgb;
  float hi = max(c.r, max(c.g, c.b));
  float lo = min(c.r, min(c.g, c.b));
  float L = dot(c, vec3(0.299, 0.587, 0.114));
  return mix(L - (hi - lo) * 1.2, L, uPaper);
}

// The photo's UV (y up) at a point of the band (0–1, y up). The focal point
// sits behind the lockup, as far as the photo's edges allow.
// Device pixels per photo pixel.
float photoScale() {
  return max(uRes.y * uFitHeight / uImgSize.y, uRes.x * uMinWidth / uImgSize.x);
}

vec2 photoUv(vec2 band, float s) {
  vec2 shown = uRes / (uImgSize * s);
  // Keep the photo's edges out of view, with a margin for the drift; where
  // the photo is smaller than the band, centre it instead.
  vec2 lo = 0.5 * shown + 0.012;
  vec2 focus = vec2(uFocusX, 1.0 - uFocusY) - vec2(0.0, uLockupY - 0.5) * shown;
  vec2 c = mix(clamp(focus, lo, max(lo, 1.0 - lo)), vec2(0.5), step(vec2(1.0), shown + 0.024));
  return c + (band - 0.5) * shown - uShift * shown;
}

void main() {
  vec2 frag = gl_FragCoord.xy;
  vec2 band = frag / uRes;

  float scale = photoScale();
  vec2 uv = photoUv(band, scale);
  float lod = max(0.0, log2(1.0 / scale));
  float sharp = keyAt(clamp(uv, 0.0, 1.0), lod);
  float soft = keyAt(clamp(uv, 0.0, 1.0), lod + 3.5);
  float k = sharp + (sharp - soft) * 0.6;
  // Fade the photo out toward its edges, for when it is smaller than the band.
  vec2 e = smoothstep(0.0, 0.08, uv) * smoothstep(0.0, 0.08, 1.0 - uv);
  float edge = e.x * e.y;

  // Thin the print over the bottom of the band, ahead of its CSS fade.
  float fog = mix(0.4, 1.0, smoothstep(0.0, 0.38, band.y));

  // 45° screen.
  vec2 r = mat2(0.7071, -0.7071, 0.7071, 0.7071) * frag / uCell * 6.2832;
  float screen = 0.5 + 0.25 * (cos(r.x) + cos(r.y));
  float fw = fwidth(screen) * 0.75;

  vec3 col;
  if (uPaper < 0.5) {
    // Ink ground: light prints as cream dots.
    float v = smoothstep(0.3, 1.0, k) * fog * edge;
    float tone = v * 0.5 * uIntro;
    tone += (hash(floor(frag / uDpr)) - 0.5) * 0.06 * smoothstep(0.02, 0.2, tone);
    // Ungated, so even dark areas keep a fine pinpoint screen.
    float on = smoothstep(1.0 - tone - fw, 1.0 - tone + fw, screen);
    col = mix(uBg, CREAM * mix(0.55, 1.0, tone), on);
  } else {
    // Paper ground: shadow prints as ink.
    float ink = (1.0 - smoothstep(0.5, 0.98, k)) * uPaperInk * fog * edge * uIntro;
    float on = smoothstep(1.0 - ink - fw, 1.0 - ink + fw, screen) * smoothstep(0.0, 0.03, ink);
    col = mix(uBg, uInk, on);
  }
  outColor = vec4(col, 1.0);
}
`;

const UNIFORMS = ["uPhoto", "uRes", "uImgSize", "uFocusX", "uFitHeight", "uMinWidth", "uFocusY", "uPaperInk", "uLockupY", "uShift", "uDpr", "uCell", "uIntro", "uPaper", "uBg", "uInk"]  as const;

function compile(gl: WebGL2RenderingContext) {
  const shader = (type: number, src: string) => {
    const s = gl.createShader(type)!;
    gl.shaderSource(s, src);
    gl.compileShader(s);
    return s;
  };
  const program = gl.createProgram()!;
  const vs = shader(gl.VERTEX_SHADER, VERTEX);
  const fs = shader(gl.FRAGMENT_SHADER, FRAGMENT);
  gl.attachShader(program, vs);
  gl.attachShader(program, fs);
  gl.bindAttribLocation(program, 0, "aPos");
  gl.linkProgram(program);
  if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
    throw new Error(gl.getShaderInfoLog(fs) || gl.getProgramInfoLog(program) || "hero photo: link failed");
  }
  gl.deleteShader(vs);
  gl.deleteShader(fs);
  gl.useProgram(program);
  // One triangle that covers the viewport.
  gl.bindBuffer(gl.ARRAY_BUFFER, gl.createBuffer());
  gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1, -1, 3, -1, -1, 3]), gl.STATIC_DRAW);
  gl.enableVertexAttribArray(0);
  gl.vertexAttribPointer(0, 2, gl.FLOAT, false, 0, 0);
  return Object.fromEntries(UNIFORMS.map((n) => [n, gl.getUniformLocation(program, n)])) as Record<(typeof UNIFORMS)[number], WebGLUniformLocation | null>;
}

/** A CSS colour as 0–1 sRGB channels, via the 2D canvas's colour parser. */
let colorProbe: CanvasRenderingContext2D | undefined;
function rgb(css: string, fallback: string): [number, number, number] {
  const ctx = (colorProbe ??= document.createElement("canvas").getContext("2d", { willReadFrequently: true })!);
  ctx.fillStyle = fallback;
  ctx.fillStyle = css || fallback;
  ctx.fillRect(0, 0, 1, 1);
  const [r, g, b] = ctx.getImageData(0, 0, 1, 1).data;
  return [r / 255, g / 255, b / 255];
}

/**
 * Loads the photo into texture unit 0, mipmapped, and resolves to its size.
 * A function of its own, so the decoded image isn't held by the render loop.
 */
async function uploadPhoto(gl: WebGL2RenderingContext, src: string): Promise<[number, number]> {
  const img = new Image();
  img.decoding = "async";
  img.src = src;
  await img.decode();
  gl.activeTexture(gl.TEXTURE0);
  gl.bindTexture(gl.TEXTURE_2D, gl.createTexture());
  gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, true);
  gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGB, gl.RGB, gl.UNSIGNED_BYTE, img);
  gl.generateMipmap(gl.TEXTURE_2D);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR_MIPMAP_LINEAR);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
  return [img.naturalWidth, img.naturalHeight];
}

/**
 * Mounts the photo on `canvas`. Returns false when WebGL2 is unavailable, so
 * the caller can fall back; `onLost` runs if the GPU context is lost later.
 */
export function mountHeroPhoto(canvas: HTMLCanvasElement, src: string, layout: PhotoLayout, onLost?: () => void): boolean {
  const gl = canvas.getContext("webgl2", { alpha: false, antialias: false, powerPreference: "low-power" });
  if (!gl) return false;
  let u: ReturnType<typeof compile>;
  try {
    u = compile(gl);
  } catch (e) {
    console.error(e);
    return false;
  }

  const reduced = matchMedia("(prefers-reduced-motion: reduce)").matches;
  const root = document.documentElement;
  let loaded = false;
  let intro = 0;

  gl.uniform1i(u.uPhoto, 0);
  // Placeholder size until the photo loads.
  gl.uniform2f(u.uImgSize, 1, 1);
  gl.uniform1f(u.uFocusX, layout.focusX);
  gl.uniform1f(u.uFitHeight, layout.fitHeight);
  gl.uniform1f(u.uMinWidth, layout.minWidth);
  gl.uniform1f(u.uFocusY, layout.focusY);
  gl.uniform1f(u.uPaperInk, layout.paperInk);
  // The lockup's centre, as a fraction of the band from the bottom.
  gl.uniform1f(u.uLockupY, 1 - LOCKUP_Y);

  function size() {
    const f = bandFrame(canvas);
    if (!f.w || !f.h) return;
    if (canvas.width !== f.W || canvas.height !== f.H) {
      canvas.width = f.W;
      canvas.height = f.H;
    }
    gl!.viewport(0, 0, f.W, f.H);
    gl!.uniform2f(u.uRes, f.W, f.H);
    gl!.uniform1f(u.uDpr, f.dpr);
    gl!.uniform1f(u.uCell, f.cell);
  }

  // Match the page's colours, so there's no seam.
  function theme() {
    const light = root.dataset.theme === "light";
    const style = getComputedStyle(root);
    gl!.uniform1f(u.uPaper, light ? 1 : 0);
    gl!.uniform3f(u.uBg, ...rgb(style.getPropertyValue("--bg").trim(), light ? "#f6f5f2" : "#141213"));
    gl!.uniform3f(u.uInk, ...rgb(style.getPropertyValue("--fg").trim(), "#231f20"));
  }

  const shift = [0, 0], target = [0, 0], drawn = [0, 0];
  function render() {
    gl!.uniform1f(u.uIntro, loaded ? intro : 0);
    gl!.uniform2f(u.uShift, shift[0], shift[1]);
    drawn[0] = shift[0];
    drawn[1] = shift[1];
    gl!.drawArrays(gl!.TRIANGLES, 0, 3);
  }

  // Listen on the window: the lockup sits on top of this canvas. The
  // position is kept in client px and placed on the band once a frame.
  const client = [0, 0], pointer = [0, 0];
  let inWindow = false, hasPointer = false;
  const listening = new AbortController();
  if (!reduced) {
    addEventListener("pointermove", (e) => {
      client[0] = e.clientX;
      client[1] = e.clientY;
      inWindow = true;
    }, { passive: true, signal: listening.signal });
    root.addEventListener("pointerleave", () => (inWindow = false), { signal: listening.signal });
  }
  function placePointer() {
    if (!inWindow) return void (hasPointer = false);
    const r = canvas.getBoundingClientRect();
    const x = (client[0] - r.left) / r.width, y = (client[1] - r.top) / r.height;
    hasPointer = x >= 0 && x <= 1 && y >= 0 && y <= 1;
    pointer[0] = x - 0.5;
    pointer[1] = y - 0.5;
  }

  let raf = 0, last = 0, time = 0, introStart = -1;
  function frame(now: number) {
    raf = requestAnimationFrame(frame);
    const dt = last ? Math.min((now - last) / 1000, 1 / 20) : 0;
    last = now;
    time += dt;
    if (loaded) {
      if (introStart < 0) introStart = now;
      intro = 1 - Math.pow(1 - Math.min((now - introStart) / INTRO_MS, 1), 3);
    }
    placePointer();
    target[0] = Math.sin(time * 0.07) * WANDER[0] + (hasPointer ? pointer[0] * PARALLAX[0] : 0);
    target[1] = Math.sin(time * 0.05 + 1.3) * WANDER[1] - (hasPointer ? pointer[1] * PARALLAX[1] : 0);
    const k = 1 - Math.exp(-dt * 2);
    shift[0] += (target[0] - shift[0]) * k;
    shift[1] += (target[1] - shift[1]) * k;
    // Once the intro is done the wander moves under a device pixel a second:
    // skip frames until the photo has moved a tenth of one.
    const moved = Math.abs(shift[0] - drawn[0]) * canvas.width + Math.abs(shift[1] - drawn[1]) * canvas.height;
    if (intro < 1 || moved >= 0.1) render();
  }
  // The loop runs only while the band is on screen, once the photo is in.
  let visible = false;
  const start = () => { if (!raf && visible && loaded) { last = 0; raf = requestAnimationFrame(frame); } };
  const stop = () => { cancelAnimationFrame(raf); raf = 0; };

  uploadPhoto(gl, src).then(([w, h]) => {
    gl!.uniform2f(u.uImgSize, w, h);
    loaded = true;
    if (reduced) intro = 1;
    render();
    start();
  }, (e) => console.error("hero photo: failed to load", e));

  const resized = new ResizeObserver(() => { size(); render(); });
  resized.observe(canvas);
  const themed = new MutationObserver(() => { theme(); render(); });
  themed.observe(root, { attributes: true, attributeFilter: ["data-theme"] });
  const seen = new IntersectionObserver(([e]) => {
    visible = e.isIntersecting;
    if (visible) start();
    else stop();
  });
  if (!reduced) seen.observe(canvas);
  canvas.addEventListener("webglcontextlost", () => {
    stop();
    resized.disconnect();
    themed.disconnect();
    seen.disconnect();
    listening.abort();
    onLost?.();
  }, { once: true });

  size();
  theme();
  render();
  return true;
}
