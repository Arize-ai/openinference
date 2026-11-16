/**
 * The trace hero's replay. One requestAnimationFrame loop advances the trace
 * time `t`; `paint` sets every part of the hero from `t` (and what is
 * hovered), using the pure functions in model.ts. No part has a timer of
 * its own. The server rendered the finished run; this takes it over.
 */
import { GLYPHS } from "./glyphs";
import {
  ANSWER_SPAN,
  answerAt,
  attributesAt,
  barWidth,
  codeAt,
  detailCandidate,
  formatClock,
  formatDuration,
  PRE_ROLL,
  SETUP_LINES,
  progressOf,
  SPANS,
  stateOf,
  statusOf,
  STEP_SPANS,
  STEPS,
  T,
  xAt,
} from "./model";

/** Trace time runs at 0.75× real time, so one run takes about 12s. */
const RATE = 0.75;
/** The finished trace holds this long before the run starts again. */
const HOLD = 5000;
/** The detail keeps a span at least this long (real ms) before following the run to the next. */
const DWELL = 2000;
/** The code's program counter holds a line at least this long (real ms). */
const LINE_DWELL = 260;
/** The loop pauses when less than this much of the hero is on screen. */
const VISIBLE_RATIO = 0.2;

const pct = (value: number) => `${Math.round(value * 100) / 100}%`;

/** Writes only what changed, so a frame touches as little of the DOM as it can. */
const setText = (element: Element, text: string) => {
  if (element.textContent !== text) element.textContent = text;
};
const setData = (element: HTMLElement, key: string, value: string | undefined) => {
  if (element.dataset[key] === value) return;
  if (value === undefined) delete element.dataset[key];
  else element.dataset[key] = value;
};
const setStyle = (element: HTMLElement, property: string, value: string) => {
  if (element.style.getPropertyValue(property) !== value) element.style.setProperty(property, value);
};

export function mount(root: HTMLElement) {
  const one = <E extends HTMLElement = HTMLElement>(selector: string) => root.querySelector<E>(selector)!;
  const all = <E extends HTMLElement = HTMLElement>(selector: string) => [...root.querySelectorAll<E>(selector)];

  const viewButtons = all<HTMLButtonElement>("[data-view]");
  const chat = one('[data-pane="chat"]');
  const code = one('[data-pane="code"]');
  const question = one(".question");
  const agent = one("[data-agent]");
  const summary = one<HTMLButtonElement>("[data-summary]");
  const steps = all("[data-step]").map((step) => ({
    step,
    text: step.querySelector(".step-text")!,
    meta: step.querySelector("[data-step-meta]")!,
  }));
  const answer = one("[data-answer]");
  const answerText = one("[data-answer-text]");
  const summaryText = one("[data-summary-text]");
  const codeBody = one("[data-code]");
  const lines = all("[data-line]");
  const waterfall = one("[data-waterfall]");
  const rows = all("[data-row]").map((row) => ({
    row,
    duration: row.querySelector<HTMLElement>("[data-duration]")!,
    /** Marks that appear partway through the run: hierarchy lines, the retry leader. */
    reveals: [...row.querySelectorAll<HTMLElement>("[data-after]")].map((mark) => ({ mark, after: Number(mark.dataset.after) })),
  }));
  const playhead = one("[data-playhead]");
  const chip = one("[data-chip]");
  const detail = {
    root: one("[data-detail]"),
    head: one(".detail-head"),
    glyph: one("[data-detail-glyph]"),
    name: one("[data-detail-name]"),
    status: one("[data-detail-status]"),
    duration: one("[data-detail-duration]"),
    list: one("[data-attributes]"),
    attributes: all("[data-attribute]").map((row) => ({ row, key: row.querySelector("dt")!, value: row.querySelector("dd")! })),
  };
  const play = one<HTMLButtonElement>("[data-play]");
  const scrubber = one<HTMLInputElement>("[data-scrubber]");
  const time = one("[data-time]");

  const reducedMotion = matchMedia("(prefers-reduced-motion: reduce)");

  // Reduced motion: the finished trace, still, until someone presses play.
  let t = reducedMotion.matches ? T : PRE_ROLL;
  let playing = !reducedMotion.matches;
  let onScreen = false;
  let hold = 0;
  /** The hovered link group, and the span that stands for it. */
  let group: string | null = null;
  let hovered: number | null = null;
  /** The step list's disclosure: collapsed under the summary until someone opens it. */
  let stepsOpen = false;
  /** The code line the lens last scrolled to. */
  let followed = 0;
  /** The span in the detail, and when (real time) it got there. */
  let shown = -2;
  let shownAt = 0;

  function paintDetail(now: number, immediate: boolean) {
    const candidate = detailCandidate(t, hovered) ?? shown;
    // Follow the run calmly: a span stays up long enough to read.
    if (candidate !== shown && (immediate || hovered !== null || now - shownAt >= DWELL)) {
      shown = candidate;
      shownAt = now;
      if (shown >= 0) {
        const span = SPANS[shown]!;
        detail.glyph.innerHTML = `<svg viewBox="0 0 16 16" aria-hidden="true">${GLYPHS[span.kind]}</svg>`;
        setText(detail.name, span.name);
        setData(detail.head, "kind", span.kind.toLowerCase());
        if (!immediate && !reducedMotion.matches) {
          detail.root.animate([{ opacity: 0.55 }, { opacity: 1 }], { duration: 320, easing: "ease-out" });
        }
      }
    }
    detail.root.style.visibility = shown < 0 ? "hidden" : "";
    if (shown < 0) return;
    const span = SPANS[shown]!;
    const done = t >= span.end;
    // Status is set when a span ends; until then there is none to show.
    const status = done ? statusOf(span) : "";
    setText(detail.status, status);
    setData(detail.status, "status", status || undefined);
    setData(detail.head, "error", done && span.error ? "" : undefined);
    setText(detail.duration, t >= span.start ? formatDuration(Math.min(t, span.end) - span.start) : "");
    const attributes = attributesAt(span, t);
    detail.attributes.forEach(({ row, key, value }, index) => {
      const attribute = attributes[index];
      row.hidden = !attribute;
      if (!attribute) return;
      setText(key, attribute.key);
      setText(value, attribute.value);
      row.classList.toggle("error", attribute.error);
    });
  }

  // --- Code: a program counter that glides, and a view that rarely moves ---
  const counter = one("[data-counter]");
  /** The line the counter is on, and when (real time) it got there. */
  let countedLine = 0;
  let countedAt = 0;
  let scrollAnimation = 0;

  /** Glides the code view to `top` with one eased motion, replacing any glide in progress. */
  function glideTo(top: number, instant: boolean) {
    cancelAnimationFrame(scrollAnimation);
    const from = codeBody.scrollTop;
    const to = Math.max(0, Math.min(top, codeBody.scrollHeight - codeBody.clientHeight));
    if (instant || Math.abs(to - from) < 1) {
      codeBody.scrollTop = to;
      return;
    }
    const start = performance.now();
    const duration = 700;
    const ease = (x: number) => (x < 0.5 ? 4 * x * x * x : 1 - (-2 * x + 2) ** 3 / 2);
    const step = (now: number) => {
      const progress = Math.min(1, (now - start) / duration);
      codeBody.scrollTop = from + (to - from) * ease(progress);
      if (progress < 1) scrollAnimation = requestAnimationFrame(step);
    };
    scrollAnimation = requestAnimationFrame(step);
  }

  /** Sets the counter over the line it is on (lines move when the lens's width rewraps them). */
  function placeCounter() {
    if (!countedLine || code.hidden) return;
    const line = lines[countedLine - 1]!;
    setStyle(counter, "transform", `translateY(${line.offsetTop}px)`);
    setStyle(counter, "height", `${line.offsetHeight}px`);
  }
  // While paused nothing repaints, so follow the lines when they reflow.
  new ResizeObserver(placeCounter).observe(codeBody);

  /** Where the code rests: on the agent's first function, so the agent leads and every line that runs shows. */
  const restTop = () => lines[SETUP_LINES + 1]!.offsetTop - 20;

  function paintCode(now: number, immediate: boolean) {
    const codeState = codeAt(t);
    const { running } = codeState;
    // Between its steps only the agent itself is open: the counter stays on
    // the last line that ran rather than jumping back to the agent's def.
    // A scrub lands where it lands: the line it stays on may be from
    // anywhere in the run.
    const rootLine = SPANS[0]!.line;
    const active = !immediate && codeState.active === rootLine && countedLine && t < T ? countedLine : codeState.active;
    // A line holds the counter a moment, so a span of a few milliseconds
    // doesn't flicker it; scrubbing moves it at once.
    if (active !== countedLine && (immediate || now - countedAt >= LINE_DWELL || !countedLine || !active)) {
      countedLine = active;
      countedAt = now;
    }
    const shown = countedLine;
    codeBody.classList.toggle("is-running", shown > 0);
    lines.forEach((line, index) => {
      line.classList.toggle("act", index + 1 === shown);
      line.classList.toggle("run", running.has(index + 1));
    });
    counter.hidden = !shown;
    if (code.hidden) return;
    const instant = immediate || reducedMotion.matches;
    if (!shown) {
      if (followed !== -1) {
        followed = -1;
        glideTo(restTop(), instant);
      }
      return;
    }
    const line = lines[shown - 1]!;
    placeCounter();
    if (shown === followed) return;
    followed = shown;
    const view = codeBody.clientHeight;
    const top = codeBody.scrollTop;
    // Scroll as little as it can: bring the line just inside the edge it
    // left, with two lines of context, never back into the setup.
    const margin = 2 * line.offsetHeight;
    if (line.offsetTop < top + margin) glideTo(Math.max(restTop(), line.offsetTop - margin), instant);
    else if (line.offsetTop + line.offsetHeight > top + view - margin) glideTo(line.offsetTop + line.offsetHeight - view + margin, instant);
  }

  function paint(now = performance.now(), immediate = false) {
    const done = t >= T;

    // Conversation
    const answerSpan = SPANS[ANSWER_SPAN]!;
    const answering = t >= answerSpan.start;
    agent.hidden = t < 0;
    question.classList.toggle("lit", group === "root");
    let live = -1;
    steps.forEach(({ step, text, meta }, index) => {
      const extent = STEP_SPANS[index]!;
      const state = t < extent.start ? "pending" : t < extent.end ? "open" : "done";
      if (state === "open") live = index;
      setData(step, "state", state);
      setText(meta, state === "done" ? [STEPS[index]!.note, formatDuration(extent.end - extent.start)].filter(Boolean).join(" · ") : "");
      // A step reads as it runs ("Searching…"), then as done ("Searched").
      setText(text, state === "open" ? STEPS[index]!.live : STEPS[index]!.text);
      step.classList.toggle("lit", group === STEPS[index]!.group);
    });
    setData(chat, "steps", stepsOpen ? "open" : "closed");
    // While the agent works, one glimmering line says what it is doing, with the
    // steps collapsed under it; once it answers, it says how long it took.
    summary.hidden = t < 0;
    summary.classList.toggle("working", !answering);
    setText(summaryText, answering ? `Worked for ${(answerSpan.start / 1000).toFixed(1)}s` : live >= 0 ? STEPS[live]!.live : "Working…");
    setData(summary, "group", !answering && live >= 0 ? STEPS[live]!.group : undefined);
    summary.classList.toggle("lit", group !== null && group === summary.dataset.group);
    summary.setAttribute("aria-expanded", String(stepsOpen));
    answer.hidden = !answering;
    const progress = progressOf(answerSpan, t);
    setText(answerText, answerAt(progress));
    answer.classList.toggle("lit", group === "answer");

    // Code: the line that is running, and the lines that called it
    paintCode(now, immediate);

    // Trace
    waterfall.classList.toggle("dim", group !== null);
    rows.forEach(({ row, duration, reveals }, index) => {
      const span = SPANS[index]!;
      const state = stateOf(span, t);
      setData(row, "state", state);
      setStyle(row, "--w", pct(barWidth(span, t)));
      setText(duration, state === "done" ? formatDuration(span.end - span.start) : "");
      for (const { mark, after } of reveals) mark.classList.toggle("off", t < after);
      row.classList.toggle("on", group !== null && (hovered === index || (group !== "root" && span.group === group)));
    });
    playhead.hidden = t < 0 || done;
    chip.hidden = playhead.hidden;
    if (!playhead.hidden) {
      setStyle(playhead, "left", pct(xAt(t)));
      setStyle(chip, "left", pct(xAt(t)));
      setText(chip, formatClock(t));
    }

    paintDetail(now, immediate);

    // Controls
    const value = String(Math.round(t));
    if (scrubber.value !== value) scrubber.value = value;
    setStyle(scrubber, "--p", pct(((t - PRE_ROLL) / (T - PRE_ROLL)) * 100));
    setText(time, `${formatClock(t)} / ${formatClock(T)}`);
    setData(play, "playing", playing ? "" : undefined);
    play.setAttribute("aria-label", playing ? "Pause" : done ? "Replay" : "Play");
  }

  // --- The loop -----------------------------------------------------------
  let raf = 0;
  let previous = 0;
  let pending = 0;

  function frame(now: number) {
    raf = requestAnimationFrame(frame);
    // A backgrounded tab mustn't jump: cap each step at 100ms.
    pending += Math.min(100, now - previous);
    previous = now;
    if (pending < 16) return;
    const step = pending;
    pending = 0;
    if (t < T) {
      t = Math.min(T, t + step * RATE);
    } else if ((hold += step) > HOLD) {
      restart();
    }
    paint(now);
  }

  /** Runs the loop only while it's playing, on screen and the tab is visible. */
  function sync() {
    const run = playing && onScreen && !document.hidden;
    if (run && !raf) {
      previous = performance.now();
      pending = 0;
      raf = requestAnimationFrame(frame);
    } else if (!run && raf) {
      cancelAnimationFrame(raf);
      raf = 0;
    }
  }

  function restart() {
    t = PRE_ROLL;
    hold = 0;
    stepsOpen = false;
  }

  new IntersectionObserver(
    ([entry]) => {
      onScreen = !!entry && entry.intersectionRatio >= VISIBLE_RATIO;
      sync();
    },
    { threshold: [0, VISIBLE_RATIO] },
  ).observe(root);
  document.addEventListener("visibilitychange", sync);
  reducedMotion.addEventListener("change", () => {
    if (!reducedMotion.matches) return;
    t = T;
    playing = false;
    sync();
    paint(performance.now(), true);
  });

  // --- Controls -----------------------------------------------------------
  for (const button of viewButtons) {
    button.addEventListener("click", () => {
      const view = button.dataset.view;
      for (const other of viewButtons) other.setAttribute("aria-pressed", String(other === button));
      chat.hidden = view !== "chat";
      code.hidden = view !== "code";
      followed = 0;
      paint(performance.now(), true);
    });
  }
  play.addEventListener("click", () => {
    playing = !playing;
    if (playing && t >= T) restart();
    hold = 0;
    sync();
    paint();
  });
  scrubber.addEventListener("input", () => {
    t = Number(scrubber.value);
    playing = false;
    hold = 0;
    sync();
    paint(performance.now(), true);
  });
  summary.addEventListener("click", () => {
    stepsOpen = !stepsOpen;
    paint();
  });

  // --- Linking ------------------------------------------------------------
  // The question, each step, the answer and each trace row name a link group
  // (and the span standing for it); hovering one lights the others.
  for (const element of [...all("[data-group]"), summary]) {
    element.addEventListener("pointerenter", () => {
      group = element.dataset.group ?? null;
      hovered = element.dataset.span === undefined ? null : Number(element.dataset.span);
      paint();
    });
    element.addEventListener("pointerleave", () => {
      group = null;
      hovered = null;
      paint();
    });
  }

  paint(performance.now(), true);
}
