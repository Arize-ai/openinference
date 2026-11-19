/**
 * The Python API reference: the data `scripts/python-api.py` extracts from
 * `python/` (see its docstring), and what the pages under `pages/python/`
 * need to render it: routes, links between objects, and docstrings as HTML.
 */
import { BRANCH, GITHUB_URL } from "../plugins/rehype-repo-links.mjs";
import {
  clip,
  escape,
  expressionHtml as html,
  fence,
  summaryOf,
  textOf,
  type DocstringItem,
  type DocstringSection,
  type Expression,
  type NamePart,
  type SearchEntry,
} from "./api";
import { pagerOf, type DocsNav } from "./docs";
import { base } from "./site";

// --- The data -----------------------------------------------------------------

export interface Parameter {
  name: string;
  /** `positional-only`, `positional or keyword`, `variadic positional`, `keyword-only`, `variadic keyword`. */
  kind: string;
  annotation: Expression | null;
  default: Expression | null;
}

interface Base {
  name: string;
  /** Where it's documented: the module it's public in, then its name. */
  path: string;
  docstring: DocstringSection[];
  source: string | null;
  labels?: string[];
}

export interface FunctionObject extends Base {
  kind: "function" | "method";
  parameters: Parameter[];
  returns: Expression | null;
}

export interface PropertyObject extends Base {
  kind: "property";
  annotation: Expression | null;
}

export interface AttributeObject extends Base {
  kind: "attribute";
  annotation: Expression | null;
  value: Expression | null;
}

export interface NestedClassObject extends Base {
  kind: "class";
  bases: Expression[];
}

export type ClassMember = FunctionObject | PropertyObject | AttributeObject | NestedClassObject;

export interface ClassObject extends Base {
  kind: "class" | "enum";
  bases: Expression[];
  /** The constructor's parameters, when it has one. */
  parameters: Parameter[] | null;
  members: ClassMember[];
  /** Where it's defined, when that isn't `path` (it was re-exported). */
  canonical?: string;
}

export interface ReexportObject {
  kind: "reexport";
  name: string;
  path: string;
  /** The object it re-exports, from a package this reference doesn't cover. */
  target: string;
  docstring?: undefined;
  source?: undefined;
}

export interface TypeAliasObject extends Base {
  kind: "type alias";
  value: Expression | null;
}

export type ApiObject =
  | ClassObject
  | FunctionObject
  | AttributeObject
  | TypeAliasObject
  | ReexportObject
  | (Base & { kind: "module" });

export interface ApiModule {
  path: string;
  docstring: DocstringSection[];
  source: string | null;
  members: ApiObject[];
}

export interface ApiPackage {
  /** The distribution name, e.g. `openinference-instrumentation-openai`. */
  name: string;
  group: "core" | "instrumentation";
  /** The top-level module, e.g. `openinference.instrumentation.openai`. */
  module: string;
  description: string;
  version: string | null;
  requiresPython: string | null;
  /** The libraries it instruments, as requirements (`openai >= 2.8.0`). */
  instruments: string[];
  /** The OpenTelemetry instrumentor it provides, if any. */
  instrumentor: { name: string; path: string; acceptsConfig: boolean } | null;
  pypi: string;
  source: string;
  readme: string | null;
  modules: ApiModule[];
}

interface ApiData {
  packages: ApiPackage[];
  /** Every object a page documents, by its paths (public and canonical). */
  index: Record<string, { package: string; anchor: string }>;
}

const GENERATED = "../generated/python-api.json";

function load(): ApiData {
  // A glob, so a missing file is an empty match with a clear error rather
  // than an unresolved import.
  const files = import.meta.glob<ApiData>("../generated/python-api.json", {
    eager: true,
    import: "default",
  });
  const data = files[GENERATED];
  if (!data) {
    throw new Error(
      "The Python API data is missing. Run `pnpm python-api` in site/ (it needs uv: https://docs.astral.sh/uv/) to write src/generated/python-api.json.",
    );
  }
  return data;
}

const data = load();

export const packages: ApiPackage[] = data.packages;

export const GROUPS = [
  { id: "core", label: "Core" },
  { id: "instrumentation", label: "Instrumentation" },
] as const;

// --- Routes -------------------------------------------------------------------

export const pythonRoot = `${base}python/`;

export const routeOfPackage = (name: string) => `${pythonRoot}${name}.html`;

const INSTRUMENTATION_PREFIX = "openinference-instrumentation-";

/** The package's name in the rail: instrumentors by what they instrument. */
export const shortName = (pkg: ApiPackage) =>
  pkg.group === "instrumentation" ? pkg.name.slice(INSTRUMENTATION_PREFIX.length) : pkg.name;

/** The page and anchor that document an object, by any of its paths. */
export function hrefOf(path: string): string | undefined {
  const entry = data.index[path];
  return entry ? `${routeOfPackage(entry.package)}#${entry.anchor}` : undefined;
}

/** The rail of packages, the pager and the source link, for a page. */
export function getPythonNav(current?: ApiPackage): DocsNav {
  const groups = [
    { items: [{ href: pythonRoot, label: "Overview", current: current === undefined }] },
    ...GROUPS.map((group) => ({
      label: group.label,
      items: packages
        .filter((pkg) => pkg.group === group.id)
        .map((pkg) => ({
          href: routeOfPackage(pkg.name),
          label: shortName(pkg),
          current: pkg === current,
        })),
    })),
  ];
  return {
    label: "Python API",
    groups,
    ...pagerOf(groups),
    source: current?.source ?? `${GITHUB_URL}/tree/${BRANCH}/python`,
    sourceLabel: current ? "View the package on GitHub" : "View the packages on GitHub",
  };
}

/** A name's link: to where this reference documents it. */
export const resolvePart = (part: NamePart) => (part.path ? hrefOf(part.path) : part.url);

/** An expression as HTML, its documented names linked. */
export const expressionHtml = (expression: Expression | null | undefined) =>
  html(expression, resolvePart);

// --- Text ---------------------------------------------------------------------

/** reStructuredText markup as plain text, for a one-line summary. */
const plain = (text: string) =>
  text
    .replace(/``([^`]+)``/g, "$1")
    .replace(/:[a-z]+:`~?([^`]+)`/g, (_, ref: string) => ref.split(".").pop() ?? ref)
    .replace(/`([^`]+)`/g, "$1");

/**
 * Docstrings are written for `help()` and editors: Google style, with a mix
 * of Markdown and reStructuredText. Turn the reStructuredText bits into
 * Markdown, outside fenced code:
 *
 * - ` ``code`` ` becomes `` `code` ``;
 * - a role, `` :func:`name` ``, becomes a link to `name` when this reference
 *   documents it (`scope` lists where to look it up), else code;
 * - a literal block (a paragraph ending `::`, then an indented block) becomes
 *   fenced Python;
 * - a section title underlined with dashes (NumPy style) becomes bold.
 */
export function toMarkdown(text: string, scope: string[] = []): string {
  return text
    .split(/(^```[\s\S]*?^```)/m)
    .map((chunk, index) => (index % 2 === 1 ? chunk : restructured(chunk, scope)))
    .join("");
}

function restructured(text: string, scope: string[]): string {
  return text
    // The block starts with an indented line, so `Note::` followed by an
    // ordinary paragraph isn't turned into an empty code block.
    .replace(/::\n\n((?:[ \t]+[^\n]*\n?)(?:(?:[ \t]+[^\n]*)?\n?)*)/g, (_, block: string) => {
      return `:\n\n${fence(dedent(block.replace(/\s+$/, "")), "python")}\n\n`;
    })
    // `[ \t]*`, not `\s*`: the blank line after the title must stay, or the
    // title runs into the paragraph under it.
    .replace(/^(\S[^\n]*)\n-{3,}[ \t]*$/gm, "**$1**")
    .replace(/:([a-z]+):`(~?)([^`]+)`/g, (_, _role, short: string, ref: string) => {
      const label = short ? (ref.split(".").pop() ?? ref) : ref;
      const href = resolve(ref, scope);
      return href ? `[\`${label}\`](${href})` : `\`${label}\``;
    })
    .replace(/``([^`]+)``/g, "`$1`");
}

/** Find a name in the reference, relative to the scopes it was written in. */
export function resolve(ref: string, scope: string[]): string | undefined {
  for (const prefix of ["", ...scope.map((path) => `${path}.`)]) {
    const href = hrefOf(`${prefix}${ref}`);
    if (href) return href;
  }
  return undefined;
}

const dedent = (text: string) => {
  const lines = text.split("\n");
  const indent = Math.min(
    ...lines.filter((line) => line.trim()).map((line) => line.match(/^\s*/)?.[0].length ?? 0),
  );
  return lines.map((line) => line.slice(indent)).join("\n");
};

// --- Signatures ---------------------------------------------------------------

/** The longest one-line signature; longer ones put a parameter per line. */
export const SIGNATURE_WIDTH = 72;

/** The parameter list with `/` and `*` markers where Python needs them. */
export function withMarkers(parameters: Parameter[]): (Parameter | "/" | "*")[] {
  const result: (Parameter | "/" | "*")[] = [];
  parameters.forEach((parameter, index) => {
    const previous = parameters[index - 1];
    if (previous?.kind === "positional-only" && parameter.kind !== "positional-only") {
      result.push("/");
    }
    if (
      parameter.kind === "keyword-only" &&
      previous?.kind !== "keyword-only" &&
      previous?.kind !== "variadic positional"
    ) {
      result.push("*");
    }
    result.push(parameter);
  });
  const last = parameters[parameters.length - 1];
  if (last?.kind === "positional-only") result.push("/");
  return result;
}

export function parameterText(parameter: Parameter | "/" | "*"): string {
  if (typeof parameter === "string") return parameter;
  const star =
    parameter.kind === "variadic positional" ? "*" : parameter.kind === "variadic keyword" ? "**" : "";
  const annotation = parameter.annotation ? `: ${textOf(parameter.annotation)}` : "";
  const assign = parameter.annotation ? " = " : "=";
  const fallback = parameter.default ? `${assign}${textOf(parameter.default)}` : "";
  return `${star}${parameter.name}${annotation}${fallback}`;
}

// --- Search -------------------------------------------------------------------

/** Every documented object, for the overview's search. */
export function searchEntries(): SearchEntry[] {
  const entries: SearchEntry[] = [];
  for (const pkg of packages) {
    const add = (object: { name: string; path: string; kind: string; docstring?: DocstringSection[] }) =>
      entries.push({
        name: object.name,
        path: object.path,
        kind: object.kind,
        href: `${routeOfPackage(pkg.name)}#${object.path}`,
        package: shortName(pkg),
        summary: clip(summaryOf(object, plain), 140),
      });
    for (const module of pkg.modules) {
      for (const member of module.members) {
        add(member);
        if (member.kind === "class" || member.kind === "enum") member.members.forEach(add);
      }
    }
  }
  return entries;
}

// --- HTML ---------------------------------------------------------------------

function parameterHtml(parameter: Parameter | "/" | "*"): string {
  if (typeof parameter === "string") return parameter;
  const star =
    parameter.kind === "variadic positional" ? "*" : parameter.kind === "variadic keyword" ? "**" : "";
  const name = `<span class="sig-param">${star}${escape(parameter.name)}</span>`;
  const annotation = parameter.annotation ? `: ${expressionHtml(parameter.annotation)}` : "";
  const assign = parameter.annotation ? " = " : "=";
  const fallback = parameter.default
    ? `${assign}<span class="sig-default">${expressionHtml(parameter.default)}</span>`
    : "";
  return `${name}${annotation}${fallback}`;
}

/**
 * A signature as HTML for a `<pre>`: `def name(a: int, b=1) -> str`, or, when
 * that would be wider than SIGNATURE_WIDTH, one parameter per line, as Black
 * formats it.
 */
export function signatureHtml({
  keyword,
  name,
  parameters,
  returns,
  bases,
}: {
  keyword: string;
  name: string;
  parameters?: Parameter[] | null;
  returns?: Expression | null;
  bases?: Expression[];
}): string {
  const head = `${keyword ? `<span class="sig-keyword">${keyword}</span> ` : ""}<span class="sig-name">${escape(name)}</span>`;
  const headText = `${keyword ? `${keyword} ` : ""}${name}`;
  // A class without a constructor of its own shows its bases instead.
  if (!parameters) {
    if (!bases?.length) return head;
    return `${head}(${bases.map(expressionHtml).join(", ")})`;
  }
  const items = withMarkers(parameters);
  const tail = returns ? ` -&gt; ${expressionHtml(returns)}` : "";
  const tailText = returns ? ` -> ${textOf(returns)}` : "";
  const oneLine = `${headText}(${items.map(parameterText).join(", ")})${tailText}`;
  if (oneLine.length <= SIGNATURE_WIDTH || items.length === 0) {
    return `${head}(${items.map(parameterHtml).join(", ")})${tail}`;
  }
  const lines = items.map((item) => `    ${parameterHtml(item)},`).join("\n");
  return `${head}(\n${lines}\n)${tail}`;
}

/** `name: annotation = value`, for an attribute or a property. */
export function attributeHtml({
  name,
  annotation,
  value,
}: {
  name: string;
  annotation?: Expression | null;
  value?: Expression | null;
}): string {
  const typed = annotation?.length ? `: ${expressionHtml(annotation)}` : "";
  const assigned = value?.length ? ` = <span class="sig-default">${expressionHtml(value)}</span>` : "";
  return `<span class="sig-name">${escape(name)}</span>${typed}${assigned}`;
}

// --- Instrumentors ------------------------------------------------------------

const BASE_INSTRUMENTOR =
  "https://opentelemetry-python-contrib.readthedocs.io/en/latest/instrumentation/base/instrumentor.html";

/**
 * `instrument()` and `uninstrument()`, which every instrumentor inherits from
 * OpenTelemetry's `BaseInstrumentor`, as methods of `path` (the class).
 */
export function instrumentorMethods(pkg: ApiPackage, path: string): FunctionObject[] {
  const library = pkg.instruments.map((requirement) => `\`${requirement.split(/[\s<>=!~;[]/)[0]}\``);
  const what = library.length ? library.join(", ") : "the library";
  const keyword = (name: string, annotation: Expression, fallback: string): Parameter => ({
    name,
    kind: "keyword-only",
    annotation,
    default: [fallback],
  });
  const parameters: Parameter[] = [
    keyword("tracer_provider", ["Optional[", { text: "TracerProvider", path: "opentelemetry.trace.TracerProvider" }, "]"], "None"),
    ...(pkg.instrumentor?.acceptsConfig
      ? [keyword("config", ["Optional[", { text: "TraceConfig", path: "openinference.instrumentation.TraceConfig" }, "]"], "None")]
      : []),
    keyword("skip_dep_check", ["bool"], "False"),
  ];
  const items: DocstringItem[] = [
    {
      name: "tracer_provider",
      annotation: null,
      default: null,
      description:
        "The tracer provider the spans are made with. Defaults to the global one, `opentelemetry.trace.get_tracer_provider()`.",
    },
    ...(pkg.instrumentor?.acceptsConfig
      ? [
          {
            name: "config",
            annotation: null,
            default: null,
            description: `What the spans record: hide inputs, outputs or messages, or limit image sizes. Defaults to \`TraceConfig()\`, which reads the \`OPENINFERENCE_*\` environment variables. See [\`TraceConfig\`](${hrefOf("openinference.instrumentation.TraceConfig") ?? "#"}).`,
          },
        ]
      : []),
    {
      name: "skip_dep_check",
      annotation: null,
      default: null,
      description: `Instrument even if the installed version of ${what} is outside the supported range.`,
    },
  ];
  const inherited = `Inherited from OpenTelemetry's [\`BaseInstrumentor\`](${BASE_INSTRUMENTOR}).`;
  return [
    {
      kind: "method",
      name: "instrument",
      path: `${path}.instrument`,
      labels: [],
      source: null,
      parameters,
      returns: ["None"],
      docstring: [
        {
          kind: "text",
          value: `Start tracing ${what}: patch it so its calls make OpenInference spans. Calling it again does nothing. ${inherited}`,
        },
        { kind: "parameters", title: null, items },
      ],
    },
    {
      kind: "method",
      name: "uninstrument",
      path: `${path}.uninstrument`,
      labels: [],
      source: null,
      parameters: [],
      returns: ["None"],
      docstring: [{ kind: "text", value: `Stop tracing: undo the patches \`instrument()\` made. ${inherited}` }],
    },
  ];
}
