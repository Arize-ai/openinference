/**
 * The JavaScript API reference: the data `scripts/js-api.mjs` extracts from
 * `js/packages/` with TypeDoc (see its header), and what the pages under
 * `pages/js/` need to render it: routes, links between objects, and
 * TypeScript signatures as HTML.
 */
import { BRANCH, GITHUB_URL } from "../plugins/rehype-repo-links.mjs";
import {
  clip,
  escape,
  expressionHtml as html,
  summaryOf,
  textOf,
  type DocstringSection,
  type Expression,
  type NamePart,
  type SearchEntry,
} from "./api";
import { pagerOf, type DocsNav } from "./docs";
import { base } from "./site";

// --- The data -----------------------------------------------------------------

export interface TypeParameter {
  name: string;
  constraint: Expression | null;
  default: Expression | null;
}

export interface Parameter {
  name: string;
  type: Expression;
  /** An inline options object's properties, to set one per line. */
  fields: { name: string; optional: boolean; type: Expression }[] | null;
  optional: boolean;
  rest: boolean;
}

export interface Signature {
  typeParameters: TypeParameter[];
  parameters: Parameter[];
  returns: Expression | null;
  docstring: DocstringSection[];
}

interface Base {
  name: string;
  /** Its id on the package's page: `Name`, `utils/Name`, `Class.member`. */
  anchor: string;
  source: string | null;
  docstring: DocstringSection[];
  flags: { static?: true; readonly?: true; optional?: true; abstract?: true; const?: true };
}

export interface FunctionObject extends Base {
  kind: "function";
  signatures: Signature[];
}

export interface ClassObject extends Base {
  kind: "class" | "interface";
  typeParameters: TypeParameter[];
  extends: Expression[];
  implements: Expression[];
  members: Member[];
}

export interface EnumObject extends Base {
  kind: "enum";
  members: PropertyMember[];
}

export interface TypeObject extends Base {
  kind: "type";
  typeParameters: TypeParameter[];
  /** The aliased type; null when it's an object type, whose members are listed. */
  type: Expression | null;
  members: Member[];
}

export interface VariableObject extends Base {
  kind: "variable";
  /** Its type; null when it's an object, whose members are listed. */
  type: Expression | null;
  /** Its value, when it's a short literal. */
  value: string | null;
  members: Member[];
}

export interface NamespaceObject extends Base {
  kind: "namespace";
  members: ApiObject[];
}

export interface ReexportObject {
  kind: "reexport";
  name: string;
  anchor: string;
  source: string | null;
  docstring?: undefined;
  /** What it re-exports. */
  target: NamePart | string;
}

export type ApiObject =
  | FunctionObject
  | ClassObject
  | EnumObject
  | TypeObject
  | VariableObject
  | NamespaceObject
  | ReexportObject;

interface MemberBase extends Base {
  /** The class it's inherited from, when it is. */
  inherited?: string | null;
}

export interface PropertyMember extends MemberBase {
  kind: "property";
  type: Expression;
  value?: string | null;
}

export interface AccessorMember extends MemberBase {
  kind: "accessor";
  type: Expression;
  readonly: boolean;
}

export interface MethodMember extends MemberBase {
  kind: "method" | "constructor";
  signatures: Signature[];
}

export type Member = PropertyMember | AccessorMember | MethodMember;

export interface ApiModule {
  /** What to import it from: `@arizeai/openinference-vercel/utils`. */
  specifier: string;
  /** Its entry in the package's `exports` map: `.`, `./utils`. */
  subpath: string;
  source: string;
  docstring: DocstringSection[];
  members: ApiObject[];
}

export interface ApiPackage {
  /** The npm name, e.g. `@arizeai/openinference-core`. */
  name: string;
  /** The name without its scope, for the page's URL. */
  slug: string;
  group: "core" | "instrumentation" | "integration";
  description: string;
  version: string | null;
  /** Its `engines.node` range. */
  node: string | null;
  /** The libraries it instruments, with the versions it supports. */
  instruments: string[];
  peerDependencies: Record<string, string>;
  /** Its OpenTelemetry instrumentation class, if it has one. */
  instrumentation: { name: string; anchor: string; manual: boolean } | null;
  npm: string;
  source: string;
  readme: string | null;
  modules: ApiModule[];
}

interface ApiData {
  packages: ApiPackage[];
  /** `package:qualifiedName` -> where it's documented. */
  index: Record<string, { package: string; anchor: string }>;
}

const GENERATED = "../generated/js-api.json";

function load(): ApiData {
  // A glob, so a missing file is an empty match with a clear error rather
  // than an unresolved import.
  const files = import.meta.glob<ApiData>("../generated/js-api.json", {
    eager: true,
    import: "default",
  });
  const data = files[GENERATED];
  if (!data) {
    throw new Error(
      "The JavaScript API data is missing. Run `pnpm js-api` in site/ (after `pnpm install`, `pnpm run -r prebuild` and `pnpm run -r build` in js/) to write src/generated/js-api.json.",
    );
  }
  return data;
}

const data = load();

export const packages: ApiPackage[] = data.packages;

export const GROUPS = [
  { id: "core", label: "Core" },
  { id: "instrumentation", label: "Instrumentation" },
  { id: "integration", label: "Integrations" },
] as const;

// --- Routes -------------------------------------------------------------------

export const jsRoot = `${base}js/`;

const slugOf = (name: string) => packages.find((pkg) => pkg.name === name)?.slug;

export const routeOfPackage = (pkg: ApiPackage) => `${jsRoot}${pkg.slug}.html`;

/** The package's name in the rail: instrumentations by what they instrument. */
export const shortName = (pkg: ApiPackage) =>
  pkg.group === "instrumentation"
    ? pkg.slug.replace(/^openinference-instrumentation-/, "")
    : pkg.slug;

/** A name's link: its page and anchor here, or its external docs. */
export function resolvePart(part: NamePart): string | undefined {
  if (part.package && part.anchor) {
    const slug = slugOf(part.package);
    if (slug) return `${jsRoot}${slug}.html#${part.anchor}`;
  }
  return part.url;
}

/** An expression as HTML, its documented names linked. */
export const expressionHtml = (expression: Expression | null | undefined) =>
  html(expression, resolvePart);

/**
 * The extractor writes `{@link Name}` as a link to `oi-ref:package:Name`;
 * point it at the page that documents `Name`, or, if none does, drop the
 * link and keep its text.
 */
export function linkRefs(markdown: string): string {
  return markdown.replace(/\[([^\]]*)\]\(oi-ref:([^)]+)\)/g, (_, label: string, ref: string) => {
    const target = data.index[decodeURIComponent(ref)];
    const href = target ? resolvePart({ text: label, ...target }) : undefined;
    return href ? `[${label}](${href})` : label;
  });
}

/** The rail of packages, the pager and the source link, for a page. */
export function getJsNav(current?: ApiPackage): DocsNav {
  const groups = [
    { items: [{ href: jsRoot, label: "Overview", current: current === undefined }] },
    ...GROUPS.map((group) => ({
      label: group.label,
      items: packages
        .filter((pkg) => pkg.group === group.id)
        .map((pkg) => ({ href: routeOfPackage(pkg), label: shortName(pkg), current: pkg === current })),
    })),
  ];
  return {
    label: "JavaScript API",
    groups,
    ...pagerOf(groups),
    source: current?.source ?? `${GITHUB_URL}/tree/${BRANCH}/js/packages`,
    sourceLabel: current ? "View the package on GitHub" : "View the packages on GitHub",
  };
}

// --- Signatures ---------------------------------------------------------------

/** The longest one-line signature; longer ones put a parameter per line. */
export const SIGNATURE_WIDTH = 80;

const keyword = (word: string) => `<span class="sig-keyword">${word}</span> `;
const nameHtml = (name: string) => `<span class="sig-name">${escape(name)}</span>`;

function typeParametersHtml(typeParameters: TypeParameter[]): { html: string; text: string } {
  if (!typeParameters.length) return { html: "", text: "" };
  const items = typeParameters.map((parameter) => {
    const constraint = parameter.constraint ? ` extends ${expressionHtml(parameter.constraint)}` : "";
    const fallback = parameter.default ? ` = ${expressionHtml(parameter.default)}` : "";
    const constraintText = parameter.constraint ? ` extends ${textOf(parameter.constraint)}` : "";
    const fallbackText = parameter.default ? ` = ${textOf(parameter.default)}` : "";
    return {
      html: `${escape(parameter.name)}${constraint}${fallback}`,
      text: `${parameter.name}${constraintText}${fallbackText}`,
    };
  });
  return {
    html: `&lt;${items.map((item) => item.html).join(", ")}&gt;`,
    text: `<${items.map((item) => item.text).join(", ")}>`,
  };
}

const parameterParts = (parameter: Parameter) => {
  const name = `${parameter.rest ? "..." : ""}${parameter.name}${parameter.optional ? "?" : ""}`;
  return {
    html: `<span class="sig-param">${escape(name)}</span>: ${expressionHtml(parameter.type)}`,
    text: `${name}: ${textOf(parameter.type)}`,
    // Too wide for its line: an options object breaks into a property per line.
    expanded: parameter.fields
      ? `<span class="sig-param">${escape(name)}</span>: {\n${parameter.fields
          .map((field) => `    ${escape(field.name)}${field.optional ? "?" : ""}: ${expressionHtml(field.type)};`)
          .join("\n")}\n  }`
      : undefined,
  };
};

/**
 * A call signature as HTML for a `<pre>`: `function name<T>(a: A, b?: B): R`,
 * or, when that's wider than SIGNATURE_WIDTH, one parameter per line, as
 * Prettier formats it.
 */
export function signatureHtml({
  prefix,
  name,
  signature,
}: {
  /** `function`, `new`, `static`... */
  prefix: string;
  name: string;
  signature: Signature;
}): string {
  const generics = typeParametersHtml(signature.typeParameters);
  const head = `${prefix ? keyword(prefix) : ""}${nameHtml(name)}${generics.html}`;
  const headText = `${prefix ? `${prefix} ` : ""}${name}${generics.text}`;
  const parameters = signature.parameters.map(parameterParts);
  const returns = signature.returns?.length ? `: ${expressionHtml(signature.returns)}` : "";
  const returnsText = signature.returns?.length ? `: ${textOf(signature.returns)}` : "";
  const oneLine = `${headText}(${parameters.map((p) => p.text).join(", ")})${returnsText}`;
  if (oneLine.length <= SIGNATURE_WIDTH || parameters.length === 0) {
    return `${head}(${parameters.map((p) => p.html).join(", ")})${returns}`;
  }
  const lines = parameters.map((p) =>
    p.expanded && p.text.length + 2 > SIGNATURE_WIDTH ? `  ${p.expanded},` : `  ${p.html},`,
  );
  return `${head}(\n${lines.join("\n")}\n)${returns}`;
}

/** `class Name<T> extends Base implements I`, `interface Name extends A, B`. */
export function declarationHtml(object: ClassObject | TypeObject | EnumObject | NamespaceObject): string {
  const word =
    object.kind === "class"
      ? `${object.flags.abstract ? "abstract " : ""}class`
      : object.kind === "type"
        ? "type"
        : object.kind;
  const generics = "typeParameters" in object ? typeParametersHtml(object.typeParameters).html : "";
  let tail = "";
  if (object.kind === "class" || object.kind === "interface") {
    if (object.extends.length) tail += ` extends ${object.extends.map(expressionHtml).join(", ")}`;
    if (object.implements.length) {
      tail += ` implements ${object.implements.map(expressionHtml).join(", ")}`;
    }
  }
  if (object.kind === "type" && object.type) tail = ` = ${expressionHtml(object.type)}`;
  return `${keyword(word)}${nameHtml(object.name)}${generics}${tail}`;
}

/** `const name: Type = value`, `readonly name?: Type`, `get name(): Type`. */
export function valueHtml(
  object: VariableObject | PropertyMember | AccessorMember,
  { prefix = "" }: { prefix?: string } = {},
): string {
  const words: string[] = [];
  if (prefix) words.push(prefix);
  if (object.flags.static) words.push("static");
  if (object.kind === "accessor") words.push(object.readonly ? "get" : "get set");
  else if (object.flags.readonly && object.kind === "property") words.push("readonly");
  const head = `${words.map(keyword).join("")}${nameHtml(object.name)}${object.flags.optional ? "?" : ""}`;
  const type = object.type?.length ? `: ${expressionHtml(object.type)}` : "";
  const value = "value" in object && object.value ? ` = <span class="sig-default">${escape(object.value)}</span>` : "";
  return `${head}${object.kind === "accessor" ? "()" : ""}${type}${value}`;
}

/** The type and value cell of a constant or property in a member table. */
export function cellHtml(object: VariableObject | PropertyMember | AccessorMember): string {
  const type = object.type?.length ? expressionHtml(object.type) : "";
  const value = "value" in object && object.value ? escape(object.value) : "";
  // A literal's type is its value; say it once.
  if (value && textOf(object.type) === object.value) return value;
  return [type, value].filter(Boolean).join(" = ");
}

/**
 * A constant: a variable whose value is a short literal and that has no
 * members. A module's constants are one table, not an object each.
 */
export const isConstant = (object: ApiObject): object is VariableObject =>
  object.kind === "variable" && object.members.length === 0 && object.value !== null;

// --- Search -------------------------------------------------------------------

/** Every documented object, for the overview's search. */
export function searchEntries(): SearchEntry[] {
  const entries: SearchEntry[] = [];
  for (const pkg of packages) {
    const add = (
      object: { name: string; anchor: string; kind: string; docstring?: DocstringSection[] },
      module: ApiModule,
      parent?: string,
    ) =>
      entries.push({
        name: object.name,
        path: `${module.specifier} · ${parent ? `${parent}.` : ""}${object.name}`,
        kind: object.kind,
        href: `${routeOfPackage(pkg)}#${object.anchor}`,
        package: shortName(pkg),
        summary: clip(summaryOf(object, linkRefs), 140),
      });
    for (const module of pkg.modules) {
      for (const member of module.members) {
        add(member, module);
        const summary = "signatures" in member ? member.signatures[0] : undefined;
        if (summary && !member.docstring?.length) {
          entries[entries.length - 1].summary = clip(summaryOf(summary, linkRefs), 140);
        }
        if ("members" in member && member.kind !== "namespace") {
          for (const child of member.members) add(child, module, member.name);
        }
      }
    }
  }
  return entries;
}
