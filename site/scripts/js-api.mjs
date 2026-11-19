// @ts-check
/**
 * Extract the public API of the JavaScript packages in `js/packages/` for the
 * site.
 *
 * Run from `site/` with `pnpm js-api`. It writes `src/generated/js-api.json`,
 * which the pages under `src/pages/js/` render as the JavaScript API
 * reference at `/js/`. The Python reference is built the same way, from
 * `scripts/python-api.py`.
 *
 * The packages are read with TypeDoc (https://typedoc.org), from the `js/`
 * workspace's own install, so run `pnpm install` there first. TypeDoc
 * resolves the types a package imports from its sibling packages through
 * their built declarations, so build them too (`pnpm run -r prebuild &&
 * pnpm run -r build` in `js/`). It doesn't type-check: the packages' own CI
 * does that.
 *
 * What counts as public: what a package's `exports` map exposes. Each
 * entry (`.`, `./utils`) is a module on the package's page, and what it
 * exports, less anything marked `@internal`, `private` or `protected`, is
 * documented.
 */
import { existsSync, mkdirSync, readFileSync, readdirSync, writeFileSync } from "node:fs";
import { createRequire } from "node:module";
import { dirname, join, relative, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const SITE = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const REPO = resolve(SITE, "..");
const JS = join(REPO, "js");
const PACKAGES = join(JS, "packages");
const OUTPUT = join(SITE, "src", "generated", "js-api.json");

const GITHUB = "https://github.com/Arize-ai/openinference";
const BRANCH = "main";
const NPM = "https://www.npmjs.com/package";

/** The longest literal value worth showing in a signature. */
const VALUE_WIDTH = 120;

// TypeDoc comes from the js/ workspace, so the site doesn't depend on it.
const require = createRequire(join(JS, "package.json"));
/** @type {typeof import("typedoc")} */
const td = await (async () => {
  // `require.resolve` throws synchronously when TypeDoc is missing, so it
  // has to be inside the try, not before a `.catch`.
  try {
    return await import(require.resolve("typedoc"));
  } catch {
    console.error("TypeDoc isn't installed in js/. Run `pnpm install` in js/ first.");
    process.exit(1);
  }
})();
const { Application, ReflectionKind, TSConfigReader } = td;

/** `externalSymbolLinkMappings` from js/typedoc.jsonc: links for OpenTelemetry types. */
function externalLinks() {
  const text = readFileSync(join(JS, "typedoc.jsonc"), "utf8")
    .replace(/^\s*\/\/.*$/gm, "")
    .replace(/,(\s*[}\]])/g, "$1");
  return JSON.parse(text).externalSymbolLinkMappings ?? {};
}

/** Read once: every package's TypeDoc run uses the same mappings. */
const EXTERNAL_LINKS = externalLinks();

// --- Packages -----------------------------------------------------------------

/**
 * @typedef {{ dir: string, json: any, entries: { subpath: string, specifier: string, file: string }[] }} Package
 */

/** @returns {Package[]} */
function discover() {
  const packages = [];
  for (const name of readdirSync(PACKAGES).sort()) {
    const dir = join(PACKAGES, name);
    const file = join(dir, "package.json");
    if (!existsSync(file) || !existsSync(join(dir, "src"))) continue;
    const json = JSON.parse(readFileSync(file, "utf8"));
    if (json.private) continue;
    packages.push({ dir, json, entries: entriesOf(dir, json) });
  }
  return packages;
}

/**
 * The source file behind each entry of the `exports` map:
 * `./dist/esm/utils.js` is `src/utils.ts` (or `src/utils/index.ts`).
 */
function entriesOf(dir, json) {
  const exports =
    typeof json.exports === "string" || json.exports === undefined
      ? { ".": json.exports ?? json.main ?? "./dist/src/index.js" }
      : json.exports;
  const entries = [];
  for (const [subpath, target] of Object.entries(exports)) {
    if (subpath.includes("*") || subpath === "./package.json") continue;
    const built =
      typeof target === "string"
        ? target
        : (target.types ?? target.import ?? target.require ?? target.default);
    if (typeof built !== "string") continue;
    const stem = built
      .replace(/^\.\/dist\/(?:esm|commonjs|cjs|src)\//, "")
      .replace(/^\.\/dist\//, "")
      .replace(/^src\//, "")
      .replace(/\.d\.ts$|\.[cm]?js$/, "");
    const file = [`src/${stem}.ts`, `src/${stem}/index.ts`]
      .map((candidate) => join(dir, candidate))
      .find(existsSync);
    if (!file) {
      console.warn(`${json.name}: no source for export ${subpath} (${built}); skipped`);
      continue;
    }
    entries.push({
      subpath,
      specifier: subpath === "." ? json.name : `${json.name}/${subpath.slice(2)}`,
      file,
    });
  }
  return entries;
}

const groupOf = (name) => {
  const short = name.replace(/^@arizeai\//, "");
  if (short === "openinference-core" || short === "openinference-semantic-conventions") {
    return "core";
  }
  return short.startsWith("openinference-instrumentation-") ? "instrumentation" : "integration";
};

/**
 * What an instrumentation patches, from its `InstrumentationNodeModuleDefinition`
 * calls (`"openai", [">=4.0.0 <7"]`), else its peer dependencies.
 */
function instrumentsOf(pkg) {
  const found = new Map();
  const walk = (dir) => {
    for (const entry of readdirSync(dir, { withFileTypes: true })) {
      const path = join(dir, entry.name);
      if (entry.isDirectory()) walk(path);
      else if (/\.ts$/.test(entry.name)) {
        const source = readFileSync(path, "utf8").replace(/\/\/[^\n]*|\/\*[\s\S]*?\*\//g, "");
        // The module's name is a string, or a constant declared in the file.
        const pattern =
          /new\s+InstrumentationNodeModuleDefinition\s*(?:<[^>]*>)?\(\s*(?:["'`]([^"'`]+)["'`]|([A-Za-z_$][\w$]*))\s*,\s*\[([^\]]*)\]/g;
        for (const match of source.matchAll(pattern)) {
          const constant = match[2]
            ? source.match(new RegExp(`const\\s+${match[2]}\\s*=\\s*["'\`]([^"'\`]+)["'\`]`))?.[1]
            : undefined;
          const module = match[1] ?? constant;
          if (!module) continue;
          // `@langchain/core/dist/callbacks/manager.cjs` is `@langchain/core`.
          const name = module.startsWith("@")
            ? module.split("/").slice(0, 2).join("/")
            : module.split("/")[0];
          if (found.has(name)) continue;
          const versions = [...match[3].matchAll(/["'`]([^"'`]+)["'`]/g)].map((m) => m[1]);
          found.set(name, versions.filter((v) => v !== "*").join(" || "));
        }
      }
    }
  };
  walk(join(pkg.dir, "src"));
  if (found.size === 0) {
    for (const [name, range] of Object.entries(pkg.json.peerDependencies ?? {})) {
      if (!name.startsWith("@opentelemetry/") && !name.startsWith("@arizeai/")) {
        found.set(name, String(range));
      }
    }
  }
  return [...found].map(([name, range]) => (range ? `${name} ${range}` : name));
}

// --- Conversion ---------------------------------------------------------------

/**
 * A part of a type: text, or a name that links, to an object in this
 * reference (`ref`, resolved to `package` and `anchor` once every package
 * is read) or to external docs (`url`).
 * @typedef {string | { text: string, ref?: string, url?: string, package?: string, anchor?: string }} Part
 */

class Extractor {
  constructor() {
    /** `package:qualifiedName` -> { package, anchor }, across packages. */
    this.index = new Map();
    /** Reflection id -> `package:qualifiedName`, for the package being read. */
    this.ids = new Map();
    /** @type {Package | undefined} */
    this.pkg = undefined;
    /** Parts and markdown links to resolve once every package is read. */
    this.pending = [];
  }

  get name() {
    return /** @type {Package} */ (this.pkg).json.name;
  }

  key(qualified) {
    return `${this.name}:${qualified}`;
  }

  /** Remember where a reflection is documented. */
  register(reflection, anchor, qualified) {
    const key = this.key(qualified);
    this.ids.set(reflection.id, key);
    if (!this.index.has(key)) this.index.set(key, { package: this.name, anchor });
  }

  sourceOf(reflection) {
    const source = reflection.sources?.[0];
    if (!source?.fullFileName) return null;
    const path = relative(REPO, source.fullFileName);
    if (path.startsWith("..") || path.includes("node_modules")) return null;
    return `${GITHUB}/blob/${BRANCH}/${path}#L${source.line}`;
  }

  // --- Types

  /** @returns {Part[]} */
  type(type) {
    if (!type) return [];
    /** @type {Part[]} */
    const parts = [];
    const push = (...items) => {
      for (const item of items) {
        if (typeof item === "string" && typeof parts[parts.length - 1] === "string") {
          parts[parts.length - 1] += item;
        } else if (item !== "") parts.push(item);
      }
    };
    const join = (types, separator) =>
      types.forEach((t, i) => {
        if (i) push(separator);
        push(...this.type(t));
      });
    const wrap = (t) => {
      const needs = ["union", "intersection", "conditional"].includes(t.type) ||
        (t.type === "reflection" && t.declaration.signatures?.length);
      if (needs) push("(");
      push(...this.type(t));
      if (needs) push(")");
    };
    switch (type.type) {
      case "intrinsic":
        push(type.name);
        break;
      case "literal":
        push(type.toString());
        break;
      case "reference": {
        push(this.reference(type));
        if (type.typeArguments?.length) {
          push("<");
          join(type.typeArguments, ", ");
          push(">");
        }
        break;
      }
      case "union":
        join(type.types, " | ");
        break;
      case "intersection":
        join(type.types, " & ");
        break;
      case "array":
        wrap(type.elementType);
        push("[]");
        break;
      case "tuple":
        push("[");
        join(type.elements, ", ");
        push("]");
        break;
      case "namedTupleMember":
        push(`${type.name}${type.isOptional ? "?" : ""}: `, ...this.type(type.element));
        break;
      case "optional":
        push(...this.type(type.elementType), "?");
        break;
      case "rest":
        push("...", ...this.type(type.elementType));
        break;
      case "typeOperator":
        push(`${type.operator} `, ...this.type(type.target));
        break;
      case "query":
        push("typeof ", ...this.type(type.queryType));
        break;
      case "indexedAccess":
        wrap(type.objectType);
        push("[", ...this.type(type.indexType), "]");
        break;
      case "conditional":
        push(...this.type(type.checkType), " extends ", ...this.type(type.extendsType));
        push(" ? ", ...this.type(type.trueType), " : ", ...this.type(type.falseType));
        break;
      case "predicate":
        push(type.asserts ? "asserts " : "", type.name);
        if (type.targetType) push(" is ", ...this.type(type.targetType));
        break;
      case "reflection":
        push(...this.inline(type.declaration));
        break;
      default:
        push(type.toString());
    }
    return parts;
  }

  /** A named type: linked to this reference, to external docs, or plain. */
  reference(type) {
    const text = type.name;
    if (type.reflection && this.ids.has(type.reflection.id)) {
      return this.later({ text, ref: this.ids.get(type.reflection.id) });
    }
    if (type.externalUrl && type.externalUrl !== "#") return { text, url: type.externalUrl };
    if (type.package && type.qualifiedName) {
      return this.later({ text, ref: `${type.package}:${type.qualifiedName}` });
    }
    // Declared in this package but not yet registered (declared later in
    // the file): resolve by name once the package is read.
    if (type.reflection) return this.later({ text, ref: this.key(type.reflection.getFullName?.() ?? text) });
    return text;
  }

  later(part) {
    this.pending.push(part);
    return part;
  }

  /** An inline type: a function type, or an object literal type. */
  inline(declaration) {
    if (declaration.signatures?.length) {
      const signature = declaration.signatures[0];
      return [
        "(",
        ...this.parameterList(signature.parameters ?? []),
        ") => ",
        ...this.type(signature.type),
      ];
    }
    const children = declaration.children ?? [];
    const index = declaration.indexSignatures ?? [];
    if (children.length === 0 && index.length === 0) return ["{}"];
    /** @type {Part[]} */
    const parts = ["{ "];
    const items = [
      ...index.map((signature) => {
        const parameter = signature.parameters?.[0];
        return [
          `[${parameter?.name ?? "key"}: `,
          ...this.type(parameter?.type),
          "]: ",
          ...this.type(signature.type),
        ];
      }),
      ...children.map((child) => [
        `${child.flags.isReadonly ? "readonly " : ""}${child.name}${child.flags.isOptional ? "?" : ""}: `,
        ...(child.type
          ? this.type(child.type)
          : child.signatures?.length
            ? this.inline(child)
            : ["unknown"]),
      ]),
    ];
    items.forEach((item, i) => {
      if (i) parts.push("; ");
      parts.push(...item);
    });
    parts.push(" }");
    return merge(parts);
  }

  parameterList(parameters) {
    /** @type {Part[]} */
    const parts = [];
    parameters.forEach((parameter, i) => {
      if (i) parts.push(", ");
      const rest = parameter.flags.isRest ? "..." : "";
      const optional = parameter.flags.isOptional || parameter.defaultValue ? "?" : "";
      parts.push(`${rest}${displayName(parameter)}${optional}: `, ...this.type(parameter.type));
    });
    return merge(parts);
  }

  // --- Comments

  /** Display parts as markdown; `{@link X}` becomes a link to resolve later. */
  markdown(displayParts = []) {
    return displayParts
      .map((part) => {
        if (part.kind !== "inline-tag") return part.text;
        if (!["@link", "@linkcode", "@linkplain"].includes(part.tag)) return part.text;
        let text = part.text;
        let target = part.target;
        // `{@link https://example.com|label}` and `{@link https://example.com label}`
        if (typeof target === "string") {
          const [url, label] = target.split(/\s*\|\s*|\s+/, 2);
          target = url;
          text = label ?? (text.startsWith(url) ? text.slice(url.length).replace(/^\s*\|?\s*/, "") || url : text);
        }
        const label = part.tag === "@linkcode" || typeof target !== "string" ? `\`${text}\`` : text;
        if (typeof target === "string") return `[${label}](${target})`;
        const ref =
          target && "id" in target && this.ids.has(target.id)
            ? this.ids.get(target.id)
            : target && "packageName" in target
              ? `${target.packageName}:${target.qualifiedName}`
              : target && "getFullName" in target
                ? this.key(target.getFullName())
                : undefined;
        if (!ref) return label;
        // Placeholder link: src/lib/js-api.ts swaps it for the page's URL,
        // or drops the link if the target isn't documented.
        this.pending.push({ markdownRef: ref });
        return `[${label}](oi-ref:${encodeURIComponent(ref)})`;
      })
      .join("");
  }

  /**
   * A comment as the sections the site's Docstring component draws (the
   * same shape the Python reference uses): the summary, then tables and
   * notes from block tags. `parameters` adds a Parameters table.
   */
  comment(comment, { parameters = [], returns } = {}) {
    const sections = [];
    if (comment?.summary?.length) {
      const value = this.markdown(comment.summary).trim();
      if (value) sections.push({ kind: "text", value });
    }
    const rows = parameters.flatMap((parameter) => this.parameterRows(parameter, comment));
    // The signature already gives names and types; a table only earns its
    // place when it says what they're for.
    if (rows.some((row) => row.description)) {
      sections.push({ kind: "parameters", title: null, items: rows });
    }
    const typeParameters = comment?.blockTags?.filter((tag) => tag.tag === "@typeParam") ?? [];
    for (const tag of comment?.blockTags ?? []) {
      const value = this.markdown(tag.content).trim();
      switch (tag.tag) {
        case "@param":
        case "@typeParam":
          break;
        case "@returns":
        case "@return":
          sections.push({
            kind: "returns",
            title: null,
            items: [{ name: null, annotation: returns ?? null, default: null, description: value }],
          });
          break;
        case "@throws":
          sections.push({
            kind: "raises",
            title: "Throws",
            items: [{ name: null, annotation: null, default: null, description: value }],
          });
          break;
        case "@example":
          sections.push({
            kind: "examples",
            items: [{ kind: "text", value: value.includes("```") ? value : "```ts\n" + value + "\n```" }],
          });
          break;
        case "@remarks":
          if (value) sections.push({ kind: "text", value });
          break;
        case "@deprecated":
          sections.push({ kind: "deprecated", version: "", value });
          break;
        case "@see":
          sections.push({ kind: "admonition", title: "See also", value });
          break;
        case "@default":
        case "@defaultValue":
          sections.push({ kind: "admonition", title: "Default", value });
          break;
        default:
          if (value) {
            const title = tag.tag.slice(1).replace(/^./, (c) => c.toUpperCase());
            sections.push({ kind: "admonition", title, value });
          }
      }
    }
    if (typeParameters.length) {
      sections.push({
        kind: "type parameters",
        title: null,
        items: typeParameters.map((tag) => ({
          name: tag.name ?? null,
          annotation: null,
          default: null,
          description: this.markdown(tag.content).trim(),
        })),
      });
    }
    return sections;
  }

  /**
   * A parameter's rows in the Parameters table. A destructured options
   * object gets a row per property (`options.tracerProvider`), as its
   * `@param options.tracerProvider` comments describe them.
   */
  parameterRows(parameter, comment) {
    const tagFor = (name) =>
      comment?.blockTags?.find((tag) => tag.tag === "@param" && tag.name === name);
    const description = (reflection, name) =>
      this.markdown(reflection.comment?.summary ?? tagFor(name)?.content ?? []).trim();
    const name = displayName(parameter);
    const rows = [
      {
        name: `${parameter.flags.isRest ? "..." : ""}${name}${parameter.flags.isOptional || parameter.defaultValue ? "?" : ""}`,
        annotation: this.type(parameter.type),
        default: parameter.defaultValue && parameter.defaultValue !== "{}" ? [parameter.defaultValue] : null,
        description: description(parameter, parameter.name),
      },
    ];
    const declaration = parameter.type?.type === "reflection" ? parameter.type.declaration : undefined;
    for (const child of declaration?.children ?? []) {
      const childName = parameter.name === "__namedParameters" ? child.name : `${name}.${child.name}`;
      rows.push({
        name: `${childName}${child.flags.isOptional ? "?" : ""}`,
        annotation: child.type ? this.type(child.type) : null,
        default: child.defaultValue ? [child.defaultValue] : null,
        description: this.markdown(
          child.comment?.summary ??
            tagFor(`${parameter.name}.${child.name}`)?.content ??
            tagFor(`${name}.${child.name}`)?.content ??
            [],
        ).trim(),
      });
    }
    // A destructured object's own row says nothing its properties don't.
    if (parameter.name === "__namedParameters" && rows.length > 1 && !rows[0].description) {
      rows.shift();
    }
    return rows;
  }

  // --- Reflections

  signature(signature) {
    const parameters = (signature.parameters ?? []).map((parameter) => {
      const declaration =
        parameter.type?.type === "reflection" ? parameter.type.declaration : undefined;
      // An options object's properties, so the site can put one per line.
      const fields =
        declaration?.children?.length && !declaration.signatures?.length
          ? declaration.children.map((child) => ({
              name: child.name,
              optional: Boolean(child.flags.isOptional),
              type: child.type ? this.type(child.type) : ["unknown"],
            }))
          : null;
      return {
        name: displayName(parameter),
        type: this.type(parameter.type),
        fields,
        optional: Boolean(parameter.flags.isOptional || parameter.defaultValue),
        rest: Boolean(parameter.flags.isRest),
      };
    });
    const returns = signature.type ? this.type(signature.type) : null;
    return {
      typeParameters: (signature.typeParameters ?? []).map((t) => this.typeParameter(t)),
      parameters,
      returns,
      docstring: this.comment(signature.comment, {
        parameters: signature.parameters ?? [],
        returns,
      }),
    };
  }

  typeParameter(parameter) {
    return {
      name: parameter.name,
      constraint: parameter.type ? this.type(parameter.type) : null,
      default: parameter.default ? this.type(parameter.default) : null,
    };
  }

  /** Register a reflection and its members before converting, so types can link to them. */
  registerTree(reflection, anchor, qualified) {
    this.register(reflection, anchor, qualified);
    for (const child of reflection.children ?? []) {
      if (!isPublic(child)) continue;
      this.registerTree(child, `${anchor}.${child.name}`, `${qualified}.${child.name}`);
    }
  }

  /** One exported object. */
  object(reflection, anchor) {
    const base = {
      name: reflection.name,
      anchor,
      source: this.sourceOf(reflection),
      flags: flagsOf(reflection),
    };
    const kind = reflection.kind;
    if (kind === ReflectionKind.Reference) {
      const target = reflection.tryGetTargetReflectionDeep?.();
      return {
        ...base,
        kind: "reexport",
        target: target
          ? this.later({ text: target.getFullName(), ref: this.ids.get(target.id) ?? this.key(target.name) })
          : reflection.name,
      };
    }
    if (kind === ReflectionKind.Function) {
      return {
        ...base,
        kind: "function",
        docstring: this.comment(reflection.comment),
        signatures: (reflection.signatures ?? []).map((s) => this.signature(s)),
      };
    }
    if (kind === ReflectionKind.Class || kind === ReflectionKind.Interface) {
      return {
        ...base,
        kind: kind === ReflectionKind.Class ? "class" : "interface",
        docstring: this.comment(reflection.comment),
        typeParameters: (reflection.typeParameters ?? []).map((t) => this.typeParameter(t)),
        extends: (reflection.extendedTypes ?? []).map((t) => this.type(t)),
        implements: (reflection.implementedTypes ?? []).map((t) => this.type(t)),
        members: this.members(reflection, anchor),
      };
    }
    if (kind === ReflectionKind.Enum) {
      return {
        ...base,
        kind: "enum",
        docstring: this.comment(reflection.comment),
        members: (reflection.children ?? []).map((member) => ({
          name: member.name,
          anchor: `${anchor}.${member.name}`,
          kind: "property",
          type: member.type ? this.type(member.type) : [],
          docstring: this.comment(member.comment),
          source: this.sourceOf(member),
          flags: {},
        })),
      };
    }
    if (kind === ReflectionKind.TypeAlias) {
      const declaration = reflection.type?.type === "reflection" ? reflection.type.declaration : undefined;
      const isObject = declaration && !declaration.signatures?.length && declaration.children?.length;
      return {
        ...base,
        kind: "type",
        docstring: this.comment(reflection.comment),
        typeParameters: (reflection.typeParameters ?? []).map((t) => this.typeParameter(t)),
        type: isObject ? null : this.type(reflection.type),
        members: isObject ? this.members(declaration, anchor) : [],
      };
    }
    if (kind === ReflectionKind.Namespace || kind === ReflectionKind.Module) {
      return {
        ...base,
        kind: "namespace",
        docstring: this.comment(reflection.comment),
        members: (reflection.children ?? [])
          .filter(isPublic)
          .map((child) => this.object(child, `${anchor}.${child.name}`)),
      };
    }
    // Variables: a constant's literal value is its documentation; an object
    // or function value is described by its type.
    const declaration = reflection.type?.type === "reflection" ? reflection.type.declaration : undefined;
    if (declaration?.signatures?.length) {
      return {
        ...base,
        kind: "function",
        docstring: this.comment(reflection.comment ?? declaration.signatures[0].comment),
        signatures: declaration.signatures.map((s) => this.signature(s)),
      };
    }
    // A constant built from others (`${prefix}.${name}` as const) has no
    // short literal value, but its type is the literal it resolves to.
    const literal = reflection.type?.type === "literal" ? reflection.type.toString() : undefined;
    const written = reflection.defaultValue;
    const value =
      written && written.length <= VALUE_WIDTH && written !== "..." && !written.includes("${")
        ? written
        : (literal ?? null);
    return {
      ...base,
      kind: "variable",
      docstring: this.comment(reflection.comment),
      type: declaration?.children?.length ? null : this.type(reflection.type),
      value,
      members: declaration?.children?.length ? this.members(declaration, anchor) : [],
    };
  }

  /** A class's, interface's or object type's members. */
  members(reflection, anchor) {
    const members = [];
    for (const child of reflection.children ?? []) {
      if (!isPublic(child)) continue;
      const childAnchor = `${anchor}.${child.name}`;
      const common = {
        name: child.name,
        anchor: childAnchor,
        source: this.sourceOf(child),
        flags: flagsOf(child),
        inherited: child.inheritedFrom ? child.inheritedFrom.toString() : null,
      };
      if (child.kind === ReflectionKind.Constructor) {
        members.push({
          ...common,
          kind: "constructor",
          docstring: [],
          signatures: (child.signatures ?? []).map((s) => this.signature(s)),
        });
      } else if (child.kind === ReflectionKind.Method || child.signatures?.length) {
        members.push({
          ...common,
          kind: "method",
          docstring: [],
          signatures: (child.signatures ?? []).map((s) => this.signature(s)),
        });
      } else if (child.kind === ReflectionKind.Accessor) {
        const getter = child.getSignature;
        const setter = child.setSignature;
        members.push({
          ...common,
          kind: "accessor",
          type: this.type(getter?.type ?? setter?.parameters?.[0]?.type),
          readonly: !setter,
          docstring: this.comment(getter?.comment ?? setter?.comment ?? child.comment),
        });
      } else {
        const declaration = child.type?.type === "reflection" ? child.type.declaration : undefined;
        if (declaration?.signatures?.length) {
          members.push({
            ...common,
            kind: "method",
            docstring: [],
            signatures: declaration.signatures.map((s) => this.signature({ ...s, comment: s.comment ?? child.comment })),
          });
        } else {
          members.push({
            ...common,
            kind: "property",
            type: this.type(child.type),
            value: child.defaultValue && child.defaultValue.length <= VALUE_WIDTH ? child.defaultValue : null,
            docstring: this.comment(child.comment),
          });
        }
      }
    }
    for (const signature of reflection.indexSignatures ?? []) {
      const parameter = signature.parameters?.[0];
      members.push({
        name: `[${parameter?.name ?? "key"}: ${parameter?.type?.toString() ?? "string"}]`,
        anchor: `${anchor}.__index`,
        kind: "property",
        type: this.type(signature.type),
        value: null,
        docstring: this.comment(signature.comment),
        source: null,
        flags: {},
      });
    }
    return members;
  }

  /** A package's modules: one per entry in its `exports` map. */
  async package(pkg) {
    this.pkg = pkg;
    this.ids = new Map();
    const app = await Application.bootstrapWithPlugins(
      {
        entryPoints: pkg.entries.map((entry) => entry.file),
        entryPointStrategy: "resolve",
        tsconfig: join(pkg.dir, "tsconfig.json"),
        skipErrorChecking: true,
        excludePrivate: true,
        excludeProtected: true,
        excludeInternal: true,
        excludeExternals: true,
        exclude: ["**/test/**", "**/tests/**", "**/examples/**", "**/*.test.ts"],
        externalSymbolLinkMappings: EXTERNAL_LINKS,
        sort: ["source-order"],
        readme: "none",
        logLevel: "Error",
        // Read links and parameter docs, but don't fail on what TypeDoc can't
        // resolve: the site resolves across packages itself.
        validation: { notExported: false, invalidLink: false, notDocumented: false },
      },
      [new TSConfigReader()],
    );
    const project = await app.convert();
    if (!project) throw new Error(`${pkg.json.name}: TypeDoc couldn't read the package`);

    // With one entry point the project is the module; with several, each is
    // a child module named after its file.
    const modules =
      pkg.entries.length === 1
        ? [{ entry: pkg.entries[0], reflection: project }]
        : pkg.entries.map((entry) => {
            const stem = relative(join(pkg.dir, "src"), entry.file).replace(/(\/index)?\.ts$/, "");
            const reflection = (project.children ?? []).find(
              (child) =>
                child.kindOf(ReflectionKind.Module) &&
                (child.name === stem ||
                  child.sources?.some((source) => source.fullFileName === entry.file)),
            );
            return { entry, reflection };
          });

    const prefixOf = (entry) => (entry.subpath === "." ? "" : `${entry.subpath.slice(2)}/`);
    // Register first, so a type can link to anything in the package.
    for (const { entry, reflection } of modules) {
      for (const child of reflection?.children ?? []) {
        if (!isPublic(child) || child.kind === ReflectionKind.Reference) continue;
        this.registerTree(child, `${prefixOf(entry)}${child.name}`, child.name);
      }
    }
    return modules.map(({ entry, reflection }) => ({
      specifier: entry.specifier,
      subpath: entry.subpath,
      source: `${GITHUB}/blob/${BRANCH}/${relative(REPO, entry.file)}`,
      docstring: this.comment(reflection?.comment),
      members: (reflection?.children ?? [])
        .filter(isPublic)
        .map((child) => this.object(child, `${prefixOf(entry)}${child.name}`)),
    }));
  }

  /** Resolve every pending link against the index of all packages. */
  resolve() {
    for (const item of this.pending) {
      if ("markdownRef" in item) continue;
      const target = item.ref ? this.index.get(item.ref) : undefined;
      if (target) Object.assign(item, target);
      delete item.ref;
    }
  }
}

const isPublic = (reflection) =>
  !reflection.flags.isPrivate &&
  !reflection.flags.isProtected &&
  !reflection.name.startsWith("_") &&
  !reflection.name.startsWith("#");

const flagsOf = (reflection) => {
  const flags = {};
  for (const flag of ["isStatic", "isReadonly", "isOptional", "isAbstract", "isConst"]) {
    if (reflection.flags[flag]) flags[flag.slice(2).toLowerCase()] = true;
  }
  return flags;
};

/**
 * TypeDoc names a destructured parameter `__namedParameters`. It has no name
 * in the source; call it `options`, as its callers do.
 */
function displayName(parameter) {
  return parameter.name === "__namedParameters" ? "options" : parameter.name;
}

/** Join adjacent strings. */
function merge(parts) {
  const merged = [];
  for (const part of parts) {
    if (typeof part === "string" && typeof merged[merged.length - 1] === "string") {
      merged[merged.length - 1] += part;
    } else if (part !== "") merged.push(part);
  }
  return merged;
}

/** The instrumentation class: one that extends OpenTelemetry's InstrumentationBase. */
function instrumentationOf(modules) {
  for (const module of modules) {
    for (const member of module.members) {
      if (member.kind !== "class") continue;
      const base = member.extends.flat().find((part) => typeof part !== "string");
      const baseName = typeof base === "object" ? base.text : "";
      if (baseName !== "InstrumentationBase") continue;
      return {
        name: member.name,
        anchor: member.anchor,
        manual: member.members.some((m) => m.name === "manuallyInstrument"),
      };
    }
  }
  return null;
}

// --- Main ---------------------------------------------------------------------

const extractor = new Extractor();
const output = [];
for (const pkg of discover()) {
  const modules = await extractor.package(pkg);
  const directory = relative(REPO, pkg.dir);
  const readme = join(pkg.dir, "README.md");
  output.push({
    name: pkg.json.name,
    slug: pkg.json.name.replace(/^@[^/]+\//, ""),
    group: groupOf(pkg.json.name),
    description: pkg.json.description ?? "",
    version: pkg.json.version ?? null,
    node: pkg.json.engines?.node ?? null,
    instruments: groupOf(pkg.json.name) === "instrumentation" ? instrumentsOf(pkg) : [],
    peerDependencies: pkg.json.peerDependencies ?? {},
    instrumentation: instrumentationOf(modules),
    npm: `${NPM}/${pkg.json.name}`,
    source: `${GITHUB}/tree/${BRANCH}/${directory}`,
    readme: existsSync(readme) ? `${GITHUB}/blob/${BRANCH}/${directory}/README.md` : null,
    modules,
  });
}
extractor.resolve();

mkdirSync(dirname(OUTPUT), { recursive: true });
writeFileSync(
  OUTPUT,
  `${JSON.stringify(
    { packages: output, index: Object.fromEntries([...extractor.index].sort()) },
    null,
    1,
  )}\n`,
);
const count = output.reduce(
  (sum, pkg) => sum + pkg.modules.reduce((n, module) => n + module.members.length, 0),
  0,
);
console.error(`Wrote ${output.length} packages, ${count} exports to ${relative(SITE, OUTPUT)}`);
