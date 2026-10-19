#!/usr/bin/env python3
"""Deterministic convention checks for one OpenInference Python instrumentor.

Usage (from anywhere in the repo):
  python3 .agents/skills/python-review/scripts/check_package.py <pkg>
  python3 .agents/skills/python-review/scripts/check_package.py <pkg> --base origin/main
  python3 .agents/skills/python-review/scripts/check_package.py --all
  python3 .agents/skills/python-review/scripts/check_package.py --validate

<pkg> is the tox factor (`openai`, `google_adk`) or the directory name
(`openinference-instrumentation-openai`). With --base, every finding is tagged
`introduced` or `pre-existing` by rerunning the checks on that git ref.
A single-package run exits 1 when a Critical finding is not pre-existing.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from collections.abc import Callable, Iterator
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Literal

Severity = Literal["Critical", "High", "Medium", "Low"]
Origin = Literal["introduced", "pre-existing", "n/a"]
Kind = Literal["wrapper", "processor", "propagator", "alias"]
ALL_KINDS: frozenset[Kind] = frozenset({"wrapper", "processor", "propagator", "alias"})
SEVERITY_ORDER = {"Critical": 0, "High": 1, "Medium": 2, "Low": 3}

INSTRUMENTATION = "python/instrumentation"
TOX_INI = "python/tox.ini"
SECRET = re.compile(r"\b(sk-(?:ant-|proj-)?[A-Za-z0-9_-]{20,}|AIza[0-9A-Za-z_-]{30,})")


@dataclass(frozen=True)
class Finding:
    check: str
    reference: str
    severity: Severity
    message: str
    location: str
    origin: Origin = "n/a"


class Tree:
    """Read files from the working tree, or from a git ref when `ref` is set."""

    def __init__(self, root: Path, ref: str | None = None) -> None:
        self.root = root
        self.ref = ref

    def read(self, path: str) -> str | None:
        if self.ref is None:
            p = self.root / path
            return p.read_text(errors="replace") if p.is_file() else None
        out = self._git("show", f"{self.ref}:{path}")
        return out.stdout if out.returncode == 0 else None

    def files(self, directory: str) -> list[str]:
        if self.ref is None:
            base = self.root / directory
            return sorted(
                str(p.relative_to(self.root))
                for p in base.rglob("*")
                if p.is_file() and ".tox" not in p.parts
            )
        out = self._git("ls-tree", "-r", "--name-only", self.ref, "--", directory)
        return out.stdout.split() if out.returncode == 0 else []

    def _git(self, *args: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["git", "-C", str(self.root), *args], capture_output=True, text=True, check=False
        )


@dataclass(frozen=True)
class Package:
    directory: str
    factor: str | None

    @property
    def dist(self) -> str:
        return Path(self.directory).name

    @property
    def name(self) -> str:
        return self.factor or self.dist.removeprefix("openinference-instrumentation-")


def packages(tree: Tree) -> list[Package]:
    tox = tree.read(TOX_INI) or ""
    factors = {
        d: f
        for f, d in re.findall(
            r"^\s+(\w+): instrumentation/(openinference-instrumentation-[\w-]+)/\s*$",
            tox,
            re.MULTILINE,
        )
    }
    dists = sorted(
        {
            f.split("/")[2]
            for f in tree.files(INSTRUMENTATION)
            if f.count("/") == 3 and f.endswith("/pyproject.toml")
        }
    )
    return [Package(f"{INSTRUMENTATION}/{d}", factors.get(d)) for d in dists]


def resolve(tree: Tree, name: str) -> Package:
    for pkg in packages(tree):
        if name in (pkg.factor, pkg.dist, pkg.name):
            return pkg
    sys.exit(f"unknown package {name!r}: no {INSTRUMENTATION}/<dir>/pyproject.toml matches")


class Ctx:
    def __init__(self, tree: Tree, pkg: Package) -> None:
        self.tree = tree
        self.pkg = pkg
        self.tox = tree.read(TOX_INI) or ""
        self.files = tree.files(pkg.directory)
        self.src = self._texts(f"{pkg.directory}/src/")
        self.tests = self._texts(f"{pkg.directory}/tests/")
        self.src_dir = f"{pkg.directory}/src"
        self.tests_dir = f"{pkg.directory}/tests"
        self.kind = self._kind()

    def _kind(self) -> Kind:
        if self.grep(self.src, r"\bBaseInstrumentor\b"):
            spans = self.grep(self.src, r"start_(as_current_)?span\(|OITracer")
            return "wrapper" if spans else "propagator"
        if self.grep(self.src, r"\(\s*SpanProcessor\s*\)"):
            return "processor"
        if self.grep(self.src, r"^from openinference\.instrumentation\.\w+ import \w+Instrumentor"):
            return "alias"
        return "wrapper"

    def _texts(self, prefix: str) -> dict[str, str]:
        return {
            f: self.tree.read(f) or ""
            for f in self.files
            if f.startswith(prefix) and f.endswith(".py")
        }

    def file(self, name: str) -> tuple[str, str | None]:
        path = f"{self.pkg.directory}/{name}"
        return path, self.tree.read(path)

    def grep(self, texts: dict[str, str], pattern: str) -> list[str]:
        rx = re.compile(pattern)
        return [
            f"{path}:{i}"
            for path, text in texts.items()
            for i, line in enumerate(text.splitlines(), 1)
            if rx.search(line)
        ]

    def tox_lines(self, factor: str) -> list[str]:
        return re.findall(rf"^\s+{re.escape(factor)}: (.+)$", self.tox, re.MULTILINE)

    def tox_loc(self, needle: str) -> str:
        for i, line in enumerate(self.tox.splitlines(), 1):
            if needle in line:
                return f"{TOX_INI}:{i}"
        return TOX_INI


Issue = tuple[Severity, str, str]


@dataclass(frozen=True)
class Check:
    id: str
    reference: str
    kinds: frozenset[Kind]
    run: Callable[[Ctx], Iterator[Issue]]


CHECKS: list[Check] = []


def check(
    id: str, reference: str, kinds: frozenset[Kind] = ALL_KINDS
) -> Callable[[Callable[[Ctx], Iterator[Issue]]], None]:
    def register(fn: Callable[[Ctx], Iterator[Issue]]) -> None:
        CHECKS.append(Check(id, reference, kinds, fn))

    return register


WRAPPER: frozenset[Kind] = frozenset({"wrapper"})
OWNS_SPANS: frozenset[Kind] = frozenset({"wrapper", "processor"})


@check("tox-registered", "ci-and-packaging.md#tox-install-pattern")
def _(c: Ctx) -> Iterator[Issue]:
    if c.pkg.factor is None:
        yield "Critical", "the package has no tox changedir entry, so CI never tests it", TOX_INI


@check("tox-install", "ci-and-packaging.md#tox-install-pattern")
def _(c: Ctx) -> Iterator[Issue]:
    if c.pkg.factor is None:
        return
    lines = c.tox_lines(c.pkg.factor)
    loc = c.tox_loc(f"  {c.pkg.factor}: uv ")
    steps = {
        "uv pip uninstall": "uninstall the pinned library",
        f"--reinstall-package {c.pkg.dist}": "reinstall the package from source",
        "-r test-requirements.txt": "install test-requirements",
    }
    missing = [
        label for needle, label in steps.items() if not any(needle in line for line in lines)
    ]
    broken = any("--reinstall {toxinidir}" in line for line in lines)
    if broken and missing:
        yield (
            "Critical",
            "commands_pre uses `--reinstall {toxinidir}/...`, which can test the wrong library version",
            loc,
        )
    elif missing:
        yield "Critical", f"commands_pre is missing steps: {', '.join(missing)}", loc
    elif broken:
        yield (
            "Low",
            "a leftover `--reinstall {toxinidir}/...` line runs before the correct steps",
            loc,
        )


@check("tox-latest", "ci-and-packaging.md#latest-env")
def _(c: Ctx) -> Iterator[Issue]:
    if c.pkg.factor is None:
        return
    latest = f"{c.pkg.factor}-latest"
    if not re.search(rf"[{{,]{re.escape(latest)}[,}}]", c.tox):
        yield (
            "High",
            f"envlist has no `{latest}` variant, so canaries never test the newest library",
            TOX_INI,
        )
    elif not any("uv pip install -U" in line for line in c.tox_lines(latest)):
        yield "High", f"`{latest}` has no `uv pip install -U <library>` step", c.tox_loc(latest)


@check("test-requirements", "ci-and-packaging.md#test-requirements")
def _(c: Ctx) -> Iterator[Issue]:
    path, text = c.file("test-requirements.txt")
    if text is None:
        yield (
            "Critical",
            "test-requirements.txt is missing; the tox install pattern depends on it",
            path,
        )
        return
    reqs = [line.split("#")[0].strip() for line in text.splitlines()]
    if not any("==" in r for r in reqs):
        yield "High", "no `==` pin, so the pinned env tests whatever resolves", path
    if not any(r.startswith("opentelemetry-sdk") for r in reqs):
        yield "Medium", "opentelemetry-sdk is not listed", path
    if c.grep(c.tests, r"pytest\.mark\.vcr") and not any(
        r.startswith("pytest-recording") for r in reqs
    ):
        yield "High", "tests use @pytest.mark.vcr but pytest-recording is not listed", path


@check("version-py", "ci-and-packaging.md#packaging")
def _(c: Ctx) -> Iterator[Issue]:
    if not any(f.startswith(c.src_dir) and f.endswith("/version.py") for f in c.files):
        yield "Critical", "no src/**/version.py; the tox `ci` env fails without it", c.src_dir


@check("instruments", "ci-and-packaging.md#packaging", WRAPPER)
def _(c: Ctx) -> Iterator[Issue]:
    if not c.grep(c.src, r"^_instruments\s*="):
        yield "High", "no `_instruments` tuple declaring the supported library range", c.src_dir
    if not c.grep(c.src, r"def instrumentation_dependencies"):
        yield "High", "the instrumentor does not define instrumentation_dependencies()", c.src_dir


@check("entry-points", "ci-and-packaging.md#packaging", WRAPPER)
def _(c: Ctx) -> Iterator[Issue]:
    path, text = c.file("pyproject.toml")
    for group in ("opentelemetry_instrumentor", "openinference_instrumentor"):
        if f"entry-points.{group}]" not in (text or ""):
            yield (
                "Medium",
                f"pyproject.toml does not register `[project.entry-points.{group}]`",
                path,
            )


@check("suppress-misuse", "runtime.md#suppression")
def _(c: Ctx) -> Iterator[Issue]:
    for loc in c.grep(c.src, r"if\s+(not\s+)?suppress_instrumentation\(\)"):
        yield (
            "Critical",
            "`suppress_instrumentation()` is a context manager, not a check; this condition is always true",
            loc,
        )


@check("suppress-check", "runtime.md#suppression", WRAPPER)
def _(c: Ctx) -> Iterator[Issue]:
    if not c.grep(c.src, r"_SUPPRESS_INSTRUMENTATION_KEY|is_instrumentation_enabled\(\)"):
        yield "High", "no suppression check anywhere in src", c.src_dir


@check("trace-config", "runtime.md#traceconfig", OWNS_SPANS)
def _(c: Ctx) -> Iterator[Issue]:
    if not c.grep(c.src, r"\bTraceConfig\b"):
        yield "High", "src never references TraceConfig, so masking flags are ignored", c.src_dir


@check("context-attributes", "runtime.md#context-attributes", OWNS_SPANS)
def _(c: Ctx) -> Iterator[Issue]:
    if not c.grep(c.src, r"\bOITracer\b|get_attributes_from_context"):
        yield (
            "High",
            "src uses neither OITracer nor get_attributes_from_context(), so using_attributes() values never reach spans",
            c.src_dir,
        )


@check("pop-assertions", "tests.md#exhaustive-assertions", OWNS_SPANS)
def _(c: Ctx) -> Iterator[Issue]:
    if not c.grep(c.tests, r"assert not \w*attributes\b|assert \w*attributes == \{\}"):
        yield (
            "High",
            "no test ends with `assert not attributes` (or `== {}`), so unexpected attributes go unnoticed",
            c.tests_dir,
        )


@check("context-tests", "tests.md#context-attributes", OWNS_SPANS)
def _(c: Ctx) -> Iterator[Issue]:
    if not c.grep(c.tests, r"\busing_(attributes|session|user|metadata|tags|prompt_template)\("):
        yield "High", "no test sets context attributes with using_attributes()", c.tests_dir


@check("error-tests", "tests.md#error-path", OWNS_SPANS)
def _(c: Ctx) -> Iterator[Issue]:
    if not c.grep(c.tests, r"StatusCode\.ERROR"):
        yield "High", "no test asserts StatusCode.ERROR on a failing call", c.tests_dir


@check("cassette-secret", "tests.md#cassettes")
def _(c: Ctx) -> Iterator[Issue]:
    for f in c.files:
        if "/cassettes/" in f:
            text = c.tree.read(f) or ""
            m = SECRET.search(text)
            if m:
                line = text.count("\n", 0, m.start()) + 1
                yield (
                    "Critical",
                    f"cassette contains a key-shaped value `{m.group(0)[:8]}...`",
                    f"{f}:{line}",
                )


def run(tree: Tree, pkg: Package) -> list[Finding]:
    ctx = Ctx(tree, pkg)
    return [
        Finding(chk.id, chk.reference, severity, message, location)
        for chk in CHECKS
        if ctx.kind in chk.kinds
        for severity, message, location in chk.run(ctx)
    ]


def tag(head: list[Finding], base: list[Finding]) -> list[Finding]:
    before = {(f.check, f.message) for f in base}
    return [
        replace(f, origin="pre-existing" if (f.check, f.message) in before else "introduced")
        for f in head
    ]


def report_all(tree: Tree, fmt: str) -> None:
    pkgs = packages(tree)
    failing: dict[str, set[str]] = {}
    for pkg in pkgs:
        for f in run(tree, pkg):
            failing.setdefault(f"{f.severity} {f.check}", set()).add(pkg.name)
    rows = sorted(failing.items(), key=lambda kv: (SEVERITY_ORDER[kv[0].split()[0]], -len(kv[1])))
    if fmt == "json":
        print(
            json.dumps(
                {"packages": len(pkgs), "failing": {k: sorted(v) for k, v in rows}}, indent=2
            )
        )
        return
    print(f"{len(pkgs)} packages checked")
    for k, v in rows:
        print(f"{k}: {len(v)} ({', '.join(sorted(v))})")


def report_one(pkg: Package, kind: Kind, findings: list[Finding], fmt: str) -> None:
    passed = sorted({c.id for c in CHECKS if kind in c.kinds} - {f.check for f in findings})
    if fmt == "json":
        print(
            json.dumps(
                {
                    "package": pkg.name,
                    "directory": pkg.directory,
                    "kind": kind,
                    "findings": [asdict(f) for f in findings],
                    "passed": passed,
                },
                indent=2,
            )
        )
        return
    print(f"{pkg.name} ({pkg.directory}, {kind}): {len(findings)} findings")
    for f in findings:
        origin = "" if f.origin == "n/a" else f" [{f.origin}]"
        print(f"{f.severity:8} {f.check:18} {f.location}{origin}")
        print(f"         {f.message} (references/{f.reference})")
    print(f"passed: {', '.join(passed)}")


def validate() -> int:
    """Every check's reference must name an existing heading in references/."""
    refs = Path(__file__).resolve().parent.parent / "references"
    broken = []
    for chk in CHECKS:
        name, _, anchor = chk.reference.partition("#")
        path = refs / name
        slugs = {
            re.sub(r"[^\w\- ]", "", line.lstrip("#").strip().lower()).replace(" ", "-")
            for line in (path.read_text().splitlines() if path.is_file() else [])
            if line.startswith("#")
        }
        if anchor not in slugs:
            broken.append(f"{chk.id}: references/{chk.reference}")
    print("\n".join(broken) or f"{len(CHECKS)} checks, all references resolve")
    return 1 if broken else 0


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("package", nargs="?")
    ap.add_argument(
        "--all", action="store_true", help="check every package and summarize repo-wide gaps"
    )
    ap.add_argument("--base", help="git ref to tag findings introduced or pre-existing")
    ap.add_argument("--format", choices=["table", "json"], default="table")
    ap.add_argument(
        "--validate", action="store_true", help="check that every check's reference heading exists"
    )
    args = ap.parse_args()
    if args.validate:
        return validate()
    top = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"], capture_output=True, text=True, check=True
    )
    head = Tree(Path(top.stdout.strip()))

    if args.all:
        report_all(head, args.format)
        return 0
    if not args.package:
        ap.error("give a package or --all")
    pkg = resolve(head, args.package)
    findings = run(head, pkg)
    if args.base:
        if head._git("rev-parse", "--verify", "--quiet", f"{args.base}^{{commit}}").returncode:
            sys.exit(f"--base {args.base!r} is not a commit here; fetch it or use --base HEAD")
        base = Tree(head.root, args.base)
        in_base = pkg.directory in {p.directory for p in packages(base)}
        findings = tag(findings, run(base, pkg) if in_base else [])
    findings.sort(key=lambda f: SEVERITY_ORDER[f.severity])
    report_one(pkg, Ctx(head, pkg).kind, findings, args.format)
    return (
        1 if any(f.severity == "Critical" and f.origin != "pre-existing" for f in findings) else 0
    )


if __name__ == "__main__":
    sys.exit(main())
