# CI and packaging

The script checks everything in this file. Read it to explain a finding or to fix one.

## Tox install pattern

`python/tox.ini` `commands_pre` for the package's factor must uninstall the pinned library,
reinstall the package from source, prove it imports, then install `test-requirements.txt`:

```
<pkg>: uv pip uninstall -r test-requirements.txt
<pkg>: uv pip install --reinstall-package openinference-instrumentation-<pkg> .
<pkg>: python -c 'import openinference.instrumentation.<pkg>'
<pkg>: uv pip install -r test-requirements.txt
```

The old `uv pip install --reinstall {toxinidir}/instrumentation/openinference-instrumentation-<pkg>[test]`
form under-resolves, so the pinned env can test the wrong library version. It is Critical
when the four steps above are missing, and Low when it is a leftover line before them. A
package with no `changedir` entry in `tox.ini` is never tested in CI, which is Critical.

## Latest env

The envlist carries both `py3{10,14}-ci-{<pkg>,<pkg>-latest}`, and the `-latest` factor
upgrades the library with `<pkg>-latest: uv pip install -U <library>`. Without it the
nightly canary never sees new library releases (High).

## Test requirements

`test-requirements.txt` sits in the package root and lists:
- the instrumented library pinned with `==` (High if not pinned)
- `opentelemetry-sdk` (Medium)
- `pytest-recording` when tests use `@pytest.mark.vcr` (High)
- any other test tools (`pytest-asyncio`, `respx`, `responses`)

A missing file is Critical, because the tox pattern depends on it.

## Packaging

Compare against `openinference-instrumentation-openai`:
- `src/openinference/instrumentation/<pkg>/version.py` defines `__version__`. The tox `ci`
  env fails without it (Critical).
- `_instruments = ("<library> >= <min>",)`, usually in `package.py`, is returned by
  `instrumentation_dependencies()`. The minimum must not be lower than the version pinned
  in `test-requirements.txt` (High).
- `pyproject.toml` registers the instrumentor under both
  `[project.entry-points.opentelemetry_instrumentor]` and
  `[project.entry-points.openinference_instrumentor]` (Medium).

Only `kind: instrumentor` packages are held to the `_instruments` and entry-point rules.
