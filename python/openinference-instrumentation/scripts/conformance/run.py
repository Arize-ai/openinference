# /// script
# requires-python = ">=3.14"
# dependencies = []
# ///
"""Run the OpenInference conformance MVP and print a console summary.

Pipeline:
  1. Start the mock LLM server (Anthropic + OpenAI + Google GenAI endpoints).
  2. Start `weaver registry live-check` against the OTel GenAI semconv registry.
  3. Run each instrumented provider script in sequence; they export OTLP traces to weaver.
  4. Stop weaver, parse its JSON output, and print the summary.

`uv` is used to manage Python dependencies for the sibling scripts via PEP 723
inline metadata. Weaver and the semantic-conventions registry are downloaded on
first run and cached under `~/.cache/oi-conformance/`.
"""

import argparse
import json
import os
import platform
import shutil
import socket
import subprocess
import sys
import tarfile
import tempfile
import time
import urllib.request
import zipfile
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
CACHE_ROOT = Path.home() / ".cache" / "oi-conformance"
SEMCONV_VERSION = "v1.40.0"
WEAVER_VERSION = "v0.22.1"
WEAVER_INACTIVITY_TIMEOUT = 90
PROVIDER_SCRIPTS = (
    "anthropic_conformance.py",
    "openai_conformance.py",
    "google_genai_conformance.py",
)


def _ensure_uv() -> str:
    uv = shutil.which("uv")
    if uv:
        return uv
    print(
        "ERROR: 'uv' is required but was not found on PATH.\n"
        "Install it from https://docs.astral.sh/uv/getting-started/installation/",
        file=sys.stderr,
    )
    sys.exit(1)


def _allocate_free_ports(count: int) -> list[int]:
    """Bind `count` loopback sockets simultaneously so the OS assigns distinct ports."""
    sockets: list[socket.socket] = []
    try:
        for _ in range(count):
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.bind(("127.0.0.1", 0))
            sockets.append(sock)
        return [sock.getsockname()[1] for sock in sockets]
    finally:
        for sock in sockets:
            sock.close()


def _is_healthy(url: str) -> bool:
    try:
        urllib.request.urlopen(url, timeout=2)
        return True
    except Exception:
        return False


def _wait_for_health(url: str, label: str, proc: subprocess.Popen, timeout: int) -> None:
    for _ in range(timeout):
        if _is_healthy(url):
            return
        if proc.poll() is not None:
            raise RuntimeError(f"{label} died during startup")
        time.sleep(1)
    raise RuntimeError(f"{label} failed to become ready after {timeout}s")


def _ensure_semconv_registry() -> Path:
    cache = CACHE_ROOT / "semconv" / SEMCONV_VERSION
    model = cache / "model"
    if model.is_dir():
        return model
    cache.parent.mkdir(parents=True, exist_ok=True)
    if cache.exists():
        shutil.rmtree(cache)
    print(f"Cloning semantic-conventions {SEMCONV_VERSION}...")
    subprocess.run(
        [
            "git",
            "clone",
            "--branch",
            SEMCONV_VERSION,
            "--depth",
            "1",
            "-q",
            "https://github.com/open-telemetry/semantic-conventions.git",
            str(cache),
        ],
        check=True,
    )
    return model


def _weaver_asset_name() -> str:
    machine = platform.machine().lower()
    if sys.platform == "darwin" and machine in {"arm64", "aarch64"}:
        return "weaver-aarch64-apple-darwin.tar.xz"
    if sys.platform == "darwin" and machine in {"amd64", "x86_64"}:
        return "weaver-x86_64-apple-darwin.tar.xz"
    if sys.platform == "linux" and machine in {"amd64", "x86_64"}:
        return "weaver-x86_64-unknown-linux-gnu.tar.xz"
    if sys.platform == "win32" and machine in {"amd64", "x86_64"}:
        return "weaver-x86_64-pc-windows-msvc.zip"
    raise RuntimeError(f"Unsupported platform for weaver: {sys.platform} / {machine}")


def _ensure_weaver() -> Path:
    binary_name = "weaver.exe" if sys.platform == "win32" else "weaver"
    on_path = shutil.which(binary_name)
    if on_path:
        return Path(on_path)

    install_dir = CACHE_ROOT / "weaver" / WEAVER_VERSION
    cached = next(install_dir.rglob(binary_name), None) if install_dir.exists() else None
    if cached and cached.is_file():
        return cached

    asset = _weaver_asset_name()
    url = f"https://github.com/open-telemetry/weaver/releases/download/{WEAVER_VERSION}/{asset}"
    install_dir.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(dir=str(install_dir.parent)) as tmp:
        tmp_path = Path(tmp)
        archive = tmp_path / asset
        extract = tmp_path / "extract"
        extract.mkdir()
        print(f"Downloading weaver {WEAVER_VERSION}...")
        with urllib.request.urlopen(url) as resp, archive.open("wb") as out:
            shutil.copyfileobj(resp, out)
        if archive.suffix == ".zip":
            with zipfile.ZipFile(archive) as zf:
                zf.extractall(extract)
        else:
            with tarfile.open(archive, "r:*") as tf:
                tf.extractall(extract, filter="data")
        binary = next(extract.rglob(binary_name), None)
        if binary is None:
            raise RuntimeError("weaver binary not found in downloaded archive")
        if install_dir.exists():
            shutil.rmtree(install_dir)
        extract.copy(install_dir)

    binary = next(install_dir.rglob(binary_name), None)
    if binary is None:
        raise RuntimeError("weaver binary not found after install")
    if sys.platform != "win32":
        binary.chmod(binary.stat().st_mode | 0o111)
    return binary


def _stop_proc(proc: subprocess.Popen | None, label: str) -> None:
    if proc is None or proc.poll() is not None:
        return
    print(f"Stopping {label}")
    proc.terminate()
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()


def _request_weaver_stop(admin_port: int) -> None:
    try:
        urllib.request.urlopen(
            urllib.request.Request(f"http://127.0.0.1:{admin_port}/stop", method="POST"),
            timeout=5,
        )
    except Exception:
        pass


def _summarize(result_dir: Path) -> int:
    statistics: dict | None = None
    for path in sorted(result_dir.rglob("*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        for obj in data if isinstance(data, list) else [data]:
            match obj:
                case {"statistics": dict() as stats}:
                    statistics = stats
                case {"registry_coverage": _} | {"advice_level_counts": _}:
                    statistics = obj

    if statistics is None:
        print("\nERROR: weaver produced no statistics.", file=sys.stderr)
        return 1

    print("\n========== OPENINFERENCE GENAI CONFORMANCE SUMMARY ==========")
    coverage = statistics.get("registry_coverage")
    if isinstance(coverage, (int, float)):
        print(f"Registry coverage: {coverage:.1%}")

    def _print_seen(label: str, key: str) -> None:
        seen = {n: c for n, c in (statistics.get(key) or {}).items() if c > 0}
        if not seen:
            return
        print(f"\n{label} ({len(seen)}):")
        for name in sorted(seen):
            print(f"  - {name}: {seen[name]}")

    _print_seen("Registry attributes seen", "seen_registry_attributes")
    _print_seen("Non-registry attributes seen", "seen_non_registry_attributes")
    _print_seen("Registry metrics seen", "seen_registry_metrics")
    _print_seen("Registry events seen", "seen_registry_events")

    missing_genai = sorted(
        name
        for name, count in (statistics.get("seen_registry_attributes") or {}).items()
        if count == 0 and name.startswith("gen_ai.")
    )
    if missing_genai:
        print(f"\nMissing registry attributes (gen_ai.*) ({len(missing_genai)}):")
        for name in missing_genai:
            print(f"  - {name}")

    advice = statistics.get("advice_level_counts") or {}
    if advice:
        print("\nAdvice levels:")
        for level, count in advice.items():
            print(f"  - {level}: {count}")

    print("=============================================================\n")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", help="Path to a local semconv registry model dir")
    args = parser.parse_args()

    uv_bin = _ensure_uv()
    weaver_bin = _ensure_weaver()
    registry = args.registry or str(_ensure_semconv_registry())

    mock_port, weaver_port, admin_port = _allocate_free_ports(3)
    result_dir = SCRIPT_DIR / "results"
    if result_dir.exists():
        shutil.rmtree(result_dir)
    result_dir.mkdir()

    mock: subprocess.Popen | None = None
    weaver: subprocess.Popen | None = None
    try:
        print(f"Starting mock server on 127.0.0.1:{mock_port}")
        mock = subprocess.Popen(
            [
                uv_bin,
                "run",
                str(SCRIPT_DIR / "mock_server.py"),
                "--host",
                "127.0.0.1",
                "--port",
                str(mock_port),
            ],
        )
        _wait_for_health(f"http://127.0.0.1:{mock_port}/health", "mock server", mock, timeout=120)

        for script in PROVIDER_SCRIPTS:
            print(f"Resolving uv environment for {script}")
            subprocess.run(
                [uv_bin, "run", str(SCRIPT_DIR / script), "--prewarm"],
                check=True,
            )

        print(f"Starting weaver live-check (otlp:{weaver_port}, admin:{admin_port})")
        weaver = subprocess.Popen(
            [
                str(weaver_bin),
                "registry",
                "live-check",
                "-r",
                registry,
                "--format",
                "json",
                "--output",
                str(result_dir),
                "--otlp-grpc-port",
                str(weaver_port),
                "--admin-port",
                str(admin_port),
                "--inactivity-timeout",
                str(WEAVER_INACTIVITY_TIMEOUT),
            ],
        )
        _wait_for_health(f"http://127.0.0.1:{admin_port}/health", "weaver", weaver, timeout=60)

        env = {
            **os.environ,
            "MOCK_LLM_URL": f"http://127.0.0.1:{mock_port}",
            "OTEL_EXPORTER_OTLP_ENDPOINT": f"http://127.0.0.1:{weaver_port}",
        }
        provider_returncode = 0
        for script in PROVIDER_SCRIPTS:
            print(f"=== Running {script} ===")
            run = subprocess.run([uv_bin, "run", str(SCRIPT_DIR / script)], env=env)
            if run.returncode != 0:
                print(f"\nERROR: {script} exited with {run.returncode}", file=sys.stderr)
                provider_returncode = run.returncode
                break

        time.sleep(1)
        _request_weaver_stop(admin_port)
        try:
            weaver_returncode = weaver.wait(timeout=30)
        except subprocess.TimeoutExpired:
            weaver.kill()
            weaver_returncode = weaver.wait()
        weaver = None

        if provider_returncode != 0:
            return provider_returncode

        if weaver_returncode != 0:
            print(
                f"Note: weaver exited with {weaver_returncode} "
                "(non-zero typically means violations were reported).",
                file=sys.stderr,
            )

        return _summarize(result_dir)
    finally:
        _stop_proc(weaver, "weaver")
        _stop_proc(mock, "mock server")


if __name__ == "__main__":
    sys.exit(main())
