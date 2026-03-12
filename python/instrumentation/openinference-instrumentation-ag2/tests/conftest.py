"""
Add openinference source packages to sys.path so the test can import
openinference.instrumentation.ag2 without a full pip install.
"""
import sys
import os

# openinference-instrumentation (core: OITracer, TraceConfig, etc.)
core_src = os.path.join(
    os.path.dirname(__file__),
    "..", "..", "..", "openinference-instrumentation", "src"
)
core_src = os.path.normpath(core_src)
if core_src not in sys.path:
    sys.path.insert(0, core_src)

# openinference-semantic-conventions
conventions_src = os.path.join(
    os.path.dirname(__file__),
    "..", "..", "..", "openinference-semantic-conventions", "src"
)
conventions_src = os.path.normpath(conventions_src)
if conventions_src not in sys.path:
    sys.path.insert(0, conventions_src)

# this package itself
pkg_src = os.path.join(os.path.dirname(__file__), "..", "src")
pkg_src = os.path.normpath(pkg_src)
if pkg_src not in sys.path:
    sys.path.insert(0, pkg_src)
