"""Sphinx configuration for the openinference-instrumentation API reference."""

from openinference.instrumentation.version import __version__

project = "OpenInference Instrumentation"
author = "OpenInference Authors"
copyright = "Arize AI"
release = __version__

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "myst_parser",
]
source_suffix = {".rst": "restructuredtext", ".md": "markdown"}

autodoc_default_options = {"members": True, "undoc-members": True}
autodoc_typehints = "description"
autodoc_member_order = "bysource"

html_theme = "pydata_sphinx_theme"
html_title = f"{project} {release}"
