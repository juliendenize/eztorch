# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
# import os
# import sys
# sys.path.insert(0, os.path.abspath('.'))

import os
import sys

sys.path.insert(0, os.path.abspath("../../"))

import eztorch

# -- Project information -----------------------------------------------------

project = "torchaug"
copyright = "2023, Julien Denize"
author = "Julien Denize"
language = "en"

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    # "sphinx.ext.doctest",
    # "sphinx.ext.intersphinx",
    "sphinx.ext.todo",
    # "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
    "myst_parser",
    "sphinx.ext.viewcode",
    "sphinx_autodoc_typehints",
    "sphinx_autodoc_defaultargs",
    # "sphinx.ext.duration",
    # "sphinx_gallery.gen_gallery",
    # "sphinx_copybutton",
    # "beta_status",
]

# sphinx_gallery_conf = {
#     "examples_dirs": "../../gallery/",  # path to your example scripts
#     "gallery_dirs": "auto_examples",  # path to where to save gallery generated output
#     "backreferences_dir": "gen_modules/backreferences",
#     "doc_module": ("torchaug",),
#     "remove_config_comments": True,
# }

napoleon_use_ivar = True
napoleon_numpy_docstring = False
napoleon_google_docstring = True
autosummary_generate = True

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# The suffix(es) of source filenames.
# You can specify multiple suffix as a list of string:
#
source_suffix = {
    ".rst": "restructuredtext",
    ".txt": "markdown",
    ".md": "markdown",
}


# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []

# The version info for the project
version = f"{eztorch.__version__}"

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "furo"
html_title = "Eztorch"

# Theme options are theme-specific and customize the look and feel of a theme
# further.  For a list of options available for each theme, see the
# documentation.
html_theme_options = {
    "source_repository": "https://github.com/juliendenize/eztorch/",
    "source_branch": "main",
    "source_directory": "docs/source/",
}


rst_prolog = (
    """
.. |default| raw:: html

    <div class="default-value-section">"""
    + ' <span class="default-value-label">Default:</span>'
)
