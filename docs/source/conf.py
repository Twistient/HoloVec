# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys
from datetime import datetime

# -- Path setup --------------------------------------------------------------

# Add the project root to sys.path so autodoc can find the modules
sys.path.insert(0, os.path.abspath('../..'))

# -- Project information -----------------------------------------------------

project = 'HoloVec'
copyright = f'{datetime.now().year}, Twistient Corp.'
author = 'Twistient'
version = '0.1.0'  # Short version
release = '0.1.0'  # Full version

# -- General configuration ---------------------------------------------------

extensions = [
    # Core Sphinx extensions
    'sphinx.ext.autodoc',        # Auto-generate documentation from docstrings
    'sphinx.ext.autosummary',    # Generate summary tables
    'sphinx.ext.napoleon',       # Support for NumPy and Google style docstrings
    'sphinx.ext.viewcode',       # Add links to highlighted source code
    'sphinx.ext.intersphinx',    # Link to other project's documentation
    'sphinx.ext.mathjax',        # Render math via MathJax
    'sphinx.ext.todo',           # Support for todo items
    'sphinx.ext.coverage',       # Check documentation coverage
    'sphinx.ext.githubpages',    # Create .nojekyll file for GitHub Pages

    # Third-party extensions
    'myst_parser',               # Markdown support
    'sphinx_copybutton',         # Add copy button to code blocks
    'sphinx_design',             # Modern UI components (cards, tabs, etc.)
    'sphinx_gallery.gen_gallery',  # Generate example gallery
    # 'nbsphinx',                # Jupyter notebook support (disabled - requires pandoc)
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
exclude_patterns = []

# The suffix(es) of source filenames.
source_suffix = {
    '.rst': 'restructuredtext',
    '.md': 'markdown',
}

# The master toctree document.
master_doc = 'index'

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.
html_theme = 'shibuya'

# Theme options are theme-specific and customize the look and feel of a theme
# further.
html_theme_options = {
    # Color scheme - using cyan accent for dark mode, blue for light
    "accent_color": "cyan",

    # Top navigation bar links
    "nav_links": [
        {
            "title": "Getting Started",
            "url": "getting-started/quickstart",
        },
        {
            "title": "Tutorials",
            "url": "tutorials/index",
        },
        {
            "title": "User Guide",
            "url": "user-guide/index",
        },
        {
            "title": "API",
            "url": "api/index",
        },
        {
            "title": "Examples",
            "url": "examples/index",
        },
    ],

    # Social/external links
    "github_url": "https://github.com/twistient/holovec",

    # Layout options
    "globaltoc_expand_depth": 2,
    "dark_code": True,  # Use dark theme for code blocks

    # UI enhancements
    "light_logo": "_static/logo.svg",
    "dark_logo": "_static/logo.svg",

    # Navigation
    "nav_links_align": "left",

    # Page metadata
    "page_layout": "compact",  # More modern, compact layout
}

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

# Custom CSS files to load (loaded after theme CSS)
html_css_files = [
    'custom.css',
]

# The name of an image file (relative to this directory) to place at the top
# of the sidebar.
html_logo = "_static/logo.svg"

# The name of an image file (within the static path) to use as favicon of the
# docs.  This file should be a Windows icon file (.ico) being 16x16 or 32x32
# pixels large.
html_favicon = "_static/logo-icon.png"

# If true, links to the reST sources are added to the pages.
html_show_sourcelink = True

# If true, "Created using Sphinx" is shown in the HTML footer. Default is True.
html_show_sphinx = True

# If true, "(C) Copyright ..." is shown in the HTML footer. Default is True.
html_show_copyright = True

# Project title shown in the header
html_title = "HoloVec"

# Short title for the navigation
html_short_title = "HoloVec"

# -- Options for autodoc -----------------------------------------------------

# This value selects what content will be inserted into the main body of an
# autoclass directive.
autoclass_content = 'both'  # Include both class docstring and __init__ docstring

# This value selects if automatically documented members are sorted
autodoc_member_order = 'bysource'  # Keep source order

# Default options for autodoc directives
autodoc_default_options = {
    'members': True,
    'member-order': 'bysource',
    'special-members': '__init__',
    'undoc-members': True,
    'exclude-members': '__weakref__'
}

# Generate stub pages for autosummary
autosummary_generate = True

# -- Options for Napoleon (NumPy/Google style docstrings) -------------------

napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = False
napoleon_use_admonition_for_notes = False
napoleon_use_admonition_for_references = False
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_preprocess_types = False
napoleon_type_aliases = None
napoleon_attr_annotations = True

# -- Options for intersphinx -------------------------------------------------

# Link to other projects' documentation
intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
    'torch': ('https://pytorch.org/docs/stable/', None),
    'jax': ('https://jax.readthedocs.io/en/latest/', None),
}

# -- Options for MyST (Markdown) ---------------------------------------------

myst_enable_extensions = [
    "dollarmath",      # Support $...$ and $$...$$ for math
    "amsmath",         # Direct math support
    "deflist",         # Definition lists
    "fieldlist",       # Field lists
    "html_admonition", # HTML-style admonitions
    "html_image",      # HTML-style images
    "colon_fence",     # ::: fence for directives
    "smartquotes",     # Smart quotes
    "replacements",    # Text replacements
    "linkify",         # Auto-link URLs
    "substitution",    # Variable substitution
]

myst_heading_anchors = 3  # Auto-generate anchors for headings up to level 3

# -- Options for sphinx-gallery ----------------------------------------------

from sphinx_gallery.sorting import FileNameSortKey

sphinx_gallery_conf = {
    'examples_dirs': '../../examples',   # Path to example scripts
    'gallery_dirs': 'examples',          # Path to generated gallery
    'filename_pattern': r'/(plot_|)\d+_',  # Pattern for example files
    'ignore_pattern': r'/(notebooks|__)',  # Ignore patterns
    'plot_gallery': False,               # Don't execute examples (too slow/dependencies)
    'download_all_examples': False,      # Don't create download-all link
    'remove_config_comments': True,      # Remove sphinx-gallery config comments
    'show_memory': False,                # Don't show memory usage
    'within_subsection_order': FileNameSortKey,  # Sort by filename
    'default_thumb_file': None,          # No default thumbnail (will use first image if any)
    'line_numbers': True,                # Show line numbers in code blocks
    'reference_url': {
        'holovec': None,                   # Don't link to external holovec docs (we are it!)
    },
    'backreferences_dir': 'gen_modules/backreferences',  # Where to store backreferences
    'doc_module': ('holovec',),            # Modules to document
    'inspect_global_variables': False,   # Don't inspect (since not executing)
    'expected_failing_examples': [],     # Examples expected to fail
    'first_notebook_cell': None,         # No special first cell
    'last_notebook_cell': None,          # No special last cell
}

# -- Options for sphinx-copybutton -------------------------------------------

# Exclude prompts from being copied
copybutton_prompt_text = r">>> |\.\.\. |\$ |In \[\d*\]: | {2,5}\.\.\.: | {5,8}: "
copybutton_prompt_is_regexp = True
copybutton_only_copy_prompt_lines = True
copybutton_remove_prompts = True

# -- Options for todo extension ----------------------------------------------

# If true, `todo` and `todoList` produce output, else they produce nothing.
todo_include_todos = True

# -- Options for coverage extension ------------------------------------------

coverage_show_missing_items = True

# -- Options for nbsphinx ----------------------------------------------------

# Allow notebooks with missing kernels (don't execute, just render)
nbsphinx_execute = 'never'  # Never execute notebooks during build
nbsphinx_allow_errors = True  # Allow notebooks with errors
