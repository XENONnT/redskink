# Configuration file for the Sphinx documentation builder.

# -- Project information

import alea

project = 'Alea'
copyright = '2023, Alea contributors, the XENON collaboration'

release = alea.__version__
version = alea.__version__

# -- General configuration

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.intersphinx',
    'sphinx.ext.viewcode',
    'nbsphinx',
]

intersphinx_mapping = {
    'python': ('https://docs.python.org/3/', None),
    'sphinx': ('https://www.sphinx-doc.org/en/master/', None),
}
intersphinx_disabled_domains = ['std']

templates_path = ['_templates']

# -- Options for HTML output

html_theme = 'sphinx_rtd_theme'

# -- Options for EPUB output
epub_show_urls = 'footnote'

def setup(app):
    # Hack to import something from this dir. Apparently we're in a weird
    # situation where you get a __name__  is not in globals KeyError
    # if you just try to do a relative import...
    import os
    import sys
    sys.path.append(os.path.dirname(os.path.realpath(__file__)))
    from build_release_notes import convert_release_notes
    convert_release_notes()
