# pylint: skip-file
# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information
# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
# import sphinxgallery
sys.path.insert(0, os.path.abspath('../../'))
sys.path.insert(0, os.path.abspath('../../example_scripts'))

project = 'modelling'
copyright = '2023, Alvenir'
author = 'Alvenir'
release = '0.1'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ['recommonmark',
              'sphinx_rtd_theme',
              # 'sphinx.ext.napoleon',
              'sphinx.ext.autodoc',
              'sphinx.ext.autosummary',
              'sphinx_gallery.gen_gallery']



# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.

exclude_patterns = []

language = 'da'

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

# html_theme = 'alabaster'
html_theme = 'sphinx_rtd_theme'

html_theme_options = {
    'sticky_navigation': False
}

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
# html_static_path = ['_static']

autodoc_member_order = 'bysource'

sphinx_gallery_conf = {
     'examples_dirs': '../../example_scripts',     # path to your example scripts
     'gallery_dirs': 'auto_examples'
    #'filename_pattern': '/execute_',
    #'expected_failing_examples': ['../example_scripts/plot_recognize.py']# path where to save gallery generated examples
}
