import os

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.coverage",
    "sphinx.ext.doctest",
    "sphinx.ext.extlinks",
    "sphinx.ext.ifconfig",
    "sphinx.ext.napoleon",
    "sphinx.ext.todo",
    "sphinx.ext.viewcode",
    "sphinx.ext.mathjax",
]
source_suffix = {".rst": "restructuredtext"}
master_doc = "index"
project = "LASER"
year = "2023-2024"
author = "Institute for Disease Modeling"
copyright = f"{year}, {author}"
version = release = "0.7.0"

pygments_style = "trac"
templates_path = ["."]
extlinks = {
    "issue": ("https://github.com/InstituteforDiseaseModeling/laser/issues/%s", "#"),
    "pr": ("https://github.com/InstituteforDiseaseModeling/laser/pull/%s", "PR #"),
}
# on_rtd is whether we are on readthedocs.org
on_rtd = os.environ.get("READTHEDOCS", None) == "True"

if not on_rtd:  # only set the theme if we are building docs locally
    html_theme = "sphinx_rtd_theme"

html_use_smartypants = True
html_last_updated_fmt = "%b %d, %Y"
html_split_index = False
html_sidebars = {
    "**": ["searchbox.html", "globaltoc.html", "sourcelink.html"],
}
html_short_title = f"{project}-{version}"

# Napoleon settings (Napolean converts Google-style docstrings to reStructuredText)
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = False
napoleon_use_admonition_for_notes = False
napoleon_use_admonition_for_references = False
napoleon_use_ivar = True  # from Cookiecutter template, False is the default
napoleon_use_param = False  # from Cookiecutter template, True is the default
napoleon_use_rtype = False  # from Cookiecutter template, True is the default
napoleon_preprocess_types = False
napoleon_type_aliases = None
napoleon_attr_annotations = True

mathjax3_config = {"TeX": {"Macros": {"small": ["{\\scriptstyle #1}", 1]}}}

# Prevent the following warning:
# sphinx/builders/linkcheck.py:86: RemovedInSphinx80Warning: The default value for 'linkcheck_report_timeouts_as_broken' will change to False in Sphinx 8, meaning that request timeouts will be reported with a new 'timeout' status, instead of as 'broken'. This is intended to provide more detail as to the failure mode. See https://github.com/sphinx-doc/sphinx/issues/11868 for details.
#   warnings.warn(deprecation_msg, RemovedInSphinx80Warning, stacklevel=1)
linkcheck_report_timeouts_as_broken = False
