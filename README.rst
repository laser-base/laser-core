========
Overview
========

.. start-badges

.. list-table::
    :stub-columns: 1

    * - docs
      - |docs|
    * - tests
      - | |github-actions|
        | |codecov|
    * - package
      - | |version| |wheel| |supported-versions| |supported-implementations|
        | |commits-since|
.. |docs| image:: https://readthedocs.org/projects/idmlaser/badge/?style=flat
    :target: https://docs.idmod.org/projects/laser/en/latest/
    :alt: Documentation Status

.. |github-actions| image:: https://github.com/laser-base/laser-core/actions/workflows/github-actions.yml/badge.svg
    :alt: GitHub Actions Build Status
    :target: https://github.com/laser-base/laser-core/actions

.. |codecov| image:: https://codecov.io/gh/InstituteforDiseaseModeling/laser/branch/main/graphs/badge.svg?branch=main
    :alt: Coverage Status
    :target: https://app.codecov.io/github/InstituteforDiseaseModeling/laser

.. |version| image:: https://img.shields.io/pypi/v/laser-core.svg
    :alt: PyPI Package latest release
    :target: https://pypi.org/project/laser-core/

.. |wheel| image:: https://img.shields.io/pypi/wheel/laser-core.svg
    :alt: PyPI Wheel
    :target: https://pypi.org/project/laser-core/

.. |supported-versions| image:: https://img.shields.io/pypi/pyversions/laser-core.svg
    :alt: Supported versions
    :target: https://pypi.org/project/laser-core/

.. |supported-implementations| image:: https://img.shields.io/pypi/implementation/laser-core.svg
    :alt: Supported implementations
    :target: https://pypi.org/project/laser-core/

.. |commits-since| image:: https://img.shields.io/github/commits-since/InstituteforDiseaseModeling/laser/v1.0.1.svg
    :alt: Commits since latest release
    :target: https://github.com/laser-base/laser-core/compare/v1.0.1...main



.. end-badges

Light Agent Spatial modeling for ERadication.

* Free software: MIT license

Installation
============

::

    pip install laser-core

You can also install the in-development version with::

    pip install https://github.com/laser-base/laser-core/archive/main.zip


Documentation
=============


https://docs.idmod.org/projects/laser/en/latest/


Development
===========

To run all the tests run::

    tox

Note, to combine the coverage data from all the tox environments run:

.. list-table::
    :widths: 10 90
    :stub-columns: 1

    - - Windows
      - ::

            set PYTEST_ADDOPTS=--cov-append
            tox

    - - Other
      - ::

            PYTEST_ADDOPTS=--cov-append tox
