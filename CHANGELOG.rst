Changelog
=========

Unreleased
----------

* ``calc_capacity()`` now returns a ``uint32`` array (previously ``int32``) and
  clamps per-node estimates that exceed ``2**32 - 1`` to the ``uint32`` maximum
  before casting, preventing silent overflow / wraparound for very large
  populations.
* Add ``test_un_member_states_2020_laserframe_exceeds_uint32_max`` exercising a
  ``LaserFrame`` whose total agent count exceeds ``uint32.max``, built from 2020
  UN population estimates for all 193 member states (one node per country). The
  test allocates ~24 GiB of property storage and self-skips on machines without
  sufficient free RAM. New fixture ``tests/data/un_member_states_2020.csv``.
* Add ``test_calc_capacity_rejects_negative_initial_pop`` covering the
  ``initial_pop >= 0`` validation in ``calc_capacity()`` (single negative,
  multiple negatives with offending values surfaced in the message, and a
  zero-allowed sanity case).
* ``calc_capacity()`` gains optional mortality. New keyword-only parameters
  ``deathrates`` (a 2D CDR matrix matching the ``birthrates`` shape; default
  ``None`` for backward compat) and ``mortality_safety_factor`` (default
  ``1.0``) switch the function from a cumulative-births bound to a
  peak-living bound appropriate for simulations that reclaim dead-agent
  slots via ``LaserFrame.squash``. Mortality is intentionally
  underestimated: only ``1 / (1 + mortality_safety_factor)`` of the death
  rate is credited against births, with the remainder held back as headroom
  against a lower-mortality realization. Per-node estimates floor at
  ``initial_pop`` when ``deathrates`` is provided. Math mirrors the
  ``calc_capacity_cdr`` function in
  `razer <https://github.com/clorton/razer>`_. New tests cover backward-
  compatibility (zero deaths == no deaths), monotonicity in
  ``mortality_safety_factor``, the net-shrinking floor, validation, and the
  keyword-only API surface.
* ``calc_capacity()`` deaths-aware path now bounds the **peak-across-time**
  net growth rather than the end-of-simulation cumulative. Fluctuating CBR /
  CDR (e.g. high CBR early followed by high CDR later) can produce an
  intermediate peak in living population well above the end-of-sim value;
  the previous formula used ``exp(sum_b - death_credit * sum_d)`` (an end-of-
  sim cumulative) and would have under-allocated against that peak. Now
  computes ``max_t cumsum(lamda_b - death_credit * lamda_d)``, floored at 0,
  and uses that exponent. Monotonic-growth scenarios (births dominate deaths
  every tick) are unchanged because peak == end when the cumulative is
  monotonic. New tests cover the spike-then-decline scenario (asserts new >
  end-of-sim) and the monotonic case (asserts new == end-of-sim).
  Illustrative script + plot under ``misc/calc_capacity_peak_vs_end.{py,png}``.
* Implemented ``LaserFrame.__contains__`` so that ``name in lf`` matches every
  name the LaserFrame documented access surface advertises, and nothing more:

  - ``True`` for property names added via ``add_scalar_property`` /
    ``add_vector_property`` / ``add_array_property`` (e.g. ``"age" in lf``).
  - ``True`` for the underscored backing arrays of registered scalar / vector
    properties (e.g. ``"_age" in lf``) — these are a documented "underlying
    array access" path (see the module docstring and
    ``test_underlying_array_access``). The rule is "an underscored name matches
    iff its public counterpart is a registered property", which automatically
    keeps internal state filtered.
  - ``True`` for non-underscored kwargs-set attributes (e.g.
    ``LaserFrame(N, start_year=1944)`` → ``"start_year" in lf``).
  - ``True`` for a user-added property whose public name itself begins with
    ``_`` (e.g. ``lf.add_scalar_property("_private_col")`` → ``"_private_col"
    in lf``) — the property registry is always honored, regardless of name.
  - ``False`` for internal state (``_count``, ``_capacity``, ``_properties``)
    — these live in ``__dict__`` but are not advertised columns or backing
    arrays. Direct access via ``lf._count`` etc. is unchanged; only the
    ``in`` advertisement is filtered.
  - ``False`` for methods and class-level ``@property`` descriptors
    (``count``, ``capacity``, ``sort``); use ``hasattr(lf, name)`` for that
    broader question.
  - ``False`` for non-string items, including unhashable ones (a short-circuit
    ``isinstance(item, str)`` check makes the test safe for lists / dicts).

  Eleven new tests pin the contract.

1.0.2 (2026-05-19)
------------------

* Prune the supported Python test matrix to 3.10 (oldest) and 3.14 (newest released).
* Remove the unused ``args`` parameter from ``kmestimator`` for a faster ``laser.core`` import.
* Update release/publish infrastructure and the GitHub Actions workflow.
* Remove deprecated files and tidy the project structure.
* Linter clean-up.

1.0.1 (2026-02-26)
------------------

* Add support for Python 3.14 now that Numba supports it.
* Fix snapshot capacity restoration and CBR handling for multi-node models (#355).
* Improve calibration documentation (#354).
* Documentation polish: fix README formatting, fix the LASER documentation link,
  clarify install/build instructions, and bring CHANGELOG up to date.
* Update copyright year in ``LICENSE``.
* Update project URLs and the required ``uv`` version.
* Remove macOS x86_64 entries from the CI configuration.
* Make Sphinx happier.

1.0.0 (2025-12-16)
------------------

* Fix documentation link in README
* Major version release with stable API

0.9.1 (2025-12-09)
------------------

* Fix bug in initialize_population() function - wasn't handling lists correctly

0.9.0 (2025-12-09)
------------------

* Add initialize_population() function to distribute population among states
* Grid creates default state columns
* Update OS versions in CI workflow
* Add tests for new population initialization features

0.8.1 (2025-12-08)
------------------

* Fix PropertySet to correctly handle lists or tuples of key value pairs
* Improve PropertySet docstrings and documentation
* Add test for PropertySet constructor exception
* Fix up docstrings for MkDocs
* Replace SortedQueue with Distributions in non-auto API reference

0.8.0 (2025-11-25)
------------------

* Implement new calc_capacity() function using geometric Brownian motion
* Update README to reference laser-generic package
* Switch MacOS runner to macos-15-intel
* Address caching issues with sample functions
* CI improvements

0.7.0 (2025-10-30)
------------------

* Add Numba-compatible distributions for model waiting times
* Change distribution signatures to take tick and node parameters
* Cache in-proc with lru_cache rather than Numba caching
* Remove redundant count from samplers
* Validate max_year argument to predict_year_of_death()
* Default properties to current count of active agents
* Remove deprecated calc_distances() in favor of distance()
* Add grid generation utility function
* Enable Python 3.13 support
* Allow NumPy integers in LaserFrame()
* Add describe() function to LaserFrame
* Update LICENSE copyright
* Adapt docs to shared top level domain change
* Update docstring formatting for mkdocs build

0.6.0 (2025-07-24)
------------------

* Fix snapshot capacity restoration
* Fix cap reload functionality
* Various bug fixes for snapshot handling

0.5.1 (2025-06-05)
------------------

* Bug fixes for version 0.5.0

0.5.0 (2025-06-02)
------------------

* Add save_snapshot and load_snapshot functionality
* Add additional examples for PropertySet operators
* Improve PropertySet functionality:
  - Update ``+=`` to only accept new keys
  - Add ``<<=`` to update existing keys
  - Add ``|=`` to add or update keys
* Fix overflow in migration matrix calculations
* Update row_normalizer() to return float32
* Update calibration documentation
* Add performance section to documentation
* Add spatial examples and migration samples
* Build wheels with GitHub Actions

0.4.1 (2025-05-27)
------------------

0.4.0 (2025-01-31)
------------------

* Remove NumPy and Numba version restrictions

0.3.0 (2025-01-28)
------------------

* Add spatial reference support
* Expose functionality directly from laser_core namespace
* Enhance distance() function options
* Implement add_array_property
* Promote methods to laser_core
* Add unit tests for utility functions
* Cleanup pyproject.toml

0.2.0 (2025-01-15)
------------------

* Add documentation content for architecture and design
* Point to PyPI package (not test site)
* Documentation enhancements for GPT integration
* Additional badges for PyPI

0.1.1 (2024-11-19)
------------------

* Downgrade Numba dependency to 0.59.1
* Point doc links to IDM sites
* Update badge links

0.1.0 (2024-11-18)
------------------

* Switch to bump-my-version for version management
* Update NumPy and Numba versions for Python 3.12 support
* Fix argsort() with kind="stable" for newer NumPy
* Update migration tests to use np.int64
* Update documentation

0.0.3 (2024-11-05)
------------------

* Add __setitem__ to PropertySet
* Add __getitem__ to PropertySet
* Modify coverage configuration to ignore if __name__ == "__main__"

0.0.2 (2024-11-04)
------------------

* Add SortedQueue types
* Add Kaplan-Meier Estimator for predicting age/year of death
* Add AliasedDistribution class for drawing from population pyramids
* Add LaserFrame class (renamed from Population)
* Add PropertySet class
* Add migration models and functions
* Add random number seed functions
* Enable testing on arm64 (Apple Silicon)
* Build fixes and improvements
* Extensive documentation cleanup
* Update GitHub action versions
* Fix author and copyright information
* Add many unit tests

0.0.1 (2023-11-18)
------------------

* Initial package structure
