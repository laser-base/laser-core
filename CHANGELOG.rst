Changelog
=========

Unreleased
----------

* Add ``benchmarks/local_compare.py`` — a local benchmark comparison tool
  that uses a git worktree to run the ``pytest-benchmark`` suite against a
  baseline ref (default ``main``) and then against HEAD on the same
  workstation, then prints a side-by-side comparison and optionally enforces
  a regression threshold. Designed to replace CI-based comparison for this
  project: GitHub Actions runner variance (30–50%) is well above any useful
  per-PR regression threshold, while back-to-back local runs cancel out
  hardware variance and let a tight threshold (e.g. ``min:15%``) be
  meaningful. The script copies the current branch's ``benchmarks/`` tree
  into the worktree so newly-added cases are still measured against the
  baseline, uses ``PYTHONPATH`` injection to point ``laser.core`` at the
  worktree source while the venv supplies all other deps, and cleans up the
  worktree on exit unless ``--keep-worktree`` is passed. Update
  ``benchmarks/README.md`` to document the new workflow. The advisory CI
  workflow at ``.github/workflows/benchmarks.yml`` is left in place
  (``continue-on-error: true``) and intentionally is **not** a regression
  gate.
* Add a derivation comment for the ``safety_multiplier`` heuristic in
  ``laser.core.utils.calc_capacity``: documents the GBM-headroom rationale,
  explains the ``sqrt(exp_mu_t) - 1`` scaling as a relative-deviation buffer
  whose magnitude is controlled by ``safety_factor`` (≈ number of "sigmas" of
  headroom), and flags the heuristic as a sizing tool rather than a
  calibrated statistical bound.
* Replace the developer-facing TODO-style preamble at the top of
  ``docs/migration.rst`` with a polished one-paragraph overview of the
  migration submodule and a single "Interpreting the output units" note (the
  user-relevant bullet from the prior preamble). Historical implementation
  notes (vectorization status, validation status, precision concerns,
  zero-distance handling) moved to ``notes.md`` with their current resolution
  status.
* Rewrite the ``PropertySet`` class docstring in Google / Markdown style
  consistent with the rest of the codebase: dropped the RST ``::`` blocks,
  added a `+= / <<= / |=` operator-semantics table, an ``Args`` /
  ``Raises`` block for the constructor, and three consolidated
  ``**Example**`` blocks (basic access, composition, save/load).
* Promote the migration-model extension story in ``docs/architecture.rst``
  from a worked example to a formally-specified protocol section: documents
  the ``model(pops, distances, **params) -> np.ndarray`` signature, the
  inputs / outputs / error-semantics contract, and the use of the shared
  ``laser.core._validation`` helpers. Updated the ``exponential_decay``
  worked example to follow the protocol explicitly (full input validation,
  return-shape guarantee, integration with ``build_network``).
* Fix stale ``distributions`` API references that pre-dated the factory + ``sample_floats``/``sample_ints`` pattern:

  - ``docs/usage.rst`` "Sampling from a distribution" snippet now shows the
    actual factory pattern and the new ``distributions.sample`` one-liner.
  - ``benchmarks/test_distributions_benchmarks.py`` rewritten to call the
    real API (``constant_float`` / ``uniform(low, high)`` / ``poisson(lam)``
    factories + ``sample_floats``/``sample_ints``). Verified end-to-end via
    ``pytest --benchmark-only``.
* Update ``LaserFrame.add``, ``LaserFrame.sort``, and ``LaserFrame.squash``
  ``Raises:`` docstrings: previously listed ``AssertionError`` but the
  implementations raise ``ValueError`` (for capacity overflow) and ``TypeError``
  (for type/shape mismatches via ``laser.core._validation``).
* Flesh out the ``SortedQueue`` class docstring with an overview, an
  ``Attributes:`` block (``indices`` / ``values`` / ``size``), a ``Raises:``
  section for capacity / empty-pop conditions, and a runnable ``**Example**``.
* Refresh the per-folder ``README.md`` files under ``src/laser/core/`` and
  ``src/laser/core/demographics/`` to reflect the current public surface:
  ``build_network``, the ``sample`` one-liner, the composition helpers
  (``mixture2`` / ``tick_modulated`` / ``node_modulated``), and the
  reworked ``distribute_population_skewed`` signature.
* Add a citation for the ``RE = 6371.0`` Earth-radius constant in the
  Haversine path of ``laser.core.migration.distance`` (IUGG mean Earth
  radius :math:`R_1 \\approx 6371.0088` km; Moritz, H. (2000), *J. Geodesy*
  74(1)); rounding to 6371.0 km is dominated by the spherical-Earth
  approximation already baked into the Haversine formula.
* Add ``laser.core.distributions.sample(fn_or_factory, n, *, dtype=None,
  tick=0, node=0, out=None, **factory_kwargs)`` — a one-liner sampler that
  allocates the output buffer and dispatches to ``sample_floats`` or
  ``sample_ints`` based on dtype. When ``factory_kwargs`` are supplied, the
  first argument is treated as a factory and invoked with those kwargs to
  build the sampler on the fly, so per-call distribution parameters do not
  need to be baked into the factory ahead of time. Seven new tests in
  ``TestSampleConvenience`` cover pre-built and factory paths, explicit
  ``out``/``dtype``, ``tick``/``node`` forwarding, and the non-numeric-dtype
  rejection.
* Switch the default ``population_fn`` in ``laser.core.utils.grid`` from
  ``np.random.uniform`` to ``laser.core.random.prng().uniform``, so the
  default population draws now honor the project-wide seeding contract
  documented in ``CLAUDE.md``.
* Add a ``Raises:`` section to ``_cumulative_at_or_closer_2d`` documenting
  that ``ValueError`` propagates from ``np.cumsum`` / ``np.take_along_axis``
  when called directly with mismatched-shape inputs (the in-package callers
  validate via ``_sanity_checks`` before delegating).
* Expand the ``LaserFrame`` class docstring from a single sentence into a
  proper overview covering the property kinds (scalar / vector / array),
  the preallocate-never-realloc design, the lifecycle helpers, and the
  documented extension points.
* Add sampler-composition helpers to ``laser.core.distributions``:
  ``mixture2(sampler_a, sampler_b, p_a)`` for two-component mixtures,
  ``tick_modulated(base_sampler, modulator)`` for tick-periodic multiplicative
  modulation (e.g. seasonality), and ``node_modulated(base_sampler, modulator)``
  for per-node multipliers. All three return Numba-compatible samplers usable
  inside ``sample_floats``/``sample_ints``. Eight new tests in
  ``tests/test_distributions.py::TestCompositionHelpers`` cover proportions,
  branch recovery, modulator-scale correctness, input validation, and the
  composition-chains case.
* Add a Claude Code skill manifest at ``.claude/skills/laser-core.md`` that
  summarizes the public API, dev workflow, and required conventions for AI
  coding assistants. Add a "Support and contact" section to ``README.rst``
  surfacing the issue tracker, maintainer emails, ``CONTRIBUTING.rst``, and
  the AI orientation files.
* Add the ``interrogate`` docstring-coverage gate to
  ``.pre-commit-config.yaml`` with configuration in ``[tool.interrogate]`` in
  ``pyproject.toml``. Floor set to ``fail-under = 95`` (current coverage
  97.9%). Fill in remaining module / class docstrings on ``extension.py``,
  ``cli.py:main``, ``demographics/__init__.py``,
  ``demographics/spatialpops.py``, and ``KaplanMeierEstimator`` so the gate
  passes on first run. Add ``interrogate`` to the ``dev`` optional-dependency
  set in ``pyproject.toml``.
* Add a "Choosing a model" section to ``docs/migration.rst`` summarizing what
  each migration model encodes, parameter intuitions, and numerical edge
  cases. Stays descriptive rather than prescriptive but flags the gravity
  model as the natural starting point (principle of least surprise).
* Swap the LaTeX formula blocks in the ``stouffer`` and ``radiation``
  docstrings in ``src/laser/core/migration.py`` — they had been attached to
  the wrong function. Implementations and ``docs/migration.rst`` were already
  correct; this only updates the inline docstrings.
* Rework ``distribute_population_skewed`` in
  ``src/laser/core/demographics/spatialpops.py`` to guarantee a minimum
  population for every node and to use the LASER-wide PRNG. Adds a new
  ``min_pop`` parameter (default 1000) with explicit feasibility validation:

  - urban share must satisfy ``(1 - frac_rural) * tot_pop >= min_pop``;
  - rural per-node average must satisfy
    ``frac_rural * tot_pop / (num_nodes - 1) >= min_pop`` (the tight case
    where it equals ``min_pop`` is permitted; the exponential surplus is then
    zero and every rural node lands at exactly ``min_pop``);
  - ``frac_rural == 0`` requires ``num_nodes == 1`` and ``frac_rural > 0``
    requires ``num_nodes >= 2`` (a rural budget needs a rural node, and vice
    versa).

  Replaces the prior ``1/U`` draw and the ``100/tot_pop`` magic clip with a
  reserve-then-distribute construction: reserve ``min_pop`` per rural node,
  draw the surplus from ``rng.exponential``, normalize the surplus to
  ``frac_rural - num_rural * min_pop / tot_pop``, and add the reservation
  back. Switches the random draw from ``np.random.rand`` to
  ``laser.core.random.prng()`` per project convention, so seeding via
  ``laser.core.random.seed`` now controls reproducibility. The sub-unit
  rounding residual is now absorbed on the urban node (previously on a rural
  node, which could dip the rural node below the floor). Updated
  ``tests/test_spatialpops.py`` accordingly — bumped legacy test populations
  to feasible values under the new default, switched seeding to
  ``laser.core.random.seed``, and added given/when/then cases for the floor
  invariant, the reproducibility contract, urban/rural infeasibility, the
  tight-budget case (every rural node pinned at ``min_pop``), the
  ``frac_rural`` / ``num_nodes`` coupling in both directions, the single-node
  degenerate case, ``min_pop`` validation, and a scipy Kolmogorov-Smirnov
  goodness-of-fit test (``tot_pop=10M``, ``num_nodes=1001``, ``min_pop=0``)
  asserting that the 1000 rural populations are consistent with an
  exponential distribution — a regression guard against future changes that
  silently swap the random-draw shape.
* Vectorize ``stouffer`` and ``radiation``: replace the per-source-node Python
  loop with a batched 2D ``_cumulative_at_or_closer_2d`` helper that does the
  per-row sort, cumulative-sum, and equidistant-tie repair across all rows in
  a single vectorized pass. Add five bit-equivalence regression tests (with and
  without ``include_home``, plus an explicit equidistant-destinations case) in
  ``tests/test_migration.py``.
* Sweep ``assert`` statements at API boundaries to raise ``ValueError`` /
  ``TypeError`` instead, so input validation survives ``python -O``. Affected:
  ``PropertySet.__iadd__`` / ``__ilshift__`` / ``__ior__`` in
  ``src/laser/core/propertyset.py`` and ``initialize_population`` in
  ``src/laser/core/utils.py``. The existing ``AssertionError`` test in
  ``test_utils.py`` was updated to ``ValueError`` to match.
* Add runnable ``**Example**`` blocks to every public distribution function in
  ``src/laser/core/distributions.py`` (beta, binomial, constant_float,
  constant_int, exponential, gamma, logistic, lognormal, negative_binomial,
  normal, poisson, uniform, weibull).
* Add a class-level docstring to ``AliasedDistribution`` and proper
  parameter-level docstrings to ``LaserFrame._save`` and ``LaserFrame._save_dict``.
* Add a ``Performance characteristics`` section to ``docs/migration.rst``
  summarizing the asymptotic cost and vectorization status of each migration
  model, and a ``Extension Points`` section to ``docs/architecture.rst`` with
  worked examples for subclassing ``LaserFrame``, extending ``PropertySet``,
  and writing a custom migration model.
* Add an advisory ``benchmarks.yml`` GitHub Actions workflow that runs the
  ``pytest-benchmark`` suite on pull requests touching ``src/laser/core/`` or
  ``benchmarks/`` and uploads the JSON results as a 90-day artifact. The
  workflow is intentionally ``continue-on-error`` until a baseline is committed.
* Vectorize ``competing_destinations`` (removed O(N^2) Python double loop) and
  ``distance`` (removed per-row Python loop); add bit-equivalence regression
  tests pinning the new implementations to the prior looped behavior.
* Add a ``benchmarks/`` directory with ``pytest-benchmark`` cases for the
  migration models, distributions, ``KaplanMeierEstimator``, and ``SortedQueue``.
* Surface ``demographics``, ``distributions``, ``migration``, ``random``, and
  ``utils`` submodules from ``laser.core.__init__`` so they can be reached
  without spelunking; flesh out ``docs/usage.rst`` with a runnable quick-start.
* Add per-folder ``README.md`` files under ``src/laser/core/``,
  ``src/laser/core/demographics/``, ``examples/``, and ``benchmarks/`` to
  document each area's public surface.
* Add a ``CLAUDE.md`` at the repo root summarizing project conventions, dev
  workflow, and the canonical public API surface (AI-assistant orientation).
* Replace bare ``assert`` calls in ``calc_capacity`` with ``ValueError`` so
  input validation survives ``python -O``.
* Extract the duplicated input-validation helpers (``_is_instance``,
  ``_is_dtype``, ``_has_shape``, ``_has_dimensions``, ``_has_values``) into a
  new package-private ``laser.core._validation`` module; ``migration`` and
  ``laserframe`` now import from it.
* Add docstrings to ``AliasedDistribution.__init__``, the
  ``alias``/``probs``/``total`` properties on ``AliasedDistribution``,
  ``_pdod`` in ``kmestimator``, and ``sum_populations_as_close_or_closer`` in
  ``migration``.
* Add lower-bound ``>=`` version pins on core runtime dependencies
  (``click``, ``numpy``, ``numba``, ``matplotlib``, ``pandas``, ``h5py``,
  ``geopandas``, ``shapely``) in ``pyproject.toml`` matching the versions
  currently exercised on CI / pinned in ``uv.lock``.

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
  - Update += to only accept new keys
  - Add <<= to update existing keys
  - Add |= to add or update keys
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
