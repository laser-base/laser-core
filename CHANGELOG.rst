Changelog
=========

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
