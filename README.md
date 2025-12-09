<p align="center" width="100%">
<img src="https://user-images.githubusercontent.com/10873335/283954768-97685a6d-7b86-4bba-a3e6-07ac22d5a2b3.png" alt="LASER logo" width="600px"/>
</p>

## Status

[![documentation](https://readthedocs.org/projects/idmlaser/badge/?style=flat)](https://docs.idmod.org/projects/laser/en/latest/)

![tests](https://github.com/InstituteforDiseaseModeling/laser/actions/workflows/github-actions.yml/badge.svg)

[![package](https://img.shields.io/pypi/v/laser-core.svg)](https://pypi.org/project/laser-core/)
![wheel](https://img.shields.io/pypi/wheel/laser-core.svg)
![python versions](https://img.shields.io/pypi/pyversions/laser-core)
![implementation](https://img.shields.io/pypi/implementation/laser-core.svg)
![license](https://img.shields.io/pypi/l/laser-core.svg)

![commits since v0.9.0](https://img.shields.io/github/commits-since/InstituteforDiseaseModeling/laser/v0.9.0.svg)

## Getting Started

`laser-core` can be installed standalone with

```bash
python3 -m pip install laser-core
```

However, it may be more instructive to install one of the disease packages built on `laser-core` to understand what `laser-core` provides and what is expected to be in a disease model. See [`laser-generic`](https://github.com/laser-base/laser-generic).

### Documentation

Documentation can be found [here](https://docs.idmod.org/projects/laser/en/latest/) at the moment.

### Development

1. clone the `laser-core` repository with
```bash
git clone https://github.com/InstituteforDiseaseModeling/laser-core.git
```
2. install [`uv`](https://github.com/astral-sh/uv?tab=readme-ov-file#installation) _in your system [Python]_, i.e. _before_ creating and activating a virtual environment
3. install `tox` as a tool in `uv` with the `tox-uv` plugin with
```bash
uv tool install tox --with tox-uv
```
4. change to the `laser-core` directory with
```bash
cd laser-core
```
5. create a virtual environment for development with
```bash
uv venv
```
6. activate the virtual environment with

**Mac or Linux:**
```bash
source .venv/bin/activate
```

**Windows:**
```batch
.venv\bin\Activate
```

Now you can run tests in the `tests` directory or run the entire check+docs+test suite with ```tox```. Running ```tox``` will run several consistency checks, build documentation, run tests against the supported versions of Python, and create a code coverage report based on the test suite. Note that the first run of ```tox``` may take a few minutes (~5). Subsequent runs should be quicker depending on the speed of your machine and the test suite (~2 minutes). You can use ```tox``` to run tests against a single version of Python with, for example, ```tox -e py310```.

## Schedule

### First 30 Days (EOY 2023)

- [x] firm up team/stakeholders/advisory committee: **kmmcarthy, krosenfeld, clorton, jbloedow**
- [x] enumerate necessary features for reproducing/supporting previous and in-progress modeling efforts
  - [Required Model Features](https://github.com/InstituteforDiseaseModeling/laser/wiki/Required-Model-Features)
- <strike>enumerate necessary features for outstanding questions and issues</strike>

### First 60 Days (January 31, 2024)

- [x] "paper search" / investigate potential existing solutions

### First 120 Days (February 29, 2024)

- technical considerations
  - [x] single laptop
  - [x] single laptop w/Nvidia GPU
  - [x] multicore
    - [x] single machine
    - [x] large machine (cloud)
    - ¿beyond?
  - [x] Numpy
  - [x] NumPy + Numba
  - [x] NumPy + Numba + C/C++
  - NumPy + Numba + CUDA

## Problem Space

The problem is inherently an issue of heterogeneity. Spatial decomposition is the easiest consideration, but not sufficient - a model of N "independent" but identical communities is generally not useful.

Spatial connectivity and the associated latencies in transmission address one dimension of heterogeneity: how "close" is a given community to a potential source of imported contagion (exogenous to the model "world", locally endogenous, e.g., an adjacent community, endogenous but at a remove - rare transmission or multi-stop chain of transmission).

Community size in a spatial model is also a consideration - what is the configuration and connectivity of sub-CCS nodes to nodes at or above CCS for the given disease?

We need configurable characteristics of the individual communities which can vary, along with their interconnectedness, to capture additional heterogeneity.

What _is_ the modeling of the individual communities? "Light-Agent" seems to limit us to an ABM, but we should consider cohorts of epidemiologically similar populations (polio >5, HIV <15, TB latents, etc.) as well as stochastic compartmental models (XLA - eXtremely Light Agents).

- [ ] Are the individual communities well-mixed or should we also provide for explicit networks at the local level?

## Technology

- Python
- high performance computing:
  - [NumPy](https://numpy.org/)
  - [NUMBA](https://numba.pydata.org/)
  - [CuPy](https://cupy.dev/)
  - [PyCUDA](https://documen.tician.de/pycuda/)
  - [mlx](https://github.com/ml-explore/mlx)
- native code
  - C++ (somewhat awkward interop with Python, but potentially accessible from other technologies, e.g., R)
  - Rust ([PyO3](https://github.com/PyO3/pyo3) is quite nice, but requires getting up to speed on Rust 😳)
- compute requirements:
  - laptop 2010+? (might inform SIMD capabilities)
  - GPU (CUDA) enabled machine laptop/desktop/cloud
  - single core/multi-core
  - largest scenarios?
- visualization
  - cross-platform
  - real-time
- existing file formats for input data
- existing file formats for output data (GeoTIFF? - works with ArcGIS?)

## Other

- community builder tool for given total population and community size distribution
- network builder given a set of communities (gravity, radiation, other algorithms in existing libraries/packages)
- independent populations w/in a community, e.g., mosquitoes or dogs along with humans
- independent or co-transmission, i.e. multiple "diseases"
- models need to be connected with real-world scenarios, not [just] hypothetical explorations

## Notes

- "light" : How light is "light"?
- "agent" : Cohorts? Stochastic compartmental?
- "spatial" : How good are the individual community models? Good enough for non-spatial questions?
- dynamic properties (e.g. GPU flu simulation)
- ¿Ace/clorton-based state machines?

> Superficial simplicity isn’t the goal of design. Some things are, by nature, complex. In such cases, you should aim for clarity rather than “simplicity.” Users will be better served if you strive to make complex systems more understandable and learnable than simply simple.

-----

## Disclaimer

The code in this repository was developed by IDM and other collaborators to support our joint research on flexible agent-based modeling. We've made it publicly available under the MIT License to provide others with a better understanding of our research and an opportunity to build upon it for their own work. We make no representations that the code works as intended or that we will provide support, address issues that are found, or accept pull requests. You are welcome to create your own fork and modify the code to suit your own modeling needs as permitted under the MIT License.
