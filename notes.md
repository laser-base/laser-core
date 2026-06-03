# LASER Motivation and Priorities

## Goals

* Support elimination and eradication programs through answering questions related to spatial transmission of disease.
  * Measles
  * Polio
  * Malaria
* Enable IDM researchers to lead the way with cutting edge capabilities.
  * Implement new features and capabilities previously unavailable.
* Enable new scenarios for existing models
  * Increased performance – explore larger parameter space
  * Increased performance – explore larger population scenarios on existing hardware
* Enable IDM researchers to model large scale scenarios.
  * GPU based compute
  * Multi-core/multi-threaded compute
  * Cluster based compute
* Enable external partners and users to model relevant scenarios on readily available hardware.
  * Maximize performance on commonly available hardware (laptops <= 10 years old).
  * Performance on laptops w/Nvidia GPUs.
  * Consider performance on Apple Silicon devices.
* Design for quick prototyping and extensibility.
  * Enable internal investigation into new features and capabilities
  * Enable timely development of necessary features
  * Enable external partners and users to extend and customize tools for their scenarios

## Technology

* Python w/NumPy
  * Numba as appropriate
  * PyCUDA as appropriate and possible
  * C++ or Rust for specific, high performance capabilties
* Agent based communities
  * Homogeneously mixed
  * Agent based communities with cohorts
  * Agent based communities with explicit networks
  * Stochastic, compartmental communities
* Flexible spatial connectivity component
  * “scoops” of contagion, i.e. “anonymous” contagion, e.g. measles
  * Traceable/trackable agents and infection, e.g. malaria vector or parasite genetics
  * Layered connectivity – local, regional, etc.

## Processes

* Documentation
* Scenarios and examples
* Test suite

## Other

* multiple independent species, e.g. humans, mosquitos, dogs, ...
* multiple independent transmission networks, intra-mural and inter-state
* event based modeling, e.g. draw for non-disease death date
* state machine based modeling

----

* toolset vs. builtin - e.g. is building network a tool in the toolset or a function in the model setup?
* relationship w/calibration

## Edward on Model Architecture and Light Agent Spatial Models

https://wiki.idmod.org/pages/viewpage.action?pageId=195791084

> The spatial dimension is critical in modeling transmission from endemic conditions through to the last cases before elimination. In some cases, it's also important to explicitly model arbitrary details of individuals, their behavior, and the combinations of interventions they have ("heavy agent"). In other cases, questions are more about meta-population dynamics and village-scale variation in connectivity, vaccination, surveillance, etc. and more simplistic representations of individual state ("light agent") can improve performance, configurability, interpretability. We expect to scope + explore some of the design considerations within the Architecture Function, and we would expect the Advanced Spatial Modeling cluster to drive a lot of the problem-space assessment as well as the engineering work.

* usability (API, syntax, portability, installation+configuration)
* performance (algorithms, approximations)
* generalizability (code design)
* scientific validity (context relevance, dynamical system behavior)

* calibration as a 1st class consideration

## Technicalities

|Minimal Payload|bytes|
|---------------|:---:|
| immune status (T/F) | 1 |
| infectious timer (\< 256 days) | 1 |
| &Sigma; | **2** |

|Realistic Payload|bytes|
|-----------------|:---:|
| age of DoB (115 years = 41,975 days) | 2 |
| susceptibility (would use float16 on GPU) | 4 |
| accessibility | 1 |
| SES/nutrition status | 1 |
| exposed timer (\< 256 days) | 1 |
| infectious timer (\< 256 days) | 1 |
| age at infection | 2 |
| community ID (up to 65K communities) | 2 |
| &Sigma; | **14** |

|Population|Minimal Payload<br>(2 bytes)|Minimally Realistic<br>Payload (14 bytes)|CPAS*|pass/sec†|notes|
|:--------:|:---------------------:|:---------------:|:-:|:-:|:-:|
|1M  (10<sup>6</sup>)|2MB|14MB|2,000|23,000/~3,300||
|10M (10<sup>7</sup>)|20MB|140MB|200|2,300/~330|megacity|
|100M (10<sup>8</sup>)|200MB|1.4GB|20|230/~33||
|200M (2x10<sup>8</sup>)|400MB|2.8GB|10|115/~16|Nigeria (231M)<br>Pakistan (242M)|
|1G (10<sup>9</sup>)|2GB|14GB|2|23/~3.3||

*CPAS - cycles/agent/second (2GHz CPU). SIMD instructions could be a multiplier on the number of instructions/cycle, e.g. AVX on Intel with 256-bit registers can operate on 8 float32s simultaneously. Apple Silicon supports ARM NEON with 128-bit registers.

†How many passes over memory/second based on memory bandwidth (Lenovo ~46GB/s, Apple Silicon up to 800GB/s).

### PRNG State Considerations

PRNG state appears to use between 16 and 268 bytes depending on the algorithm. CPU based modeling can probably allocate state for each CPU code in use (vanishingly small memory use). GPU based modeling will probably need to group agents to use draws from a shared PRNG to amortize the memory usage across several agents. E.g., 16 agents using the same PRNG with 16 bytes of state only incur +1 byte of state/agent.

|Algorithm|State Size (bytes)|
|:--------|:----------------:|
|XOROSHIRO128+| 16 |
|other XOROSHIRO variations| 16+ |
|XORWOW| 44 |
|MRG32k3a| 44 |
|Sobel32| 140 |
|Sobel64| 268 |

## Cohorts

- How many properties are tracked?
- How do cohorts progress over time?
- How do cohorts interact with other entities/agents?
- How do agents exit the cohort?

## Other

### Compose Rather than Configure

Choose the necessary components for the model and for reporting rather than enabling/disabling features from a large feature set.

## Migration module — historical implementation notes

Moved here from the top of `docs/migration.rst` so the user-facing reference page
stays focused on the model semantics. These are the original developer-side
considerations from when the migration module was written; most are now resolved
in the implementation:

- **Vectorization vs. loops.** "I'm aiming to do most of this with element-by-element
  numpy functions when I can, though loops would probably translate more obviously to
  numba/c. I don't expect that computation of these matrices will be a substantial
  part of overall computational spend on a model either way."
  **Status:** all four migration models and `distance` are now fully vectorized
  NumPy; see `docs/migration.rst` Performance characteristics for the matrix.

- **Input validation.** "I have not tested nor written code to enforce conditions
  on the inputs. This should be done — e.g., populations coming in as integers can
  present wrapping issues when we start exponentiating and multiplying them (signed
  integers in particular can be a problem because you may wrap into negative numbers)."
  **Status:** `_sanity_checks` (and the shared `_validation` helpers) now validate
  inputs at every public migration entry point, raising `TypeError` / `ValueError`.
  Populations are promoted to `np.float64` inside each model before exponentiation
  to avoid integer overflow.

- **Numerical precision.** "It's also worth investigating whether using large
  floats makes sense when computing these formulas, or whether we should put
  operations in a specific order. This is because depending on the choice of
  someone's metapop network and spatial model parameters, and the order of
  computations, we can end up multiplying, dividing, summing over numbers that
  can be across really different scales, so weirdness might happen with loss
  of precision?"
  **Status:** still a known concern at large `N` × extreme parameter values;
  the bit-equivalence regression tests in `tests/test_migration.py` lock the
  current ordering against a loop-based reference within `rtol=1e-10`.

- **Zero distances.** "Distances on the diagonal of the distance matrix should
  always be 0. We should check for 0s elsewhere and throw an error. It's also
  nice to be able to use numpy element-by-element math without constant
  div-by-zero errors for the diagonal elements, so maybe each function should
  start by adding epsilon to the diagonal of the distance matrix? We're going
  to zero out those terms in the network anyway..."
  **Status:** each migration function copies the distance matrix and sets the
  diagonal to 1 before exponentiating, then zeroes the network diagonal at the
  end. Off-diagonal zeros are still the user's responsibility (documented in
  the "Numerical edge cases to watch for" section of `docs/migration.rst`).
