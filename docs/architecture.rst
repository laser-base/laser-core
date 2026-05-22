==================================
Software Architecture & Design
==================================

Summary
=======

The LASER framework supports the development of agent-based, spatio-temporal infectious disease models, where the system is represented as a mutable dataframe. Each column corresponds to a numerical property (e.g., ``node_id``, ``age``, ``infection_status``, ``infectious_timer``), with rows representing individual agents. The ``node_id`` property is built-in to accommodate LASER's geospatial focus, though single-node simulations are possible by assigning a uniform ``node_id`` (e.g., ``0`` or ``10000``). (We may use the term patch interchangeably with node.)

Users define the set of agent properties and implement model behavior through **components**, which consist of an initialization function and a **step function**. Components process one or more properties and run at each timestep to update the 'dataframe'. They can be implemented in **NumPy** (default), **Numba**, or **C** with optimizations like **OpenMP** or **SIMD** for performance. Declarative behavior is encouraged, with step functions optionally described in SQL-like syntax for clarity and maintainability.

The modular architecture enables developers to easily extend the framework by adding custom properties and components, making LASER adaptable to diverse modeling requirements.

Principles
==========

The core principle of LASER's design is to optimize computational efficiency by aligning the system with what modern CPUs and GPUs excel at—performing billions of floating-point operations per second—while minimizing costly operations like runtime memory allocation and random memory access. To achieve this:

1. **Preallocate Memory:**
   All required memory (e.g., arrays, data structures) is allocated at initialization. This eliminates the need for dynamic memory allocations during runtime, which can introduce latency and fragmentation.

2. **Sequential Array Access:**
   Data is processed by iterating sequentially through preallocated arrays, ideally only once per timestep. This design ensures cache-friendly operations and minimizes the overhead of random memory access.

3. **Fixed Data Structures:**
   Instead of resizing data structures (e.g., appending to lists), the system works with fixed-size arrays where the data for all entities (e.g., agents, reports) is pre-allocated. For instance:

   - Agents are marked as "dead" rather than removed, allowing the array size to remain constant.
   - For births, "preborn" agents are included in the array from the start, with their activation deferred until the appropriate timestep.

4. **Time-Specific Data Slots:**
   Reports and outputs allocate data slots for every timestep and location in advance. This enables efficient insertion of results during runtime without requiring dynamic resizing.

By adhering to these principles, LASER achieves a highly efficient, scalable system, minimizing the bottlenecks caused by memory management and ensuring smooth timestep progression.

Layout
======

The ``laser-core`` module includes the following core components:

- **LaserFrame**: A custom dataframe class with additional methods tailored for LASER, such as ``add_scalar_property`` and ``add_vector_property``.
- **Demographics Utilities**: Tools to help initialize population demographic data, including dates of birth and expected dates of death.
- **SortedQueue**: A high-performance priority queue class, implemented in the ``sortedqueue`` submodule, for managing scheduled events like non-disease deaths.
- **PropertySet**: A "smart dictionary" implemented in the ``propertyset`` submodule that allows dot-notation access to dictionary keys.
- **Migration Module**: A submodule for modeling agent migration using approaches like gravity and radiation models.
- **Visualization Utilities**: Tools for visualizing and analyzing simulation results.

In addition to these core components, users will develop:

- **Top-Level Script**: An application layer, such as ``disease.py``, which orchestrates the simulation.
- **Components (Step Functions)**: Modular units that define the dynamics of the simulation, updating one or more agent properties (e.g., age, infection timers) at each timestep.


Application Layer
=================

The top-level script serves as the orchestrator for the simulation and consists of the following key components:

1. **Configuration Handling**:
   Gather configuration parameters from the user. This can be achieved via command-line arguments, a ``config.json`` file, or other input methods. For simple use cases, parameters such as the number of timesteps or the population per node can be hardcoded directly into the script.

2. **Model Initialization**:
   Create the main Model object, which acts as the container for the simulation's data. This includes the population dataframe (a ``LaserFrame`` object) and associated metadata or global variables. You can define a detailed Model class or just a minimal stub class that serves as a container.

   Below is an example of the minimal code required to initialize the simplest population dataframe:

   .. code-block:: python

       # Declare a very simple Model class to house our model pieces.
       class Model:
           pass

       # Initialize the model and its population
       from laser.core import LaserFrame
       model = Model()
       # Create the agent population with max size 1000
       model.population = LaserFrame(capacity=1000)
       # Add our properties, which can be thought of as the columns of our dataframe.
       model.population.add_scalar_property("disease_state")
       # Explicitly add the total population size, in this case the same as our max capacity
       model.population.add(1000)

   This code initializes a ``LaserFrame`` capable of holding 1000 agents, with a single property named ``disease_state``. The values for this property will be initialized later, such as through a specific component or during setup. This represents the simplest functional structure for running a simulation with a basic population and one agent-level property.


3. **Component Setup**:
   Import and initialize all components (also referred to as phase or step functions). Components define the simulation's logic, such as updating age, managing infections, or handling migration. If all component code resides within the same script, importing may not be necessary.

4. **Simulation Loop**:
   Execute the main simulation loop. For each timestep, call the step function for every active component in the order defined by your simulation logic. This loop progresses the simulation, updating agent properties and state variables. Counters to track (and record) model state are also updated during each step.

5. **Output and Analysis**:
   After the simulation completes, generate outputs such as reports, visualizations, or summary statistics. These outputs should provide insights into the simulation's results, such as disease spread, migration patterns, or demographic changes.


The Model Object
================

The ``Model`` object is the central data structure in LASER, encapsulating both agent and node-level information using ``LASERFrame``. This structure is a specialized dataframe designed for managing and updating model properties efficiently.

- **Agent Population**: Represented as a ``LASERFrame``, where each agent corresponds to a row and each property (e.g., age, infection status) is stored as a column. This allows for efficient per-agent computations during each timestep.

- **Node Data**: Another ``LASERFrame`` is used to manage node-level information.
  - **Input Values**: These are typically static scalar properties (e.g., geographic coordinates, demographic constants) provided at initialization.
  - **Output Values**: These include dynamic vector properties (e.g., total population or infected count at each timestep) that are updated as the simulation progresses.

As the ``Model`` evolves, additional data and methods may be incorporated into the class to better support the specific needs of your simulation. This flexibility ensures the ``Model`` can serve as a robust container for all simulation-related data and computations.


Components
==========

Components are modular units of functionality within the simulation, responsible for performing specific updates or computations on the agent population or node-level data. Each component is implemented as a class with an initialization function to set up any required state and a step function to execute the component's logic during each timestep.

As demonstrated in the "Model Initialization" section, the ``LaserFrame`` object contains the agent properties. Components operate on these properties to modify their values or derive new ones based on simulation logic.

Example: Infection Timer Component
----------------------------------
The example below shows a component that decrements the ``infection_timers`` property for all agents. When a timer reaches zero, the component sets the corresponding agent's ``susceptibility`` flag to reflect that they have recovered or gained immunity.

.. code-block:: python

    class InfectionTimerComponent:
        def __init__(self, model):
            self.population = model.population

        def step(self):
            timers = self.population.infection_timers
            susceptibility = self.population.susceptibility

            # Decrement all non-zero timers
            timers[:] = np.maximum(timers - 1, 0)

            # Update susceptibility based on timer state
            susceptibility[:] = np.where(timers == 0, 1, susceptibility)

After defining this component, it would typically be initialized and added to the simulation loop:

.. code-block:: python

    # Initialize the component
    infection_timer_component = InfectionTimerComponent(model)

    # Run the component step function during the simulation
    for timestep in range(total_timesteps):
        infection_timer_component.step()

Explanation
-----------
1. **Initialization**: The component retrieves a reference to the ``LaserFrame`` from the model. This allows direct access to the agent properties.
2. **Step Function**: The logic modifies the ``infection_timers`` array in place, ensuring that values do not go below zero, and updates the ``susceptibility`` flag based on timer state.
3. **Integration**: The component is called once per timestep, ensuring its behavior aligns with the simulation's temporal dynamics.

By defining components in this modular fashion, the LASER framework supports reusable and extensible functionality, allowing developers to add new behavior to simulations efficiently.


Input Files
===========
There is no requirement for any particular input files for laser-core. You're free to provide, load and parse input data in preferred formats for values such as input populations, age structure, fertility, mortality, and migration rates.

Output Files
============
`laser-core` does not output data to disk. It's up to you to collect and write csv or other data files as needed. HDF5 file format is preferred for large output files.

Demographics
============

Age Structure
-------------

  If you want to work with age structure for a short simulation which doesn't need births you can just give everyone an age (based on distribution) and increment it each timestep. The laser.core.demographics.pyramid module is provided to support the initialization of agents with plausible initial ages.


Births
------

Preborn Management in LASER
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

LASER's design philosophy emphasizes contiguous and fixed-size arrays, meaning all agents—both currently active and preborn—are created at the start of the simulation. Preborns are "activated" as they are born, rather than being dynamically added. Several approaches to handling preborns while adhering to these principles are outlined below:

1. **Negative and Positive Birthdays**:
   - Assign ``date_of_birth`` values in the past (negative) for active agents.
   - Assign ``date_of_birth`` values in the future (positive) for preborns.

2. **Unified Preborn Marker**:
   - Set all preborns' ``date_of_birth`` to a placeholder value (e.g., ``-1``).
   - Update the ``date_of_birth`` to the current timestep when a preborn is born.

3. **Active Flag Only** (if not modeling age structure):
   - If the model doesn't require age structure, you can skip ``date_of_birth`` entirely. Instead, use an ``active`` flag. Preborns start with ``active = False`` and are switched to ``active = True`` during the fertility step. This simplifies implementation while remaining consistent with LASER principles.

Calculating Age from Birthday
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If calculating age isn't frequent or essential, you can avoid explicitly tracking an ``age`` property. Instead, compute age dynamically as the difference between the current timestep (``now``) and ``date_of_birth``. For models that depend on age-specific dynamics (e.g., fertility rates by age group), consider adding a dedicated ``age`` property that updates at each timestep.

Deaths
------
  The recommended way of doing mortality in LASER is by precalculating a lifespan for each agent, rather than probabilistically killing agents as the simulation runs. This can take different forms: If you prefer to track agent age, you can also have an agent lifespan. Alternatively, if you are just using `date_of_birth` you can have a `date_of_death`, where theses 'dates' are really simulation times ('sim day of birth' and 'sim day of death'). Also, in LASER, as mentioned in the 'Principles' section, we strive to leave the contiguous arrays of agent data in place, without adding or deleting elements (allocating or freeing). This means that to model mortality, we prefer to 'kill' agents by doing either 1) check that their age is greater than their lifespan (or that the current timestep is greater than their 'sim day of death') in each component that cares, or 2) Set an active flag to false or a dead flag to true. The second approach is simpler, and avoids doing millions of comparison operations, at the cost of an additional property. Note that many component operations (step functions) can be done without checking whether the agent is alive, because, for example, as long as transmission never infects a dead person, decrementing all non-zero infection timers will only operate on live agents. Finally, while you can set lifespans using any algorith you want, laser.core.demographics.kmestimator is provided to support these calculations.


User Customizability
====================

1. **Config Params**
LASER doesn't have a set of pre-existing configuration params. You are free to add code to let the user set params like R-nought or simulation duration in code, in a settings file, on the command line, or even in environment variables. We suggest you collect these early in the sim and store them in a PropertySet which is then stored as a member of the model.

2. **Input Files**
LASER doesn't have a set of pre-defined input files or file formats but it's likely as you develop your model that you will want to load population data (by node/patch) and other demographics from csv files. This can provide a convient data-driven way of modifying model behavior.

3. **Code**
As discussed above, LASER modelers are expected to write their own application-level scripts and their own components.

New Modeler Workflow
====================

Here’s how you should break down your modeling problem to model a disease with LASER:

1. Figure out how your disease model maps to a set of agent properties.
2. Add code to add those properties to the population LASERFrame.
3. Figure out the updates you'll need to do each timestep, as declarations.
4. Add component code for each of those updates.

.. _extension-points:

Extension Points
================

LASER is intentionally a small set of composable primitives rather than a rigid
framework — most extension lives in *your* model script, not in a subclass tree
inside ``laser.core``. That said, three patterns recur often enough to call out:

Subclassing ``LaserFrame``
--------------------------

Subclass when you want a model-flavored frame that pre-registers its properties
and exposes domain methods. Avoid subclassing just to add helper methods that
could live on the model class itself.

.. code-block:: python

    import numpy as np
    from laser.core import LaserFrame

    class SIRFrame(LaserFrame):
        """A LaserFrame pre-populated with the properties an SIR model needs."""

        def __init__(self, capacity, initial_count=-1):
            super().__init__(capacity=capacity, initial_count=initial_count)
            self.add_scalar_property("state", dtype=np.uint8, default=0)  # 0=S, 1=I, 2=R
            self.add_scalar_property("infection_timer", dtype=np.int16, default=0)
            self.add_scalar_property("date_of_birth", dtype=np.int32, default=0)

        def infected_count(self) -> int:
            """Number of currently-infected agents in the active slice."""
            return int(np.count_nonzero(self.state == 1))

    frame = SIRFrame(capacity=1_000_000, initial_count=100_000)
    start, end = frame.add(0)  # already initialized; no new activation needed
    frame.state[:100] = 1      # seed 100 infections

Extending ``PropertySet``
-------------------------

``PropertySet`` is a flat parameter bag. The recommended pattern is to keep
parameters in a vanilla ``PropertySet`` and compose them with ``+=``, ``<<=``,
or ``|=`` rather than to subclass. If you do subclass — for instance to attach
validation or a ``to_yaml`` method — keep the public attributes flat so the
existing operators continue to work:

.. code-block:: python

    from laser.core import PropertySet

    class ValidatedSIRParams(PropertySet):
        """SIR parameters with a single ``validate`` hook called by your model setup."""

        def validate(self):
            if not (0.0 <= self.beta <= 10.0):
                raise ValueError(f"beta must be in [0, 10] (got {self.beta!r})")
            if not (0.0 <= self.gamma <= 1.0):
                raise ValueError(f"gamma must be in [0, 1] (got {self.gamma!r})")

    pars = ValidatedSIRParams({"beta": 0.4, "gamma": 0.1})
    pars |= {"seed": 20260514}   # composition still works
    pars.validate()

Adding a custom migration model
-------------------------------

The migration models in :mod:`laser.core.migration` are plain functions that
take ``(pops, distances, ...)`` and return a 2-D network. There is no plugin
registry — to add a model, just write a function with the same signature, and
optionally re-use the shared validation helpers in
:mod:`laser.core._validation` (package-private but stable for the package's
own code) or the public sanity-check pattern in :func:`laser.core.migration.gravity`.

.. code-block:: python

    import numpy as np
    from laser.core import migration

    def exponential_decay(pops, distances, k, scale, **_):
        """A custom migration model: flows decay exponentially with distance.

        network[i, j] = k * pops[j] * exp(-distance[i, j] / scale), with zero diagonal.
        """
        # Use the same boundary-validation pattern as the built-in models.
        if not isinstance(distances, np.ndarray) or distances.ndim != 2:
            raise TypeError("distances must be a 2D NumPy array")

        pops = pops.astype(np.float64)
        distances = distances.astype(np.float64)
        network = k * pops[np.newaxis, :] * np.exp(-distances / scale)
        np.fill_diagonal(network, 0)
        return network

    # Plug it in just like one of the built-ins:
    net = exponential_decay(pops, distances, k=0.01, scale=100.0)
    net = migration.row_normalizer(net, max_rowsum=0.05)

When in doubt, prefer **composition** (a regular function or model class that
*uses* ``LaserFrame``, ``PropertySet``, and the migration helpers) over
inheritance.

Glossary of Terms
=================
- **Patch**
  Something...
