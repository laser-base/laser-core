===========================
Vital Dynamics Model
===========================

Overview
========
This module implements a vital dynamics model using the LASER framework. The model simulates
population dynamics, including births and deaths, over a specified period. The key point with vital dynamics in LASER is that we don't want to add to or remove from our agent population so we preallocate inactive or preborn agents in anticipation of birthing/activating them mid-simulation, and with deaths we don't actually remove them or deallocate memory. One possible solution is a constant population scenario where every death results in a birth, so the agents are recycled in place. That has some of its own complications and is not the general case, so we are not doing that here. Key features include:

- **Births**: New agents are added to the population based on a crude birth rate (CBR).
- **Deaths**: Agents are marked as dead based on their predicted lifespan, using a Kaplan-Meier estimator.
- **Reporting**: Tracks births and deaths over time using LaserFrames.
- **Visualization**: Plots population trends over time.

Classes
=======
The following classes are implemented:

1. **VitalDynamicsModel**: The main model class that manages the population and coordinates components. You do not have to use an enclosing Model class but it's recommended. You can call it whatever you like.
2. **BirthsComponent**: Handles births in the simulation. Can be called whatever you like, but should have a constructor and a step function. Can also have reporting and/or visualization methods.
3. **DeathsComponent**: Manages deaths in the simulation. Can be called whatever you like, but should have a constructor and a step function. Can also have reporting and/or visualization methods.

Functions
=========
The following utility functions are provided:

1. **create_cumulative_deaths**: Generates a cumulative deaths array for use with the Kaplan-Meier estimator. In practice, you will probably provide cumulative deaths data from a file.

Sections
========

1. **Model Class**:
   - Describes and intializes the population and tracks its dynamics. Holds the main methods which manage population lifecycle.
2. **Births Component**:
   - Handles the addition/activation of new agents.
3. **Deaths Component**:
   - Handles agent removal/deactivation based on their lifespan.
4. **Utility Functions**:
   - Provides helper methods such as cumulative deaths generation.

Model Class
===========

Code for VitalDynamicsModel
---------------------------

.. note::

   ``calc_capacity`` returns a 1-D ``numpy.uint32`` array of per-node capacity
   estimates. Any per-node estimate that exceeds the maximum representable
   ``uint32`` value (``2**32 - 1`` ≈ 4.29 × 10\ :sup:`9`) is clamped to that
   maximum *before* the cast, so the returned array never silently overflows or
   wraps around. For scenarios with very large per-node populations (a single
   node modelling billions of agents), be aware that the returned capacity may
   underestimate the true growth — split the population across nodes or use a
   wider integer container yourself if you need the full count.

.. code-block:: python

   class VitalDynamicsModel:
       """
       Represents a vital dynamics model for simulating births and deaths in a population.

       Parameters
       ----------
       params : dict
           Dictionary containing simulation parameters:
           - `population_size`: int, initial population size.
           - `timesteps`: int, number of simulation timesteps.
           - `cbr`: float, crude birth rate per 1000 individuals per year.
       death_estimator : KaplanMeierEstimator
           Estimator used to predict lifespans based on a cumulative deaths array.
       pyramid : np.ndarray
           Population pyramid array, used to sample initial ages for the population.
       sampler : AliasedDistribution
           Distribution object for sampling initial age bins from the pyramid.

       Attributes
       ----------
       population : LaserFrame
           Frame containing properties of the population, such as `date_of_birth`, `date_of_death`, and `alive`.
       report : LaserFrame
           Frame for tracking births and deaths over time.
       components : list
           List of components (e.g., `BirthsComponent`, `DeathsComponent`) added to the model.
       """

       def __init__(self, params, death_estimator, pyramid, sampler):
           """
           Initialize the vital dynamics model and its population.
           """
           self.params = params
           # Add 1% 'fudge factor'
           capacity = int(1.01*calc_capacity(params["population_size"], params["timesteps"], params["cbr"]))

           # Initialize the population LaserFrame
           self.population = LaserFrame(capacity=capacity, initial_count=params["population_size"])
           self.population.add_scalar_property("date_of_birth", dtype=np.int32, default=-1)
           # date_of_death will be the simulation timestep where death occurs
           self.population.add_scalar_property("date_of_death", dtype=np.uint16, default=0)
           self.population.add_scalar_property("alive", dtype=np.int8, default=1)

           # Sample initial ages for the population
           n_agents = params["population_size"]
           samples = sampler.sample(n_agents)
           ages = np.zeros(n_agents, dtype=np.int32)

           for i in range(len(pyramid)):
               mask = samples == i
               ages[mask] = np.random.randint(
                   pyramid[i, 0] * 365, (pyramid[i, 1] + 1) * 365, size=mask.sum()
               )

           # Set date_of_birth and predict lifespans using Kaplan-Meier estimator
           dobs = -ages # for code clarity
           self.population.date_of_birth[:n_agents] = dobs

           lifespans = death_estimator.predict_age_at_death(ages, max_year=100)
           dods = lifespans - ages # we could check that dods is non-negative to be safe
           self.population.date_of_death[:n_agents] = dods
           # Note: We could set up a PriorityQueue with the date_of_death values sorted
           # while throwing away all those which don't lie in the realm of our simulation.
           # In this implementation we will be simpler but less efficient and check all
           # dods each timestep against tick.

           # Initialize a reporting LaserFrame for births and deaths
           self.report = LaserFrame(capacity=1)
           self.report.add_vector_property("births", length=params["timesteps"], dtype=np.int32)
           self.report.add_vector_property("deaths", length=params["timesteps"], dtype=np.int32)

           # Components (Births and Deaths)
           self.components = []

       def add_component(self, component):
           """
           Add a simulation component to the model.

           Parameters
           ----------
           component : object
               A component such as `BirthsComponent` or `DeathsComponent`.
           """
           self.components.append(component)

       def track_results(self, tick):
           """
           Record results from all components at the current timestep.

           Parameters
           ----------
           tick : int
               The current timestep.
           """
           for component in self.components:
               component.log(tick)

       def run(self):
           """
           Run the simulation for the specified number of timesteps.
           """
           for tick in range(self.params["timesteps"]):
               for component in self.components:
                   component.step(tick)
               self.track_results(tick)

       def plot_results(self):
           """
           Visualize the births and deaths over time as a plot.
           """
           plt.figure(figsize=(10, 6))
           plt.plot(self.report.births, label="Births", color="green")
           plt.plot(self.report.deaths, label="Deaths", color="red")
           plt.title("Vital Dynamics Over Time")
           plt.xlabel("Time (Days)")
           plt.ylabel("Count")
           plt.legend()
           plt.grid()
           plt.show()

Births Component
================

Code for BirthsComponent
-------------------------
.. code-block:: python

   class BirthsComponent:
       """
       Handles births in the simulation, adding new agents to the population.

       Parameters
       ----------
       model : VitalDynamicsModel
           The vital dynamics model.
       cbr : float
           Crude birth rate per 1000 individuals per year.

       Methods
       -------
       step(tick)
           Simulate births at the current timestep.
       log(tick)
           Record the number of births at the current timestep.
       """

       def __init__(self, model, cbr, death_estimator):
           self.population = model.population
           self.birth_rate_per_tick = cbr / (365 * 1000)
           self.report = model.report
           self.death_estimator = model.death_estimator

       def step(self, tick):
           births = int(self.birth_rate_per_tick * len(self.population))
           if births > 0:
               start, end = self.population.add(births)
               self.population.date_of_birth[start:end] = tick
               newborn_ages = np.zeros(births, dtype=np.int32)
               lifespans = self.death_estimator.predict_age_at_death(newborn_ages,max_year=100)
               self.population.date_of_death[start:end] = lifespans + tick
               self.population.alive[start:end] = 1

       def log(self, tick):
           births = int(self.birth_rate_per_tick * len(self.population))
           self.report.births[tick] = births

Deaths Component
================

Code for DeathsComponent
-------------------------
.. code-block:: python

   class DeathsComponent:
       """
       Handles deaths in the simulation, marking agents as dead based on their predicted date_of_death.

       Parameters
       ----------
       model : VitalDynamicsModel
           The vital dynamics model.
       death_estimator : KaplanMeierEstimator
           Estimator used to predict lifespans.

       Methods
       -------
       step(tick)
           Simulate deaths at the current timestep.
       log(tick)
           Record the number of deaths at the current timestep.
       """

       def __init__(self, model, death_estimator):
           self.population = model.population
           self.report = model.report

       def step(self, tick):
           alive = self.population.alive[:self.population.count] == 1
           dying = alive & (self.population.date_of_death[:self.population.count] <= tick)
           self.population.alive[:self.population.count][dying] = 0

       def log(self, tick):
           deaths = (self.population.alive[:self.population.count] == 0) & \
                    (self.population.date_of_death[:self.population.count] == tick)
           self.report.deaths[tick] = deaths.sum()

Utility Functions
=================

Code for Utility Functions
--------------------------
.. code-block:: python

   def create_cumulative_deaths(total_population, max_age_years):
       """
       Generate a cumulative deaths array with back-loaded mortality.

       Parameters
       ----------
       total_population : int
           Total population size.
       max_age_years : int
           Maximum age in years for the cumulative deaths array.

       Returns
       -------
       cumulative_deaths : np.ndarray
           Cumulative deaths array.
       """
       ages_years = np.arange(max_age_years + 1)
       base_mortality_rate = 0.0001
       growth_factor = 2
       mortality_rates = base_mortality_rate * (growth_factor ** (ages_years / 10))
       cumulative_deaths = np.cumsum(mortality_rates * total_population).astype(int)
       return cumulative_deaths

Simulation Parameters
~~~~~~~~~~~~~~~~~~~~~~

The simulation parameters are defined using the `PropertySet` class.

.. code-block:: python

    params = PropertySet({
        "population_size": 100_000,
        "cbr": 15, # Crude Birth Rate: 15 per 1000 per year
        "timesteps": 365*10 # Run for 10 years
    })

Running the Simulation
~~~~~~~~~~~~~~~~~~~~~~~

The model is initialized with the defined parameters, components are added, and the simulation is run for the specified timesteps. Results are then visualized.

.. code-block:: python

    # Load example population pyramid
    laser.core_path = importlib.util.find_spec("laser.core").origin
    laser.core_dir = os.path.dirname(laser.core_path)
    pyramid_file = os.path.join(laser.core_dir, "data/us-pyramid-2023.csv")
    pyramid = load_pyramid_csv(pyramid_file)

    # Build cumulative deaths array
    sampler = AliasedDistribution(pyramid[:, 2])
    cumulative_deaths = create_cumulative_deaths(params["population_size"])

    # Initialize the model
    model = VitalDynamicsModel(params, death_estimator, pyramid, sampler )

    # Initialize and add components
    model.add_component(BirthsComponent(model, params["cbr"], death_estimator))
    model.add_component(DeathsComponent(model))

    # Run the simulation
    model.run()

    # Plot results
    model.plot_results()

Conclusion
----------

The Vital Dynamics example demonstrates how to use LASER's modular components to simulate realistic population dynamics over time, including births, deaths, and age-structured demographics. By combining the ``KaplanMeierEstimator`` for mortality predictions with dynamic birth rates and agent-based properties, this example highlights the flexibility and scalability of the LASER framework for demographic modeling. Users can extend this baseline example with additional components, such as migration or disease dynamics, to create more complex simulations tailored to their specific research questions. This example serves as a foundational building block for models requiring detailed population structure and temporal dynamics.
