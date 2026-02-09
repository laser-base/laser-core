Calibration Workflow for LASER Models
=====================================

This guide explains how to calibrate a LASER model using Optuna. Calibration is the process of adjusting model parameters (e.g., transmission rate, R0) so that simulation outputs match reference data (e.g., case counts, prevalence curves).

**Important Principle:** Calibration only perturbs parameters within a fixed scenario structure. The scenario logic, geography, and core mechanisms remain constant—calibration searches for the parameter values that best fit observed data within that structure.

Prerequisites
-------------
Before beginning calibration, ensure you have:

- **A functioning, tested LASER model** - Your model must run successfully end-to-end with default parameters
- **Validated single simulation** - Always test a single model run before attempting calibration
- Python environment with `laser-core`, `optuna`, `pandas`, and `numpy` installed
- **Reference data** - Observed data (CSV format) with time series, case counts, prevalence, etc.
- **Clear parameter bounds** - Scientifically justified ranges for parameters to calibrate
- (Optional) Docker Desktop installed if running distributed calibration

Calibration Workflow: Three Stages
-----------------------------------

The LASER calibration workflow progresses through three stages, each building on the previous. **Validate each stage before moving to the next.**

Stage 1: Local Calibration (Fast Iteration)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Purpose:** Rapid prototyping and debugging on a single machine.

**When to use:** Initial development, testing objective functions, understanding parameter sensitivity.

**When to move on:** Once you have a working calibration configuration and understand which parameters matter most.

**Steps:**

1. **Verify Your Model Runs**
   Before calibrating, confirm your model executes successfully with default parameters:

   .. code-block:: shell

       python run_model.py --config default_config.yaml

   If this fails, fix the model before attempting calibration.

2. **Expose Parameters in Your Model**
   Ensure your LASER model can load and apply parameters you wish to calibrate. These are typically passed through a `params` dictionary or a `PropertySet` and might include:

   - Basic reproduction number (R0)
   - Duration of infection
   - Seeding prevalence

   **Remember:** Calibration should only vary parameter *values*, not toggle mechanisms or change scenario structure.

3. **Write Post-Processing Code**
   Modify your model to save key outputs (e.g., number of infected individuals over time) to a CSV file. For example, use:

   .. code-block:: python

       save_results_to_csv(sim.results)

   This CSV will be used later by the objective function.

3. **Create the Objective Function**
   Write a Python script, usually named `objective.py`, containing a function like this:

   .. code-block:: python

       def objective(trial):
           # Load trial parameters
           R0 = trial.suggest_float("R0", 1.0, 3.5)

           # Run model (via subprocess, or function call)
           run_model(R0)

           # Load model output and reference data
           model_df = pd.read_csv("output.csv")
           ref_df = pd.read_csv("reference.csv")

           # Compare and return score
           error = np.mean((model_df["I"] - ref_df["I"])**2)
           return error

   **Tip:** You can write unit tests for your objective function by mocking model outputs.

4. **Test Objective Function Standalone**
   Before integrating with Optuna, run your objective function directly to ensure it works:

   .. code-block:: python

       from objective import objective
       from optuna.trial import FixedTrial

       score = objective(FixedTrial({"R0": 2.5}))
       print(f"Test score: {score}")

   **Expected Result:** A numeric score. If it crashes, check CSV paths and data types.

5. **Run Simple Calibration (SQLite, No Docker)**
   Create a `calib/worker.py` helper to run a local test study with a small number of trials. This helper should:

   - parse command-line arguments (for example, ``--num-trials`` and optionally ``--storage-url``),
   - read the ``STORAGE_URL`` environment variable if no storage URL is passed,
   - create or load an Optuna study using that storage URL (e.g., via ``optuna.create_study(..., load_if_exists=True)``), and
   - call ``study.optimize(objective, n_trials=args.num_trials)`` with your objective function.

   For examples of configuring storage backends and running studies, see the Optuna documentation: https://optuna.readthedocs.io/en/stable/
   **Linux/macOS (Bash or similar):**

   .. code-block:: shell

       export STORAGE_URL=sqlite:///example.db && python3 calib/worker.py --num-trials=10

   **Windows (PowerShell):**

   .. code-block:: powershell

       $env:STORAGE_URL="sqlite:///example.db"; python calib/worker.py --num-trials=10

   This is helpful for debugging. Consider running a scaled-down version of your model to save time.

Stage 2: Dockerized Local Calibration (Environment Parity)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Purpose:** Validate reproducibility in a containerized environment before scaling.

**Why Docker?** Provides environment parity and dependency reproducibility between local development and production cloud runs. When random seeds and execution settings are controlled, this should lead to outputs that match within numerical tolerance, while catching environment-specific issues early.

**When to use:** After local calibration works and you're preparing to scale to distributed computing.

**When to move on:** When Docker runs complete successfully and produce consistent results with your local runs (within numerical tolerance).

**Steps:**

1. **Dockerize Your Model and Objective**
   Use the provided `Dockerfile` to build a container that includes both your model and objective function. Do this from the main directory.

   .. code-block:: shell

       docker build . -f calib/Dockerfile -t your-registry/laser-model:latest

2. **Create Docker Network**
   You'll need a shared network so your workers and database container can communicate:

   .. code-block:: shell

       docker network create optuna-network

3. **Launch MySQL Database Container**

   .. code-block:: shell

       docker run -d --name optuna-mysql --network optuna-network -p 3306:3306 \
         -e MYSQL_ALLOW_EMPTY_PASSWORD=yes \
         -e MYSQL_DATABASE=optuna_db mysql:latest

4. **Launch Calibration Worker**

    .. code-block:: shell

        docker run --rm --name calib_worker --network optuna-network \
          -e STORAGE_URL="mysql://root@optuna-mysql:3306/optuna_db" \
          your-registry/laser-model:latest \
          --study-name test_calib --num-trials 1

    If that works, you can change the study name or number of trials.

    **Troubleshooting:** If this fails, try running the worker interactively and debug inside:

    .. code-block:: shell

        docker run -it --network optuna-network --entrypoint /bin/bash your-registry/laser-model:latest

5. **Monitor Calibration Progress**

    Use Optuna CLI. You should be able to pip install optuna.

    .. code-block:: shell

        optuna trials \
          --study-name=test_calib \
          --storage "mysql+pymysql://root:@localhost:3306/optuna_db"

        optuna best-trial \
          --study-name=test_calib \
          --storage "mysql+pymysql://root:@localhost:3306/optuna_db"

6. **Verify Results Match Local Runs**

    Compare the best-fit parameters and likelihood scores between Docker and local runs. They should match within numerical tolerance (accounting for floating-point precision).

Stage 3: Cloud Calibration (Production Scale)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Purpose:** Distributed parameter search with hundreds or thousands of parallel trials.

**When to use:** After Docker validation succeeds and you need to explore a large parameter space or run many iterations.

**When to move on:** When calibration converges to stable best-fit parameters and likelihood scores plateau.

**Steps:**

1. **Push Docker Image to Registry**

    If you've built a new docker image, you'll want to push it so it's available to your cloud cluster (e.g., AKS, GKE).

    .. code-block:: shell

        docker push your-registry/laser-model:latest

2. **Cloud Deployment**

    This step assumes you have secured access to a Kubernetes cluster (e.g., AKS, GKE). You may need to obtain or generate a kube config file. Detailed instructions for that are not included here.

    .. code-block:: shell

       cd calib/cloud

    - Edit config file. Edit `cloud_calib_config.py` to set the storage_url to your cloud MySQL/PostgreSQL instance:

    .. code-block:: python

        "mysql+pymysql://optuna:password@mysql-host:3306/optunaDatabase"

    And set the study name and number of trials per your preference.

    - Launch multiple workers:

      .. code-block:: shell

          python3 run_calib_workers.py

3. **Monitor and Analyze Results**

    - Forward port to local machine (requires `kubectl` installed):

      .. code-block:: shell

          kubectl port-forward mysql-0 3306:3306 &

    - Use Optuna CLI to check results:

      .. code-block:: shell

          optuna trials \
            --study-name=your_study_name \
            --storage "mysql+pymysql://optuna:password@localhost:3306/optunaDatabase"

          optuna best-trial \
            --study-name=your_study_name \
            --storage "mysql+pymysql://optuna:password@localhost:3306/optunaDatabase"

    - Generate a report on disk about the study (can be run during study or at end):

      .. code-block:: shell

          python3 report_calib_aks.py

    - Launch Optuna Dashboard for interactive visualization:

      .. code-block:: shell

          python -c "import optuna_dashboard; optuna_dashboard.run_server('mysql+pymysql://optuna:password@127.0.0.1:3306/optunaDatabase')"

      **Note:** If port 8080 is already in use, specify a different port in the dashboard command.

Common Pitfalls
---------------

**Trying to calibrate before validating a single run**
   Always test your model with default parameters first. If a single simulation fails, calibration will fail hundreds of times. Use ``python run_model.py`` or equivalent to verify your model works end-to-end.

**Allowing calibration to toggle mechanisms on/off**
   Calibration should only vary parameter *values*, not change scenario structure. Keep mechanism switches (e.g., "enable_seasonality") fixed in your configuration. If you need to compare different model structures, run separate calibration studies for each.

**Skipping Stage 2 (Docker validation)**
   Don't jump directly from local calibration to cloud deployment. Docker catches environment-specific issues early—reproducibility problems that surface at scale are much harder to debug. Validate Docker locally first.

**Debugging cloud deployment before Docker works locally**
   If calibration fails in the cloud, first verify the exact same Docker image works on your local machine. Cloud failures are often environment issues (networking, permissions, resource limits), not calibration logic problems.

**Not validating Docker results match local results**
   Even if Docker runs complete, verify the numerical results match your local runs (within floating-point tolerance). Unexpected differences indicate environment issues that will cause problems at scale.

**Using poorly justified parameter bounds**
   Wide parameter ranges increase search space exponentially. Use scientific literature, expert knowledge, or preliminary sensitivity analysis to set reasonable bounds. Document the rationale for each range.

**Insufficient trials for the parameter space**
   A 10-dimensional parameter space needs many more trials than a 3-dimensional one. If calibration isn't converging, you may need more trials or tighter parameter bounds.

**Ignoring failed trials**
   Failed trials (model crashes, timeouts) provide information. If many trials fail in a specific parameter region, that region may be scientifically invalid. Investigate the pattern, don't just ignore failures.

Expected Output
---------------
- A best-fit parameter set (e.g., `R0`, transmission rates, etc.) that minimizes your objective function
- An Optuna study database (MySQL or SQLite) containing all trial results
- Log files showing convergence over time
- Visualizations from Optuna dashboard showing parameter relationships and optimization progress

Troubleshooting
---------------

**Missing CSVs or output files**
   Ensure your model writes output files before the objective function tries to read them. Test the output path in a single run first.

**Model crashes during calibration**
   - Check Docker logs: ``docker logs <container>``
   - Run the container interactively: ``docker run -it --entrypoint /bin/bash your-image``
   - Test the failing parameter combination in a standalone run
   - Check for resource limits (memory, disk space)

**Database connection errors**
   - Confirm Docker network exists: ``docker network ls``
   - Check container health: ``docker ps`` and ``docker logs optuna-mysql``
   - Verify MySQL is listening: ``docker exec optuna-mysql mysql -e "SELECT 1"``
   - Test connection string from within worker container

**Optuna study not found**
   - Verify study name matches exactly (case-sensitive)
   - Check you're connecting to the correct database
   - Confirm at least one trial has been submitted

**Poor convergence or no improvement**
   - Verify objective function returns correct values (not NaN or Inf)
   - Check parameter bounds are reasonable for your model
   - Increase number of trials
   - Examine parameter distributions in Optuna dashboard to identify if certain regions are unexplored

**Docker image build failures**
   - Check Dockerfile syntax
   - Verify all dependencies are listed
   - Test each RUN command separately
   - Check for network issues during package downloads

Best Practices
--------------

**Start small, scale gradually**
   Begin with just 1-2 trials to validate your objective function. Then scale to 10-20 trials locally, verify Docker produces consistent results, and only then move to cloud scale with hundreds or thousands of trials.

**Use scaled-down models for debugging**
   When testing your calibration setup, use a reduced version of your model (shorter time period, smaller population, fewer nodes). This speeds up iteration and makes debugging practical.

**Version control your configurations**
   Keep calibration configurations (parameter bounds, study settings) in version control alongside your model code. Document the rationale for parameter ranges.

**Monitor resource usage**
   Track memory, CPU, and disk usage during calibration. Models that barely fit in memory locally may fail at scale. Plan resource requests accordingly for cloud deployment.

**Document your objective function**
   Clearly document what your objective function measures, how it weights different data sources, and what "better" means (lower or higher values). This helps others understand and modify your calibration.

**Preserve intermediate results**
   Configure your model to save intermediate outputs even if a simulation fails partway through. This helps diagnose parameter regions that cause instability.

**Use scientific priors**
   Don't treat calibration as a black box. Use domain knowledge to set reasonable parameter bounds and validate that best-fit parameters are scientifically plausible.

Continuous Integration Tips
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If you're using CI/CD pipelines (GitHub Actions, etc.):

1. Test calibration code in CI with 1-2 trials using a fast model
2. Build Docker images automatically after merging to main
3. Tag images with git commit hash for reproducibility
4. Keep cloud deployment scripts in version control
5. Archive best-fit parameters and study databases for each major calibration run

Next Steps
----------
Once you've completed calibration:

- **Validate best-fit parameters** - Verify they're scientifically reasonable
- **Examine parameter correlations** - Use Optuna dashboard to understand parameter relationships
- **Run posterior checks** - Simulate with best-fit parameters and compare to held-out data
- **Document results** - Record parameter values, likelihood scores, and any insights
- **Use in scenario analysis** - Apply calibrated parameters to your research questions
