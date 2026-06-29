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

Distributed Calibration Patterns
--------------------------------

This section collects patterns, anti-patterns, and deployment heuristics
for running distributed Optuna calibrations of LASER models on a
Kubernetes cluster with a shared MySQL backend. The material is
drawn from one extended deployment cycle in mid-2026 and is presented
here as general guidance for future LASER calibration deployments,
not as a project-specific post-mortem. Concrete configuration values
shown below are illustrative; the corresponding values for your
cluster come from your cluster admin.

The section is split into nine categories: assumptions and cluster
handoff, the local-validation discipline, deployment, builds,
shared-backend discipline, the probe-and-diagnose workflow,
calibration-objective design, the identifiability workflow, and a
pre-deployment checklist. The first two — assumptions and local
validation — are the cheapest investments and the ones most likely
to prevent days of debugging time on a shared cluster. The
deployment and build patterns are the ones most likely to consume
real time during a first deployment; the objective-design and
identifiability sections capture the most durable scientific
lessons.

Assumptions and Cluster Handoff
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This subsection is written for LASER users who are new to running
on a managed Kubernetes cluster. It does **not** explain Kubernetes,
Azure, AKS, GKE, EKS, or any other specific cloud or
Kubernetes-management product; upstream Kubernetes and your cluster
vendor's documentation cover those. The patterns below assume you
have already obtained access to a cluster through your organisation
and need to know how to make that access productive without
becoming a cluster administrator.

**What this section assumes you already have:**

- A LASER calibration that runs end-to-end on your workstation as
  described in *Stage 1: Local Calibration*.
- A working local Docker installation. The same calibration runs
  inside a container against a local MySQL container, as described
  in *Stage 2: Dockerized Local Calibration*. (If you have not yet
  reached this point, do not push anything to a shared cluster.
  See the local-validation discipline subsection below.)
- ``kubectl`` installed on your workstation and a ``KUBECONFIG``
  pointing at the target cluster.
- Whatever network access the cluster requires (corporate VPN,
  bastion host, IP allow-list) — confirm with your cluster admin.

**What this section deliberately does not require you to learn first:**

- Kubernetes internals, YAML schema details, or the administrative
  concepts of any specific Kubernetes-management product. The
  patterns below tell you which primitives matter for calibration
  workloads; upstream Kubernetes documentation explains how they
  work in general.

**What to ask your cluster administrator for.** A single ~15-minute
handoff conversation with the cluster admin, with the following
explicit list of questions, will prevent most of the time-consuming
debugging cycles below. Bring this list with you:

- A ``kubeconfig`` file that grants access to a working cluster and
  a working namespace on it.
- The container image registry URL you should push to, and the
  name of an ``imagePullSecret`` that is pre-provisioned in your
  namespace and is wired to that registry.
- The simulation node pool's nodeSelector label (key and value),
  and the corresponding toleration key, value, and effect. Pods
  that lack the toleration will land on the cluster's general
  pool — which is typically also where shared stateful services
  like the calibration MySQL pod live. *(See item 1 below.)*
- The name of the namespace ``Secret`` that holds the shared MySQL
  connection details, and the list of keys it contains
  (typically ``MYSQL_USER``, ``MYSQL_PASSWORD``, ``MYSQL_HOST``,
  ``MYSQL_DB``).
- Any network restrictions: corporate VPN access required, an IP
  allow-list, a proxy, or any outbound-egress restrictions.
- Any namespace-level resource quotas (total CPU, total memory,
  PVC count) that constrain how large you can scale your worker
  pool.
- Whether the cluster autoscales nodes on pod demand, and if so,
  whether your workload's expected scale-out has any constraints
  (e.g. you might need approval before requesting 100 simultaneous
  pods).
- Any conventions your cluster admin asks you to follow for
  cleaning up completed Jobs, log retention, and so on.

Once you have those answers — they fit on one page — you have
everything the patterns below assume. None of the patterns require
you to administer the cluster yourself.

Validate Locally with Docker Before Pushing to the Cluster
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Rule:** Do not use the cluster as your debugging environment.
Every category of build, image, and integration failure in the
patterns that follow is reproducible — and ten to a hundred times
cheaper to iterate on — with a small ``docker-compose`` stand on
your workstation. A failed cluster deployment cycle takes five to
twenty minutes (rebuild the image, push it, redeploy the Job, wait
for the pod to come up, fetch logs, diagnose); the same failure
caught locally takes seconds.

The local-validation stand has three containers, all running on
your workstation:

- a MySQL container running the same major version your cluster uses,
- a one-shot study-creator container running **your calibration
  image** — the exact image you would push to the cluster — pointed
  at the local MySQL,
- a worker container running the same image with a tiny trial
  budget (one to three trials).

A ``docker-compose.yml`` template that captures this pattern
(illustrative, not normative):

.. code-block:: yaml

    services:
      mysql:
        image: mysql:8.0
        environment:
          MYSQL_ALLOW_EMPTY_PASSWORD: "yes"
          MYSQL_DATABASE: optunaDatabase
          MYSQL_USER: optuna
          MYSQL_PASSWORD: localpw
        healthcheck:
          test: ["CMD", "mysqladmin", "ping", "-h", "localhost"]
          interval: 2s
          retries: 30

      study-creator:
        image: <your-image>:dev
        depends_on:
          mysql:
            condition: service_healthy
        environment:
          MYSQL_USER: optuna
          MYSQL_PASSWORD: localpw
          MYSQL_HOST: mysql
          MYSQL_DB: optunaDatabase
          STAGE_MODULE: <your.calibration.stage_module>
          STUDY_NAME: local-smoke
        command: ["python3", "/app/study_creator.py"]

      worker:
        image: <your-image>:dev
        depends_on:
          study-creator:
            condition: service_completed_successfully
        environment:
          MYSQL_USER: optuna
          MYSQL_PASSWORD: localpw
          MYSQL_HOST: mysql
          MYSQL_DB: optunaDatabase
          STAGE_MODULE: <your.calibration.stage_module>
          STUDY_NAME: local-smoke
          WORKER_N_TRIALS: "3"
        command: ["python3", "/app/worker.py"]

The local stand catches, in roughly the order they show up across
deployment cycles:

- Build context bloat from a naïve ``COPY .`` *(see item 4)*.
- Wheel-build errors and missing dependencies in your package
  metadata.
- The shell-glob extras parsing trap *(see item 8)*.
- Application data assets not present at the path your loader
  expects inside the container *(see item 7)*.
- Native extensions crashing on import for any reason — wrong
  ``-march``, missing system libraries, ABI mismatch *(see item 5)*.
- MySQL connection URL parsing bugs from a malformed env var or
  badly URL-encoded password *(see item 2)*.
- ``optuna.create_study`` / ``optuna.load_study`` round-trip
  failures.
- Stage-module interface mismatches — missing ``_make_objective``,
  unexpected signature, no ``get_warm_starts`` *(see item 12)*.
- A single trial failing to complete for any model-level reason
  (model assertions, divergent simulation, etc.).

**What the local stand does not catch**, and which therefore
warrant separate attention in the cluster phase:

- Node-pool placement issues — your local Docker has no notion of
  taints or selectors *(see item 1)*.
- Behaviour of the worker against a long-history shared MySQL
  backend with millions of accumulated rows *(see items 9–10)*.
- Optuna client-vs-server version-skew issues — your local Optuna
  matches your image's by construction *(see item 11)*.
- Kubernetes-specific Pod-spec quirks like the
  ``enableServiceLinks`` env-injection collision *(see item 2)*.

**Discipline:** every change to your stage module, every Dockerfile
change, and every dependency adjustment should pass the local
stand before being pushed to the cluster. The cluster is a scaling
environment, not a debugging environment. Most of the items in the
patterns below were discovered in deployment cycles where the local
stand was skipped; explicitly establishing the discipline at the
start of a project prevents most of them from being rediscovered
the hard way.

Kubernetes Deployment Patterns
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

1. **Always declare both ``nodeSelector`` and a matching
   ``toleration`` for the simulation node pool.** Shared clusters
   typically reserve a tainted, dedicated simulation node pool so
   calibration workloads do not interfere with system services
   (databases, ingress, monitoring). Pods that omit the toleration
   default-schedule to the general pool — which is also where shared
   stateful services like MySQL usually live, and where heavy
   per-pod workloads can drive eviction. The selector and the
   toleration are inseparable: the selector chooses the pool, the
   toleration earns the right to land there.

   .. code-block:: yaml

       spec:
         nodeSelector:
           pool-name: <your-sim-pool>
         tolerations:
           - effect: NoSchedule
             key: pool-name
             operator: Equal
             value: <your-sim-pool>

   Apply to **every** Pod spec in the calibration, including the
   one-shot study-creator Job and any auxiliary probe Pods.

2. **Set ``enableServiceLinks: false`` on every Pod spec when there
   is a Service whose name matches an env-var prefix you also set
   from a Secret.** Kubernetes' legacy Docker-links compatibility
   shim auto-injects environment variables like
   ``<SERVICE>_PORT=tcp://<ip>:<port>`` for every Service in the
   namespace. If your application also relies on a ``<SERVICE>_PORT``
   from a Secret, the two collide silently; one wins, your URL
   parser fails non-obviously, and your worker logs read as if
   the database hostname is wrong. ``enableServiceLinks: false``
   disables the injection cleanly. A defensive ``int(port)``
   coercion in the application's connection-URL builder catches
   the same class of bug if the flag is ever forgotten.

3. **Verify pod placement on every first deployment.** A
   one-line ``kubectl get pods ... -o
   jsonpath='{.spec.nodeName}'`` plus a lookup of the node's pool
   label is the cheapest possible verification that your nodeSelector
   is doing its job. Run it on the first deploy after any
   manifest change.

Container Build Patterns
~~~~~~~~~~~~~~~~~~~~~~~~~

4. **Build the package as a wheel; copy the wheel; never rsync the
   source tree.** A naïve ``COPY .`` from the working directory of a
   LASER model can pull in multi-gigabyte sibling directories
   (scratch outputs, HDF5 caches, dev-only datasets) and produce
   tens-of-GB build contexts. Build a local wheel first with
   ``pip wheel --no-deps``, copy that wheel into the build context,
   and install it inside the image:

   .. code-block:: dockerfile

       # On the host, before docker build:
       python -m pip wheel --no-deps --wheel-dir /tmp/whl <pkg-root>

       # In the Dockerfile:
       COPY /tmp/whl/*.whl /wheels/
       RUN pip install /wheels/*.whl

   Build context drops from gigabytes to under 10 MB; image build
   time drops correspondingly; the image contains exactly what you
   intended.

5. **Compile native extensions with a portable ``-march``, not
   ``-march=native``.** LASER models that ship compiled C / OpenMP
   kernels and build them inside the image will bake the build
   host's CPU instruction set into the resulting ``.so`` files. If
   the build host advertises a newer instruction set than any
   target cluster node, the kernels crash with ``SIGILL`` (exit
   132) the moment they are exercised on the cluster.

   .. code-block:: bash

       # Single-machine development build:
       CFLAGS_ARCH="-march=native" ./build.sh

       # Cluster image build:
       CFLAGS_ARCH="-march=x86-64-v2" ./build.sh

   ``x86-64-v2`` (Nehalem + SSE 4.2 + POPCNT) is portable to
   virtually every cloud x86 server. Use ``x86-64-v3`` (Haswell +
   AVX2 + BMI) only when you have measured evidence the cluster
   advertises it. A ``CFLAGS_ARCH`` environment variable on your
   build script that defaults to ``-march=native`` and is overridden
   by the cluster-image Dockerfile is the cleanest configuration.

6. **Use a pure-Python MySQL driver (PyMySQL) unless you have
   measured evidence the C-extension driver is the bottleneck.**
   ``mysqlclient`` is the canonical, fastest Python MySQL driver,
   but it requires ``libmysqlclient-dev`` and a working C toolchain
   in the image. On a slim base image these are not always
   straightforwardly available, and the resulting per-trial
   round-trip times are dominated by network and Optuna logic, not
   by the driver. PyMySQL is pure Python, installs with one
   ``pip install``, has zero apt dependencies, and is fast enough.
   The Optuna URL scheme is ``mysql+pymysql://``.

7. **Stage application data assets explicitly; do not rely on
   workstation-relative paths inside the image.** Loaders that
   reference paths relative to ``__file__`` or to the user's home
   directory work fine in development and fail inside containers,
   where ``__file__`` resolves to a site-packages path that is
   completely unrelated to the working directory. The cleanest fix
   is to parameterise loader paths via a ``data-dir`` argument and
   pass the staged location explicitly. If the loader code is not
   yours to modify, a Dockerfile-time symlink from the staged
   location to the expected path works as a more compact
   workaround.

8. **Shell-glob ``[extras]`` parses as a character class, not as
   pip extras.** This:

   .. code-block:: bash

       pip install /wheels/*.whl[full]

   does *not* mean "install all wheels with the [full] extras".
   The shell expands ``[full]`` as a character class matching one of
   ``f``, ``u``, or ``l``. Use a for-loop to defer extras parsing
   to pip:

   .. code-block:: dockerfile

       RUN set -e && \
           for w in /wheels/*.whl; do \
               pip install --no-cache-dir "$w[full]"; \
           done

   The double quotes around ``"$w[full]"`` cause the shell to skip
   glob expansion on the bracket portion; pip then parses it as
   the extras specifier.

Shared Optuna Backend Patterns
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A shared Optuna MySQL backend used by multiple teams accumulates
trial history indefinitely. By the time the next team arrives, the
``trial_params`` and ``trial_values`` tables may have millions of
rows. Several routine-looking Optuna API calls are scan-and-join
queries that are fine for a 100-trial study and pathologically slow
against a million-trial database.

9. **Calls that are safe against a long-history shared backend:**

   - ``optuna.load_study(study_name=NAME, storage=URL)`` — single-row
     ``SELECT ... WHERE study_name = ?``, fast at any scale.
   - ``study.best_trial`` — single-row lookup once the study is
     loaded.
   - ``study.enqueue_trial({...})`` — single INSERT.
   - ``study.optimize(objective, n_trials=N)`` — ask/tell loop
     scoped to the loaded study.

10. **Calls that are unsafe against a long-history shared backend:**

    - ``optuna.get_all_study_summaries(storage)`` and the
      ``optuna studies`` CLI that wraps it — both join against
      ``trial_params`` and ``trial_values`` for every study, and
      can hang for tens of minutes. Replace any liveness probe
      based on ``optuna studies`` with a ``python -c
      'optuna.load_study(name=..., storage=...)'`` that raises
      ``KeyError`` if the study does not exist.
    - ``len(study.trials)`` and ``study.trials_dataframe()`` —
      fetch all trials and parameters. Avoid in worker code; use
      only post-hoc on a completed study.

11. **Local vs cluster Optuna version skew is a real failure mode.**
    Optuna 4.x and 3.6.x have incompatible schema reads. A study
    written by Optuna 3.6 can produce ``KeyError: 'Record does not
    exist'`` when queried from a workstation running Optuna 4.7,
    even though the row is plainly present in the database. If you
    need to probe a cluster study from outside the cluster, either
    pin your local Optuna to the image's version or run the probe
    as an in-cluster one-shot Pod — same image, same Optuna, same
    schema reads. See the probe-and-diagnose pattern below.

Probe-and-Diagnose Workflow
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When a calibration is going through multiple iteration cycles
(redesigning the objective, widening parameter bounds), invest up
front in two patterns that pay back across every iteration.

12. **Stage-agnostic scaffold via a ``STAGE_MODULE`` env var.**
    Make the worker and study-creator Pods parameterised in two
    runtime values:

    - ``STAGE_MODULE``: a dotted Python import path to the module
      defining the calibration's free parameters, warm-starts, and
      objective function.
    - ``STUDY_NAME``: the Optuna study name to write into.

    Have the worker dynamically ``importlib.import_module`` the
    stage module and call a well-known function on it (e.g.
    ``_make_objective(targets, n_seeds)``); have the study-creator
    call ``stage_module.get_warm_starts()`` to enqueue trials.
    Each iteration of the calibration then requires only:

    - a new ``stage_N.py`` module exposing the conventional
      interface,
    - a new wheel build,
    - a new image tag,
    - and an env-var bump in the manifest.

    No scaffolding edits per iteration. This is a one-hour
    investment that compounds across many iteration cycles.

13. **In-cluster probe Pod for live-study inspection.** Build a
    small helper script (``probe.sh``, ``probe_cluster.sh``, etc.)
    that:

    - spawns a one-shot Pod using the same calibration image and
      the same node-selector / toleration as the workers,
    - runs a small Python script (``probe_best.py``) inside the
      Pod that loads the live Optuna study, retrieves the
      best-so-far trial, re-runs a small ensemble of seeds at
      those parameters, and writes a figure plus a
      machine-readable summary to a known path,
    - copies the artifacts back to the host with ``kubectl cp``.

    Running entirely inside the cluster sidesteps both
    ``kubectl port-forward`` brittleness and the Optuna version-skew
    failure mode from item 11. A typical probe run is well under a
    minute including a few model seeds.

14. **Automated ditch-detection heuristics on every probe.** A
    probe is most useful when it answers "is the optimizer heading
    into a ditch?" deterministically, not by visual inspection
    alone. Useful generic heuristics:

    - **Wide miss:** any cell of your evaluation grid (e.g. data
      point, age band × time, region × outcome) where the
      model-vs-target gap exceeds a fixed tolerance. Count, list
      the worst K, flag the total fraction.
    - **Structural miss:** any sub-set of cells with ≥ K consecutive
      wide misses along a meaningful axis (time, age, etc.).
      Distinguishes localised initialisation transients from
      systematic mis-fit.
    - **Constraint check:** any physical or domain constraint
      your objective enforces should be re-evaluated on the
      probe sample independently of the loss value, so a probe
      catches it even if the loss penalty has been silently
      defeated.
    - **Per-seed CV:** any cell with standard deviation across
      seeds divided by mean exceeding a threshold (e.g. 0.5)
      indicates the optimiser is finding a noisy minimum and the
      result is suspect.

    A single human-readable verdict line of the form ``"looks
    good — no wide-miss, no structural, constraint in band"`` is a
    clean target to write for and a clean signal for humans
    glancing at probe artifacts.

Calibration Objective Design
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The most durable scientific lesson from the deployment cycle that
generated this section is about *what to put in the loss*, not how
to deploy it.

15. **Do not calibrate to pooled summary statistics alone.**
    Longitudinal epidemiological datasets routinely contain severe
    *trajectory* misfit that disappears when aggregated. A model
    can match a pooled per-age prevalence mean exactly while
    exhibiting catastrophic temporal behaviour — for example,
    saturating to near-1.0 in the first few months of the
    observation window and collapsing to near-0 by the end, such
    that the pooled mean accidentally hits the target by averaging
    opposite-sign residuals.

    Always inspect, and where possible explicitly penalise:

    - per-survey (or per-time-point) trajectories,
    - per-stratum (age, region, intervention arm) trajectories,
    - intervention-era dynamics versus pre-intervention dynamics.

    An in-loss "trajectory shape penalty" — e.g. "for each
    (arm, stratum, time-point) cell, add a quadratic excess penalty
    whenever the absolute prevalence gap exceeds a tolerance" — is
    cheap to compute and forces the optimiser to fit the shape
    rather than the aggregate.

16. **Constrain physically meaningful quantities, not just
    statistical fit.** When the model has an emergent quantity that
    is independently measurable in the literature (basic
    reproduction number, annual entomological inoculation rate,
    case-detection rate, etc.), add an explicit penalty for that
    emergent quantity falling outside a defensible range. Without
    such constraints, a complex model can routinely fit the
    aggregate target at a wildly implausible value of the emergent
    quantity — typically by compensating with another mechanism
    such as infection duration or chronic-carriage strength. The
    resulting parameter set is a good fit but is not usable for
    predictive intervention analysis.

17. **Add mechanism only after demonstrating that a simpler model
    fails.** Each layer of model complexity should be motivated by
    a specific characterised failure of the simpler version, not by
    biological plausibility alone. Keep stage-by-stage records of
    what failure was observed and what mechanism was added in
    response, so a reviewer can trace each mechanism back to its
    motivating data. Models built this way are easier to defend
    against complexity-skeptics and easier to simplify later.

Identifiability Workflow
~~~~~~~~~~~~~~~~~~~~~~~~~

18. **Run a 1D identifiability sweep before widening the search
    space.** When you suspect a parameter is at the edge of its
    search range, do not just widen the range and re-launch a
    multi-day cluster run. Instead, hold all other parameters at
    the current best and sweep the suspect parameter across its
    range (and a candidate widened range) on a small grid, with
    multiple seeds per grid point. Compute a signal-to-noise ratio:

    .. math::

        \mathrm{S/N} = \frac{\max\_\mathrm{grid}\,L - \min\_\mathrm{grid}\,L}{\sigma_L(\text{at centre across seeds})}

    Three outcomes are useful:

    - **Interior minimum, large S/N:** parameter is well-constrained;
      the calibration result is scientifically defensible for that
      parameter.
    - **Edge minimum:** parameter is pinned at the boundary;
      widening the range may help.
    - **Flat (small S/N):** parameter is unidentifiable in the
      current objective; consider fixing it at a prior value or
      adding a constraint that gives it traction.

    A small grid sweep takes minutes on a single workstation and
    saves cluster hours; it should be the default step between a
    completed calibration and any decision to widen search ranges.

19. **Interpret edge minima carefully.** A 1D sweep is a slice
    through the full parameter space. When TPE has already explored
    a higher-dimensional basin and your sweep returns "edge
    minimum, widen", verify in a small cluster run that the widened
    range actually produces a lower loss in the full higher-
    dimensional context. It is common for a 1D slice to suggest
    widening a parameter that, when widened, requires compensating
    moves in other parameters that net out to no improvement.

Pre-Deployment Checklist
~~~~~~~~~~~~~~~~~~~~~~~~~

Before pushing a first image to a shared cluster, spend 30 minutes
working through the following questions explicitly. Each question
maps to one of the patterns above.

- **Cluster handoff:** have I obtained, from my cluster admin in a
  single 15-minute conversation, every item in the handoff list —
  ``kubeconfig``, registry URL + ``imagePullSecret`` name,
  simulation-pool ``nodeSelector`` and ``toleration``, MySQL
  ``Secret`` name and keys, network restrictions, and resource
  quotas?
- **Local validation:** has my exact image (the one I am about to
  push) been validated end-to-end on the local docker-compose
  stand against a local MySQL? Did a small trial run to completion
  there?
- **Node pool:** does my pod spec declare both a ``nodeSelector``
  and a matching ``toleration`` for the cluster's simulation node
  pool? Have I confirmed the resulting pod placement on the first
  deploy?
- **Service-link injection:** does every Pod spec set
  ``enableServiceLinks: false``? Does my application's connection-
  URL builder also defensively coerce the port to an integer?
- **MySQL driver:** is my image using ``mysql+pymysql://`` with
  PyMySQL installed?
- **Native extensions:** are my C / OpenMP kernels compiled with a
  portable ``-march`` for the cluster image?
- **Build context:** is the application installed from a locally-
  built wheel, with no source-tree COPY?
- **Data paths:** are application data assets staged at paths the
  code actually reads inside the container?
- **Optuna API discipline:** does my worker use
  ``optuna.load_study(name)`` rather than
  ``optuna.get_all_study_summaries()`` to determine whether a study
  exists? Does any call in the worker hot path fetch all trials?
- **Optuna version parity:** does the Optuna version on my
  workstation match the image's, or am I committed to probing only
  from inside the cluster?
- **Stage scaffold:** is my worker parameterised by ``STAGE_MODULE``
  and ``STUDY_NAME``, so the next iteration is a wheel-rebuild and
  an env-var bump rather than a code change?
- **Objective design:** does my loss penalise trajectory shape, not
  just pooled aggregates? Does it constrain at least one
  physically meaningful emergent quantity?
- **Identifiability:** have I run 1D sweeps over each free
  parameter at the current best before committing to widening any
  search ranges?

Working through this checklist once per cluster (not once per
iteration) typically saves hours of debugging time that the
patterns above were extracted from.
