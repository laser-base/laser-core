"""Local benchmark comparison via git worktree.

Compares the current checkout's benchmark numbers against a baseline ref
(default `main`) by spinning up a detached-HEAD worktree, running the
benchmark suite in both checkouts with matched flags, and reporting the
diff via `pytest-benchmark compare` plus an optional threshold gate.

Why local rather than CI?
    GitHub Actions runners are shared VMs whose performance varies by 30-50%
    between runs (noisy neighbours, throttling, cold caches). That noise
    floor is well above the per-PR regressions we actually want to catch.
    Running both halves of the comparison back-to-back on the same workstation
    cancels out hardware variance and lets a tight threshold (e.g. `min:15%`)
    be useful.

Usage:
    python3 benchmarks/local_compare.py                       # HEAD vs main
    python3 benchmarks/local_compare.py --baseline=v1.0.2     # HEAD vs a tag
    python3 benchmarks/local_compare.py --no-fail             # report only
    python3 benchmarks/local_compare.py --threshold=min:15%   # tighter gate
    python3 benchmarks/local_compare.py --rounds=5            # quick run
    python3 benchmarks/local_compare.py --keep-worktree       # leave worktree for inspection

Environment expectations:
    - A `.venv/` virtualenv at the repo root with `pytest-benchmark` installed.
    - A clean (or at least committable) working tree — the script does not
      touch your current files; the worktree lives in `.benchmarks/`.
    - The baseline ref must already exist locally (`git fetch origin main`
      first if you don't have it).
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path


def _run(cmd: list[str], cwd: Path | None = None, env: dict[str, str] | None = None, check: bool = True) -> subprocess.CompletedProcess:
    """Echo a command, run it, and return the completed process."""
    pretty_cwd = f"(cd {cwd}) " if cwd else ""
    print(f"  $ {pretty_cwd}{' '.join(cmd)}", flush=True)
    return subprocess.run(cmd, cwd=cwd, env=env, check=check, text=True)


def _check_regressions(baseline_json: Path, head_json: Path, threshold: str) -> list[tuple[str, float, float, float]]:
    """Compare two pytest-benchmark JSON files and return regressions worse than `threshold`.

    Args:
        baseline_json: Path to the baseline-run pytest-benchmark JSON.
        head_json: Path to the HEAD-run pytest-benchmark JSON.
        threshold: A `stat:pct%` string (e.g. `min:25%`). `stat` is one of the
            keys under each benchmark's `stats` block (commonly `min`, `mean`,
            `median`, `stddev`).

    Returns:
        A list of `(name, baseline_value, head_value, fractional_slowdown)` tuples
        for every benchmark whose `stat` slowed by more than the threshold. Empty
        list means no regressions worse than threshold.

    Raises:
        ValueError: If `threshold` is not in the form `stat:pct%`.
    """
    try:
        stat, pct_str = threshold.split(":")
    except ValueError as e:
        raise ValueError(f"--threshold must be in form 'stat:pct%' (got {threshold!r})") from e
    if not pct_str.endswith("%"):
        raise ValueError(f"--threshold percent must end with '%' (got {threshold!r})")
    pct = float(pct_str.rstrip("%")) / 100.0

    with baseline_json.open() as f:
        baseline = {b["name"]: b for b in json.load(f)["benchmarks"]}
    with head_json.open() as f:
        head = {b["name"]: b for b in json.load(f)["benchmarks"]}

    regressions: list[tuple[str, float, float, float]] = []
    for name, head_bench in head.items():
        if name not in baseline:
            continue  # new benchmark; nothing to compare against
        baseline_v = baseline[name]["stats"].get(stat)
        head_v = head_bench["stats"].get(stat)
        if baseline_v is None or head_v is None or baseline_v == 0:
            continue
        ratio = (head_v - baseline_v) / baseline_v
        if ratio > pct:
            regressions.append((name, baseline_v, head_v, ratio))
    return regressions


def _git_rev_parse(ref: str, cwd: Path) -> str:
    """Resolve a git ref to its commit SHA. Raises on unknown ref."""
    result = subprocess.run(["git", "rev-parse", ref], cwd=cwd, check=True, capture_output=True, text=True)
    return result.stdout.strip()


def main() -> int:
    """Entry point — see module docstring for usage."""
    parser = argparse.ArgumentParser(description="Local benchmark comparison via git worktree")
    parser.add_argument("--baseline", default="main", help="Git ref to use as the comparison baseline (default: main)")
    parser.add_argument("--rounds", type=int, default=10, help="--benchmark-min-rounds value (default: 10)")
    parser.add_argument("--threshold", default="min:25%", help="Regression threshold in the form 'stat:pct%%' (default: min:25%%)")
    parser.add_argument("--no-fail", action="store_true", help="Report the comparison but do not exit non-zero on regression")
    parser.add_argument("--keep-worktree", action="store_true", help="Leave the worktree in place after running (for debugging)")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    worktree_dir = repo_root / ".benchmarks" / "baseline-worktree"
    local_dir = repo_root / ".benchmarks" / "local"
    local_dir.mkdir(parents=True, exist_ok=True)

    safe_ref = args.baseline.replace("/", "_").replace("\\", "_")
    baseline_json = local_dir / f"baseline-{safe_ref}.json"
    head_json = local_dir / "head.json"

    venv_python = repo_root / ".venv" / "bin" / "python3"
    if not venv_python.exists():
        print(f"ERROR: venv python not found at {venv_python}; activate the project virtualenv first", file=sys.stderr)
        return 2

    # Sanity check: warn if baseline resolves to the same commit as HEAD.
    head_sha = _git_rev_parse("HEAD", repo_root)
    try:
        baseline_sha = _git_rev_parse(args.baseline, repo_root)
    except subprocess.CalledProcessError:
        print(f"ERROR: baseline ref {args.baseline!r} not found locally; try `git fetch origin {args.baseline}` first", file=sys.stderr)
        return 2
    if head_sha == baseline_sha:
        print(f"WARNING: HEAD ({head_sha[:8]}) is the same commit as {args.baseline} ({baseline_sha[:8]}); the comparison will be trivial.")

    # 1. Set up worktree.
    if worktree_dir.exists():
        print(f"[INFO] Removing stale worktree at {worktree_dir}")
        _run(["git", "worktree", "remove", "--force", str(worktree_dir)], cwd=repo_root, check=False)
    print(f"[INFO] Creating worktree for {args.baseline} ({baseline_sha[:8]}) at {worktree_dir}")
    _run(["git", "worktree", "add", "--detach", str(worktree_dir), args.baseline], cwd=repo_root)

    try:
        # 2. Copy the current branch's `benchmarks/` tree into the worktree so
        #    newly-added benchmark cases are still measured against the baseline source.
        worktree_bench = worktree_dir / "benchmarks-from-head"
        if worktree_bench.exists():
            shutil.rmtree(worktree_bench)
        shutil.copytree(repo_root / "benchmarks", worktree_bench)

        # 3. Baseline run — invoked from the worktree directory so pytest picks
        #    up the baseline's pyproject.toml config. PYTHONPATH points at the
        #    worktree's `src/` so `import laser.core` resolves to the baseline
        #    source while the venv supplies all other dependencies.
        print(f"\n[INFO] Running baseline ({args.baseline}) — pytest from {worktree_dir}")
        env = {**os.environ, "PYTHONPATH": str(worktree_dir / "src")}
        _run(
            [
                str(venv_python),
                "-m",
                "pytest",
                str(worktree_bench),
                "--benchmark-only",
                "--benchmark-warmup=on",
                "--benchmark-warmup-iterations=2",
                f"--benchmark-min-rounds={args.rounds}",
                f"--benchmark-json={baseline_json}",
                "-q",
            ],
            cwd=worktree_dir,
            env=env,
        )

        # 4. HEAD run — invoked from the main checkout with the normal venv path.
        print(f"\n[INFO] Running HEAD ({head_sha[:8]}) — pytest from {repo_root}")
        _run(
            [
                str(venv_python),
                "-m",
                "pytest",
                "benchmarks/",
                "--benchmark-only",
                "--benchmark-warmup=on",
                "--benchmark-warmup-iterations=2",
                f"--benchmark-min-rounds={args.rounds}",
                f"--benchmark-json={head_json}",
                "-q",
            ],
            cwd=repo_root,
        )

        # 5. Print the pytest-benchmark comparison table (read-only).
        print(f"\n[INFO] Comparison table: HEAD vs. {args.baseline}\n")
        _run(
            [
                str(venv_python),
                "-m",
                "pytest_benchmark",
                "compare",
                str(baseline_json),
                str(head_json),
                "--columns=min,mean,median,stddev,rounds",
            ],
            cwd=repo_root,
            check=False,
        )

        # 6. Optionally enforce the threshold.
        if args.no_fail:
            return 0

        regressions = _check_regressions(baseline_json, head_json, args.threshold)
        if regressions:
            print(f"\nFAIL: {len(regressions)} benchmark(s) regressed beyond {args.threshold} (on `{args.threshold.split(':')[0]}`):")
            for name, baseline_v, head_v, ratio in regressions:
                print(f"  {name}: {baseline_v * 1e6:.1f}us -> {head_v * 1e6:.1f}us ({ratio:+.1%})")
            return 1

        print(f"\nPASS: no regressions beyond {args.threshold}")
        return 0

    finally:
        if args.keep_worktree:
            print(f"\n[INFO] --keep-worktree set; leaving {worktree_dir} in place")
        else:
            print(f"\n[INFO] Removing worktree at {worktree_dir}")
            _run(["git", "worktree", "remove", "--force", str(worktree_dir)], cwd=repo_root, check=False)


if __name__ == "__main__":
    sys.exit(main())
