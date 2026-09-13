#!/usr/bin/env python3
"""Require complete, successful main-branch CI for the exact release commit."""
import argparse
import json
import os
import pathlib
import re
import subprocess


WORKFLOW_PATH = ".github/workflows/ci.yml"
# Keep these names in sync with ci.yml. Additional jobs must also pass.
REQUIRED_JOBS = {
    "Lightweight quality checks",
    "WASI and browser execution",
    *(f"features ({features})" for features in ("none", "exafs", "exafs,sfconv", "full", "all")),
}


def api(endpoint, paginate=False):
    command = ["gh", "api", endpoint]
    if paginate:
        command.extend(["--paginate", "--slurp"])
    return json.loads(subprocess.check_output(command, text=True))


def is_release_ci(run, repository, commit, workflow_id):
    return (
        run.get("workflow_id") == workflow_id
        and run.get("path") == WORKFLOW_PATH
        and run.get("head_sha") == commit
        and run.get("head_branch") == "main"
        and run.get("event") in ("push", "workflow_dispatch")
        and run.get("repository", {}).get("full_name") == repository
        and run.get("head_repository", {}).get("full_name") == repository
    )


def select_run(runs, repository, commit, workflow_id):
    candidates = [run for run in runs if is_release_ci(run, repository, commit, workflow_id)]
    if not candidates:
        raise ValueError("No main-branch CI run exists for this commit; run ci.yml on main first")
    # Never fall back to an older success when a newer run is failing or pending.
    return max(candidates, key=lambda run: run["id"])


def verify_run(run, repository, commit, workflow_id):
    if not is_release_ci(run, repository, commit, workflow_id):
        raise ValueError("CI must come from this repository's ci.yml on main at the release commit")
    if run.get("status") != "completed" or run.get("conclusion") != "success":
        raise ValueError("Latest CI run has not completed successfully; finish or rerun CI first")
    if not isinstance(run.get("run_attempt"), int) or run["run_attempt"] < 1:
        raise ValueError("CI run lacks a valid attempt number")


def verify_jobs(jobs, run):
    names = [job.get("name") for job in jobs]
    if not REQUIRED_JOBS.issubset(names) or len(names) != len(set(names)):
        raise ValueError("CI is missing required jobs or contains duplicate job names")
    for job in jobs:
        if (job.get("status") != "completed" or job.get("conclusion") != "success"
                or job.get("run_id") != run["id"]
                or job.get("run_attempt") != run["run_attempt"]
                or job.get("head_sha") != run["head_sha"]):
            raise ValueError(f"CI job did not pass in this attempt: {job.get('name')}")


def require_ci(repository, commit):
    if not re.fullmatch(r"[A-Za-z0-9_-]+/[A-Za-z0-9_.-]+", repository):
        raise ValueError("Expected an owner/repository name")
    if not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise ValueError("Expected a full release commit SHA")
    base = f"repos/{repository}/actions"
    workflow = api(f"{base}/workflows/ci.yml")
    if workflow.get("path") != WORKFLOW_PATH or workflow.get("state") != "active":
        raise ValueError("The release CI workflow must be active")
    pages = api(f"{base}/workflows/ci.yml/runs?branch=main&head_sha={commit}&per_page=100", True)
    run = select_run([run for page in pages for run in page["workflow_runs"]],
                     repository, commit, workflow["id"])
    endpoint = f"{base}/runs/{run['id']}"
    run = api(endpoint)
    verify_run(run, repository, commit, workflow["id"])
    pages = api(f"{endpoint}/attempts/{run['run_attempt']}/jobs?per_page=100", True)
    verify_jobs([job for page in pages for job in page["jobs"]], run)
    # A rerun must not start while its previous attempt's jobs are being checked.
    current = api(endpoint)
    verify_run(current, repository, commit, workflow["id"])
    if current["run_attempt"] != run["run_attempt"]:
        raise ValueError("CI was rerun during verification; retry after it finishes")
    return f"https://github.com/{repository}/actions/runs/{run['id']}"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", required=True)
    parser.add_argument("--commit", required=True)
    args = parser.parse_args()
    try:
        url = require_ci(args.repository, args.commit)
    except (ValueError, KeyError, TypeError, subprocess.CalledProcessError) as error:
        parser.exit(1, f"CI verification failed: {error}\n")
    message = f"Reused successful CI for {args.commit}: {url}"
    print(message)
    if summary := os.environ.get("GITHUB_STEP_SUMMARY"):
        with pathlib.Path(summary).open("a") as output:
            output.write(message + "\n")


if __name__ == "__main__":
    main()
