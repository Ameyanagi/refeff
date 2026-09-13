"""Contracts for reusing CI without weakening the release gate."""
import copy
import importlib.util
import pathlib
import unittest
from unittest.mock import patch


spec = importlib.util.spec_from_file_location(
    "require_ci", pathlib.Path(__file__).with_name("require-ci.py"))
ci = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ci)
REPOSITORY = "owner/refeff"
COMMIT = "a" * 40
WORKFLOW = {"id": 7, "path": ci.WORKFLOW_PATH, "state": "active"}


def successful_run():
    return {
        "id": 42, "workflow_id": 7, "path": ci.WORKFLOW_PATH,
        "head_sha": COMMIT, "head_branch": "main", "event": "push",
        "repository": {"full_name": REPOSITORY},
        "head_repository": {"full_name": REPOSITORY},
        "status": "completed", "conclusion": "success", "run_attempt": 1,
    }


def successful_jobs(run):
    return [{"name": name, "run_id": run["id"], "run_attempt": run["run_attempt"],
             "head_sha": run["head_sha"], "status": "completed", "conclusion": "success"}
            for name in sorted(ci.REQUIRED_JOBS)]


class ReusedCiContracts(unittest.TestCase):
    def test_successful_push_and_manual_ci(self):
        for event in ("push", "workflow_dispatch"):
            run = dict(successful_run(), event=event)
            ci.verify_run(run, REPOSITORY, COMMIT, WORKFLOW["id"])
            ci.verify_jobs(successful_jobs(run), run)

    def test_wrong_source_or_unsuccessful_run_is_rejected(self):
        mutations = [
            {"workflow_id": 8}, {"path": ".github/workflows/publish.yml"},
            {"head_sha": "b" * 40}, {"head_branch": "feature"},
            {"event": "pull_request"}, {"event": "pull_request_target"},
            {"repository": {"full_name": "someone/refeff"}},
            {"head_repository": {"full_name": "someone/refeff"}},
            {"status": "in_progress"}, {"conclusion": "failure"},
            {"conclusion": "cancelled"}, {"conclusion": "skipped"},
            {"run_attempt": 0},
        ]
        for mutation in mutations:
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                ci.verify_run(dict(successful_run(), **mutation), REPOSITORY, COMMIT, WORKFLOW["id"])

    def test_missing_duplicate_skipped_and_wrong_attempt_jobs_are_rejected(self):
        run = successful_run()
        jobs = successful_jobs(run)
        invalid = [[], jobs[:-1], jobs + [jobs[0]]]
        for mutation in ({"status": "in_progress"}, {"conclusion": "failure"},
                         {"conclusion": "skipped"}, {"conclusion": "cancelled"},
                         {"run_id": 1}, {"run_attempt": 2}, {"head_sha": "b" * 40}):
            changed = copy.deepcopy(jobs)
            changed[0].update(mutation)
            invalid.append(changed)
        for changed in invalid:
            with self.subTest(jobs=changed), self.assertRaises(ValueError):
                ci.verify_jobs(changed, run)

    def test_additional_jobs_must_also_pass(self):
        run = successful_run()
        jobs = successful_jobs(run)
        extra = dict(jobs[0], name="future check")
        ci.verify_jobs(jobs + [extra], run)
        with self.assertRaises(ValueError):
            ci.verify_jobs(jobs + [dict(extra, conclusion="failure")], run)

    def test_parallel_builds_and_parity_are_required_for_release(self):
        run = successful_run()
        jobs = successful_jobs(run)
        for name in ("Native parity reference", "WASI and browser execution",
                     "Native and WASM spectrum parity"):
            with self.subTest(job=name):
                with self.assertRaises(ValueError):
                    ci.verify_jobs([job for job in jobs if job["name"] != name], run)
                for conclusion in ("failure", "skipped"):
                    with self.assertRaises(ValueError):
                        ci.verify_jobs([
                            dict(job, conclusion=conclusion) if job["name"] == name else job
                            for job in jobs
                        ], run)

    def test_newer_pending_or_failed_run_never_falls_back_to_older_success(self):
        older = successful_run()
        for status, conclusion in (("in_progress", None), ("completed", "failure")):
            newer = dict(older, id=43, status=status, conclusion=conclusion)
            selected = ci.select_run([older, newer], REPOSITORY, COMMIT, WORKFLOW["id"])
            self.assertEqual(selected["id"], newer["id"])
            with self.assertRaises(ValueError):
                ci.verify_run(selected, REPOSITORY, COMMIT, WORKFLOW["id"])

    def test_pr_and_other_commit_cannot_supply_release_ci(self):
        for runs in ([], [dict(successful_run(), event="pull_request")],
                     [dict(successful_run(), head_sha="b" * 40)]):
            with self.assertRaises(ValueError):
                ci.select_run(runs, REPOSITORY, COMMIT, WORKFLOW["id"])

    def test_paginated_runs_and_jobs_are_verified_with_latest_attempt(self):
        run = dict(successful_run(), run_attempt=2)
        jobs = successful_jobs(run)
        responses = [WORKFLOW, [{"workflow_runs": []}, {"workflow_runs": [run]}], run,
                     [{"jobs": jobs[:3]}, {"jobs": jobs[3:]}], run]
        with patch.object(ci, "api", side_effect=responses) as api:
            self.assertEqual(ci.require_ci(REPOSITORY, COMMIT),
                             f"https://github.com/{REPOSITORY}/actions/runs/42")
        self.assertIn("/attempts/2/jobs?", api.call_args_list[3].args[0])
        self.assertTrue(api.call_args_list[3].args[1])

    def test_rerun_started_during_verification_is_rejected(self):
        run = successful_run()
        for current in (dict(run, run_attempt=2), dict(run, status="in_progress", conclusion=None)):
            responses = [WORKFLOW, [{"workflow_runs": [run]}], run,
                         [{"jobs": successful_jobs(run)}], current]
            with patch.object(ci, "api", side_effect=responses), self.assertRaises(ValueError):
                ci.require_ci(REPOSITORY, COMMIT)

    def test_disabled_ci_is_rejected(self):
        with patch.object(ci, "api", return_value=dict(WORKFLOW, state="disabled_manually")):
            with self.assertRaises(ValueError):
                ci.require_ci(REPOSITORY, COMMIT)

    def test_invalid_arguments_do_not_call_github(self):
        with patch.object(ci, "api") as api:
            for repository, commit in (("owner/repo/extra", COMMIT), (REPOSITORY, "short")):
                with self.assertRaises(ValueError):
                    ci.require_ci(repository, commit)
            api.assert_not_called()


if __name__ == "__main__":
    unittest.main()
