"""Subprocess contracts for the publishing gate; all evidence here is synthetic."""
import copy
import hashlib
import json
import pathlib
import subprocess
import sys
import tempfile
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]


class EvidenceContracts(unittest.TestCase):
    def test_valid_record_and_independent_tampering(self):
        manifest = ROOT / "compatibility/feff10.json"
        inventory = json.loads(manifest.read_text())
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
        compiler = subprocess.check_output(["rustc", "--version"], cwd=ROOT, text=True).strip()
        with tempfile.TemporaryDirectory(prefix="refeff-evidence-test-") as temporary:
            directory = pathlib.Path(temporary)
            hashes = {}
            for workflow in inventory["stock_workflows"]:
                name = workflow + "/feff.inp"
                target = directory / "inputs" / name
                target.parent.mkdir(parents=True)
                target.write_bytes(b"synthetic contract input\n")
                hashes[name] = hashlib.sha256(target.read_bytes()).hexdigest()
            valid = {
                "schema_version": 2, "commit": commit, "dirty": False,
                "complete": True, "binary_unchanged": True,
                "reference_commit": inventory["upstream"]["revision"],
                "toolchain": compiler,
                "input_manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
                "input_sha256": hashes,
                "workflows": [{"id": name, "status": "pass"} for name in inventory["stock_workflows"]],
                "provenance": {
                    "rustCommit": commit, "dirty": False, "rustCompiler": compiler,
                    "feffCommit": inventory["upstream"]["revision"],
                    "rustBinarySha256": "1" * 64, "feffDriverSha256": "2" * 64,
                },
                "release_readiness": {
                    "commit": commit, "toolchain": compiler,
                    "report": {
                        "passed": True,
                        "filters": {"modules": [], "rows": [], "open_only": False, "detail": False},
                        "production_scope": {"passed": True, "error": None},
                        "port_status": {"passed": True, "error": None},
                        "compatibility_matrix": {"passed": True, "error": None},
                        "open_compatibility_items": [],
                    },
                },
            }

            def verify(record):
                path = directory / "evidence.json"
                path.write_text(json.dumps(record))
                return subprocess.run(
                    [sys.executable, str(ROOT / "scripts/release-evidence.py"), str(path)],
                    cwd=directory, capture_output=True, text=True,
                )

            self.assertEqual(verify(valid).returncode, 0)
            mutations = [
                lambda data: data.update(commit="0" * 40),
                lambda data: data["workflows"].pop(),
                lambda data: data["workflows"][0].update(status="fail"),
                lambda data: data["provenance"].update(dirty=True),
                lambda data: data["provenance"].update(rustCompiler="different compiler"),
                lambda data: data["input_sha256"].update({"../outside": "0" * 64}),
                lambda data: data.update(schema_version=1),
                lambda data: data.update(complete=False),
                lambda data: data.update(binary_unchanged=False),
                lambda data: data.pop("release_readiness"),
                lambda data: data["release_readiness"].update(commit="0" * 40),
                lambda data: data["release_readiness"].update(toolchain="different compiler"),
                lambda data: data["release_readiness"]["report"].update(passed=False),
                lambda data: data["release_readiness"]["report"]["filters"].update(rows=["one-row"]),
                lambda data: data["release_readiness"]["report"].update(open_compatibility_items=[{"id": "unfinished"}]),
            ]
            for gate in ["production_scope", "port_status", "compatibility_matrix"]:
                mutations.append(lambda data, gate=gate: data["release_readiness"]["report"][gate].update(passed=False))
                mutations.append(lambda data, gate=gate: data["release_readiness"]["report"][gate].update(error="gate failed"))
            for mutate in mutations:
                record = copy.deepcopy(valid)
                mutate(record)
                self.assertNotEqual(verify(record).returncode, 0)
            next((directory / "inputs").rglob("feff.inp")).write_bytes(b"changed after testing")
            self.assertNotEqual(verify(valid).returncode, 0)


if __name__ == "__main__":
    unittest.main()
