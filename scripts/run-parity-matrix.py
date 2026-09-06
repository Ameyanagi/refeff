#!/usr/bin/env python3
"""Run every stock workflow against pinned golden references on a clean checkout.

HIGHZ expands the upstream template over Z=1..138 and checks its 1s energies
using the existing engine-test tolerance. All calculation cwds are disposable.
Partial results and failures are retained; this script never certifies a subset.
"""
import argparse
import hashlib
import json
import os
import pathlib
import re
import signal
import subprocess
import tempfile
import time


def sha256(filename):
    return hashlib.sha256(filename.read_bytes()).hexdigest()


def command_text(root, *command):
    return subprocess.check_output(command, cwd=root, text=True).strip()


def invoke(command, cwd, env, timeout):
    started = time.perf_counter()
    process = subprocess.Popen(command, cwd=cwd, env=env, text=True,
                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                               start_new_session=True)
    try:
        output, _ = process.communicate(timeout=timeout)
        return process.returncode, output, time.perf_counter() - started
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        output, _ = process.communicate()
        return 124, output + "\nParity workflow timed out.\n", time.perf_counter() - started


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=pathlib.Path, default=pathlib.Path.cwd())
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--timeout", type=int, default=1800, help="seconds per stock workflow")
    parser.add_argument("--atomic-timeout", type=int, default=120, help="seconds per HIGHZ module invocation")
    parser.add_argument("--resume", action="store_true", help="retain completed results after checking their provenance and inputs")
    args = parser.parse_args()
    root, output = args.root.resolve(), args.output.resolve()
    if args.threads < 1 or args.timeout < 1 or args.atomic_timeout < 1:
        parser.error("threads and timeout must be positive")
    if command_text(root, "git", "status", "--porcelain", "--untracked-files=no"):
        parser.error("the tested checkout must have no tracked changes")
    inventory = json.loads((root / "compatibility/feff10.json").read_text())
    subprocess.run(["cargo", "build", "--release", "--locked", "-p", "refeff-cli", "--bin", "refeff", "-p", "xtask", "--bin", "xtask"], cwd=root, check=True)
    metadata = json.loads(command_text(root, "cargo", "metadata", "--format-version", "1", "--no-deps", "--locked"))
    target = pathlib.Path(metadata["target_directory"]) / "release"
    binary, xtask = target / "refeff", target / "xtask"
    golden = root / "reference-work/golden"
    output.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, REFEFF_THREADS=str(args.threads), REFEFF_CACHE="audit")
    summary = {
        "schema_version": 1,
        "provenance": {
            "rustCommit": command_text(root, "git", "rev-parse", "HEAD"), "dirty": False,
            "rustCompiler": command_text(root, "rustc", "--version"),
            "rustBinarySha256": sha256(binary),
            "feffCommit": command_text(root / "feff10", "git", "rev-parse", "HEAD"),
            "feffDriverSha256": sha256(root / "feff10/bin/feff"),
            "referenceMode": "pinned golden fixtures; native FEFF was not rerun",
            "threads": args.threads,
        },
        "input_sha256": {}, "reference_manifest_sha256": {}, "workflows": [],
    }
    if args.resume:
        previous = json.loads((output / "workflow-summary.json").read_text())
        if previous.get("provenance") != summary["provenance"]:
            parser.error("resume requires the same clean source, executable, compiler, reference revision and threads")
        for name, expected in previous.get("input_sha256", {}).items():
            retained = (output / "inputs" / name).resolve()
            if not retained.is_relative_to((output / "inputs").resolve()) or sha256(retained) != expected:
                parser.error("retained input changed: " + name)
            current = golden / name
            if current.is_file() and sha256(current) != expected:
                parser.error("reference input changed: " + name)
        for name, expected in previous.get("reference_manifest_sha256", {}).items():
            if sha256(golden / name / "manifest.json") != expected:
                parser.error("reference manifest changed: " + name)
        for row in previous["workflows"]:
            if row["id"] == "HIGHZ" and "reference_sha256" in row and row["reference_sha256"] != sha256(golden / "HIGHZ/HighZ.out"):
                parser.error("HIGHZ binding-energy reference changed")
        summary = previous
    summary["complete"] = False
    summary.setdefault("invocations", []).append({
        "resume": args.resume, "workflow_timeout_seconds": args.timeout,
        "atomic_timeout_seconds": args.atomic_timeout,
        "runner_sha256": sha256(pathlib.Path(__file__).resolve()),
    })

    def save():
        (output / "workflow-summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    def retain_input(name, source, expected_hash=None):
        data = source.read_bytes()
        actual = hashlib.sha256(data).hexdigest()
        if expected_hash is not None and actual != expected_hash:
            raise RuntimeError("input changed since execution: " + name)
        destination = output / "inputs" / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(data)
        summary["input_sha256"][name] = actual

    for workflow in inventory["stock_workflows"]:
        record = next((row for row in summary["workflows"] if row["id"] == workflow), None)
        if record is not None and record["status"] != "running":
            continue
        if record is None:
            record = {"id": workflow, "status": "running"}
            summary["workflows"].append(record)
        save()
        reference_manifest = golden / workflow / "manifest.json"
        if reference_manifest.is_file():
            summary["reference_manifest_sha256"][workflow] = sha256(reference_manifest)
        report_path = output / "reports" / (workflow + ".json")
        report_path.parent.mkdir(parents=True, exist_ok=True)
        if workflow != "HIGHZ":
            code, log, seconds = invoke([str(xtask), "parity", "--example", workflow, "--json-out", str(report_path)], root, env, args.timeout)
            report_path.with_suffix(".log").write_text(log)
            record.update(status="pass" if code == 0 else "timeout" if code == 124 else "fail", seconds=seconds, exit_code=code)
            if report_path.is_file():
                report = json.loads(report_path.read_text())
                for name, expected_hash in report.get("input_sha256", {}).items():
                    retain_input(workflow + "/" + name, golden / workflow / name, expected_hash)
        else:
            template = golden / "HIGHZ/feff.inp"
            reference = golden / "HIGHZ/HighZ.out"
            retain_input("HIGHZ/feff.inp", template)
            record["reference_sha256"] = sha256(reference)
            references = {int(parts[0].rstrip(":")): parts for line in reference.read_text().splitlines() if (parts := line.split())}
            elements = json.loads(report_path.read_text())["elements"] if args.resume and report_path.is_file() else []
            for element in elements:
                number = element["atomic_number"]
                name = f"HIGHZ/{number}/feff.inp"
                retained = output / "inputs" / name
                if retained.read_text() != template.read_text().replace("XXX", str(number)):
                    parser.error("completed HIGHZ input changed: " + name)
                if "reference_ev" in element and element["reference_ev"] != float(references[number][2]):
                    parser.error("completed HIGHZ reference changed: " + name)
                retain_input(name, retained)
            save()
            completed_elements = {row["atomic_number"] for row in elements}
            for number in range(inventory["highz"]["first_atomic_number"], inventory["highz"]["last_atomic_number"] + 1):
                if number in completed_elements:
                    continue
                with tempfile.TemporaryDirectory(prefix="refeff-highz-parity-") as temporary:
                    work = pathlib.Path(temporary)
                    source = work / "feff.inp"
                    source.write_text(template.read_text().replace("XXX", str(number)))
                    retain_input(f"HIGHZ/{number}/feff.inp", source)
                    commands = [[str(binary), "-C", str(work), "module", module, "-i", str(source), "--json", "--threads", str(args.threads)] for module in ["rdinp", "atomic"]]
                    code, log, seconds = invoke(commands[0], work, env, args.atomic_timeout)
                    if code == 0:
                        code, atomic_log, atomic_seconds = invoke(commands[1], work, env, args.atomic_timeout)
                        log += atomic_log
                        seconds += atomic_seconds
                    element = {"atomic_number": number, "exit_code": code, "seconds": seconds, "status": "timeout" if code == 124 else "fail"}
                    reference_row = references[number]
                    if number in inventory["highz"]["known_reference_failures"]:
                        expected_failure = code not in [0, 2, 3, 124] and bool(re.search(r"converg|wavefunction|SCF", log, re.I))
                        element.update(status="pass" if expected_failure else "fail", expected_reference_failure=True)
                    elif code == 0 and (work / "atom00.dat").is_file():
                        row = next((line.split() for line in (work / "atom00.dat").read_text().splitlines() if line.strip().startswith("1s ")), None)
                        if row is not None:
                            actual, expected = float(row[2]), float(reference_row[2])
                            relative = abs(actual - expected) / abs(expected)
                            element.update(status="pass" if relative <= 1e-3 else "fail", actual_ev=actual, reference_ev=expected, relative_error=relative, tolerance=1e-3)
                    elements.append(element)
                    (report_path.parent / f"HIGHZ-{number}.log").write_text(log)
                    report_path.write_text(json.dumps({"elements": elements}, indent=2) + "\n")
                    save()
                    print(f"HIGHZ Z={number}: {element['status']}", flush=True)
            record.update(status="pass" if all(row["status"] == "pass" for row in elements) else "fail", seconds=sum(row["seconds"] for row in elements), elements=len(elements))
        save()
        print(f"{workflow}: {record['status']} ({record['seconds']:.1f}s)", flush=True)
    summary["complete"] = len(summary["workflows"]) == len(inventory["stock_workflows"])
    summary["provenance"]["dirty"] = bool(command_text(root, "git", "status", "--porcelain", "--untracked-files=no"))
    summary["binary_unchanged"] = sha256(binary) == summary["provenance"]["rustBinarySha256"]
    save()
    if summary["provenance"]["dirty"] or not summary["binary_unchanged"] or any(row["status"] != "pass" for row in summary["workflows"]):
        raise SystemExit("Full matrix did not pass; retained every result in " + str(output))


if __name__ == "__main__":
    main()
