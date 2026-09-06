#!/usr/bin/env python3
"""Interleave release CLI builds in disposable, isolated calculation directories."""
import argparse
import hashlib
import json
import os
import pathlib
import platform
import re
import statistics
import subprocess
import tempfile
import time


def named_path(value):
    name, separator, filename = value.partition("=")
    if not separator or not name or not pathlib.Path(filename).is_file():
        raise argparse.ArgumentTypeError("expected NAME=existing-file")
    return name, pathlib.Path(filename).resolve()


def digest(filename):
    return hashlib.sha256(filename.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", type=named_path, action="append", required=True)
    parser.add_argument("--case", type=named_path, action="append", required=True)
    parser.add_argument("--threads", type=int, nargs="+", default=[1, 2, 4])
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--provenance-variants", nargs="*", default=[])
    parser.add_argument("--output", type=pathlib.Path, required=True)
    args = parser.parse_args()
    if args.samples < 1 or any(value < 1 for value in args.threads):
        parser.error("samples and threads must be positive")
    if len(dict(args.variant)) != len(args.variant) or len(dict(args.case)) != len(args.case):
        parser.error("case and variant names must be unique")
    if not set(args.provenance_variants) <= dict(args.variant).keys():
        parser.error("unknown provenance variant")
    report = {
        "schema_version": 1, "complete": False, "platform": platform.platform(),
        "method": "Sequential processes; rotated variant order per repetition; two discarded reuse warmups; no calculation runs from the repository cwd.",
        "binaries": {name: {"path": str(binary), "sha256": digest(binary)} for name, binary in args.variant},
        "inputs": {name: {"path": str(source), "sha256": digest(source)} for name, source in args.case},
        "samples": [], "summary": [],
    }
    expected = {}
    outputs = ["chi.dat", "xmu.dat", "paths.dat", "feff.bin", "phase.bin", "pot.bin", "fms.bin"]
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        args.output.write_text(json.dumps(report, indent=2) + "\n")

    with tempfile.TemporaryDirectory(prefix="refeff-comparison-") as temporary:
        root = pathlib.Path(temporary)

        def run(case, source, variant, binary, threads, mode, directory, verify=True):
            env = dict(os.environ, REFEFF_CACHE="provenance" if mode == "provenance" else "audit")
            command = [str(binary), "run", "-i", str(source), "-o", str(directory), "--threads", str(threads), "--json"]
            timer = ["/usr/bin/time", "-l" if platform.system() == "Darwin" else "-v"]
            load_before = os.getloadavg()
            started = time.perf_counter()
            result = subprocess.run(timer + command, cwd=root, env=env, capture_output=True, text=True)
            seconds = time.perf_counter() - started
            if result.returncode:
                raise RuntimeError(f"{case}/{variant}/{mode}: {result.stdout}\n{result.stderr}")
            payload = json.loads(result.stdout)
            data = payload.get("data", payload)
            hashes = {name: digest(directory / name) for name in outputs if (directory / name).is_file()}
            # Legacy audit reuse can re-encode FEFF handoffs. Compare matching
            # modes; full provenance hits must preserve their cold outputs.
            key = (case, threads, "cold" if mode == "provenance" else mode)
            baseline = expected.setdefault(key, hashes) if verify else hashes
            if verify and hashes != baseline:
                report["all_scientific_files_identical"] = False
                report["parity_failure"] = {"case": case, "variant": variant, "mode": mode, "expected": baseline, "actual": hashes}
                save()
                raise RuntimeError("scientific output hashes differ; see parity_failure")
            pattern = r"(\d+)\s+maximum resident set size" if platform.system() == "Darwin" else r"Maximum resident set size \(kbytes\):\s*(\d+)"
            match = re.search(pattern, result.stderr)
            return {
                "case": case, "variant": variant, "threads": threads, "mode": mode,
                "seconds": seconds, "load_before": load_before,
                "peak_rss_bytes": int(match[1]) * (1 if platform.system() == "Darwin" else 1024) if match else None,
                "stages": data.get("stages", []), "diagnostics": data.get("diagnostics", []),
                "scientific_sha256": hashes,
            }

        for case_index, (case, source) in enumerate(args.case):
            for threads in args.threads:
                conditions = [(variant, binary, mode) for variant, binary in args.variant for mode in ["cold", "reuse"]]
                conditions += [(variant, binary, "provenance") for variant, binary in args.variant if variant in args.provenance_variants]
                workspaces = {}
                for condition_index, (variant, binary, mode) in enumerate(conditions):
                    directory = root / f"{case_index}-{threads}-{condition_index}"
                    workspaces[condition_index] = directory
                    if mode != "cold":
                        for _ in range(2):
                            run(case, source, variant, binary, threads, mode, directory, verify=False)
                for sample in range(args.samples):
                    order = list(range(len(conditions)))
                    offset = sample % len(order)
                    order = order[offset:] + order[:offset]
                    for index in order:
                        variant, binary, mode = conditions[index]
                        directory = workspaces[index]
                        if mode == "cold":
                            directory = directory / str(sample)
                        row = run(case, source, variant, binary, threads, mode, directory)
                        row["repetition"] = sample + 1
                        report["samples"].append(row)
                        save()
                        print(f"{case} {threads}t {variant} {mode} {sample + 1}/{args.samples}: {row['seconds']:.3f}s", flush=True)
                for variant, _, mode in conditions:
                    rows = [row for row in report["samples"] if (row["case"], row["threads"], row["variant"], row["mode"]) == (case, threads, variant, mode)]
                    report["summary"].append({
                        "case": case, "threads": threads, "variant": variant, "mode": mode,
                        "median_seconds": statistics.median(row["seconds"] for row in rows),
                        "stdev_seconds": statistics.pstdev(row["seconds"] for row in rows),
                        "min_seconds": min(row["seconds"] for row in rows),
                        "max_seconds": max(row["seconds"] for row in rows),
                        "median_peak_rss_bytes": statistics.median(row["peak_rss_bytes"] for row in rows if row["peak_rss_bytes"] is not None),
                    })
                report["all_scientific_files_identical"] = True
                save()
        report["complete"] = True
        save()


if __name__ == "__main__":
    main()
