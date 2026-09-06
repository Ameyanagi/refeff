#!/usr/bin/env python3
"""Measure isolated clean library builds across Cargo feature combinations.

Each sample has a fresh target directory; the Cargo download cache stays warm.
This measures compilation, not dependency download or network latency.
"""
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

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output", type=pathlib.Path, required=True)
parser.add_argument("--features", nargs="+", default=["none", "exafs", "exafs,sfconv", "full", "all"])
parser.add_argument("--profile", choices=["dev", "release"], default="dev")
parser.add_argument("--samples", type=int, default=3)
args = parser.parse_args()
if args.samples < 1:
    parser.error("samples must be positive")
root = pathlib.Path(__file__).resolve().parents[1]
sources = subprocess.check_output(["git", "ls-files", "-z"], cwd=root).split(b"\0")
digest = hashlib.sha256()
# Include new source files, too: these changes may not yet have been committed.
files = {root / os.fsdecode(name) for name in sources if name}
files.update((root / "crates").rglob("*.rs"))
for source in sorted(files):
    if source.is_file() and (source.suffix == ".rs" or source.name in ["Cargo.toml", "Cargo.lock", "rust-toolchain.toml"]):
        digest.update(str(source.relative_to(root)).encode())
        digest.update(source.read_bytes())
report = {
    "schema_version": 1, "profile": args.profile, "platform": platform.platform(),
    "source_sha256": digest.hexdigest(),
    "compiler": subprocess.check_output(["rustc", "--version"], cwd=root, text=True).strip(),
    "download_cache": "warm", "measurements": [],
}
for feature in args.features:
    flags = ["--all-features"] if feature == "all" else ["--no-default-features"]
    if feature not in ["all", "none"]:
        flags += ["--features", feature]
    samples = []
    for sample in range(args.samples):
        with tempfile.TemporaryDirectory(prefix="refeff-build-benchmark-") as target:
            timing = ["/usr/bin/time", "-l" if platform.system() == "Darwin" else "-v"]
            command = ["cargo", "build", "-p", "refeff", "--lib", "--locked", "--profile", args.profile, "--target-dir", target, *flags]
            started = time.perf_counter()
            result = subprocess.run(timing + command, cwd=root, capture_output=True, text=True)
            seconds = time.perf_counter() - started
            if result.returncode:
                raise SystemExit(result.stderr)
            pattern = r"(\d+)\s+maximum resident set size" if platform.system() == "Darwin" else r"Maximum resident set size \(kbytes\):\s*(\d+)"
            match = re.search(pattern, result.stderr)
            samples.append({"seconds": seconds, "peak_rss_bytes": int(match[1]) * (1 if platform.system() == "Darwin" else 1024) if match else None})
            print(f"{feature}, sample {sample + 1}: {seconds:.2f}s", flush=True)
    report["measurements"].append({"features": feature, "samples": samples, "median_seconds": statistics.median(row["seconds"] for row in samples)})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
