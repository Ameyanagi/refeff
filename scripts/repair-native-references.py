#!/usr/bin/env python3
"""Rebuild two defective native reference stages without changing physical inputs.

Builds patched copies outside the pinned FEFF checkout. The Cr2GeC patch only
expands STRVECGEN work arrays; the CeO2 patch initializes PHASE_H's undefined
DFOVRG endpoint. Replays downstream native stages and records complete provenance.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import time

PIN = "0a4fbd797cf72938f64dda034a438ce009ec6eb7"
FLAGS = ["-ffree-line-length-none", "-cpp", "-O3", "-fallow-argument-mismatch"]
PROVENANCE = ".native-reference-repair.json"
REPAIRS = {
    "KSPACE/Cr2GeC": {
        "module": "fms", "source": "KSPACE/strvecgen.f90",
        "sha256": "b5d6974c079294d0f2f34aa917588bc75854da652231ae264ad7cbf35dcfcd8f",
        "description": "Expand only local STRVECGEN reciprocal-vector work arrays from 5000 to 20000.",
        "outputs": ["gg.bin", "fms.bin", "gtr.dat", "xmu.dat"],
    },
    "HUBBARD/CeO2": {
        "module": "xsph", "source": "XSPH/phase_h.f90",
        "sha256": "96482ccc5e416f9aadc36e745cd27239f312920734146abcf7cb8161505237c0",
        "description": "Initialize the Hubbard DFOVRG integration endpoint ilast=jri before each solve.",
        "outputs": ["phase.bin", "aphase_hubbard.bin", "xsect.dat", "gg.bin", "fms.bin", "gtr.dat", "xmu.dat"],
    },
}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def patched_source(example, source):
    if example == "KSPACE/Cr2GeC":
        # Rename the imported bound, leaving every other module constant intact.
        needle = "use boundaries,only : NGRLMAX,"
        if source.count(needle) != 1:
            raise ValueError("unexpected STRVECGEN boundaries import")
        source = source.replace(needle, "use boundaries,only : NGRLMAX_UPSTREAM => NGRLMAX,")
        needle = "! PARAMETER definitions\n"
        count = source.count(needle)
        source = source.replace(needle, needle + "      integer, parameter :: NGRLMAX = 4*NGRLMAX_UPSTREAM\n")
    else:
        needle = "                call dfovrg (ncycle, ikap, rmt, ilast, jri, p2, dx,"
        count = source.count(needle)
        source = source.replace(needle, "                ilast = jri\n" + needle)
    if count != 1:
        raise ValueError("native repair no longer matches the pinned source")
    return source


def validate_outputs(work):
    phase = (work / "phase.bin").read_text().splitlines()[0].split()
    energy_count, real_count = int(phase[1]), int(phase[2])
    gg = (work / "gg.bin").read_bytes()
    if len(re.findall(rb"(?m)^#SN#", gg)) != energy_count:
        raise ValueError("fresh GG section count differs from phase contour")
    fms = (work / "fms.bin").read_text().splitlines()
    if int(fms[1].split()[0]) != energy_count:
        raise ValueError("fresh FMS contour count differs from phase contour")
    rows = []
    for line in (work / "xmu.dat").read_text().splitlines():
        if line.strip() and not line.lstrip().startswith("#"):
            row = [float(x) for x in line.split()]
            if len(row) != 6 or not all(math.isfinite(x) for x in row):
                raise ValueError("invalid fresh absorption spectrum")
            rows.append(row)
    if len(rows) != real_count or not any(row[5] != 0.0 for row in rows):
        raise ValueError("fresh spectrum is missing the FMS contribution")
    return {"contour_points": energy_count, "spectrum_points": real_count}


def repair(reference, case, example):
    spec = REPAIRS[example]
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=reference, text=True).strip()
    dirty = subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"], cwd=reference, text=True).strip()
    if revision != PIN or dirty:
        raise ValueError("reference repair requires the clean pinned native source")
    src = reference / "src"
    original = src / spec["source"]
    if digest(original) != spec["sha256"]:
        raise ValueError("native source checksum does not match the reviewed repair")
    compiler = shutil.which("gfortran")
    if not compiler:
        raise ValueError("gfortran is required to reproduce native reference repairs")
    compiler_version = subprocess.check_output([compiler, "--version"], text=True).splitlines()[0]
    with tempfile.TemporaryDirectory(prefix="refeff-native-reference-") as temporary:
        build = Path(temporary) / "build"
        work = Path(temporary) / "work"
        build.mkdir()
        shutil.copytree(case, work)
        source = build / original.name
        source.write_text(patched_source(example, original.read_text()))
        obj = source.with_suffix(".o")
        binary = build / spec["module"]
        includes = ["-I" + str(p) for p in sorted(src.iterdir()) if p.is_dir()]
        subprocess.run([compiler, *FLAGS, *includes, "-c", str(source), "-o", str(obj)], cwd=build, check=True, capture_output=True)
        objects = []
        object_hashes = {}
        dependency = src / "DEP" / (spec["module"] + ".mk")
        for name in re.findall(r"\./([\w/]+)\.f90", dependency.read_text()):
            path = obj if name + ".f90" == spec["source"] else src / (name + ".o")
            if path not in objects:
                objects.append(path)
                object_hashes[name + ".o"] = digest(path)
        for directory in ["BLAS", "LAPACK"]:
            for path in sorted((src / directory).glob("*.o")):
                if path not in objects:
                    objects.append(path)
                    object_hashes[path.relative_to(src).as_posix()] = digest(path)
        subprocess.run([compiler, *FLAGS, "-o", str(binary), *map(str, objects)], cwd=build, check=True, capture_output=True)
        inputs = {p.name: digest(p) for p in sorted(work.iterdir()) if p.is_file()
                  and (p.suffix == ".inp" or p.name in [".dimensions.dat", "geom.dat", "pot.bin", "v_hubbard.bin", "transformation_hubbard.bin", "phase.bin", "xsect.dat"])
                  and p.name not in spec["outputs"]}
        # Remove all products before the first native stage. Fortran STOP can
        # return success, so exit status alone cannot establish a valid oracle.
        for name in spec["outputs"]:
            (work / name).unlink(missing_ok=True)
        stages = ["fms", "mkgtr", "ff2x"]
        if spec["module"] == "xsph":
            stages.insert(0, "xsph")
        runs = []
        logs = []
        required = {"xsph": ["phase.bin", "aphase_hubbard.bin", "xsect.dat"], "fms": ["gg.bin"], "mkgtr": ["fms.bin", "gtr.dat"], "ff2x": ["xmu.dat"]}
        for stage in stages:
            executable = binary if stage == spec["module"] else reference / "bin/Seq" / stage
            start = time.monotonic()
            result = subprocess.run([str(executable)], cwd=work, capture_output=True, timeout=5400)
            for channel in ["stdout", "stderr"]:
                name = stage + "-reference-repair." + channel
                (work / name).write_bytes(getattr(result, channel))
                logs.append(name)
            output = result.stdout + result.stderr
            if result.returncode or b"NG > NGRLMAX" in output or b"Fortran runtime error" in output or b"Error:" in output:
                raise ValueError(f"native {stage} failed: {output.decode(errors='replace')[-2000:]}")
            for name in required[stage]:
                if not (work / name).is_file() or not (work / name).stat().st_size:
                    raise ValueError(f"native {stage} did not create fresh {name}")
            runs.append({"module": stage, "exit_code": result.returncode, "seconds": time.monotonic() - start, "executable_sha256": digest(executable)})
        validation = validate_outputs(work)
        provenance = {"schema_version": 1, "native_commit": revision, "example": example,
                      "generator": "scripts/repair-native-references.py", "generator_sha256": digest(Path(__file__)),
                      "operation": spec["description"], "source": spec["source"], "original_source_sha256": digest(original),
                      "patched_source_sha256": digest(source), "compiler": compiler_version, "flags": FLAGS,
                      "linked_object_sha256": object_hashes, "input_sha256": inputs, "runs": runs, "validation": validation,
                      "output_sha256": {name: digest(work / name) for name in spec["outputs"] + logs}}
        # Install only after every native stage and scientific handoff check passes.
        for name in spec["outputs"] + logs:
            shutil.copy2(work / name, case / name)
        (case / PROVENANCE).write_text(json.dumps(provenance, indent=2) + "\n")
        return provenance


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--case", type=Path, required=True)
    parser.add_argument("--example", choices=REPAIRS, required=True)
    args = parser.parse_args()
    result = repair(args.reference.resolve(), args.case.resolve(), args.example)
    print(json.dumps({"example": args.example, "validation": result["validation"]}))


if __name__ == "__main__":
    main()
