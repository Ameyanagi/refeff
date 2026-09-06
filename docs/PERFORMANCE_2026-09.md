# September performance follow-up

The completed comparison contains 150 timed runs: two inputs, three thread
settings, five execution conditions and five repetitions per condition. At one
thread, ZnSe fresh calculations improved by 14.2% and semantic audit reuse by
16.2%. BN ordinary runtimes remained close to the baseline. Valid opt-in
provenance hits took 0.058–0.076 seconds across the six input/thread combinations.

The checked scientific files match between original and updated executables in
all measured conditions. Fresh-run peak memory increased in several cases; this
is not an across-the-board memory reduction. Full stock scientific validation
still fails five workflow entries, so these changes are not release-certified.

## Before and after

Times below are five-run medians in seconds. Negative changes mean faster.
“Fresh” means a new calculation directory, with ordinary OS file caches still
available. “Audit” is the default semantic cache validation. Provenance is opt-in
via `REFEFF_CACHE=provenance` and requires a valid complete workspace fingerprint.

| Input | Threads | Fresh: original → updated | Change | Audit: original → updated | Change | Provenance hit |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| ZnSe | 1 | 1.379 → 1.182 | -14.2% | 1.947 → 1.632 | -16.2% | 0.059 |
| ZnSe | 2 | 1.379 → 1.229 | -10.9% | 2.018 → 1.622 | -19.6% | 0.058 |
| ZnSe | 4 | 1.682 → 1.207 | -28.2% | 2.017 → 1.653 | -18.0% | 0.059 |
| BN | 1 | 30.285 → 30.631 | +1.1% | 22.323 → 22.409 | +0.4% | 0.067 |
| BN | 2 | 20.311 → 20.155 | -0.8% | 14.110 → 14.297 | +1.3% | 0.075 |
| BN | 4 | 14.937 → 14.641 | -2.0% | 9.614 → 9.683 | +0.7% | 0.076 |

BN's ordinary changes are small relative to its observed run-to-run spread.
ZnSe benefited little from adding threads on this machine; BN ran faster with
more threads. These observations apply to the tested inputs and hardware.
Every sample, standard deviation, minimum and maximum is retained in the raw JSON.

Median peak process memory, in MiB:

| Input | Threads | Fresh: original → updated | Audit: original → updated | Provenance |
| --- | ---: | ---: | ---: | ---: |
| ZnSe | 1 | 34.2 → 36.8 | 35.5 → 34.3 | 5.3 |
| ZnSe | 2 | 34.2 → 36.9 | 35.5 → 34.3 | 5.2 |
| ZnSe | 4 | 34.7 → 36.8 | 36.0 → 34.4 | 5.3 |
| BN | 1 | 233.4 → 246.9 | 79.9 → 80.4 | 6.1 |
| BN | 2 | 268.5 → 265.6 | 100.1 → 100.8 | 6.1 |
| BN | 4 | 299.6 → 309.9 | 135.0 → 142.4 | 6.1 |

## What changed

The reuse profile identified repeated POT/APOT parsing and canonical rendering,
plus atomic recomputation during semantic cache validation. Prepared no-SCF POT
state now retains its canonical text once per run. Exact matching payloads avoid
another decode/render cycle; differently formatted files use the previous
semantic comparison. Every check still reads the payload, so replacing a file
invalidates the comparison. ATOM discovery also avoids a duplicate cache check.

`REFEFF_CACHE=provenance` now fingerprints a complete successful file pipeline
after RDINP regenerates the current input handoffs. A matching executable,
execution policy, input handoffs and output payloads bypass scientific discovery
and computation. Cached stages still emit progress and honor cancellation.
Their stage durations are zero because the scientific stages did not execute;
whole-process timing includes fingerprint validation. Native typed captures bypass
both pipeline and stage file caches so their requested values reach observers,
including spectra intentionally omitted from text output. Unset or
`REFEFF_CACHE=audit` retains semantic validation.

The broader sweep also exposed dense full-potential FMS matrix construction as
the dominant Hubbard/CeO2 cost. Columns with mostly structural zero scattering
terms now visit only nonzero terms, preserving their original ascending index
and summation order. Dense columns and non-finite propagators retain the dense
path. All 73 FMS tests pass, including an equivalence test against the previous
dense formula covering signed zeros, subnormals, overflow and non-finite values.
Finite outputs match bit for bit; NaN results retain their IEEE classification.

Broader Hubbard regressions also exposed a pre-existing LDOS recovery bug:
discovery accepted an algebraically recoverable magnetic pair, and execution
then skipped writing its missing or malformed member. Execution now checks that
both files actually parse before skipping repair. Missing-source diagnostics
retain LDOS context and the underlying file error. All 81 Hubbard unit tests
pass; two old POT test counts now include the existing `chemical.dat` output.

## Validation and remaining failures

The full 44-entry sweep ran on clean revision
`5ec0cfde930b0dc150c72bd08d68eefba7289c7b`: **39 passed, four failed, one timed out**.
HIGHZ expanded to all 138 atomic numbers: 135 met their expected outcome,
including the two documented native failure cases; 71, 137 and 138 timed out.
The tested source stayed clean and its executable stayed unchanged.

After the FMS and LDOS fixes, clean final revision
`eaf009ff085b6e21174ec2be2fc6bbbe9faaca12` was tested on the two Hubbard workflows.
NiO passed and its six checked scientific files are byte-identical to the earlier
clean build. CeO2 completed in 342.10 seconds but still missed the golden
spectrum tolerance. Its output files match the earlier sparse implementation.
The pre-sparse CeO2 run timed out after 1,800 seconds. These workflow runtimes
were diagnostic runs with overlapping work, not controlled speedup measurements.

The entire stock matrix was not rerun on the final revision. It received the
focused Hubbard checks above. During implementation, targeted checks passed:
73 core FMS tests, 81 engine Hubbard tests, 86 LDOS tests, six cache tests, four
cache tests with optional features disabled, and five public API tests with
provenance enabled. The earlier implementation also passed 216 cache regressions
and the five feature configurations. Clean final commit checks passed formatting,
compilation, documentation and Clippy across all targets and features with
warnings denied.

| Workflow | Outstanding result |
| --- | --- |
| COMPTON/Cu | `compton.dat` exceeds the existing tolerance (maximum relative difference about 1.13e-6 against 1e-6); the full pipeline also stops at an unsupported XSPH boundary. |
| HIGHZ | Z=71, 137 and 138 exceeded the 60-second per-module deadline. |
| HUBBARD/CeO2 | Earlier build timed out; final build's mu relative L2 error is 7.172e-5 against 5e-5. |
| KSPACE/Cr2GeC | Relative-energy L2 error is 5.305e-5 against 5e-5. |
| KSPACE/Graphite | EELS total relative L2 error is 6.069e-2 against 5e-5; its absolute-error fallback also fails. |

No numerical tolerance was relaxed. These scientific discrepancies have not all
been classified against the original baseline. The separate release-readiness
check also rejects the existing unsupported reciprocal-FMS/active-Hubbard guard;
that guard was verified unchanged from the original source. No release was made.

## Method and provenance

Calculations ran sequentially during the benchmark, without other agent builds,
tests or calculations. Ordinary desktop activity and a pre-existing CPU-bound
Python process remained; load averages are recorded per sample. The script
rotates condition order between repetitions and discards two reuse warmups.
Every calculation uses absolute input/output paths and a disposable cwd.

Hashes cover `chi.dat`, `xmu.dat`, `paths.dat`, `feff.bin`, `phase.bin`, `pot.bin`
and `fms.bin` when produced. Matching input/thread/mode conditions are compared
between builds. Legacy audit reuse can re-encode handoffs, so fresh and audit
outputs are compared separately; provenance hits must preserve fresh output
bytes. Binary and input hashes were checked again after all 150 samples.

- Original source: `7517915464c72e0769d51cfb2ef7f6cd7592d360`.
- Final benchmark source: `eaf009ff085b6e21174ec2be2fc6bbbe9faaca12`.
- Compiler: `rustc 1.95.0 (59807616e 2026-04-14)`.
- Reference FEFF revision: `0a4fbd797cf72938f64dda034a438ce009ec6eb7`.
- Reference mode: pinned golden fixtures; native FEFF was not rerun.
- Original executable SHA-256: `259b9d4e8745fe7602563b909703195474262d3751a19470fe0394399b637cce`.
- Updated executable SHA-256: `293e3a963c13e3ff8be5454ca329159f0b8a385d0ded20e0fc6ae0866cd8acd6`.

Local evidence is under `target/performance-2026-09-06/`:

- `comparison.json` and `analysis-summary.json`: all 150 samples and medians.
- `initial-stage-profile.json`, `reuse-stack-sample.txt`: initial reuse bottleneck.
- `hubbard-ceo2-stack-sample.txt`, `sparse-hubbard-stack-sample.txt`: Hubbard profiles.
- `parity/workflow-summary.json` and `parity/reports/`: complete earlier stock sweep.
- `final-hubbard/`: final clean checks, hashes and scientific-file comparisons.
- `baseline-hubbard-failures.json`: baseline reproduction of the six fixed unit failures.
- `sparse-fms-tests.log`, `hubbard-repair-tests.log`, `ldos-repair-tests.log`:
  final numerical and recovery regressions.
- `final-commit-checks.log`, `final-release-build.log`: final clean build checks.

The matrix runner requires a clean checkout, defaults to a 30-minute workflow
limit and has a separate HIGHZ module deadline. `--resume` checks source,
executable, compiler, threads, retained inputs and references before retaining
completed results. Timeouts fail validation. Returned parity reports include
staged input hashes; outer workflow timeouts can lack a completed parity report.

```sh
python3 scripts/benchmark-comparison.py \
  --variant original=/absolute/path/to/original-refeff \
  --variant updated=/absolute/path/to/updated-refeff \
  --case ZnSe=crates/refeff/tests/data/znse.inp \
  --case BN=feff10/examples/XANES/BN/feff.inp \
  --threads 1 2 4 --samples 5 --provenance-variants updated \
  --output target/performance-2026-09-06/comparison.json
python3 scripts/run-parity-matrix.py --root /absolute/path/to/clean-checkout \
  --threads 4 --atomic-timeout 60 \
  --output target/performance-2026-09-06/parity
```
