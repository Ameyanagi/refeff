# September 2026 improvement work

This implements the repository review from 6 September. The numerical algorithms,
FEFF legacy constants and existing file codecs remain compatibility references.
Status below distinguishes implemented interfaces from follow-up measurement.

The subsequent [performance follow-up](PERFORMANCE_2026-09.md) profiles reuse,
removes repeated POT comparisons, and adds complete-pipeline provenance hits
plus controlled before/after and full-workflow validation tools.

| Review | Delivered change | Evidence and scope |
| --- | --- | --- |
| 1–3 | Explicit module workspace; one JSON envelope for results, errors, help and completions; semantic `check` and syntax-only mode | Nine real CLI subprocess contracts; 43 valid stock inputs accepted, one existing HIGHZ `XXX` placeholder rejected |
| 4–7 | Live stage/SCF/FMS events, cancellation/deadlines, owned Rayon pools, scoped faer policy, source-preserving errors, shared exit codes | Facade cancellation-before-write, deadline, sequential thread settings and terminal-event contracts |
| 8–9 | Reuse/recompute/conflict policies, read-only requested-stage plan, init example, artifact inspection, effective thread counts | CLI contracts; major-stage plan explicitly identifies conditional sub-stages |
| 10–14 | Evidence-aware report status, accurate timing labels, accessible filters, inspect/zoom/CSV/SVG/PNG, responsive/print tables, lazy decimated gallery | Five Node tests; Edge desktop and 390 px viewport inspection; real filter, zoom, point and SVG/PNG export checks |
| 15 | POT timer surrounds preparation, execution and publication | Fresh BN run reports POT at 7.63 seconds, instead of timing only its bookkeeping |
| 16–17 | Opt-in SHA-256 provenance cache; owned PATH and GENFMT preparation; no path-energy computation during discovery; generated GENFMT comparisons reused for publication | Cache invalidation/symlink/retained-artifact contracts; repeated PATH/GENFMT/FF2X fast hits; legacy semantic audit remains default |
| 18–20 | Dependency-only FF2X scratch with immediate RAII cleanup; in-place LU RHS; reusable propagator scratch and ordered bounded FMS batches | 121 FF2X and 72 core FMS tests; BN scientific files identical to baseline |
| 21 | Early artifact selection; typed chi/xmu and PATH results; ordinary typed-only EXAFS omits final chi/xmu text | Release integration compares typed-only and serialized EXAFS; intermediate stage handoffs still use files |
| 22 | Inline short path candidates and borrowed GENFMT finalizations | 39 core PATH and 182 core GENFMT tests; preserved traversal order and legacy normalization |
| 23 | Runtime cold/reuse/recompute/memory/typed/provenance/thread matrix; isolated clean-build matrix tool; RSS, artifact bytes and build/input provenance | Local ZnSe distributions and before/after output checks; compile tool distinguishes a warm download cache from a clean target directory |
| 24 | Symlink-safe artifact inventory; cached-stage artifact manifests; unchanged unrelated files excluded; artifact selection before payload reads | Facade cycle/escape and selection tests; retained registered files require a completed producer |
| 25–28 | Scheduler, module registry, FF2X workspace and large test-module extraction; shared identities; codec stream helpers; checked units and phase/spectrum accessors | Workspace, all-feature Clippy, feature matrix and documentation checks; public root aliases retained |
| 29 | Strict opt-in PAD encoding with unchanged legacy default | Fourteen PAD tests; pinned Fortran reproduces the documented boundary anomalies |
| 30–31 | Behavioral CI and feature matrix; exact-revision release parity gate, input payload hashes, metadata-derived versions | Local contract and tampering tests; Actions and registry publication were not invoked |
| 32 | Actual full-run and typed-spectrum examples; refreshed entry-point documentation and current backlog record | Examples compile and the typed-spectrum path runs in the release integration test |

## Validation record

Local artifacts are under `target/review-2026-09-06/` (ignored build data):

- `final-checks.log`: facade/CLI/linalg contracts, cached-workflow regressions,
  cache fingerprints and release typed-output integration.
- `final-interface-checks.log`: final all-feature Clippy, path-aware codec errors,
  cache artifact retention, CLI/facade contracts and release typed-output checks.
- `artifact-manifest-comparison.json`: a fresh audit run and three provenance
  runs report the same 86 artifacts; the last run has no cache-fallback diagnostics.
- `build-smoke.json`: one successful clean-target, no-default-features library
  build exercises the compile harness. This overlapped other validation and is
  a smoke check, not an uncontended compile-time baseline.
- `before-after.json`: three samples for cold and warm ZnSe, at one and two
  threads. All six checked files (`chi.dat`, `xmu.dat`, `paths.dat`, `feff.bin`,
  `phase.bin`, `pot.bin`) match the clean baseline byte for byte.
- `bn-before-after.json`: fresh SCF/FMS BN, two threads. All six checked scientific
  files, including `fms.bin`, match the baseline byte for byte.
- `runtime-final.json`: three runtime/RSS samples per mode and thread count,
  collected after the local builds/tests finished. At one thread, median cold
  generation was 1.44 s; memory mode 1.44 s; typed-only 1.42 s; validated reuse
  2.11 s. Typed-only returned zero raw artifact bytes versus about 5.81 MB for
  memory mode. These small runtime differences do not establish a general
  speedup. These samples precede the final cache-artifact bookkeeping correction.
  Earlier `benchmark-matrix.json` and before/after timing samples
  overlapped other local work and are preliminary.
- `ui-validation.html`, `ui-mobile.html`, `ui-export.svg`, `ui-export.png`:
  explicitly synthetic presentation fixtures and real browser exports.

The broader cache suite exposed old assertions that omitted the already-produced
`chemical.dat`. These now check the current count and its presence. The MnF2
fixture mixed loose FEFF10 inputs with a different archived potential result;
its test now regenerates modern handoffs from the archived input. The thermal
XANES fixture likewise now carries the same 17-energy contour in every handoff.
Numerical tolerances were not relaxed.

This validation covers the changed interfaces, numerical kernels and representative
full calculations. It is not a newly certified complete FEFF release matrix.
The publishing gate still requires clean, exact-revision local parity evidence.
Full screen-reader qualification, every print driver, and a complete clean-build
performance distribution remain outside this local verification sample.

## Measurement commands

```sh
cargo build --release -p refeff --example benchmark
python3 scripts/benchmark-matrix.py --binary target/release/examples/benchmark \
  --input crates/refeff/tests/data/znse.inp --threads 1 2 4 --samples 5 \
  --output target/runtime-matrix.json
python3 scripts/benchmark-builds.py --profile release --samples 3 \
  --output target/compile-matrix.json
```

The runtime harness creates disposable workspaces and uses absolute input/output
paths. Use self-contained inputs for its memory modes. Run performance measurements
without other builds or calculations, and retain all samples rather than just
reporting the fastest run.

## Execution limits

Cancellation is cooperative. It is checked at scheduler, SCF and FMS energy
boundaries; an active matrix factorization must return before it can stop.
A calculation owns its Rayon pool. ReFEFF serializes its faer set/run/restore
scope because faer's high-level APIs read a global policy; unrelated faer users
must coordinate externally. Recompute replaces the entire output directory.

`REFEFF_CACHE=provenance` opts into fingerprint validation. Unset or `audit` uses
legacy regeneration comparisons. Fast hits restore the cached stage’s generated-file
manifest as well as its count. Fingerprints include executable identity,
features, thread/environment policy, workspace filenames and payload hashes.
External auxiliary references, symlinks, or excessive workspace size disable the
fast path. This conservative first version may invalidate when unrelated files
change; it must never claim a speedup without measurement.

## PAD compatibility evidence

The pinned `feff10/src/COMMON/padlib.f90` compiled with local gfortran reproduces:

- `1.1099547846496504e-18`, width 3: negative decoded value (sign carry).
- `-0.0009999999999980088`, width 9: byte values 33 and 225, outside printable PAD;
  Rust's legacy encoder rejects the out-of-range byte rather than wrapping it.
- `1e38`, width 8: decodes as zero.

`encode_f64_strict` accepts finite zero or magnitudes between the exclusive
`1e-38` and `1e38` boundaries, widths 4–8, and verifies the result by a round trip.
The default encoder remains unchanged. Detailed local probe output and source
hash are in `target/review-2026-09-06/pad-fortran-probe.json`.

## Release evidence workflow

Run the complete existing FEFF parity workflow matrix on the intended local
hardware after committing the tested revision. Its summary must identify that
revision in `provenance.rustCommit`, set `dirty: false`, record `rustCompiler`,
`feffCommit`, `rustBinarySha256` and `feffDriverSha256`, list each stock workflow
once and mark each `status` as `pass`. Capture `input_sha256` (relative filename
to SHA-256) for every actual input/auxiliary file at test time. The packager
rejects missing provenance or inputs that changed afterward. It also runs the
unfiltered strict `release-readiness` audit locally, where pinned golden fixtures
and their provenance are available, and bundles its successful report. Package
the summary and the actual retained input files:

```sh
python3 scripts/package-release-evidence.py --summary workflow-summary.json \
  --inputs path/to/parity/inputs --output parity-evidence.zip
```

Store that ZIP at an HTTPS URL accessible to Actions, and run **Record local
parity evidence** for the tested revision. It verifies source revision,
toolchain, reference revision, complete inventory, every bundled input hash and
the complete successful local readiness audit,
then produces the `release-parity-evidence` artifact. Supply that successful run
ID to the publish workflow. Publishing repeats ordinary CI and source-only
module, compatibility and evidence-reference checks. Fixture presence and
provenance remain mandatory in the bundled local audit; the publishing runner
does not contain those fixtures. Schema version 2 rejects old evidence without
that audit. All evidence must match the exact release commit, and package
versions come from Cargo metadata. No release has been published by this change.
