# TODO — API & Functionality Improvements

Findings from a multi-agent review (2026-07-08) of the workspace, focused on API
design and functionality — not on finishing the port itself (that is tracked in
`docs/FEFF_RUST_PORT_PLAN.md` and the xtask gates). Items are grouped into
workstreams that can be worked on in parallel; ordering constraints are noted
per workstream. Tags: priority (P1 > P3), effort (S/M/L).

Cross-reviewer consensus (flagged independently by 3+ reviewers):

1. Both library crates flatten ~2000 symbols into their crate roots — the
   internal porting surface looks like the stable API (A1, B4).
2. There is no programmatic way to run FEFF — all orchestration is
   `pub(crate)` in refeff-cli behind `println!` and anyhow (A3).
3. `IoError` is a 1009-line, ~168-variant enum and parse errors don't carry
   the file path (C1, C2).
4. rayon is declared but used exactly once in the whole workspace; every
   per-energy / per-path loop is sequential (E1–E6).
5. `refeff run` prints nothing on success and has no plan/dry-run, check,
   JSON, or progress output (D2–D5).

---

## Wave-1 status (2026-07-08, multi-agent implementation)

Checked items were implemented and verified (fmt/check/clippy/doc gates clean
workspace-wide). Known partials and caveats:

- **G1** is metadata-only: descriptions/keywords/dep-versions are in, but the
  **LICENSE decision is still open** (FEFF10 upstream is restrictive; the
  derived-work-vs-clean-room statement needs an owner decision) — no LICENSE
  file or `license` field yet.
- **A7**: enums under `src/fms/` were deliberately left out of
  `#[non_exhaustive]`/the `Error` umbrella (documented in
  `refeff-core/src/error.rs`) — finish once fms work settles.
- **E3**: `retain_*` flags landed; the caller-provided scratch-buffer reuse
  for the system matrix was deferred (parity risk under time pressure).
- **F3/F6**: helpers landed with a representative subset of call sites
  migrated; remaining `assert_close`/silent-skip sites migrate mechanically.
- **F7** documented 3 genuine `pad.rs` `encode_f64` bugs as `known_bug_*`
  tests (npack=3 rounding-carry sign flip, wide-npack rounding-drift error,
  huge-boundary values decoding as zero) — triage against FEFF10's Fortran PAD
  before "fixing": they may be faithful ports.

Port-completion note (2026-07-26): the strict compatibility matrix is closed
at 98/98. The canonical fresh `XANES/BN` port item is closed at approximately
`1–2e-5` relative L2 parity after keeping POT
independent centers in slot zero and preserving saved SCMT retry state,
freezing `sqrt(rhoint)` as the plasmon value, carrying raw-Hartree `emu` with
separate XSPH `ixc0`/`ixc` roles, and applying FMS reversed-axis rotations
from the original vectors. The unchecked items below are API, maintenance,
performance, or packaging improvements rather than missing FEFF production
branches.

---

## A. Public API surface & programmatic use

Ordering: A1/A2 are independent. A3 and A4 should agree on the driver shape
first (one design note), then individual stages migrate in parallel.

- [ ] **A1 (P1, L)** Replace the flat root re-exports with module-scoped API + curated prelude.
  `crates/refeff-core/src/lib.rs` re-exports ~2000 items (976 pub fns, 1186 pub
  types) with prefix pseudo-namespacing (`AtomicDiracEnergyDisagreementCorrectionInput`,
  `BandKkrBandEnergiesFromKspacePhaseNonRelGridInput`); `crates/refeff-io/src/lib.rs`
  spends ~500 of 599 lines on `pub use` blocks. Make module paths canonical
  (`refeff_core::fms::ScatteringInput`), shorten type names by dropping the
  module prefix, keep the root to `Real`/`Complex`, `FeffDimensions`, errors,
  and a small `prelude`. For refeff-io, add grouped facades
  (`spectra`, `module_inputs`, `bins`) or a prelude with the ~20 types users
  actually need. Mechanical per module — parallelizable one module at a time
  with temporary deprecated root aliases.

- [x] **A2 (P1, S)** Centralize physical constants with documented legacy variants.
  Hartree/eV appears as 27.211_396, 27.21160, and 27.211_396_132; three
  different Bohr radii exist across `debye.rs`, `sfconv.rs`,
  `screen/constants.rs`, `fms/internals/geometry.rs`. Add
  `refeff_core::constants` with canonical values plus explicitly named legacy
  variants (`HARTREE_EV_SFCONV_LEGACY`, …) whose docs cite the Fortran file
  that hardcodes each value. Anchor for A5.

- [ ] **A3 (P1, L)** Extract a typed programmatic run API (facade/driver crate).
  All sequencing lives in private refeff-cli modules (`pot.rs` is 3875 lines)
  keyed on files-on-disk + anyhow; the only entry points are `run_cli`/`run_rdinp`
  which print to stdout. Create a `refeff` (or `refeff-driver`) library crate:
  `RunConfig` builder, `Module` enum, `RunReport`/`StageReport` returned as
  values, thiserror errors, file handoffs optional at the edges. refeff-cli
  becomes a thin clap wrapper. Prove the shape on one source-backed stage
  (fms or genfmt), then migrate stages in parallel. Prerequisite for PyO3
  bindings / GUI / batch drivers.

- [ ] **A4 (P1, L)** Add composed per-module entry points in refeff-core above the micro-kernels.
  genfmt exports ~100 step functions mirroring the Fortran call graph and only
  refeff-cli (`crates/refeff-cli/src/genfmt.rs:580+`) knows the calling order;
  same for xsph's ~90 `xsph_xsect_*` kernels and the density `PotScf*` steps.
  Add one or two composed in-memory drivers per module
  (`genfmt::evaluate_ordinary_paths(...)`, `fms::solve(...)`) and refactor the
  CLI to call them so sequencing has a single source of truth. Module-by-module,
  parallel-safe.

- [ ] **A5 (P2, M)** Introduce unit newtypes at driver boundaries.
  Everything is bare f64/f32 with units only in inconsistent field-name
  suffixes; FMS geometry is f32 Angstrom while atomic solvers are f64 Bohr.
  Add `#[repr(transparent)]` `Hartree(f64)`, `ElectronVolt(f64)`, `Bohr(f64)`,
  `Angstrom(f64)` wired to A2's constants. Scope to top-level Input/Result
  structs and the new composed entry points (A4) — not a wholesale retrofit;
  bulk ndarray storage keeps documented-unit fields.

- [ ] **A6 (P2, M)** Give Fortran-named public kernels Rust-canonical names with `#[doc(alias)]`.
  `besjh`/`terp`/`conv`/`xstar`/`strap` leak into the public API. Invert the
  existing `conv as lorentz_convolve` pattern: descriptive name canonical,
  `#[doc(alias = "besjh")]` keeps the Fortran name searchable for people
  cross-referencing `feff10/`. Mechanical, module-local, parallel-safe.

- [x] **A7 (P2, S)** Mark all public error enums `#[non_exhaustive]` and add a crate-level `Error` umbrella.
  40+ per-module thiserror enums in refeff-core, none non_exhaustive — every
  new variant is a semver break while the port is still growing. Add
  `refeff_core::Error` with `#[from]` variants and a `Result<T>` alias.

- [ ] **A8 (P3, M)** Wrap FEFF packed-array layouts / 1-based index conventions in typed accessors.
  E.g. `PhaseShiftTable` with `by(energy, l, spin, potential)` matching
  `FeffDimensions::phase_shape`, a `ScatteringMatrices` wrapper for
  `FmsScatteringResult.scattering` ("Packed gg(channel1,channel2,potential)"
  is currently prose-only). Apply to the most-consumed tables first.

## B. refeff-io codec architecture

Ordering: B1 first (B2, B3, B6 build on it or fall out of it). B5 independent.

- [ ] **B1 (P1, L)** Introduce a `FeffCodec` trait unifying the parse/render/read/write quadruplet.
  ~100 codec modules hand-write the same four functions; 109 `read_*` fns with
  93 identical `fs::read_to_string(...).map_err(...)` sites. Trait with
  `parse`/`render` (+ bytes variant for binary codecs) and provided
  `read`/`write` centralizing IO and error-path attachment; existing free
  functions stay as thin wrappers. Enables a generic golden round-trip harness
  and the registry (B2). Also resolve the two competing parse-API styles
  (`PotInput::parse_str(source, text)` vs free `parse_x(text)` vs
  no-source `GridInput::parse_str`) as part of the same convention.

- [ ] **B2 (P1, M)** Add a FEFF file-format registry; generalize `refeff inspect` to all known files.
  Filename knowledge (`chipNNNN.dat`, `ldosNN.dat`, `gtrNN.bin`) is scattered
  in refeff-cli string literals. Static `FormatDescriptor { name, matcher,
  kind, producing_module, parse_fn }` table in refeff-io; `refeff inspect
  <file>` identifies, parses, and summarizes any FEFF file with validation
  errors. Single source of truth for cached-output validators and the
  compatibility matrix.

- [x] **B3 (P2, M)** Provide a feff.inp writer (`FeffDocument -> text`).
  Every derived module input has a `*_inp_string`, but the flagship input has
  no inverse — programmatic input generation means string templating. Emit
  canonical cards reusing `format.rs` helpers; unlocks
  `refeff convert structure.cif feff.inp` on top of the existing
  `expand_cif_structure`, plus parse→render→parse round-trip tests.

- [ ] **B4 (P1, L)** Extract pipeline "handoff" glue out of refeff-io into a bridge layer.
  48 of 112 top-level modules contain handoff builders that are physics wiring,
  not codecs (`screen_dat.rs` is 5604 lines; 73 files import refeff_core;
  `IoError` embeds `BandError`/`RhorrpError`/`SfconvError`). Move
  `*_handoff*`/`*_from_handoffs` into a `refeff-bridge` crate (or refeff-cli),
  restoring clean layering (core has no io, io has no physics). Coordinate
  with A3. Not parallel-safe with B1/C1 — sequence it.

- [ ] **B5 (P2, L)** Split `FeffDocument`'s ~100-field flat struct into grouped sections with builders.
  Group into `Structure`, `Spectroscopy`, `ScfControls`, `PolarizationGeometry`,
  `ModuleToggles` with per-group `Default` + builder so programmatic
  construction (needed by B3) names only deviations from FEFF defaults;
  `from_input` delegates per group so cross-card fixups live next to their
  fields.

- [x] **B6 (P3, S)** Replace ad-hoc `(width, precision)` Fortran-format args with a `FortranField` spec type.
  Column layouts — the thing golden compatibility depends on — are invisible
  magic numbers (`write_fortran_exp(out, *first, 13, 6)`). Named consts like
  `const CHI_ROW: [FortranField; 4]` document FEFF's format strings in code.
  Incrementally adoptable per codec.

## C. Errors & diagnostics

Ordering: C1 and C2 together define the error shape; C3–C6 are independent.

- [ ] **C1 (P1, L)** Collapse the 1009-line, ~168-variant `IoError` into a format-tagged structured error.
  Most variants are copy-pasted per-format families (`{Fmt}Parse`, `{Fmt}Shape`,
  `{Fmt}RowWidth`, …). Replace with `IoError::Codec { format: FileFormat,
  path: Option<PathBuf>, kind: CodecErrorKind }`; ~85% shrink, uniform
  messages, matching by error class. Keep genuinely unique variants (PAD,
  include-depth) separate.

- [ ] **C2 (P1, M)** Attach the file path at the read boundary.
  `parse_*` errors carry line/field but no path; the CLI compensates with
  ~190 `with_context` sites in `ff2x.rs` alone, inconsistently — a bad
  `chipNNNN.dat` fails without naming which of dozens of files. Wrap parse
  errors with the path inside `read_*` (natural with B1/C1), then delete the
  redundant CLI context boilerplate.

- [ ] **C3 (P1, M)** Replace stringly rdinp card errors with structured card diagnostics.
  140 `line: 0` sentinel sites; `rdinp/log.rs:106-160` dispatches on message
  prose (`message.starts_with("HOLE requires")`) so rewording an error breaks
  legacy log.dat emission. Add `IoError::Card { card, path, line:
  Option<usize>, kind: CardErrorKind }`; the log renderer matches on
  (card, kind).

- [ ] **C4 (P1, M)** Add physics context to numerical failures at solver loop boundaries.
  Bare `#[from] LinalgError` means an FMS failure reads "matrix is singular at
  pivot 42" with no energy/spin/potential — the exact info needed to decide to
  shrink the FMS radius or shift the grid. Add wrap variants like
  `FmsError::EnergyPointSolve { energy_index, energy_ev, spin, source }` in
  fms/xsph/band/kspace drivers; extend `IterativeSolverNoConvergence` with
  energy point and achieved-vs-requested tolerance.

- [ ] **C5 (P2, M)** Introduce a structured warning sink (FEFF WARNING equivalent).
  No logging infra at all; FEFF emits non-fatal WARNING lines and continues
  (cf. `warn_ion` plumbed through `atomic.rs:6730` with nowhere to report).
  `Diagnostics` collected per module run, rendered as `WARNING(pot): ...` on
  stderr, mirrored into the FEFF-compatible `log*.dat`, counted in the module
  summary.

- [ ] **C6 (P2, M)** Give feff.inp parse errors caret-style rendering in the CLI.
  The data (path, line, raw `FeffLine`) already exists in
  `rdinp/log_helpers.rs:161-230`; render the offending line, a marker, and a
  hint (did-you-mean via Levenshtein against the card table). Hand-rolled
  ~50-line renderer or miette behind a feature flag. Builds on C3.

- [x] **C7 (P2, S)** Eliminate the remaining panic paths in refeff-core; escalate lints to deny.
  `.expect("g0 shape")` at `fms/pairs.rs:234,322`, `unreachable!()` at
  `density/scf.rs:149`, `xsph/planning.rs:235,261`, `debye/spring.rs:176,195`.
  A panic in an hours-long SCF run loses everything with no diagnostic. Fix
  the sites, then deny `unwrap_used`/`expect_used`/`panic` at crate level with
  per-site justified allows for tests/benches.

- [ ] **C8 (P2, M)** Standardize module-stage error framing in refeff-cli; fix the misleading run summary.
  `run_feff_to_dir`'s outermost failure frame reads like a success report
  ("FEFF run completed rdinp for N cards…"). Add `run_stage(stage, dir, f)`
  applying one canonical frame; reword the outer context to state failure
  first.

## D. CLI UX

All independent and mostly small — good first tasks. D3 depends on nothing but
pairs well with C-series.

- [x] **D1 (P1, S)** Write help text for every subcommand and argument.
  `refeff --help` currently shows four subcommands with blank description
  columns. Doc comments on the `Command` variants/fields, `after_help` on
  `module` listing supported names, top-level typical-workflow blurb, `about`
  for the standalone bins.

- [x] **D2 (P1, M)** Make `refeff run` report per-stage progress instead of succeeding silently.
  A full run producing ~50 files prints zero lines; `SupportedModuleReport` is
  only surfaced on failure. One line per stage
  (`[3/20] xsph: generated phase.bin (4 files, 1.2s)` / `pot: reused cached
  pot.bin`), final summary on success, `-v/-q` flags, print the rdinp summary
  at run start.

- [ ] **D3 (P1, L)** Add `refeff plan` / `run --dry-run` explaining cache and handoff decisions.
  The scheduler is a cascade of boolean predicates that swallow errors into
  `Ok(false)`, so users can't see why a stage was skipped or regenerated.
  Introduce `StageDecision` (RunFromCache / RegenerateFromHandoff / RunFresh /
  Skip{reason}) returned by each module's predicate; print the decision table
  in dry-run and the same reasons during real runs.

- [x] **D4 (P1, M)** Add `refeff check` — validate feff.inp with no side effects.
  Today semantic errors surface only during rdinp/run, which writes
  `.feff.error` and `log.dat` even when the user just wants validation.
  Parse + build `FeffDocument`, report card-located problems, exit nonzero for
  scripts/CI; extend with scientist-facing warnings (unused ipot, missing
  HOLE/EDGE, suspicious lattice constants). Fold `inspect` in as an alias.

- [x] **D5 (P2, S)** Replace the free-form module-name string with a clap `ValueEnum`.
  `refeff module bogus` currently parses the input first, then errors with
  "module bogus is not implemented yet; parsed 3 active lines". ValueEnum with
  `#[value(alias)]` for the existing aliases gives possible-values in help and
  did-you-mean for free; keep a distinct message for recognized-but-unported
  modules. Public API becomes `run_module(ModuleName, ...)`.

- [x] **D6 (P2, M)** Add `--json` machine-readable reports for inspect/check/run/module.
  `RdinpReport`/`SupportedModuleReport` are already structured; derive
  Serialize, emit one JSON document per invocation (per-stage
  `{name, status, count, outputs, duration_ms}`), human text to stderr when
  active. Reuse the xtask `--json-out` conventions. See also G3 (serde).

- [x] **D7 (P2, M)** Unify working-directory and `--output` semantics across run and module.
  `run -i a/feff.inp -o b` and `module pot -i a/feff.inp` write to different,
  undocumented places (`module` silently uses the input's parent dir). Add a
  global git-style `-C/--dir` or give `module` the same `-o`; state the rule in
  help text.

- [x] **D8 (P2, S)** Complete (or intentionally drop) FEFF10-style standalone module binaries.
  Cargo.toml ships bins for an arbitrary subset (rdinp, pot, atomic, band,
  mdff) but not xsph/fms/path/genfmt/ff2x that FEFF10 also ships. Either
  generate all wrappers from one shared `module_main` or remove the extras and
  document the `refeff module <name>` mapping.

- [x] **D9 (P3, S)** Shell completions + documented exit-code taxonomy.
  `clap_complete` subcommand; exit codes (0 ok, 2 usage, 3 invalid input,
  4 unsupported/not-yet-ported stage) so snakemake/nextflow pipelines can
  distinguish failure classes.

## E. Performance & parallelism

Ordering: E7 (benchmarks) should land first to quantify the rest. E2 → E1.
Everything is per-module and parallelizable across workers.

- [x] **E7 (P1, S)** Add realistic-size FMS benchmarks first.
  Current FMS benches use 2-3 atoms / matrix order 8; real XANES clusters are
  order thousands. Synthesize FCC/rocksalt clusters (~87 and ~177 atoms,
  lmax=3 → orders ~1400/~2800): one LU solve, assembly alone, and an 8-energy
  sweep. Prerequisite for judging E1–E5.

- [x] **E2 (P1, M)** Split energy-independent setup out of `fms_real_space_energy` into a reusable `FmsRealSpacePlan`.
  `fms_driver_setup` rebuilds state kets / lmax tables / offsets on every one
  of hundreds of energy points. Plan is built once, is `Sync`, and directly
  enables E1. Keep the existing function as a thin wrapper.

- [x] **E1 (P1, M)** Parallelize per-energy FMS loops with rayon.
  FEFF10's main speedup is MPI over energy points; the port runs all five
  per-energy loops in `crates/refeff-cli/src/fms.rs` (lines ~1317, 1470, 1648,
  1838, 2063) sequentially even though `fms_real_space_energy` is pure. Add a
  core-level `fms_real_space_spectrum(plan, energies)` doing
  `into_par_iter().map(...)`. Single highest-leverage performance change; no
  algorithmic risk.

- [x] **E3 (P2, M)** Make heavy FMS intermediates opt-in.
  `FmsRealSpaceEnergyResult` unconditionally returns pair tables, the full
  N² free propagator, t-matrix, and system matrix; the CLI drops all of it
  each iteration. Under a parallel loop this multiplies peak memory by thread
  count. Flags → `Option<...>` fields, plus caller-provided scratch for the
  system matrix.

- [x] **E4 (P2, S)** Eliminate per-call ndarray↔faer copies in refeff-linalg.
  `Mat::from_fn` element-wise copies + RHS/solution copies on every solve,
  thousands of avoidable N² copies per run. FMS matrices are already
  column-major (`.f()` throughout) — add zero-copy
  `MatRef::from_column_major_slice` paths and `*_solve_in_place` variants.

- [x] **E5 (P2, S)** Expose thread-count control (`--threads`, `REFEFF_THREADS`).
  faer already multithreads the LU silently while everything else is serial;
  HPC users need bounded, deterministic threads (SLURM). Build the rayon
  global pool + `faer::set_global_parallelism`; `--threads 1` gives a
  deterministic run for golden validation. Add
  `refeff_linalg::set_parallelism` for library users.

- [ ] **E6 (P2, L)** Decouple GENFMT per-path evaluation from the sequential normalization chain.
  `current_normalization` threading forces sequential evaluation of an
  embarrassingly parallel loop over hundreds of paths, but normalization only
  affects importance filtering. Split into parallel
  `genfmt_ordinary_path_compute` + cheap serial `finalize_path_sequence`
  keeping exact FEFF-order semantics and golden output.

- [ ] **E8 (P2, M)** Parallelize XSPH per-energy loops after hoisting the Fermi cache.
  The only cross-iteration dependency in the loops at `xsph.rs:7029, 5123,
  5463, 8558` is the lazily-populated `fermi_cache`. Compute it explicitly up
  front (`xcpot_fermi_cache(...)`), then `into_par_iter()` the energy grid.
  Independent of the FMS work.

- [ ] **E9 (P3, L)** In-memory handoff cache (`RunContext`) for full runs.
  `run_feff_to_dir` chains modules purely through the filesystem — `phase.bin`
  is parsed 3+ times per run. Cache parsed handoff structs keyed by path,
  populated when a stage writes; disk files still written for FEFF
  compatibility. Also the seam for running independent modules (ldos, dmdw,
  eels) concurrently. Coordinate with A3.

- [ ] **E10 (P3, M)** Move the parallel Dirac channel solver out of `refeff-io/src/screen_dat.rs` into refeff-core.
  The workspace's only rayon usage is a FOVRG solve loop inside a codec module
  (io doing physics). Extract `fovrg_solve_channels` into refeff-core so pot
  SCF / rhorrp can reuse it. Overlaps with B4.

## F. Testing & parity infrastructure

Ordering: F5 (GoldenCase) supports F1; otherwise independent.

- [x] **F1 (P1, L)** Add `cargo xtask parity --example XANES/BN` — run Rust, gate the canonical workflow output and report every file diff against golden.
  Parity evidence currently lives only inside 11,100-line test modules; a
  first failing assert aborts with no overall picture. Per-file table (max
  abs/rel, RMS, first divergence, pass/fail) + JSON artifact. Auxiliary FEFF
  diagnostics remain visible without overriding the workflow's canonical
  output contract; per-format tolerance dispatch uses refeff-io readers with a
  generic Fortran-float text differ fallback. Becomes the parity front door
  the release gate cites.

- [x] **F2 (P1, M)** Write provenance manifests for golden fixtures and validate them.
  Golden trees and the 47 REFERENCE.zip usages record nothing about which
  FEFF10 commit/compiler/flags produced them — and compiler choice measurably
  changes FEFF numerics. `generate-golden` emits `manifest.json` (feff10 rev,
  compiler, flags, checksums); compatibility-matrix verifies, with
  `--fail-on-stale-fixtures`.

- [x] **F3 (P1, M)** Make fixture-gated test skips visible and enforceable.
  ~151 tests silently `return Ok(())` when fixtures are absent — a renamed
  fixture dir would disable dozens of parity tests forever, invisibly.
  `require_fixture!` helper: panics under `REFEFF_REQUIRE_FIXTURES=1` (CI
  parity job), otherwise appends to a skip ledger reported as "N parity tests
  skipped".

- [x] **F4 (P1, M)** Add CI with tiered gates.
  `.github/workflows/ci.yml` runs the Rust 1.95 quality suite on Linux and
  macOS (including `cargo clippy --workspace --all-targets --all-features
  --locked -- -D warnings`), builds provenance-tracked fixtures from the
  pinned FEFF10 revision, then runs required-fixture release tests, strict
  release readiness, and every runnable stock-workflow parity comparison on
  both platforms. The fixture-skip ledger must be empty in the parity tier.

- [x] **F5 (P2, M)** Replace the 38 bespoke fixture-lookup helpers with a `GoldenCase` API; drop the `unzip` subprocess.
  `GoldenCase::locate("XANES/BN")?.require_files([...])` backed by the pure-Rust
  `zip` crate (the current per-entry `unzip` subprocess is slow and
  Windows-hostile). Serves F1 and F3.

- [x] **F6 (P2, M)** Centralize numeric tolerance policy.
  ~120 `assert_close` call sites with unexplained magic tolerances (58× 1e-12,
  22× 5e-5, one-off 4.2e-6…), reporting only the first failing pair. Named
  profiles (`Tol::PHASE_SHIFT`, `Tol::XMU`) combining rel+abs floors; array
  comparators reporting max-abs/max-rel/RMS + offending index. Shared with F1.

- [x] **F7 (P2, M)** Add proptest round-trip coverage for PAD and high-traffic codecs.
  No property testing anywhere; PAD (`pad.rs` encode/decode) is exactly the
  lossy fixed-width codec where hand-picked values miss subnormals/exponent
  boundaries. Start with pad.rs + 3-5 codecs; commit regression seeds.

- [x] **F8 (P2, S)** Verify compatibility-matrix evidence strings against the real test inventory.
  ~92 hardcoded rows name test functions in free-form strings; a test rename
  silently rots the release gate. xtask self-check greps the `#[test]`
  inventory and fails on dangling references.

- [x] **F9 (P3, S)** Replace hand-rolled JSON in xtask with serde_json.
  ~400 lines of manual escaping/trailing-comma bookkeeping for machine-consumed
  artifacts. Derive Serialize on the report structs, delete `xtask/src/json.rs`.

- [ ] **F10 (P3, M)** Adopt insta snapshot tests for Fortran-formatted text writers.
  Whitespace-sensitive expected outputs are hand-embedded `concat!` string
  literals; `cargo insta review` replaces hand-editing them. Keep
  golden-derived byte comparisons for parity.

- [ ] **F11 (P3, M)** Extend `bench-e2e` beyond rdinp to per-module pipeline timing.
  `--modules pot,xsph,fms` selector timing Rust vs the Fortran per-module
  binaries, JSON output for CI regression tracking.

## G. Packaging, docs & distribution

All independent, mostly small.

- [ ] **G1 (P1, S)** Add publication metadata and a LICENSE decision to every crate.
  No `description`/`license`/`keywords`/`categories` anywhere; no LICENSE file;
  path deps lack `version` so `cargo publish` fails outright. NOTICE.md
  acknowledges FEFF10's restrictive copyright but never states the port's own
  license — **the licensing/derived-work question blocks any distribution and
  needs an explicit decision.**

- [x] **G2 (P1, M)** Rewrite README as a user-facing document; generate the status section.
  Lines 13–684 are an ever-growing kernel changelog; there's no install,
  quickstart, or "which spectroscopies work end-to-end today". Cut to pitch +
  quickstart + per-module support table generated from the existing
  `port-status`/`release-readiness` JSON so it can't rot; move the bullet log
  to docs/CHANGELOG.md.

- [ ] **G3 (P2, M)** Add an optional `serde` feature; versioned JSON schema for results.
  serde is absent from the whole workspace. Feature-gated derives on the typed
  result structs (`XmuDatData`, `ChiDatData`, core Input/Result structs),
  `refeff export --format json <file.dat>`, and a `schema_version` field so
  external tools (xraylarch-style pipelines, plotting) stop shipping their own
  FEFF parsers. Pairs with D6 and F9.

- [x] **G4 (P2, S)** Populate the empty `examples/` directories.
  `crates/refeff-cli/examples/` exists and is empty; zero doctests in ~340
  refeff-core files. 3–5 examples (parse feff.inp, read xmu/chi into ndarray,
  full run into a tempdir, one core kernel with units documented), built in CI.

- [ ] **G5 (P2, M)** Turn on `missing_docs` (warn) + `rustdoc::broken_intra_doc_links` (deny); write real crate-level docs.
  refeff-core's lib.rs has a 4-line comment for 50 public modules; refeff-io
  never documents its module-per-FEFF-file naming convention. Each lib.rs gets
  a module map (physics pipeline: atomic → pot → xsph → fms/path → genfmt →
  ff2x, and which FEFF Fortran dir each module ports).

- [ ] **G6 (P3, M)** Progress/logging hooks for long-running stages.
  Neither `log` nor `tracing` exists; SCF and FMS run for minutes with no
  feedback and no way for a future GUI/PyO3 consumer to observe progress.
  `tracing` in refeff-cli (`-v/-q`), a light `ProgressSink` callback trait in
  refeff-core's long drivers (pot SCF loop, FMS energy loop) to stay
  dependency-light. Pairs with D2 and C5.

- [ ] **G7 (P3, S)** Keep clap out of the public library API.
  `Cli`/`Command` are `pub` with clap derives, so a clap major bump breaks
  library consumers. Once A3 lands, clap types become private to the binaries.
