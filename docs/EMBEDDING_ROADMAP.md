# Embedding roadmap

This roadmap turns the XrayTsubaki integration feedback into incremental work
that preserves FEFF numerical parity at each step.

## Completed foundation

- `refeff-engine` owns the numerical modules and pipeline scheduler.
- `refeff-cli` is a thin Clap frontend and FEFF-compatible executable layer.
- The typed `refeff` facade depends directly on `refeff-engine`.
- CI rejects any normal `refeff` dependency on `refeff-cli` or Clap.

Moving the stages and scheduler together is important: private POT/atomic/FMS
handoff types stay inside one crate, so the extraction does not require a broad
new public API or a dependency cycle.

## Recommended implementation order

### 1. Typed EXAFS outputs and early artifact selection

Add an engine result that retains the owned FF2X output before it is written:

```rust
pub struct ExafsResult {
    pub chi: ChiDatData,
    pub xmu: XmuDatData,
    pub paths: Vec<ExafsPath>,
}
```

Each path must distinguish the raw `feffNNNN.dat` scattering table from the
damped per-path chi contribution. It must also retain the numerical path
identity, geometry, criterion, and whether the path was included in the sum.
Filename parsing is not a safe identity mechanism because FEFF's four-digit
suffix truncates larger indices.

Build artifact selection into the same data flow:

```rust
pub enum ArtifactPolicy {
    PathsOnly,
    SpectrumOnly,
    All,
    Selected(ArtifactSelection),
}
```

Selection must happen before serialization and before artifacts are loaded
into the facade result. Cached runs need an explicit contract: either reparse
requested typed outputs, return `None`, or require recomputation. The API
must not silently claim a no-round-trip result after a cache-only run.

### 2. Real EXAFS/XANES Cargo features

Keep `default = ["full"]` for compatibility, then gate engine modules and
scheduler branches rather than adding no-op feature labels. Measure both clean
compile time and a minimal embedding binary for every feature combination.

Feature validation must cover:

- EXAFS with and without SFCONV;
- XANES/FMS as an additive pipeline, not only the `fms` and `mkgtr` stages;
- unsupported module requests returning a typed error;
- `--all-features`, `--no-default-features`, and default builds;
- unchanged default/full numerical output.

The exact dependency graph should be derived from scheduler prerequisites.
The initially suggested feature list is directionally correct but incomplete
if interpreted as independently runnable stages.

### 3. Cancellation, deadlines, and scoped execution

Introduce a clonable execution context carrying a cancellation token, absolute
deadline, and runner-owned Rayon pool. Start with checkpoints before and after
each pipeline stage, then add deeper checkpoints to POT SCF iterations, FMS
energy/k-point loops, KSPACE retries, and outer RIXS rows.

Cancellation must be a typed facade error and must never be converted into an
"unsupported source" cache miss. Recompute mode must keep its current atomic
publication property when cancelled.

Faer's current high-level APIs read process-global parallelism. Until explicit
per-call parallelism is threaded through `refeff-linalg`, protect the
set/run/restore region with a mutex and document that concurrent calculations
are serialized. A runner-owned Rayon pool alone does not make faer settings
independent.

### 4. Memory-native execution

Replace the private temporary directory transport stage by stage. Keep the
file-backed engine as a compatibility adapter and parity oracle while typed
handoffs become the primary path. Do not attempt this as one rewrite: POT,
GENFMT/FF2X, and path outputs have different ownership and cache semantics.

### 5. Prepared calculations and instrumentation

Add a typed `PreparedCalculation` only after stage inputs have stable typed
identities. Cache keys must include every physics-affecting input, engine
version, enabled features, and numerical policy. Report cache-hit reasons,
thread counts, cancellation/deadline state, and optional tracing spans.

## Release discipline

Every phase keeps the current public wrapper as a compatibility path. A change
is ready only after:

- default/full workspace checks, tests, docs, and Clippy pass;
- dependency and feature-combination checks pass;
- representative EXAFS and XANES outputs remain within registered FEFF parity
  tolerances;
- release-build performance is measured separately for cold generation and
  warm/cache-reuse paths.
