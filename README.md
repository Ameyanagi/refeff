# refeff

`refeff` is a pure-Rust port of FEFF10 in progress. The local `feff10/`
directory is used only as a reference checkout and is intentionally ignored by
Git.

The implementation targets Rust 1.95, uses `ndarray` as the primary numerical
array layer, and uses `faer` for pure-Rust linear algebra kernels.

## Current Status

The repository currently contains the workspace scaffold and the first
compatibility layer:

- FEFF-style input line reading with `include`/`load` support.
- FEFF card and section-data tokenization.
- Packed ASCII Data (PAD) encoding/decoding.
- Fortran-style formatting helpers.
- `ndarray` type aliases and allocation helpers.
- `faer` bridge helpers.
- Core numerical helpers ported from FEFF common routines, including radial
  grids, phase unwrapping, core-hole widths/quantum numbers, vector rotations,
  Legendre normalization tables, Wigner 3j coefficients, and state-ket
  construction.
- `rdinp` text output generation for the current FEFF handoff set, including
  `.dimensions.dat`, `geom.dat`, `atoms.dat`, `global.inp`, `pot.inp`, and
  module `.inp` files, checked against generated FEFF10 outputs when present.
- `xtask generate-golden`, which can build/run the ignored FEFF10 Fortran
  reference and place generated outputs under `reference-work/golden/`.

Numerical FEFF modules are being ported incrementally behind compatibility
tests against the ignored `feff10/` reference tree.

The default `refeff run` command currently executes the supported `rdinp`
compatibility stage and writes the known handoff files, then exits with a clear
error before unported numerical modules.

## Reference Outputs

The Rust implementation does not vendor FEFF10 outputs. When a golden test is
missing, generate it from the local Fortran reference:

```sh
cargo run -p xtask -- generate-golden --ref-dir feff10 --example EXAFS/Cu --force
```

Use `--program rdinp` to generate only FEFF's input-preparation handoff files
for broader, faster compatibility checks while downstream numerical modules are
still being ported.

Use `--no-build` only when `feff10/bin/Seq/feff` or `feff10/bin/feff` already
exists. The generated work directories are ignored by Git.

## Commit Hooks

Use the repository-managed Git hooks before committing:

```sh
git config core.hooksPath .githooks
```

The pre-commit hook runs `cargo fmt --all --check`,
`cargo check --workspace --all-targets --locked`, `cargo test --workspace
--locked`, and `cargo clippy --workspace --all-targets --locked -- -D
warnings`.

## Benchmarks

Parser and `rdinp` rendering baselines live in the `refeff-io` crate:

```sh
cargo bench -p refeff-io --bench rdinp
```

Core numerical table baselines live in the `refeff-core` crate:

```sh
cargo bench -p refeff-core --bench core_tables
```

The benchmark uses `feff10/examples/EXAFS/Cu/feff.inp` when the ignored
reference tree is present and falls back to a small embedded Cu input in clean
checkouts.
