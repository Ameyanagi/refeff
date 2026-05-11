# reff

`reff` is a pure-Rust port of FEFF10 in progress. The local `feff10/`
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
- `xtask generate-golden`, which can build/run the ignored FEFF10 Fortran
  reference and place generated outputs under `reference-work/golden/`.

Numerical FEFF modules are being ported incrementally behind compatibility
tests against the ignored `feff10/` reference tree.

## Reference Outputs

The Rust implementation does not vendor FEFF10 outputs. When a golden test is
missing, generate it from the local Fortran reference:

```sh
cargo run -p xtask -- generate-golden --ref-dir feff10 --example EXAFS/Cu --force
```

Use `--no-build` only when `feff10/bin/Seq/feff` or `feff10/bin/feff` already
exists. The generated work directories are ignored by Git.
