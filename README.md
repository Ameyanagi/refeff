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

Numerical FEFF modules are being ported incrementally behind compatibility
tests against the ignored `feff10/` reference tree.
