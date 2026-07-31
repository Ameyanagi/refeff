# refeff

`refeff` is a pure-Rust, from-scratch port of [FEFF10](https://feff.phys.washington.edu/),
the ab initio X-ray spectroscopy code used to simulate EXAFS, XANES, RIXS,
EELS, Compton profiles, and related core-level spectra from a structure file.
It reads the same `feff.inp` input FEFF10 does and reproduces FEFF10's module
pipeline and file formats (`pot.bin`, `phase.bin`, `xmu.dat`, `chi.dat`, ...) in
safe Rust, using `ndarray` for numerical arrays and `faer` for linear algebra,
with no `unsafe` code and no dependency on a Fortran toolchain at runtime.

The local `feff10/` directory, when present, is used only as a reference
checkout for compatibility testing and is intentionally ignored by Git; no
FEFF10 source or generated output is vendored into this repository.

## Build & install

Requires Rust 1.95 (see `rust-toolchain.toml`; edition 2024).

```sh
git clone <this repository>
cd refeff
cargo build --release -p refeff-cli --bin refeff --bin feff
```

This produces two equivalent binaries in `target/release/`:

- `refeff`, with an explicit `--input`/`--output` CLI (`refeff run`, `refeff
  module <name>`, ...).
- `feff`, a drop-in FEFF10-style entry point that reads `feff.inp` from the
  current directory.

The workspace also provides FEFF-compatible standalone executables, including
`dym2feffinp`, `mkgtr`, and `opconsat`; build the complete executable set with
`cargo build --release -p refeff-cli --bins`.

## Quickstart

```sh
cargo build --release -p refeff-cli --bin refeff
target/release/refeff run --input path/to/feff.inp --output run/refeff
```

`refeff run` executes the full supported workflow (RDINP through FF2X) and
writes every generated FEFF-format file under `--output` instead of the input
directory. The final spectra land at:

- `run/refeff/xmu.dat` — normalized absorption spectrum (XANES/EXAFS/DANES).
- `run/refeff/chi.dat` — EXAFS fine-structure `chi(k)`.

Other module outputs (`pot.bin`, `phase.bin`, `paths.dat`, `feff.bin`, ...)
land alongside them in the same `--output` directory. For FEFF10-style usage
from a calculation directory, copy or create `feff.inp` there and run
`feff run` with no arguments; outputs are written next to the input.

Run a single module directly (e.g. to inspect `xsph`'s `phase.bin`/`xsect.dat`
handoff) with:

```sh
target/release/refeff module xsph --input path/to/feff.inp
```

See `crates/refeff-engine/examples/full_run.rs` for a scripted end-to-end example
and `crates/refeff-io/examples/` for reading `feff.inp` and `xmu.dat`
programmatically.

## Embedding architecture

The workspace separates computation from command-line concerns:

- `refeff-engine` owns the FEFF stages and scheduler and has no Clap
  dependency.
- `refeff` provides the typed embedding facade used by applications such as
  XrayTsubaki and depends directly on `refeff-engine`.
- `refeff-cli` contains argument parsing, completion generation, and the
  FEFF-compatible executable wrappers.

This keeps Clap and other frontend-only code out of embedded applications
while preserving the same computational pipeline for library and CLI callers.
CI checks this dependency boundary with `cargo tree`.

Embedded EXAFS consumers can also exclude the full/XANES scheduler:

```toml
refeff = { version = "0.2.0", default-features = false, features = ["exafs"] }
```

The `full` feature remains the default for compatibility. `sfconv` is additive
and optional in a reduced build. Known modules outside the selected feature
set return a typed feature-disabled error. The next typed-output,
cancellation, and memory-native phases are tracked in
[`docs/EMBEDDING_ROADMAP.md`](docs/EMBEDDING_ROADMAP.md).

## Module support

FEFF10's module pipeline is fully source-backed in the current inventory. The
scope audit tracks 22 production executables, 3 Rust extensions, 110 input-card
tokens, 44 stock workflows, and the 138-case HIGHZ range. Separately, the
module-status inventory has 22 entries: 21 workflow stages with source
handoffs plus `dym2feffinp`, a standalone converter that consumes a `.dym`
file directly rather than a pipeline handoff.

| Module | Role | Key outputs | Status |
|---|---|---|---|
| `pot` | Self-consistent muffin-tin potentials | `pot.bin`, `potNN.dat` | Supported |
| `atomic` | Free-atom potentials/wavefunctions | `apot.bin` | Supported |
| `xsph` | Phase shifts and cross sections | `phase.bin`, `xsect.dat` | Supported |
| `fms` | Full multiple scattering / Green's function | `gg.bin`, `fms.bin` | Supported |
| `paths` | Scattering path finder | `paths.dat` | Supported |
| `genfmt` | Path scattering-amplitude tables | `feff.bin`, `list.dat` | Supported |
| `ff2x` | Final spectrum assembly (EXAFS/XANES/DANES/FPRIME) | `xmu.dat`, `chi.dat` | Supported |
| `ldos` | Local density of states | `ldosNN.dat`, `rhocNN.dat` | Supported |
| `band` | Band structure / KKR | `bandstructure.dat`, `kmesh.dat` | Supported |
| `screen` | Core-hole screening / Hubbard-U response | `wscrn.dat` | Supported |
| `crpa` | Constrained-RPA Hubbard parameters | `crpa.dat` | Supported |
| `rhorrp` | Charge-density grid | `density.inp` outputs | Supported |
| `compton` | Compton profiles | `compton.dat` | Supported |
| `fullspectrum` | Optical constants across the full spectral range | `xmu.dat`, `opcons.dat`, `sumrules.dat` | Supported |
| `opcons` | Optical constants from elemental dielectric data | `opcons*.dat`, `loss.dat` | Supported |
| `dmdw` | Dynamical-matrix Debye-Waller factors | path/atom Debye-Waller tables | Supported |
| `dym2feffinp` | Dynamical-matrix to centered FEFF input conversion | `feff.inp`, centered `.dym` | Supported standalone executable |
| `eels` | Electron energy-loss spectroscopy | `eels.dat` | Supported |
| `eelsmdff` | EELS mixed dynamic form factor | `mdff.dat` | Supported |
| `rixs` | Resonant inelastic X-ray scattering | `rixsET.dat`, `herfd.dat` | Supported |
| `sfconv` | Many-body spectral-function convolution | `xmu.dat`, `chi.dat` (convolved) | Supported |
| `wpot` | Potential-file rendering | `potXX.dat` | Supported |

`rdinp` (input parsing, CIF/lattice expansion, and module `.inp` handoff
generation) sits ahead of this table and is always available; it is the
foundation every module above builds on.

The current compatibility inventory closes all 98 tracked rows. The canonical
fresh `XANES/BN` workflow now completes from `feff.inp` with approximately
`1–2e-5` relative L2 parity against FEFF. The decisive final corrections were
POT's independent-center FMS slot-zero layout and saved SCMT retry state,
FEFF's frozen `sqrt(rhoint)` plasmon value, raw-Hartree `emu` plus the
`ixc0`-versus-`ixc` XSPH selector split, and FMS reversed-axis rotations built
from the original vectors.

Other closure work includes bundled OPCONS `epsdb` generation for Z=1 through
Z=99, external `bphl.dat` support for broadened Hedin-Lundqvist exchange, XSPH
`MULTIPOLES=3` and nonlocal/two-spin TDLDA paths, and FULLSPECTRUM final
`xmu.dat` and `CONTROL(6)` behavior. HIGHZ scope remains precise: data and
configuration cover Z=1 through Z=138, representative successful binding
energies have reference parity, and the upstream Z=119 failure remains typed.
The release gate does not claim production completion for the pinned report's
Z=118 failure or for Z=138.
Additional fixtures can still broaden numerical evidence without representing
an unported production branch.

For the live, generated version of this table (and the branch-level
compatibility backlog beneath it), run:

```sh
cargo run -p xtask -- port-status --detail
cargo run -p xtask -- compatibility-matrix --detail
```

Full details, including the module inventory, follow-up parity backlog, and
test cadence, are tracked in
[`docs/FEFF_RUST_PORT_PLAN.md`](docs/FEFF_RUST_PORT_PLAN.md). The porting
acceptance criteria are in [`docs/PORTING.md`](docs/PORTING.md). The historical
"Current Status" bullet log that used to live in this README is now in
[`docs/CHANGELOG.md`](docs/CHANGELOG.md).

## Reference outputs & release gates

The Rust implementation does not vendor FEFF10 outputs. When a golden test is
missing, generate it from the local Fortran reference (`--no-build` if
`feff10/bin/.../feff` is already built):

```sh
cargo run -p xtask -- generate-golden --ref-dir feff10 --example EXAFS/Cu --force
```

Audit which module gates and branch-level FEFF10 workflows still need work:

```sh
cargo run -p xtask -- port-status --detail                       # module inventory
cargo run -p xtask -- compatibility-matrix --open-only --detail  # branch-level backlog
```

Add `--fail-on-unported`, `--fail-on-guarded-branches`, or
`--fail-on-ignored-parity` (port-status) and `--fail-on-open` /
`--fail-on-missing-fixtures` (compatibility-matrix) to use either as a strict
CI gate; both accept `--json-out <path>` for machine-readable reports.

The composed release gate runs both, strictly:

```sh
cargo run --profile release -p xtask -- release-readiness --detail --open-only
```

This is the definitive release audit: it passes only when the module and
compatibility inventories are closed and every required local fixture is
present with valid provenance. Generate the pinned reference fixtures first;
missing fixture groups intentionally make the command fail.

## Commit hooks

```sh
git config core.hooksPath .githooks
```

The pre-commit hook runs `cargo fmt --all --check`, `git diff --check
--cached`, `cargo check --workspace --all-targets --locked`, `cargo doc
--workspace --no-deps --locked`, and `cargo clippy --workspace --all-targets
--all-features --locked -- -D warnings`. Full numerical, reference-generation,
and parity test suites are run explicitly on local release hardware rather
than in commit hooks or GitHub Actions.

## Benchmarks

```sh
cargo bench -p refeff-io --bench rdinp        # parser, CIF import, rdinp, potXX.dat
cargo bench -p refeff-core --bench core_tables # core numerical tables
cargo bench -p refeff-linalg --bench linalg    # linear-algebra bridge/solvers
cargo run -p xtask -- bench-e2e --example EXAFS/Cu --iterations 5 [--reference]
```

`bench-e2e --reference` additionally times the FEFF10 `rdinp` reference binary
when it has been built. The `refeff-io` benchmark uses
`feff10/examples/EXAFS/Cu/feff.inp` when the reference tree is present and
falls back to a small embedded Cu input otherwise.

## License

ReFEFF is dual-licensed under your choice of the
[Apache License, Version 2.0](LICENSE-APACHE) or the [MIT License](LICENSE-MIT).
