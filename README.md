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
  grids, SCMT complex-energy grid construction, Dirac spinor and
  potential/density grid interpolation, Loucks-grid spherical overlap sums,
  FEFF spherical-overlap cap/lens volumes, muffin-tin overlap-matrix
  construction, and overlap-to-muffin-tin projection, Norman-radius integration
  from overlapped densities, FEFF potential/density overlap assembly,
  Coulomb-potential correction, valence density and LDOS accumulation, Broyden
  SCF density mixing, interstitial shell averages, four-point Coulomb radial
  integration, interstitial Fermi-level calculation, overlap-density tail indexing,
  atomic weight/symbol/mass lookup, phase
  unwrapping, core-hole widths/quantum numbers, vector rotations, hydrogen
  bond adjustment for potential geometry, Legendre normalization tables,
  Wigner 3j coefficients, exchange-potential,
  Perdew-Zunger, Perrot-Dharma-Wardana, Karasiev-Sjostrom-Dufty-Trickey, and
  Hedin-Lundqvist scalar helpers, self-energy dispersion, branch-log, complex
  Hartree-Fock exchange, many-pole fitting, and Hedin-Lundqvist integrand
  kernels and adaptive quadrature, Debye/Einstein cumulants, Debye displacement
  correlations and path Debye-Waller factors, path packing, geometry, hashing,
  heap helpers, phase-derived path criteria tables, output path parameters,
  standard-frame and canonical time-reversal path coordinates, path pruning
  criteria and decisions, output-path importance, companion-index sorting,
  state-ket construction, GENFMT lambda-index selection, central-atom
  plane-wave factors, `snlm` Legendre-normalization tables, `rdpath` path
  angle/leg-length tables, curved-wave polynomial factors, scattering-amplitude
  F-matrices, energy-independent and polarized scattering-amplitude matrices,
  and initial-state rotation matrices.
- Initial FMS numerical and cluster-preparation helpers for Rehr-Albers
  polynomial tables, z-axis propagator terms, pair angles, and radial atom
  and representative atom ordering, FMS rotation matrices, and pair
  `rho`/`xclm` tables, plus off-diagonal free-propagator elements and
  matrices, same-site T-matrix elements, compact T-matrix tables, iterative
  FMS system-matrix assembly, BiCGStab, recursion-method, Graves-Morris/Salam,
  and TFQMR FMS scattering, compact and full-potential LU FMS scattering solves.
- Initial RIXS numerical helpers for the FEFF KK and double-Lorentz analytic
  integrals and complex 2-D bilinear interpolation.
- `rdinp` text output generation for the current FEFF handoff set, including
  `.dimensions.dat`, `geom.dat`, `atoms.dat`, `global.inp`, `pot.inp`, and
  module `.inp` files, checked against generated FEFF10 outputs when present.
- FEFF `wpot`-compatible `potXX.dat` potential-output rendering from
  `ndarray` density and potential grids.
- FEFF `m_mtdp` muffin-tin density/potential text read/write support.
- FEFF `pot.bin` formatted text/PAD read/write support for potential-state
  handoff data.
- FEFF `phase.bin` formatted text/PAD read/write support for XSPH phase-shift
  and transition-moment handoff data.
- FEFF v03 `feff.bin` formatted text/PAD read/write support for GENFMT path
  handoff data.
- FEFF `feffl.bin` formatted text/PAD read/write support for NRIXS/LDEC
  path-decomposition companion data.
- FEFF `paths.dat` text read/write support for the PATH to GENFMT handoff.
- FEFF DMDW type-1/type-4 `.dym` dynamical-matrix read/write support with
  ndarray-backed coordinate, force-constant, and mass-weighted matrix data.
- FEFF `grid.inp` energy-grid read/write support for XSPH user EGRID handoff
  files.
- FEFF `config.inp` electron-configuration read/write support for `CONFIG card`
  payload handoff files.
- FEFF `spring.inp` Debye force-field read/write support with ndarray accessors
  for stretch and angle rows.
- FEFF `list.dat` path-selection read/write support for the GENFMT to FF2X
  handoff.
- FEFF `xsect.dat` cross-section read/write support for the XSPH to FF2X
  handoff.
- FEFF `xmu.dat` spectrum read/write support for final normalized absorption
  output tables.
- FEFF `chi.dat`/`chipNNNN.dat` EXAFS spectrum read/write support for final
  and per-path output tables.
- FEFF `eels.dat` spectrum read/write support for orientation-averaged and
  tensor-resolved EELS output tables.
- FEFF `danes.dat` anomalous-scattering read/write support for FPRIME/DANES
  output tables.
- FEFF `ldosNN.dat` local density-of-states read/write support for orbital and
  spin-resolved output tables.
- FEFF `compton.dat`, `rhozzp.dat`, and `jzzp.dat` Compton profile,
  diagnostic, and cache read/write support.
- FEFF `fms.bin` formatted text/PAD read/write support for MKGTR FMS trace
  handoff data.
- FEFF `fmsl.bin` formatted text/PAD read/write support for NRIXS/LDEC FMS
  decomposition handoff data.
- FEFF `xsecl.bin` formatted text/PAD read/write support for NRIXS/LDEC atomic
  cross-section decomposition handoff data.
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

Parser, `rdinp`, and `potXX.dat` rendering baselines live in the `refeff-io`
crate:

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
