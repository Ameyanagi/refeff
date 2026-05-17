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
- FEFF card, section-data, and `bwords`-style control-file tokenization.
- Packed ASCII Data (PAD) encoding/decoding.
- Fortran-style formatting helpers.
- `ndarray` type aliases and allocation helpers.
- `faer` bridge helpers, FEFF-compatible LU solves, complex `polyfit`, and
  the single-precision `SLAE2`/`SLAEV2`/`SSYEV` symmetric eigensolver surface
  from `MATH/seigen.f90`.
- Core numerical helpers ported from FEFF common routines, including radial
  grids, SCMT complex-energy grid construction, Dirac spinor and
  potential/density grid interpolation, Loucks-grid spherical overlap sums,
  SCREEN logarithmic radial-grid, complex-energy contour, and local-density
  exchange-correlation helpers,
  FEFF spherical-overlap cap/lens volumes, muffin-tin overlap-matrix
  construction, and overlap-to-muffin-tin projection, Norman-radius integration
  from overlapped densities, FEFF potential/density overlap assembly,
  Coulomb-potential correction, valence density and LDOS accumulation, Broyden
  SCF density mixing, interstitial shell averages, four-point Coulomb radial
  integration, interstitial Fermi-level calculation, overlap-density tail indexing,
  atomic weight/symbol/mass lookup, ATOM polynomial, convergence-mixing,
  Thomas-Fermi density, occupation-product, and Coulomb angular-coefficient
  helper kernels, phase
  unwrapping, core-hole widths/quantum numbers, Williams/Elam edge-energy
  table adapters for `getedg`/`preved`/`nexted`, vector rotations, hydrogen
  bond adjustment for potential geometry, Legendre normalization tables,
  Wigner 3j coefficients, relativistic Clebsch-Gordan coefficient tables,
  `MKGTR/calclbcoef.f90` Clebsch-Gordan coefficient tables,
  `BAND/ikapmue.f90` relativistic state indexing, angular
  basis-transformation matrices, XSPH final-state calculation planning,
  angular-channel need flags, angular density-coefficient tables,
  longitudinal and relativistic multipole factors, NRIXS transition weights,
  q-Bessel tables, and four-step complex radial Simpson integration,
  NRIXS angular-decomposition, angular-channel, and final-state spectrum
  updates, AXAFS background extraction tables, initial-state occupation
  normalization, initial-hole orbital interpolation, and XSPH phase-mesh
  primitive, FEFF84 EXAFS/XANES/XES/FPRIME-grid, no-FMS horizontal-grid,
  vertical-grid, default `phmesh2` mesh construction, user `grid.inp`
  phase-mesh composition, and finite-temperature `phmesh2T` normal-mesh
  composition,
  exchange-potential,
  Perdew-Zunger, Perrot-Dharma-Wardana, Karasiev-Sjostrom-Dufty-Trickey, and
  Hedin-Lundqvist scalar helpers, self-energy dispersion, branch-log, complex
  Hartree-Fock exchange, many-pole fitting, SFCONV real-coefficient polynomial
  roots, and Hedin-Lundqvist integrand kernels, adaptive quadrature,
  Gauss-Legendre meshes, exact interpolation
  polynomial coefficients, table-backed and derivative-assisted Brent
  minimization, reciprocal-space
  Bravais classification, KSPACE Bravais basis construction,
  reciprocal-lattice vector generation, K-path segment generation, k-mesh
  division selection, arbitrary-mesh generation, irreducible-point reduction,
  tetrahedron cell division and record counting, and common-factor reduction,
  symmetry-operation lattice relabeling and basis transformation,
  point-group operation discovery, symmetry-operation closure checks, and
  lattice-coordinate reduction helpers, Debye/Einstein cumulants, Debye
  displacement
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
  integrals and complex 2-D bilinear interpolation, plus EELS electron
  wavelength, Euler rotation, q-mesh integration helpers, and COMPTON
  rotation/grid/xy-integration/rhozzp/profile helpers, and FULLSPECTRUM
  effective-electron-count sum-rule integration, the active `egrid_lin.f90`
  linear energy grid, edge-restarted `egrid.f90` grid generation with
  Elam-backed component edge-list adaptation, `rdop.f90` default energy-window
  inference for explicit edge sets, `rdval.f90` valence `xmu.dat` eps2
  projection, `rddens.f90` number-density estimation from `pot.bin`,
  `gtedgs.f90` occupied-edge selection, and FEFF
  `fullspectrum.f90` scattering-to-dielectric conversion,
  `kk.f90`/`hamaker.f90` dielectric transforms, and `opcons.f90`
  optical-constant generation, plus `FF2X/exconv.f90` excitation-spectrum
  convolution, `FF2X/xscorratan.f90` arctangent correction, and the
  `FF2X/fprime.f90` logarithmic/integral FPRIME helpers, with initial
  `FF2X/xscorr.f90` contour-kernel primitives.
- Initial SFCONV numerical helpers for FEFF `mkrmu` Kramers-Kronig real-part
  reconstruction, `plset.f90` pole/electron-gas parameter setup,
  `ppole.f90` pole dispersion/coupling helpers, `qlimits.f90` momentum
  limit selection, `grater.f90` adaptive real quadrature,
  `rdeps.f90` excitation-pole loading/fallback conversion, `mksat.f90`
  satellite spectral-function helpers, `senergies.f90` beta, real/imaginary
  self-energy, and first-derivative primitives, `mkspectf.f90` spectral energy
  grid construction, quasiparticle and satellite finite-element row assembly,
  satellite extrinsic-region split, weight clipping correction, and final
  spectral-weight vector assembly, `SO2CONV` minimal momentum grid,
  material/header unit conversion, photoelectron momentum refinement, EXAFS
  channel preparation and energy padding, XANES signal padding and
  Kramers-Kronig phase preparation,
  `feffNNNN.dat`
  path-column interpolation and raw EXAFS path-signal construction, cached
  spectral-function momentum interpolation, EXAFS post-convolution
  amplitude/phase row assembly, XANES absorption/background row assembly, and
  path-grid amplitude/phase averaging,
  `interpsf.f90` spectral-function interpolation, and `sfconvsub.f90`
  spectral-function convolution.
- Initial FEFF orbital-configuration helper for applying `getorb` core-hole,
  screening-electron, ionicity, compacting, and high-l valence-freezing rules
  to selected occupation rows.
- Initial FOVRG numerical helpers for FEFF `diff` C3 radial derivative
  construction, `yzktec` radial `yk`/`zk` exchange-kernel transform, and
  `yzkrdc` exchange source construction, `dsordc` overlap integration,
  `ortdac` Schmidt orthogonalization, `potex` exchange-potential accumulation,
  `nucdec` point-nucleus radial mesh and potential construction, `potdvp`
  potential development coefficients, `aprdep`/`aprdec` polynomial product
  coefficients, `muatcc` angular exchange coefficients, `inmuac` orbital
  bookkeeping, plus `dfovrg/flatv` flat-potential radial propagation and
  `intout` outward Dirac radial
  integration, `solout` regular outgoing radial solution assembly, `solin`
  irregular inward radial solution assembly, and `wfirdc` initial
  photoelectron orbital assembly, with a `dfovrg`-level Dirac photoelectron
  driver covering WKB switching, potential flattening, exchange cycles, and
  muffin-tin propagation.
- Initial OPCONS numerical helper for FEFF `AddEps` weighted epsilon-table
  combination and optical loss-function evaluation.
- Initial RHORRP density-grid traversal/evaluation, `density.inp` read/write
  support and Bohr-grid adapters, nearest-atom and FMS-radius inclusion
  counts, radial-grid and wavefunction interpolation, process partitioning,
  irregular-solution
  smoothing, core atomic-density, same-site and scattering Green's-function
  terms, point-pair energy-density assembly and contour integration, energy
  prefactors, energy-density finishing, and Fermi-contour integration helpers,
  plus ASCII/binary density-output read/write, Bohr-to-Angstrom output
  conversion, filename-based output selection, and nearest-atom text
  diagnostics.
- `rdinp` text output generation for the current FEFF handoff set, including
  CIF-derived potential and atom-cluster generation, reciprocal-lattice
  real-space cluster expansion, `.dimensions.dat`, `geom.dat`, `atoms.dat`,
  `global.inp`, `pot.inp`, auxiliary Debye `spring.inp` and DMDW `.dym`
  carry-through, and module `.inp` files, checked against generated FEFF10
  outputs when present.
- FEFF `EQUIVALENCE 2` and `EQUIVALENCE 4` CIF import support for collapsing
  CIF potential types by atomic number when requested or required by FEFF's
  potential-count limit.
- FEFF structural handoff read/write support for `.dimensions.dat`,
  `atoms.dat`, and `geom.dat`.
- FEFF `wpot`-compatible `potXX.dat` potential-output rendering from
  `ndarray` density and potential grids, including `pot.bin`/`apot.bin`
  handoff bridging and `refeff module wpot` output generation.
- FEFF `compton.dat` profile output generation from an existing
  `jzzp.dat` COMPTON cache via `refeff module compton`.
- FEFF `FULLSPECTRUM` optical-table output generation from an existing
  `eps.dat` dielectric cache via `refeff module fullspectrum`, including
  `opcons.dat`, `opconsKK.dat`, `opcons0.dat`, and `sumrules.dat` when a
  `pot.bin` density cache is available.
- FEFF `SFCONV` module startup compatibility via `refeff module sfconv`,
  including `sfconv.inp` parsing and disabled-path `logsfconv.dat`
  creation. Enabled S0^2 convolution still requires the unported `SO2CONV`
  driver.
- FEFF `m_mtdp` muffin-tin density/potential text read/write support.
- FEFF `apot.bin` atomic-potential TXT section-stream read/write support for
  `WriteData`, `WriteArrayData`, and `Write2D` payloads.
- FEFF `pot.bin` formatted text/PAD read/write support for potential-state
  handoff data, plus a borrowed `FULLSPECTRUM/rdpotp_fs.f90` view of the
  title, multiplicity, Norman-radius, and atomic-number fields.
- FEFF `phase.bin` formatted text/PAD read/write support for XSPH phase-shift
  and transition-moment handoff data.
- FEFF v03 `feff.bin` and `feffNN.bin` formatted text/PAD read/write support
  for GENFMT path handoff data.
- FEFF `feffl.bin` formatted text/PAD read/write support for NRIXS/LDEC
  path-decomposition companion data.
- FEFF `MDFF` parsing with NRIXS `global.inp` mixed-DFF handoff output,
  including `MDFF 2` q-prime generation.
- FEFF `paths.dat` text read/write support for the PATH to GENFMT handoff and
  RDINP-generated single-scattering output from `SS` cards.
- FEFF `OVERLAP` geometry parsing and `pot.inp` overlap-shell handoff output.
- FEFF `FOLP` manual overlap-factor parsing and `pot.inp` handoff output.
- FEFF `ION` ionization parsing and `pot.inp` `xion` handoff output.
- FEFF four-character block-card aliases (`POTENTIALS`, `ATOMS`, `OVERLAP`,
  `LATTICE`, `EGRID`, `ELNES`, `EXELFS`, `NRIXS`, `MDFF`) with section-row
  routing.
- FEFF `JUMPRM` jump-removal parsing and `pot.inp` handoff output.
- FEFF `EXTPOT` and `RESTART` parsing with `pot.inp` logical handoff output.
- FEFF common-control aliases (`TITLE`, `CONTROL`, `PRINT`, `EXCHANGE`,
  `CORRECTIONS`, `RGRID`, `COREHOLE`, `UNFREEZEF`, `ABSOLUTE`) with module
  handoff output.
- FEFF four-character control-card aliases for path, Debye-Waller, SCF/FMS,
  criteria, many-pole, EELS magic-angle, and overlap-factor controls.
- FEFF `CHSHIFT` parsing with `pot.inp` and `xsph.inp` handoff output.
- FEFF `CORVAL`, `HIGHZ`, and `WARNION` parsing with `pot.inp` handoff output.
- FEFF `SCFTH`, `SCFR`, and `TOLS` parsing with `pot.inp` handoff output.
- FEFF `INTERSTITIAL` parsing with `pot.inp` handoff output.
- FEFF `MBCONV`, `SIG2`, `SIG3`, and `SIGGK` parsing with `ff2x.inp`
  and `fms.inp` handoff output.
- FEFF `SFCONV`/`SO2CONV`, `SELF`, `SFSE`, and `RCONV` parsing with
  `sfconv.inp` handoff output.
- FEFF `SO2CONV` fixed-width spectrum-header extraction for `Gam_ch`,
  `Rs_int`, `Vint`, `Mu`, and `kf` material inputs.
- FEFF `SO2CONV` spectrum-header preflight for detecting prior
  `# Convoluted with A(omega).` processing before the data separator.
- FEFF `SO2CONV` target-file discovery from `sfconv.inp`, `eels.inp`, and
  `list.dat`, including ELNES polarization filenames and path-expanded
  `chipNNNN.dat`/`feffNNNN.dat` selection.
- FEFF `IORDER`/`IORD`, `NSTAR`, and `RPHASES` parsing with `genfmt.inp`
  and `xsph.inp` handoff output.
- FEFF spectroscopy-grid and polarization aliases (`XANES`, `DANES`,
  `FPRIME`, `EXAFS`, `POLARIZATION`, `ELLIPTICITY`, `MULTIPOLE`) with
  `global.inp` and `xsph.inp` handoff output.
- FEFF `CFAVERAGE` parsing with `global.inp`, `pot.inp`, and `geom.dat`
  handoff output.
- FEFF `SYMMETRY` parsing with `paths.inp` handoff output.
- FEFF `NRIXS` multi-q parsing with `global.inp` handoff output.
- FEFF `BANDSTRUCTURE` parsing with `band.inp` handoff output.
- FEFF XSPH core-hole controls (`CHBROADENING`, `CHWIDTH`, `EPS0`, `EGAP`,
  `SETEDGE`, `RLPRINT`, `ICORE`) with `pot.inp`, `xsph.inp`, and `ff2x.inp`
  handoff output.
- FEFF `TDLDA` and `PMBSE` parsing with XSPH advanced-control handoff output.
- FEFF `OPCONS`/`NUMDENS`/`PREPS` parsing with aliases and `opcons.inp`
  handoff output.
- FEFF `SCREEN` parsing with `screen.inp` handoff output.
- FEFF `FULLSPECTRUM` parsing with `fullspectrum.inp` handoff output.
- FEFF `FULLSPECTRUM/rdop.f90` option-card parsing for standalone
  `fullspectrum.inp` component, edge, Drude, valence, EELS, and energy-grid
  controls.
- FEFF `FULLSPECTRUM/rdxmu.f90`, `rdxmunorm.f90`, and `rdst.f90` adapters
  for `xmu.dat` absorption and fine-structure segments.
- FEFF `FULLSPECTRUM/hamaker.f90` `hamaker.dat` read/write support and
  imaginary-axis dielectric transform adapter.
- FEFF `HUBBARD` parsing with `hubbard.inp` handoff output.
- FEFF `REAL`/`RECIPROCAL` input-order parsing with reciprocal-space handoff
  selection.
- FEFF `COORDINATES` parsing for reciprocal `LATTICE` atom-coordinate
  conversion.
- FEFF `NOGEOM` parsing to suppress `geom.dat` output.
- FEFF `TEMP`/`SCXC` parsing with finite-temperature and SCF exchange
  selector handoff output.
- FEFF DMDW type-1/type-4 `.dym` dynamical-matrix read/write support with
  ndarray-backed coordinate, force-constant, and mass-weighted matrix data.
- FEFF `dmdw.out` Debye-Waller diagnostic read/write support for PDOS poles,
  Einstein summaries, moments, and path/atom result tables.
- FEFF `grid.inp` energy-grid read/write support for XSPH user EGRID handoff
  files.
- FEFF `DENSITY` parsing with `density.inp` payload handoff output.
- FEFF `COMPTON`/`RHOZZP`/`CGRID` parsing with aliases and `compton.inp`
  handoff output.
- FEFF `band.inp`, `fullspectrum.inp`, `opcons.inp`, `crpa.inp`,
  `hubbard.inp`, `screen.inp`, `paths.inp`, `sfconv.inp`, `dmdw.inp`,
  `fms.inp`, `genfmt.inp`, `xsph.inp`, `pot.inp`, `global.inp`, `compton.inp`,
  `eels.inp`, `ff2x.inp`, `ldos.inp`, and `rixs.inp` module-control
  read/write support.
- FEFF `config.inp` electron-configuration read/write support for
  `CONFIG`/`CONFIGURATION card` payload handoff files, including expansion of
  grouped orbital labels and noble-gas shorthand bases into FEFF's 40-slot
  occupation rows plus potential-index table application.
- FEFF `config.dat` electron-configuration output read/write support for
  post-core-hole and post-ionicity occupation arrays, including expansion from
  compacted `getorb` orbital configurations.
- FEFF potential-stage diagnostic read/write support for `convergence.scf`,
  `convergence.scf.fine`, and `fort.16`.
- FEFF `misc.dat` quick-reference header read/write support for potential-stage
  `wthead` title records.
- FEFF `spring.inp` Debye force-field read/write support with ndarray accessors
  for stretch and angle rows.
- FEFF `list.dat` and `listNN.dat` path-selection read/write support for the
  GENFMT to FF2X handoff.
- FEFF `log.dat` run-summary read/write support with parsed version,
  core-hole, feature, title, and card metadata, plus raw module-log read/write
  support for `log1.dat`, `logdos.dat`, and related `log*.dat` outputs.
- FEFF `feff.stdout`, `feff.stderr`, `rdinp.stderr`, and `fort.11` run-output
  diagnostic read/write support with module-completion and floating-point
  exception metadata.
- FEFF `xsect.dat` cross-section read/write support for the XSPH to FF2X
  handoff, including the `FF2X/ff2gen.f90` `rdxbin` unit-conversion adapter.
- FEFF `xmu.dat` spectrum read/write support for final normalized absorption
  output tables, plus `FULLSPECTRUM/rdxmu.f90`, `rdxmunorm.f90`, `rdbkg.f90`,
  and `rdst.f90` adapters.
- FEFF `xmul.dat` NRIXS angular-decomposition spectrum read/write support.
- FEFF `chi.dat`/`chipNNNN.dat` EXAFS spectrum read/write support for final
  and per-path output tables.
- FEFF `SO2CONV` target-data parsing and rendering for selected `xmu.dat`,
  `chi.dat`/`chipNNNN.dat`, and plain-text `feffNNNN.dat` path files,
  including `reff` metadata, seven-column path rows, and adapters for applying
  row-level SO2CONV XANES, EXAFS, and path-average results.
- FEFF `eels.dat` spectrum read/write support for orientation-averaged and
  tensor-resolved EELS output tables.
- FEFF `danes.dat` anomalous-scattering read/write support for FPRIME/DANES
  output tables.
- FEFF `ldosNN.dat` and `rhocNN.dat` local density-of-states read/write support
  for orbital, spin-resolved, and embedded-reference output tables, plus the
  `FULLSPECTRUM/rdldos.f90` Hartree-unit adapter.
- FEFF `compton.dat`, `rhozzp.dat`, and `jzzp.dat` Compton profile,
  diagnostic, and cache read/write support.
- FEFF `crpa.dat` constrained-RPA Hubbard parameter read/write support.
- FEFF `loss.dat` MPSE/OPCONS loss-function table read/write support.
- FEFF `eps.dat` FULLSPECTRUM dielectric-function table read/write support and
  `fullspectrum.f90` scattering-to-dielectric row generation.
- FEFF `osc_str.dat` FULLSPECTRUM oscillator-strength summary read/write
  support and `fullspectrum.f90` edge-summary row generation.
- FEFF `opcons.dat`, `opconsKK.dat`, and `opcons0.dat` FULLSPECTRUM optical
  constants read/write support and `FULLSPECTRUM/opcons.f90` optical-constant
  generation to FEFF-compatible output rows.
- FEFF `sumrules.dat` FULLSPECTRUM optical sum-rule read/write support and
  cumulative sum-rule generation from `opconsKK.dat`-style tables.
- FEFF `drude.dat` FULLSPECTRUM Drude free-electron term read/write support
  and generation from FEFF energy grids.
- FEFF `hamaker.dat` FULLSPECTRUM read/write support for reference files; the
  cached CLI runner leaves it disabled by default to match FEFF10's compiled
  `dohamaker = .false.` branch.
- FEFF `FULLSPECTRUM/rdbkg.f90` FPRIME background scattering-factor assembly
  with FEFF-compatible segment precedence and effective-electron integration.
- FEFF `FULLSPECTRUM/rdst.f90` FMS/path fine-structure interpolation with
  FEFF-compatible transition weighting for real and imaginary components.
- FEFF `FULLSPECTRUM/addedg.f90` single-edge assembly with FEFF-compatible
  sign convention, background/fine-structure smoothing, and `fp0` shift.
- FEFF `exc.dat` excitation-pole table read/write support for SELF and
  SFCONV handoff data.
- FEFF `specfunct.dat` SO2CONV spectral-function cache read/write support,
  including Fortran sequential-unformatted records, pole tables, momentum
  metadata, spectral weights, cache-reuse compatibility checks, and a typed
  bridge to the core momentum spectral interpolation and EXAFS/XANES
  row-convolution kernels.
- FEFF `mpse.dat` many-pole self-energy read/write support for complex
  self-energy and optional renormalization tables.
- FEFF RIXS map and line-spectrum output read/write support for `rixsET.dat`
  and `herfd*.dat` tables.
- FEFF `edges.dat`, `chemical.dat`, and `emesh.dat` scalar/energy-grid
  read/write support for potential, phase, and downstream RIXS handoff data.
- FEFF `emesh.bin` Fortran-unformatted complex energy-grid handoff read/write
  support.
- FEFF `fpf0.dat` atomic form-factor and oscillator-strength read/write
  support for anomalous-scattering handoff data.
- FEFF XSCORR intermediate table read/write support for `prexmu.dat`,
  `residue.dat`, `contour.dat`, `curve.dat`, and `raw.dat`.
- FEFF screened-core-hole radial table read/write support for `wscrn.dat` and
  `vtot.dat`.
- FEFF `gtr.dat`, `gtrNN.bin`, and `gtrl.dat` FMS Green's-function trace
  diagnostic and LDOS handoff read/write support.
- FEFF `gg.bin`/`gg.dat` `Write2D` complex Green's-function matrix handoff and
  diagnostic read/write support.
- FEFF RHORRP `gg_slice.bin` and `gg_diag.bin` sequential unformatted FMS
  matrix handoff read/write, core-ready block extraction, and FEFF
  `rhoerrp` pair-selection support.
- FEFF `xsecl.dat` and `xsecl2.dat` NRIXS angular cross-section text
  read/write support.
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

Parser, CIF import, `rdinp`, and `potXX.dat` rendering baselines live in the
`refeff-io` crate:

```sh
cargo bench -p refeff-io --bench rdinp
```

Core numerical table baselines live in the `refeff-core` crate:

```sh
cargo bench -p refeff-core --bench core_tables
```

Linear algebra bridge and solver baselines live in the `refeff-linalg` crate:

```sh
cargo bench -p refeff-linalg --bench linalg
```

The `xtask` end-to-end benchmark runs the supported Rust `rdinp` path across
the ignored FEFF example tree and can optionally time the FEFF10 `rdinp`
reference binary when it has been built:

```sh
cargo run -p xtask -- bench-e2e --example EXAFS/Cu --iterations 5
cargo run -p xtask -- bench-e2e --example EXAFS/Cu --iterations 5 --reference
```

The `refeff-io` benchmark uses `feff10/examples/EXAFS/Cu/feff.inp` when the
ignored reference tree is present and falls back to a small embedded Cu input
in clean checkouts.
