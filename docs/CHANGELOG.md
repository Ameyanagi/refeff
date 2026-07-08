# Changelog

This file tracks the module-porting bullet log that used to live in the
project README under "Current Status". README.md now carries a condensed,
generated per-module support table instead; this file preserves the detailed
history verbatim for reference.

## 2026-07-08 — Current Status snapshot moved from README.md

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
  SCREEN logarithmic radial-grid, radial active-prefix and `getph` bounds,
  phase-potential reference shift, per-energy wave-number state,
  `getph` angular cutoff, radial-solution normalization, irregular-solution scaling,
  exact radial continuation, `rdgeom` atomic-unit setup, complex-energy contour, local-density
  exchange-correlation, radial Coulomb-kernel, bare core-hole, radial Coulomb
  potential, FMS cluster Green trace, atomic/FMS/CRPA response-slice, response
  integration/solve, shared SCREEN/CRPA contour response assembly, CRPA
  orbital-density, density-weight, and Hubbard-summary helpers,
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
  JAS Bessel functions, q-Bessel and X-ray Bessel tables, JAS orthogonality
  corrections, overlap quadrature, reduced radial matrix elements, and
  central-atom double radial integrals, phase angular-cutoff planning,
  muffin-tin radial-index setup, per-energy phase wave-number setup,
  per-angular-channel phase setup, post-`phamp` phase cutoff handling, XSPH
  reference-energy tail finalization, `mpse.dat` self-energy summary values,
  MPSE plasmon-pole scaling, Hubbard phase-potential shift, `aph` assignment,
  and reference-tail setup, `PrintRl` header and radial-output normalization,
  empty-cell phase matching, unreferenced normal-potential phase-grid
  preparation, regular FOVRG-to-`phamp` phase-channel matching, non-JAS reduced
  and cross-section radial matrix elements, and four-step complex radial Simpson
  integration, regular/irregular XSPH cross-section FOVRG channels, and
  ordinary nonstandard-potential `xsect.dat` row accumulation,
  NRIXS angular-decomposition, angular-channel, and final-state spectrum
  updates, typed NRIXS `xsecl.dat`/`xsecl2.dat`/`xsecl.bin` output handoff
  assembly, typed NRIXS `xmul.dat` decomposition-output assembly with
  xsect-backed photon-energy/momentum grid conversion, AXAFS background
  extraction tables, decomposed FF2X/JAS path summation from
  `feffl.bin` channel amplitudes/phases plus file-backed `fmsl.bin` FMS trace
  combination and NRIXS `S^0(q,w)` row totals from channel backgrounds,
  initial-state occupation normalization,
  initial-hole orbital interpolation, and XSPH phase-mesh
  primitive, FEFF84 EXAFS/XANES/XES/FPRIME-grid, no-FMS horizontal-grid,
  vertical-grid, JAS/NRIXS `phmeshjas` mesh construction, default `phmesh2`
  mesh construction, user `grid.inp`
  phase-mesh composition, and finite-temperature `phmesh2T` normal-mesh
  composition,
  exchange-potential, `xcpot` static-potential branch, MPSE density-grid setup,
  MPSE enable/pole-count setup, MPSE delta-self-energy table shaping and
  row-delta selection, local density/momentum scale setup, nested `sigma`
  dispatcher, Fermi-level self-energy cache setup, Dyson self-energy
  correction, self-energy delta application, composed dynamic `xcpot` potential
  update including non-BPR computed `CSigZ` MPSE data, plus
  reference-potential finalization,
  Perdew-Zunger, Perrot-Dharma-Wardana, Karasiev-Sjostrom-Dufty-Trickey, and
  Hedin-Lundqvist scalar helpers, self-energy dispersion, branch-log, complex
  Hartree-Fock exchange, non-BPR `CSigZ` single/many-pole self-energy
  accumulation, many-pole fitting, SFCONV real-coefficient polynomial roots,
  and Hedin-Lundqvist integrand kernels, adaptive quadrature,
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
  correlations and path Debye-Waller factors, DMDW path descriptor expansion,
  center-of-mass/inertia and
  rigid-body projection helpers, force-block mass weighting, Lanczos tridiagonal recursion,
  polynomial helpers, pole/weight extraction, pole/moment Einstein summaries,
  and pole-table thermal
  Debye-Waller/free-energy accumulation, path reduced-mass, rigid-mode seed
  projection, initial-vector setup, and DMDW type-2 PDS/`a2f` phonon-coupling
  normalization, PATH `phase.bin`/`geom.dat` handoffs, path packing, geometry,
  hashing, pathfinder atom/neighbor preparation and heap candidate search,
  hash-range degeneracy grouping, `pathsd` retention/range and outer-loop
  reduction plus output assembly, heap helpers, phase-derived path
  criteria tables, output path parameters, standard-frame and canonical
  time-reversal path coordinates, path pruning criteria and decisions,
  output-path importance, companion-index sorting,
  state-ket construction, GENFMT lambda-index selection, central-atom
  plane-wave factors, `snlm` Legendre-normalization tables, `rdpath` path
  angle/leg-length tables, curved-wave polynomial factors, scattering-amplitude
  F-matrices, energy-independent and polarized scattering-amplitude matrices,
  and initial-state rotation matrices.
- Initial FMS numerical and cluster-preparation helpers for Rehr-Albers
  polynomial tables, z-axis propagator terms, pair angles, and radial atom
  and representative atom ordering, `yprep` absorber-centered cluster prefixes,
  `yprep` pair-rotation tables, FMS rotation matrices, pair `rho`/`xclm`
  tables, and spin-resolved pair tables, plus off-diagonal free-propagator
  elements and scalar/spin-resolved matrices, same-site T-matrix elements,
  compact T-matrix tables, iterative FMS setup/state-ket prelude, `minv`
  method selection, compact solver dispatch, one-energy real-space FMS
  assembly, MKGTR Green's-function trace folding, system-matrix assembly,
  `gg_full` LU side output, BiCGStab, recursion-method, Graves-Morris/Salam,
  and TFQMR FMS scattering, compact and full-potential LU FMS scattering
  solves.
- Initial RIXS numerical helpers for the FEFF KK and double-Lorentz analytic
  integrals, complex 2-D bilinear interpolation, `rdxsphrxs.f90`
  `phase.bin` handoff normalization, and the final `rixs.f90`
  wave-number preparation, radial-grid and screened-core `DeltaV` setup,
  `setkap`/`bcoef` transition-matrix setup, `rl.dat` radial-function table
  assembly, signed-`l` transition phase-shift selection, radial `DeltaV`
  overlap integration, initial `TLb` amplitude assembly, direct final-transition
  term, incident-amplitude convolution, raw cross-section assembly, `mpse.dat`
  self-energy grid preparation,
  incident-energy broadening, final-energy broadening, per-edge broadening
  pipeline, multi-edge spectrum summation, post-raw standard spectrum pipeline,
  MBConv satellite convolution, `edges.dat`/default pole normalization, and
  shared `refeff-io` xas/HERFD/RIXS spectrum-output assembly, plus EELS
  electron wavelength, Euler rotation, 3x3
  matrix-vector product, q-vector mesh
  construction, q-mesh integration helpers, `readsp` spectrum assembly,
  spectrum accumulation, angular-dependence tables, collection-angle
  dependence tables, GOS table helpers, and COMPTON
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
- FEFF `ATOM` cached-output validation for `apot.bin` plus optional
  `config.dat`, `fpf0.dat`, and `log1.dat` sidecars via `atomic` and
  `refeff module atomic` when existing atomic-potential caches are available,
  including Rust-backed section-5 core-hole `dvcoul` refresh from `drho`; full
  `refeff run` also generates the source-backed `config.dat` handoff when a
  `pot.bin` cache is present but `apot.bin`/`config.dat` are not.
- FEFF `wpot`-compatible `potXX.dat` potential-output rendering from
  `ndarray` density and potential grids, including `pot.bin`/`apot.bin`
  handoff bridging and `pot`, `refeff module pot`, and `refeff module wpot`
  output generation, direct `pot.inp` source-handoff validation before the
  remaining SCF solver gate, shared `log1.dat` POT wrapper refresh, and
  optional potential-stage diagnostic cache sidecars.
- FEFF `XSPH` cached-output validation and `phase.bin`/`xsect.dat` rendering
  via `refeff module xsph`, including optional NRIXS `xsecl.dat`,
  `xsecl2.dat`, `xsecl.bin`, AXAFS `axafs.dat`, MPSE `mpse.dat`, and
  phase-mesh `emesh.dat`/`emesh.bin` cache sidecars, source-backed empty-cell
  `phase.bin` generation when all `pot.bin` potential slots have `iz=0`,
  source-backed normal-potential `phase.bin` generation from
  `pot.bin` plus `config.dat`, including active Hubbard `mldos_hubb = 2`
  `phase_h`/`aph` generation when a compatible `v_hubbard.bin` handoff is
  present, with typed
  `v_hubbard.bin`/`aphase_hubbard.bin`/`transformation_hubbard.bin` handoff
  codecs and generated/cached `aphase_hubbard.bin` preservation now available,
  including the `loss.dat`/`MkExc` MPSE pole handoff for `ixc = 0`, two-spin
  phase-only
  handoffs, ordinary and M1/E2 higher-multipole `ispin = +/-1` two-spin
  `xsect.dat` spin-merge
  handoffs including the unfiltered XMCD `ic3 = 1` cross-term retry, and generated `PrintRl` `rl.dat`
  radial-function sidecars, recovery of missing `rl.dat` from the same
  handoff when cached `phase.bin`/`xsect.dat` are preserved, source-backed
  EXAFS/XANES/XES
  normal-mesh/user-grid single-spin and ordinary plus M1/E2 two-spin
  nonstandard normal-potential `xsect.dat` generation with ordinary `l2lp = -1,
  0, 1` transition filtering, the same MPSE pole handoff, and matching
  `phase.bin` transition moments, opportunistic missing-`mpse.dat` generation
  only from complete compatible `phase.bin`/`pot.bin` handoffs so unsuitable
  optional sidecar inputs do not block cached base outputs, full-run scheduling
  of those complete
  `pot.bin`/`config.dat` handoffs before cached `phase.bin`/`xsect.dat` exist,
  including the XES screened-core-hole source path that consumes `wscrn.dat`,
  full-run `xsph-emesh` scheduling of partial `phase.bin` caches to generate
  `emesh.dat`/`emesh.bin` before later solver gates, additive
  `xsph-phase-text` scheduling of `PRINT 2` `phaseNN.dat`/`phminNN.dat`
  sidecars from the same partial phase caches, XANES default-mesh
  horizontal-plus-vertical contour preservation, global/RDINP NRIXS `l2lp = 30`
  default-mesh capacity handoff, typed NRIXS `xsecl.dat`/`xsecl2.dat`/
  `xsecl.bin` assembly from completed `xsectjas` rows, screened-core-hole
  `wscrn.dat` handling for XANES/XES, zip-backed XES/Cu phase/xsect/AXAFS
  reference coverage, plus optional `log2.dat` diagnostic preservation.
- FEFF `FMS`/`MKGTR` cached-output validation and Green's-function handoff
  rendering via `refeff module fms` when existing `gg.bin`/`gg.dat`,
  `fms.bin`, `fmsl.bin`, `gtr.dat`, `gtrNN.bin`, or `gtrl.dat` caches are
  available, including cached `transformation_hubbard.bin` preservation when
  `hubbard.inp`/`phase.bin` dimensions are present, MKGTR generation of
  missing `fms.bin`/`gtr.dat` from cached absorber `gg` plus `phase.bin` and
  non-NRIXS `global.inp`; missing
  `gg.bin`/`gg.dat` can also be generated from `phase.bin`, `geom.dat`,
  `global.inp`, and supported Debye-Waller controls (`idwopt < 0`, `0`, `3`,
  `4`, or `5`) for non-Hubbard source phases; active Hubbard FMS/GENFMT/FF2X
  source generation remains behind the unported `fms_h` transform path, plus
  optional `log3.dat` diagnostic preservation.
- FEFF `BAND` cached-output validation and `bandstructure.dat`/`kmesh.dat`
  rendering via `refeff module band` when existing band-structure caches are
  available, plus shared `refeff-io` `kmesh.dat` generation from
  no-symmetry `reciprocal.inp` handoffs for BAND and reciprocal-space LDOS,
  including BAND pre-solver/full-run scheduling that reports generated
  `kmesh.dat` through `band-handoff` when reciprocal BAND handoffs are
  compatible before earlier unported numerical stages stop the run,
  with an explicit-operation symmetry-reduction handoff available for future
  parsed FEFF symmetry matrices, and source-backed `bandstructure.dat`
  assembly from solved k-point eigenvalue rows and typed BAND setup/result
  handoffs, plus FEFF `bandtot.f90`
  K-path point sampling, energy-search mesh setup, band-identification tail,
  `phase.bin` BAND handoff normalization, reference-energy and phase-shift
  interpolation onto the BAND search mesh, and combined CLI pre-solver setup
  from `phase.bin` plus high-symmetry K-path setup from `reciprocal.inp` before
  the remaining numerical solver boundary, full-run `band` scheduling/reporting
  when compatible source handoffs generate `bandstructure.dat`, validation-only
  `band-handoff` reporting for pre-solver handoffs, structurally stale
  `bandstructure.dat` regeneration when cached K-point rows or per-row band
  counts drift from compatible source handoffs, plus lattice
  T-matrix expansion,
  search-energy T-matrix grid assembly, phase/structure-factor grid solve
  composition, KKR `G - T^-1` work-matrix setup, and
  `structurefactor.f90` FEFF-basis block/grid conversion plus KSPACE-backed
  `STRBBDD -> STRSET -> G` grid assembly and ordinary KSPACE-plus-phase solve
  composition through final rows, plus spin-degenerate and non-degenerate
  spin-resolved multi-spin KSPACE `G` expansion using FEFF final-spin scalar
  `fmsband` wave-number semantics, plus one-point and `(energy,kpoint)` KKR
  solve composition, including the `kkrband.f90` `freeprop` raw-`G` branch,
  through final band-energy rows, plus `fmsband.f90` general complex eigenvalue
  extraction,
  single-solve and `(energy,kpoint)` KKR eigenvalue-grid orchestration through
  final band-energy rows, and
  `KSPACE/strbbdd.f90` reciprocal/direct lattice-sum accumulation with
  `strharpol.f90` real-harmonic polynomial generation, plus composed
  `STRBBDD -> STRSET` non-relativistic Gaunt contraction into `TAUKINV` and
  `strvecgen.f90`
  q-pair, direct-lattice `R`/`INDR`, reciprocal-lattice `G`, and `straa.f90`
  `EXPGNQ` reciprocal pair phase plus base `QQMLRS`/`GGJLRS` direct-term setup
  plus fixed-`ETA` `strcc.f90` `IILERS`/`D1TERM3`/`D300` energy setup for the
  future BAND solver, including FEFF `change_eta.f90` retry policy for the
  Ewald table rebuild and composed `STRBBDD -> STRSET` relativistic `SRREL`
  transform.
- FEFF `RIXS` cached-output validation and `rixsET.dat`/`rixsEE.dat` map plus
  `herfd.dat`/`xas*.dat` line-spectrum rendering via `refeff module rixs`
  when existing RIXS result caches are available, missing diagonal
  `herfd.dat`/`herfd-sat.dat`/`xas0.dat`/`xas1.dat` line caches from cached
  `rixsET.dat`/`rixsET-sat.dat`/`rixs0.dat`/`rixs1.dat`, shared
  `refeff-io` `SkipCalc`
  derivation of `herfd.dat`, `xasEI.dat`, `xasEF.dat`, and `rixsEE.dat`
  from cached `rixsET.dat` where the FEFF edge offsets are available, the same
  missing final-output derivation for cached `rixsET.dat` and
  `rixsET-sat.dat` directories with `edges.dat` in both regular cached and
  `SkipCalc` paths, full-run regression coverage for the regular cached
  `edges.dat` final-output route, plus uncached pre-solver validation of the RIXS
  `phase.bin` handoff, `global.inp` transition setup, signed-`l`
  transition phase-shift selection, and
  `rl_1.dat`/`rl_2.dat` radial-function handoffs with shared `rl.dat` fallback,
  `wscrn_1.dat`/`wscrn_2.dat` screened-core handoffs with shared `wscrn.dat`
  incident-core fallback, source-backed radial `DeltaV` overlap, initial `TLb`
  amplitude assembly, wave-number preparation, incident-amplitude convolution,
  raw cross-section assembly, optional `ReadSigma` self-energy preparation from
  cached `mpse.dat` or in-memory XSPH `xsph.inp`/`phase.bin`/`pot.bin` source
  state, post-raw standard spectrum assembly, optional MBConv satellite
  output assembly from `XES/xmu.dat`, and standard/satellite map/line output
  writing through the module and `run_for_input` paths when the transition,
  radial, screened-core, Green-function, xsect, and pole data/default-pole
  handoffs are all available,
  `gg_1.bin`/`gg_2.bin` Green-function handoffs with shared `gg.bin` fallback,
  and final-edge `xsect_2.dat`
  cross-section handoffs with shared `xsect.dat` fallback before the remaining
  numerical solver boundary, full-run `rixs-handoff` scheduling/reporting for
  those validation-only source handoffs, including full-run regression coverage
  for direct `vtot.dat`/`apot.bin` recovery of shared `wscrn.dat` and
  SCREEN-recovered shared `wscrn.dat` feeding RIXS validation, and
  optional `logrixs.dat` sidecar preservation.
- FEFF `RHORRP` cached-output validation and ASCII/binary density-grid
  rendering via `refeff module rhorrp` when existing `density.inp`-named
  output files are available, plus missing core-grid generation from
  `pot.bin`/`geom.dat` and missing non-core-grid generation from RHORRP
  density-table handoff files.
- FEFF `compton.dat` profile output generation from a validated/re-rendered
  `jzzp.dat` COMPTON cache via `refeff module compton`, plus `jzzp.dat` and
  `rhozzp.dat` generation from RHORRP density callback handoff files when
  available, cached `rhozzp.dat` diagnostic preservation when RHOZZP is
  requested, and optional `logcompton.dat` sidecar preservation.
- FEFF `FULLSPECTRUM` optical-table output generation from an existing
  `eps.dat` dielectric cache via `refeff module fullspectrum`, including
  `opcons.dat`, `opconsKK.dat`, `opcons0.dat`, and `sumrules.dat` when a
  `pot.bin` density cache is available, plus optional `drude.dat`,
  `osc_str.dat`, `hamaker.dat`, and `logfullspectrum.dat` cache sidecars.
- FEFF `CRPA` cached-output validation and `crpa.dat` rendering via
  `refeff module crpa` when an existing CRPA result cache is available, shared
  `refeff-io` paired `crpa.dat`/`wscrn.dat` generation from completed
  screened-Hubbard arrays or per-energy response slices, direct-module
  `wscrn.dat` recovery from `vtot.dat` plus `apot.bin` before the remaining
  Hubbard-U solver gate, plus optional `logscrn.dat` diagnostic preservation.
- FEFF `SCREEN` cached-output validation and `wscrn.dat` rendering via
  `refeff module screen` when an existing screened-core-hole cache is
  available, shared `refeff-io` `vtot.dat` derivation from `wscrn.dat` plus
  `pot.bin`, missing `wscrn.dat` recovery from `vtot.dat` plus `apot.bin`
  core-hole columns in module and full-run dispatch, per-energy response-slice
  handoff to `wscrn.dat`, and optional `logscreen.dat` sidecar preservation.
- FEFF `LDOS` cached-output validation and `ldosNN.dat`/`rhocNN.dat`
  rendering via `refeff module ldos` when existing density-of-states caches
  are available, source-backed non-spin and spin-resolved `ff2rho` table
  adapters for completed LDOS work arrays, no-FMS recovery in both
  `rhocNN.dat` to `ldosNN.dat` and `ldosNN.dat` to `rhocNN.dat` directions,
  source-backed FEFF header metadata for generated LDOS tables,
  full-run promotion of complete no-FMS radial handoffs without `fms.inp` and
  supported FMS source-grid handoffs to a completed `ldos` supported stage
  before the remaining solver gate,
  plus optional `logdos.dat` sidecar preservation.
- FEFF `EELS` cached-output validation and source-backed `eels.dat` generation
  via `refeff module eels` from typed `xmu*.dat` or `opconsKK*.dat` spectra,
  optional `magic.dat` and `gos1.txt`/`gos2.txt` sidecar generation, and
  optional `logeels.dat` sidecar preservation.
- FEFF `EELSMDFF` cached-output validation and source-backed complex
  `mdff.dat` generation via `mdff` and `refeff module mdff` from typed
  `xmu*.dat` or `opconsKK*.dat` spectra, including manual `q_input=1` and the
  hardcoded two-position automatic `q_input=2` branch.
- FEFF `DMDW` cached-output validation, run-type 0 single- and
  multi-temperature path Debye-Waller generation, run-type 1 atom-local and
  total vibrational-free-energy generation, run-type 2 PDS/`a2f` coupling
  sidecar generation, and run-type 3 atom-local `u^2` generation, run-type 4
  type-3-`.dym` IR Lanczos diagnostics, plus run-type 5 projected-DOS
  generation via `refeff module dmdw`; unsupported DMDW branches remain
  explicit errors.
- FEFF `PATH` `paths.dat` generation via `refeff module path` from
  `phase.bin`, `geom.dat`, and `global.inp` handoffs, plus cached-output
  validation, FEFF-compatible zero-path generation for the `rmax < 1.0`
  branch, full-run source scheduling guarded by actual handoff generation
  preflight, and optional `log4.dat` diagnostic preservation.
- FEFF `GENFMT` cached-output validation and `feff.bin`/`list.dat`
  rendering via `refeff module genfmt` when existing path-format caches are
  available, plus optional `log5.dat` diagnostic preservation.
- FEFF `FF2X` final-spectrum generation via `refeff module ff2x` from
  `xsect.dat` plus matching `feff.bin`/`list.dat` path handoffs for EXAFS,
  regular XANES, DANES, and FPRIME outputs, plus cached-output validation for
  existing `xmu.dat`/`chi.dat`/`xmul.dat`/`danes.dat` files, typed
  source-output assembly of NRIXS `xmul.dat` rows from completed decomposition
  arrays with xsect-backed photon-energy/momentum grid conversion, decomposed
  path-sum assembly from `feffl.bin` channel
  amplitudes/phases plus file-backed `fmsl.bin` FMS trace combination and
  reference-backed NRIXS `S^0(q,w)` row totals from channel backgrounds, optional
  XSCORR `prexmu.dat`, `residue.dat`,
  `contour.dat`, `curve.dat`, and `raw.dat` diagnostic sidecars, full-run
  source scheduling guarded to require FF2X-compatible contour rows after the
  horizontal
  `xsect.dat` grid, and `log6.dat` module-log preservation.
- FEFF `SFCONV` module startup compatibility via `refeff module sfconv`,
  including `sfconv.inp` parsing and disabled-path `logsfconv.dat`
  creation, enabled-path missing-target skipping, and `specfunct.dat` cache
  compatibility preflight, plus cached `chi.dat`/`chipNNNN.dat`, `xmu.dat`,
  and `feffNNNN.dat` table assembly helpers. Enabled S0^2 convolution now uses
  the Rust SO2CONV spectral-function generator when selected target files are
  present.
- FEFF `SELF` cached-output validation and `exc.dat` excitation-pole table
  rendering via `refeff module self` when an existing SELF cache is available.
- FEFF `m_mtdp` muffin-tin density/potential text read/write support.
- FEFF `apot.bin` atomic-potential TXT section-stream read/write support for
  `WriteData`, `WriteArrayData`, and `Write2D` payloads, plus a shared
  section-5 core-hole Coulomb refresh adapter backed by the ported `potslw`
  transform.
- FEFF `pot.bin` formatted text/PAD read/write support for potential-state
  handoff data, plus a borrowed `FULLSPECTRUM/rdpotp_fs.f90` view of the
  title, multiplicity, Norman-radius, and atomic-number fields.
- FEFF `phase.bin` formatted text/PAD read/write support for XSPH phase-shift
  and transition-moment handoff data, plus PATH `prcrit`, RHORRP, and RIXS
  handoff normalization for downstream criteria, density, and spectrum setup.
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
- FEFF DMDW type-1/type-3/type-4 `.dym` dynamical-matrix read/write support
  with ndarray-backed coordinate, type-3 dipole-derivative, force-constant, and
  mass-weighted matrix data.
- FEFF DMDW run-type 2 self-energy `dmdw.inp` parsing/rendering for
  displacement, electron-energy, PDS, and `a2f` handoff fields.
- FEFF DMDW run-type 2 PDS, `a2f`, and generated `dmdw_A2.dat` coupling-table
  read/write support with ndarray-backed phonon-coupling normalization.
- FEFF DMDW run-type 2 `dmdw_a2f.info`, `dmdw_spectral.info`, `dmdw_Egrid.info`,
  `dmdw_reSE_a2F.dat`, `dmdw_imSE_a2F.dat`, and `dmdw_Akw.dat`
  self-energy/spectral sidecar read/write support, plus the core pole-weight
  `a2f` diagnostic transform used to populate `dmdw_a2f.info`.
- FEFF `dmdw.out` Debye-Waller diagnostic read/write support for PDOS poles,
  Einstein summaries, moments, path/atom result tables, and the run-type 2
  mass-enhancement output marker.
- FEFF `grid.inp` energy-grid read/write support for XSPH user EGRID handoff
  files.
- FEFF `DENSITY` parsing with `density.inp` payload handoff output.
- FEFF `COMPTON`/`RHOZZP`/`CGRID` parsing with aliases and `compton.inp`
  handoff output.
- FEFF `band.inp`, `fullspectrum.inp`, `opcons.inp`, `crpa.inp`,
  `hubbard.inp`, `screen.inp`, `paths.inp`, `sfconv.inp`, `dmdw.inp`,
  `fms.inp`, `genfmt.inp`, `xsph.inp`, `pot.inp`, `global.inp`, `compton.inp`,
  `eels.inp`, `mdff.inp`, `ff2x.inp`, `ldos.inp`, and `rixs.inp` module-control
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
  handoff, including the XSPH per-spin row merge and the `FF2X/ff2gen.f90`
  `rdxbin` unit-conversion adapter.
- FEFF `xmu.dat` spectrum read/write support for final normalized absorption
  output tables, plus `FULLSPECTRUM/rdxmu.f90`, `rdxmunorm.f90`, `rdbkg.f90`,
  and `rdst.f90` adapters.
- FEFF `xmul.dat` NRIXS angular-decomposition spectrum read/write support,
  plus typed source-output assembly from completed decomposition arrays.
- FEFF `chi.dat`/`chipNNNN.dat` EXAFS spectrum read/write support for final
  and per-path output tables.
- FEFF `SO2CONV` target-data parsing and rendering for selected `xmu.dat`,
  `chi.dat`/`chipNNNN.dat`, and plain-text `feffNNNN.dat` path files,
  including previous-convolution marker detection/writing, `reff` metadata,
  seven-column path rows, and adapters for applying row-level SO2CONV XANES,
  EXAFS, and path-average results.
- FEFF `eels.dat` spectrum read/write support for orientation-averaged and
  tensor-resolved EELS output tables.
- FEFF `magic.dat` EELS collection-angle table read/write support for the
  `MAGIC` sidecar generated by `EELS/writeangulardependence2.f90`.
- FEFF `gos1.txt` and `gos2.txt` generalized-oscillator-strength read/write
  support for the EELS mode-9 sidecars from `writeangulardependence3.f90`.
- FEFF `danes.dat` anomalous-scattering read/write support for FPRIME/DANES
  output tables.
- FEFF `ldosNN.dat` and `rhocNN.dat` local density-of-states read/write support
  for orbital, spin-resolved, and embedded-reference output tables, plus the
  `FULLSPECTRUM/rdldos.f90` Hartree-unit adapter.
- FEFF `compton.dat`, `rhozzp.dat`, and `jzzp.dat` Compton profile,
  diagnostic, and cache read/write support.
- FEFF `crpa.dat` constrained-RPA Hubbard parameter read/write support,
  including the paired CRPA `wscrn.dat` sidecar handoff for solved response
  arrays.
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
  SFCONV handoff data, plus `SO2CONV` `apl.dat` pole-diagnostic rendering.
- FEFF `specfunct.dat` SO2CONV spectral-function cache read/write support,
  including Fortran sequential-unformatted records, pole tables, momentum
  metadata, spectral weights, cache-reuse compatibility checks, and a typed
  bridge to the core momentum spectral interpolation, EXAFS/XANES
  row-convolution kernels, and cached `xmu.dat` XANES table assembly.
- FEFF `mpse.dat` many-pole self-energy read/write support for complex
  self-energy and optional renormalization tables.
- FEFF RIXS map and line-spectrum output read/write support for `rixsET.dat`
  and `herfd*.dat` tables, including typed cached `SkipCalc` output assembly.
- FEFF `edges.dat`, `chemical.dat`, and `emesh.dat` scalar/energy-grid
  read/write support for potential, phase, and downstream RIXS handoff data.
- FEFF `emesh.bin` Fortran-unformatted complex energy-grid handoff read/write
  support.
- FEFF `fpf0.dat` atomic form-factor and oscillator-strength read/write
  support for anomalous-scattering handoff data.
- FEFF XSCORR intermediate table read/write support for `prexmu.dat`,
  `residue.dat`, `contour.dat`, `curve.dat`, and `raw.dat`.
- FEFF screened-core-hole radial table read/write support for `wscrn.dat` and
  `vtot.dat`, including typed `vtot.dat` derivation from SCREEN/POT handoffs.
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

Numerical FEFF modules are source-backed behind compatibility tests against
the ignored `feff10/` reference tree. Module-level gates are clean in the
current inventory, while branch-level FEFF10 parity broadening is tracked
separately.

The default `refeff run` command executes the supported Rust full-workflow
orchestration. Use `--output` to keep generated FEFF files out of the input
directory:

```sh
cargo build --release -p refeff-cli --bin refeff --bin feff
target/release/refeff run --input path/to/feff.inp --output run/refeff
```

For FEFF-style usage from a calculation directory, copy or create `feff.inp`
there and run:

```sh
target/release/feff run
```
