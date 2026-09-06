# September 2026 scientific parity corrections

These corrections follow the performance work recorded in
[PERFORMANCE_2026-09.md](PERFORMANCE_2026-09.md). Reference calculations use
FEFF revision `0a4fbd797cf72938f64dda034a438ce009ec6eb7`.

## Calculation changes

- Reciprocal FMS now converts Cartesian k-points in inverse Bohr to the reduced
  coordinates required by FEFF's lattice sum. An independent native structure
  factor regression covers diagonal and intersite matrix entries. In the
  Graphite workflow, the chemical-potential error fell from about 0.028 eV to
  0.0000013 eV and total EELS relative L2 fell from 0.06069 to 0.00001297.
- Reciprocal FMS retains FEFF's increased Ewald parameter across the energy
  contour. Resetting it for each point incorrectly changed the vertical contour
  after high-energy points triggered `change_eta`. A regression covers a
  high-energy retry followed by a lower-energy vertical point.
- ATOM reproduces the single-precision expression used to initialize FEFF's
  atomic radius, then interpolates potentials, densities and spinors onto the
  output grid for both point and finite nuclei. Fresh total energies and overlap
  integrals retain their solver grid and finite-nucleus origin powers. Energy
  reconstruction from cached APOT spinors uses their output grid instead.
  Core-hole Coulomb integration separately reproduces APOT's double-precision
  origin and promoted single-precision step; a native `potslw` oracle covers
  both its radii and potentials.
- XSPH potential and spinor resampling preserves IFUNS' promoted REAL
  origin while retaining the caller's double-precision grid spacing. Native
  FIXVAR radii and automatic potential jumps are checked at the existing
  `1e-12` limit. The separate WFIRDC photoelectron mesh reproduces the REAL
  exponential and nuclear-charge multiplication before promotion to double.
  With an identical CeO2 POT handoff, ordinary and Hubbard phase relative L2
  errors fall from about `1e-5` to `1e-13`; transition moments agree to `3e-14`.
- Real-space FMS reproduces the fused squared-radius expression in native
  GETANG. A regression links the original native object and catches the
  previous six-ulp polar-angle difference for an off-axis CeO2 pair.
  POT also converts double-precision coordinates to Bohr before narrowing,
  retaining native inverse-Bohr wave numbers and cluster cutoffs in FMSIE.
- POT spherical overlap uses COMMON/xx's promoted single-precision grid
  consistently for both integration radii and interpolation caps. The existing
  native SUMAX oracle now has a `1e-10` check, catching the previous `3.8e-9`
  outer-cap drift without changing its expected values.
- POT's magnetic-channel trace accumulates and applies its phase factor in
  single precision, matching `POT/fmsie`; SCREEN retains its double-precision
  projection. Native trace values and a cancellation regression distinguish
  the two producers.
- Finite-nucleus grids can end before the output grid. Beyond their last point,
  bound quantities continue their terminal exponential decay and Coulomb
  potentials continue as `1/r`. This replaces unbounded cubic extrapolation,
  which produced a negative integrated density for isolated Be. The regression
  checks positive density, the existing electron-normalization budget and
  constant exterior Coulomb charge. Interpolation within the source grid is
  unchanged.
- Weakly bound Dirac states leave the matching-point fallback directly, as in
  FEFF, instead of revisiting the same search indefinitely. SCF retries a failed
  Dirac solve once after Schmidt orthogonalization. The formerly failing HIGHZ
  cases Z=71, 103, 137 and 138 complete and reproduce the printed native 1s energies.
- Compton XSPH produces its cross-section handoff and preserves the zero Fermi
  index of its contour. The RHORRP contour also preserves FEFF's single-precision
  temperature floor. Its native phase/cross-section regression passes without
  changing that regression's numerical limits.
- Compton's spatial and momentum grids reproduce native REAL expressions
  before promotion to double precision. The spatial regression links the
  original native `compton_build_grid` object. With the same Jzzp density input,
  the corrected Fourier profile agrees with native to relative L2 `8.0e-13`.
  Compton contours also skip the GENFMT/FF2X photoabsorption stages, so a stock
  `CONTROL 1 1 1 1 1 1` input can finish without nonexistent EXAFS path caches.
- Compton's RHORRP density integration evaluates independent columns using
  the configured Rayon worker count. Each column retains its original
  quadrature order, and a one-worker/four-worker regression requires identical
  output values. The generic mutable callback API retains sequential execution.
- Reciprocal FMS validates active Hubbard handoffs and follows FEFF's spherical
  phase dispatch. An isolated native calculation produced identical ordinary
  and active-Hubbard reciprocal Green functions for the same spherical phases.

POT output-inventory tests now include the existing `chemical.dat` handoff.
Finite-nucleus pipeline fixtures request POT-only output; the corrected density
tail lets them finish. A controlled retry regression separately checks SCMT
first-call state, successful recovery and exhaustion without final output.
Positive-IZSTD scheduler fixtures disable POT regeneration so they consume
their deliberately supplied synthetic POT/config handoffs.

The bounded BN density-preservation test now compares its SCF scalars to an
isolated one-iteration native calculation, using the existing scientific
relative budget of `5e-5`. Its prior scalar expectations were snapshots from
Rust's unremapped grid. The independent density-preservation assertions remain
at `1e-10`, and the initial density still has an absolute `1e-6` check.

## EELS comparison at the floating-point floor

The physical EELS relative L2 limit remains `5e-5`, including every diagonal
tensor channel, the total, atomic background and fine structure. Energy and
identity checks are unchanged.

The previous fixed `1e-20` near-zero fallback was insufficient for cancellation
in off-diagonal entries of the single-precision reciprocal FMS matrix. Reversing
only the order of native FEFF's identical k-points changed Graphite's total by
relative L2 `2.90e-7`, but changed its xz/zx channels by absolute L2
`4.84e-19`/`5.71e-19`. Both cross terms failed the previous comparison even though
the physical quadrature was unchanged. The experiment's inputs, output hashes
and measurements are recorded in [EELS_ROUNDOFF_2026-09.json](EELS_ROUNDOFF_2026-09.json).

The comparison now adds an off-diagonal roundoff floor of one `f32::EPSILON`
times the L2 norm of `sqrt(abs(reference_ii * reference_jj))`. It uses reference
diagonals, so a candidate cannot enlarge its own allowance. Resolved cross-term
changes remain subject to the existing relative limit. Regression tests retain
the previous rejection of a `1e-18` cross-term error in the smaller synthetic
spectrum, and reject larger changes in resolved tensor channels.

Native and Rust EELS, when given identical spectrum inputs, agree in their main
printed spectra and within roughly `1e-22` in the cross terms. This isolates the
observed Graphite floor to upstream reciprocal FMS rather than EELS integration.

## Native reference defects

Two original reference spectra were unsuitable for validation. Their corrections
are reproducible with `scripts/repair-native-references.py`, invoked automatically
by `xtask generate-golden` for these cases. The pinned FEFF checkout is unchanged;
patched objects are compiled in a temporary directory.

- **KSPACE/Cr2GeC:** native STRVECGEN exceeded its 5,000-vector local array bound
  and returned through Fortran STOP with status zero. The archived spectrum had
  no FMS contribution (`mu == mu0`, `chi == 0`). The repair expands only the local
  reciprocal-vector work arrays to 20,000; inputs, equations and summation order
  remain identical. FMS, MKGTR and FF2X are regenerated.
- **HUBBARD/CeO2:** native PHASE_H passed an uninitialized `ilast` to DFOVRG,
  which reads that endpoint before assigning its output. This changed the
  high-energy WKB transition in the Hubbard phases. The repair initializes
  `ilast = jri` before each magnetic-channel solve, matching the regular phase
  call's endpoint. XSPH, FMS, MKGTR and FF2X are regenerated. For Ce, the phase
  table relative L2 difference fell from approximately `1.1e-2` to `4.0e-6`.

The repair removes downstream products before execution and requires fresh GG,
FMS and nonzero fine structure with the correct contour lengths. It records
original and patched source hashes, compiler flags, linked object hashes, native
stage executable hashes, consumed inputs, outputs and validation counts in
`.native-reference-repair.json`. Reference manifests and these repair records
travel with release evidence and are checked against the tested manifest hashes.
The existing physical spectrum limits are unchanged.

## Release validation

Diagnostic runs establish the corrections above; they are not a substitute for
the release gate. Publishing requires a complete clean-revision replay of all
44 stock workflows, all 138 HIGHZ elements, the strict readiness audit and green
CI. Performance numbers from concurrent diagnostic runs are not speedup claims.
