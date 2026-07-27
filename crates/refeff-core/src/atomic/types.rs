use crate::angular::AngularError;
use crate::exchange::ExchangeError;
use crate::quadrature::QuadratureError;
use ndarray::{Array1, Array2, ArrayView1, ArrayView2, ArrayView3};
use thiserror::Error;

use crate::Real;
use crate::configuration::OrbitalConfiguration;

/// Error returned by FEFF atomic lookup helpers.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Error)]
#[non_exhaustive]
pub enum AtomicError {
    /// The requested FEFF atomic lookup table does not contain this atomic number.
    #[error("atomic number {z} is not present in the requested FEFF table")]
    InvalidAtomicNumber { z: usize },
}

/// Error returned by FEFF ATOM helper kernels.
#[derive(Debug, Clone, Copy, PartialEq, Error)]
#[non_exhaustive]
pub enum AtomMathError {
    /// `aprdev` needs the requested 1-based term to fit both coefficient rows.
    #[error(
        "atomic polynomial term {term_count} is invalid for coefficient lengths {left_len} and {right_len}"
    )]
    InvalidPolynomialTerm {
        term_count: usize,
        left_len: usize,
        right_len: usize,
    },
    /// Scalar inputs must be finite.
    #[error("atomic {field} must be finite, got {value}")]
    NonFiniteScalar { field: &'static str, value: Real },
    /// Positive scalar inputs must be greater than zero.
    #[error("atomic {field} must be positive, got {value}")]
    NonPositiveScalar { field: &'static str, value: Real },
    /// FEFF ATOM helper dimensions have fixed lower bounds.
    #[error("atomic {field} must be at least {minimum}, got {actual}")]
    InvalidCount {
        field: &'static str,
        minimum: usize,
        actual: usize,
    },
    /// FEFF ATOM helpers require a positive atomic number.
    #[error("atomic number must be positive, got {atomic_number}")]
    InvalidAtomicNumber { atomic_number: usize },
    /// FEFF `fpf0` requires a positive absorber atomic number.
    #[error("atomic form-factor atomic number must be positive, got {atomic_number}")]
    InvalidFormFactorAtomicNumber { atomic_number: usize },
    /// FEFF ATOM nuclear-potential construction requires positive finite values.
    #[error("atomic nuclear-potential {field} must be positive finite, got {value}")]
    InvalidNuclearPotentialScalar { field: &'static str, value: Real },
    /// FEFF ATOM nuclear-potential dimensions have fixed lower bounds.
    #[error("atomic nuclear-potential {field} must be at least {minimum}, got {actual}")]
    InvalidNuclearPotentialCount {
        field: &'static str,
        minimum: usize,
        actual: usize,
    },
    /// FEFF ATOM finite-nucleus branch needs a radius index inside the radial grid.
    #[error(
        "atomic nuclear radius index {nucleus_index} is outside radial grid length {radial_count}"
    )]
    NuclearRadiusOutOfRange {
        nucleus_index: usize,
        radial_count: usize,
    },
    /// FEFF atomic lookup data was unavailable while running an ATOM helper.
    #[error("atomic lookup failed")]
    AtomicLookup(#[from] AtomicError),
    /// Radial tables integrated together must have identical active lengths.
    #[error(
        "atomic radial table {table} length mismatch: expected {expected_len}, got {actual_len}"
    )]
    RadialTableLengthMismatch {
        table: &'static str,
        expected_len: usize,
        actual_len: usize,
    },
    /// FEFF radial quadrature failed while evaluating an atomic helper.
    #[error("atomic radial quadrature failed")]
    Quadrature(#[from] QuadratureError),
    /// FEFF exchange-correlation helper failed while evaluating an ATOM kernel.
    #[error("atomic exchange-correlation helper failed")]
    Exchange(#[from] ExchangeError),
    /// Relativistic kappa values must be nonzero and fit FEFF's integer algebra.
    #[error("invalid atomic relativistic kappa {kappa}")]
    InvalidKappa { kappa: i32 },
    /// FEFF `tabrat` has fixed spectroscopic labels through the `g` shell.
    #[error("atomic orbital label is unavailable for relativistic kappa {kappa}")]
    OrbitalLabelKappaOutOfRange { kappa: i32 },
    /// FEFF active orbitals must have a positive principal quantum number.
    #[error(
        "atomic principal quantum number for orbital {orbital_1based} must be positive, got {principal_quantum_number}"
    )]
    InvalidPrincipalQuantumNumber {
        orbital_1based: usize,
        principal_quantum_number: usize,
    },
    /// Two FEFF relativistic kappa values overflowed while forming `kap(j)-kap(i)`.
    #[error("atomic kappa difference overflow for left={left_kappa}, right={right_kappa}")]
    KappaDifferenceOutOfRange { left_kappa: i32, right_kappa: i32 },
    /// The Breit rank must fit the FEFF integer arithmetic used by `bkmrdf`.
    #[error("atomic Breit angular rank {rank} is outside FEFF integer range")]
    BreitRankOutOfRange { rank: usize },
    /// FEFF `etotal` Breit exchange branch arithmetic overflowed.
    #[error("atomic Breit exchange branch rank is outside FEFF integer range")]
    BreitBranchOutOfRange,
    /// FEFF `vlda` only defines exchange modes `1`, `2`, `5`, and `6`.
    #[error("atomic local-density exchange mode idfock={idfock} is undefined")]
    InvalidExchangeMode { idfock: i32 },
    /// FEFF `inmuat` checks that the compacted occupation count matches the requested ion.
    #[error(
        "atomic electron count mismatch: expected {expected} for Z={atomic_number} ionicity={ionicity}, got {actual} (tolerance {tolerance})"
    )]
    ElectronCountMismatch {
        atomic_number: usize,
        ionicity: Real,
        expected: Real,
        actual: Real,
        tolerance: Real,
    },
    /// FEFF `inmuat` only accepts orbitals with angular momentum below `n` and through `g`.
    #[error(
        "atomic orbital {orbital_1based} has invalid angular momentum {angular_momentum} for n={principal_quantum_number}, kappa={kappa}"
    )]
    OrbitalAngularMomentumOutOfRange {
        orbital_1based: usize,
        principal_quantum_number: usize,
        kappa: i32,
        angular_momentum: usize,
    },
    /// FEFF `fdrirk` needs a saved first radial factor for sentinel requests.
    #[error("atomic radial integral sentinel request requires a previous first factor")]
    MissingRadialFirstFactor,
    /// FEFF `fdrirk` integer rank/index arithmetic overflowed.
    #[error("atomic radial integral rank/index arithmetic overflowed")]
    RadialIntegralIndexOutOfRange,
    /// Wigner 3j construction failed while building Breit angular coefficients.
    #[error("atomic Breit angular coefficient construction failed")]
    BreitAngular(#[from] AngularError),
    /// The FEFF `muatco` multipole rank must fit integer Wigner arithmetic.
    #[error("atomic Coulomb angular rank {rank} is outside FEFF integer range")]
    CoulombRankOutOfRange { rank: usize },
    /// Wigner 3j construction failed while building Coulomb angular coefficients.
    #[error("atomic Coulomb angular coefficient construction failed")]
    CoulombAngular {
        /// Source Wigner-3j error.
        source: AngularError,
    },
    /// Thomas-Fermi density approximation divides by the radius.
    #[error("atomic radius must be positive, got {radius}")]
    NonPositiveRadius { radius: Real },
    /// Occupation and kappa tables must have identical orbital counts.
    #[error(
        "atomic occupation/kappa length mismatch: occupations={occupation_len}, kappas={kappa_len}"
    )]
    OccupationKappaLengthMismatch {
        occupation_len: usize,
        kappa_len: usize,
    },
    /// FEFF atomic orbital tables must all use the same active orbital count.
    #[error(
        "atomic orbital table {table} length mismatch: expected {expected_len}, got {actual_len}"
    )]
    OrbitalTableLengthMismatch {
        table: &'static str,
        expected_len: usize,
        actual_len: usize,
    },
    /// ATOM total-energy accumulation requires at least one orbital.
    #[error("atomic total energy requires at least one orbital")]
    EmptyOrbitalTable,
    /// The requested one-based active orbital is outside the active table.
    #[error("atomic active orbital {active_orbital_1based} is outside 1..={orbital_count}")]
    ActiveOrbitalOutOfRange {
        active_orbital_1based: usize,
        orbital_count: usize,
    },
    /// FEFF triangular orbital-pair storage would overflow Rust indexing.
    #[error("atomic triangular orbital table is too large for {orbital_count} orbitals")]
    OrbitalPairTableTooLarge { orbital_count: usize },
    /// Some FEFF ATOM kernels divide by active orbital occupations.
    #[error(
        "atomic occupation for orbital {orbital_1based} must be positive in {context}, got {occupation}"
    )]
    NonPositiveOccupation {
        context: &'static str,
        orbital_1based: usize,
        occupation: Real,
    },
    /// Matrix inputs must match the active radial or coefficient dimensions.
    #[error(
        "atomic matrix {table} must be {expected_rows}x{expected_columns}, got {rows}x{columns}"
    )]
    MatrixShape {
        table: &'static str,
        expected_rows: usize,
        expected_columns: usize,
        rows: usize,
        columns: usize,
    },
    /// Active radial lengths must fit the supplied component matrices.
    #[error(
        "atomic active length {active_len} for orbital {orbital_1based} exceeds row count {row_count}"
    )]
    ActiveLengthOutOfRange {
        orbital_1based: usize,
        active_len: usize,
        row_count: usize,
    },
    /// FEFF `dsordf` needs a positive odd active radial length for Simpson integration.
    #[error(
        "atomic differential integral active length {active_len} is invalid for radial grid length {radial_count}"
    )]
    InvalidDifferentialIntegralActiveLength {
        active_len: usize,
        radial_count: usize,
    },
    /// FEFF `soldir` normalization needs a positive odd active radial length.
    #[error(
        "atomic Dirac normalization active length {active_len} is invalid for radial grid length {radial_count}"
    )]
    InvalidDiracNormalizationActiveLength {
        active_len: usize,
        radial_count: usize,
    },
    /// FEFF `soldir` final normalization scales only a positive active radial prefix.
    #[error(
        "atomic Dirac solution-normalization active length {active_len} is invalid for radial grid length {radial_count}"
    )]
    InvalidDiracSolutionNormalizationActiveLength {
        active_len: usize,
        radial_count: usize,
    },
    /// FEFF `soldir` node counting scans through one-based in-grid indices.
    #[error("atomic Dirac node-count {field} index {index_1based} is outside 1..={radial_count}")]
    InvalidDiracNodeCountIndex {
        field: &'static str,
        index_1based: usize,
        radial_count: usize,
    },
    /// FEFF `soldir` node-search energy scaling reached the zero-energy cutoff.
    #[error("atomic Dirac node-search energy {energy} is below cutoff {precision}")]
    DiracNodeEnergyTooSmall { energy: Real, precision: Real },
    /// FEFF `soldir` node-search energy dropped below the apparent-potential floor.
    #[error(
        "atomic Dirac node-search energy {energy} is below apparent-potential floor {energy_floor}"
    )]
    DiracNodeEnergyBelowPotentialFloor { energy: Real, energy_floor: Real },
    /// FEFF `soldir` node-search bracket collapsed before finding the target node count.
    #[error(
        "atomic Dirac node-search bracket collapsed: einf={energy_inf}, esup={energy_sup}, precision={precision}"
    )]
    DiracNodeEnergyBracketCollapsed {
        energy_inf: Real,
        energy_sup: Real,
        precision: Real,
    },
    /// FEFF `soldir` node-search retry count overflowed Rust indexing.
    #[error("atomic Dirac node-search attempt count {search_attempt_count} overflowed")]
    DiracNodeEnergyAttemptCountOutOfRange { search_attempt_count: usize },
    /// FEFF `soldir` energy correction uses one-based matching-point indexing.
    #[error(
        "atomic Dirac energy-correction matching index {matching_index_1based} is outside 1..={radial_count}"
    )]
    DiracEnergyCorrectionMatchingIndexOutOfRange {
        matching_index_1based: usize,
        radial_count: usize,
    },
    /// FEFF `soldir` energy correction divides by the previous trial energy.
    #[error("atomic Dirac energy-correction denominator became zero")]
    ZeroDiracEnergyCorrectionDenominator,
    /// FEFF `soldir` reports zero energy when backtracking makes the step too small.
    #[error("atomic Dirac energy-correction relative step became too small: {relative_step}")]
    DiracEnergyCorrectionTooSmall { relative_step: Real },
    /// FEFF `soldir` shooting-pass setup divides by the current trial energy.
    #[error("atomic Dirac shooting-pass energy became zero")]
    ZeroDiracShootingPassEnergy,
    /// FEFF `soldir` small-component rematch retry count overflowed Rust indexing.
    #[error("atomic Dirac rematch attempt count {match_attempt_count} overflowed")]
    DiracRematchAttemptCountOutOfRange { match_attempt_count: usize },
    /// FEFF `soldir` method retry state overflowed Rust integer arithmetic.
    #[error("atomic Dirac method {method} overflowed during abnormal-exit recovery")]
    DiracMethodOutOfRange { method: i32 },
    /// FEFF `soldir` matching algebra updates only an active radial prefix.
    #[error(
        "atomic Dirac match active length {active_len} is invalid for radial grid length {radial_count}"
    )]
    InvalidDiracMatchActiveLength {
        active_len: usize,
        radial_count: usize,
    },
    /// FEFF `soldir` matching algebra uses one-based matching-point indexing.
    #[error("atomic Dirac match index {matching_index_1based} is outside 1..={active_len}")]
    DiracMatchMatchingIndexOutOfRange {
        matching_index_1based: usize,
        active_len: usize,
    },
    /// FEFF `soldir` matching algebra divides by homogeneous solution values.
    #[error("atomic Dirac match denominator {field} became zero")]
    ZeroDiracMatchDenominator { field: &'static str },
    /// FEFF `soldir` method-2 energy-disagreement source updates an active radial prefix.
    #[error(
        "atomic Dirac energy-disagreement source active length {active_len} is invalid for radial grid length {radial_count}"
    )]
    InvalidDiracEnergyDisagreementActiveLength {
        active_len: usize,
        radial_count: usize,
    },
    /// FEFF `soldir` method-2 energy correction uses Simpson integration.
    #[error(
        "atomic Dirac energy-disagreement correction active length {active_len} is invalid for radial grid length {radial_count}"
    )]
    InvalidDiracEnergyDisagreementCorrectionActiveLength {
        active_len: usize,
        radial_count: usize,
    },
    /// FEFF `soldir` method-2 energy correction divides by the cross integral.
    #[error("atomic Dirac energy-disagreement correction integral became zero")]
    ZeroDiracEnergyDisagreementCorrectionIntegral,
    /// FEFF `soldir` method-2 energy correction origin exponent became singular.
    #[error("atomic Dirac energy-disagreement correction origin exponent became zero")]
    ZeroDiracEnergyDisagreementCorrectionOriginExponent,
    /// FEFF `soldir` matching-point relocation needs a nonzero large component.
    #[error(
        "atomic Dirac matching-point update found no nonzero large component in {active_len} rows"
    )]
    DiracMatchingPointNotFound { active_len: usize },
    /// FEFF `intdir` needs enough active radial rows for its five-point history.
    #[error(
        "atomic Dirac integration active length {active_len} is invalid for radial grid length {radial_count}"
    )]
    InvalidDiracIntegrationActiveLength {
        active_len: usize,
        radial_count: usize,
    },
    /// FEFF `soldir` method-1 normalization adjusts an in-grid matching point.
    #[error(
        "atomic Dirac normalization matching index {matching_index_1based} is outside 1..={active_len}"
    )]
    DiracNormalizationMatchingIndexOutOfRange {
        matching_index_1based: usize,
        active_len: usize,
    },
    /// FEFF `intdir` fixed/inward modes need a usable one-based matching point.
    #[error(
        "atomic Dirac integration matching index {matching_index_1based} is outside the supported range for active length {active_len}"
    )]
    DiracIntegrationMatchingIndexOutOfRange {
        matching_index_1based: usize,
        active_len: usize,
    },
    /// FEFF `intdir` fixed/inward modes need a usable one-based inward start.
    #[error(
        "atomic Dirac integration max index {max_index_1based} is outside the supported range for matching index {matching_index_1based} and active length {active_len}"
    )]
    DiracIntegrationMaxIndexOutOfRange {
        max_index_1based: usize,
        matching_index_1based: usize,
        active_len: usize,
    },
    /// FEFF `intdir` search could not find the second matching-point sign change.
    #[error("atomic Dirac integration failed to find a matching point in {active_len} rows")]
    DiracIntegrationMatchingPointNotFound { active_len: usize },
    /// FEFF `intdir` inward start search ran into the matching-point window.
    #[error(
        "atomic Dirac integration inward start is too close to matching index {matching_index_1based}"
    )]
    DiracIntegrationInwardStartTooClose { matching_index_1based: usize },
    /// FEFF `intdir` requires a bound-state energy compatible with the Dirac tail.
    #[error(
        "atomic Dirac integration energy {energy} is invalid for speed of light {speed_of_light}"
    )]
    InvalidDiracIntegrationEnergy { energy: Real, speed_of_light: Real },
    /// FEFF `scfdat` computes a relative energy change using the updated orbital energy.
    #[error("atomic SCF orbital energy became zero for orbital {orbital_1based}")]
    ZeroScfOrbitalEnergy { orbital_1based: usize },
    /// FEFF `scfdat` stops when the active-orbital iteration budget is exhausted.
    #[error(
        "atomic SCF exceeded iteration limit {iteration_limit} after {iteration_count} active-orbital iteration(s)"
    )]
    ScfIterationLimitExceeded {
        iteration_count: usize,
        iteration_limit: usize,
    },
    /// FEFF `scfdat` iteration budget is derived from a per-orbital limit.
    #[error(
        "atomic SCF iteration limit overflow: max_orbital_iterations={max_orbital_iterations}, orbital_count={orbital_count}"
    )]
    ScfIterationLimitOverflow {
        max_orbital_iterations: usize,
        orbital_count: usize,
    },
    /// FEFF `scfdat` rejects converged tables if a `soldir` match remains failed.
    #[error("atomic SCF Dirac matching remained failed for orbital {orbital_1based}")]
    ScfDiracAttemptFailed { orbital_1based: usize },
    /// Some FEFF `intdir` branches do not produce outward matching values.
    #[error("atomic Dirac integration did not produce matching value {field}")]
    MissingDiracIntegrationMatchingValue { field: &'static str },
    /// FEFF `intdir` origin-development recurrence divides by this denominator.
    #[error(
        "atomic Dirac integration development denominator became zero at coefficient {coefficient_1based}"
    )]
    ZeroDiracIntegrationDevelopmentDenominator { coefficient_1based: usize },
    /// FEFF `soldir` setup needs at least one active radial row for its energy floor.
    #[error(
        "atomic Dirac solver setup active length {active_len} is invalid for radial grid length {radial_count}"
    )]
    InvalidDiracSolverSetupActiveLength {
        active_len: usize,
        radial_count: usize,
    },
    /// FEFF `soldir` setup needs a principal quantum number that fits signed node arithmetic.
    #[error("atomic Dirac solver principal quantum number {principal_quantum_number} is invalid")]
    InvalidDiracSolverPrincipalQuantumNumber { principal_quantum_number: usize },
    /// FEFF `soldir` setup found no attractive apparent potential.
    #[error("atomic Dirac solver apparent potential is non-negative: floor {energy_floor}")]
    DiracSolverPotentialNotAttractive { energy_floor: Real },
    /// FEFF `soldir` setup divides by this point-nucleus kappa/power denominator.
    #[error("atomic Dirac solver initial coefficient denominator became zero")]
    ZeroDiracSolverInitialCoefficientDenominator,
    /// One-dimensional coefficient vectors must match the FEFF origin development order.
    #[error(
        "atomic coefficient table {table} length mismatch: expected {expected_len}, got {actual_len}"
    )]
    CoefficientTableLengthMismatch {
        table: &'static str,
        expected_len: usize,
        actual_len: usize,
    },
    /// FEFF `dsordf` origin correction divides by each shifted power.
    #[error("atomic differential integral origin exponent became zero")]
    ZeroDifferentialIntegralOriginExponent,
    /// FEFF `soldir` normalization origin correction divides by shifted powers.
    #[error("atomic Dirac normalization origin exponent became zero")]
    ZeroDiracNormalizationOriginExponent,
    /// FEFF `dsordf` raises the radial grid to `n + 1`.
    #[error("atomic differential integral power {power} is outside FEFF integer range")]
    DifferentialIntegralPowerOutOfRange { power: i32 },
    /// FEFF `yzkteg` angular momentum must fit signed exponent arithmetic.
    #[error("atomic yk/zk angular momentum {angular_momentum} is outside FEFF integer range")]
    YkZkAngularMomentumOutOfRange { angular_momentum: usize },
    /// FEFF `yzkteg` origin-development algebra divides by shifted powers.
    #[error("atomic yk/zk denominator {field} became zero")]
    ZeroYkZkDenominator { field: &'static str },
    /// Schmidt orthogonalization requires a positive finite norm.
    #[error("atomic Schmidt norm for orbital {orbital_1based} must be positive, got {norm}")]
    NonPositiveNorm { orbital_1based: usize, norm: Real },
    /// The requested one-based core-hole orbital is outside the active table.
    #[error("atomic hole orbital {hole_orbital_1based} is outside 1..={orbital_count}")]
    HoleOrbitalOutOfRange {
        hole_orbital_1based: usize,
        orbital_count: usize,
    },
    /// FEFF `s02at` uses fixed 8x8 work matrices for each kappa group.
    #[error("atomic kappa group {kappa} has {count} orbitals, exceeding FEFF limit {limit}")]
    KappaGroupTooLarge {
        kappa: i32,
        count: usize,
        limit: usize,
    },
    /// Relaxed-overlap matrices must be square over the active orbital table.
    #[error("atomic overlap matrix must be {expected}x{expected}, got {rows}x{columns}")]
    OverlapMatrixShape {
        expected: usize,
        rows: usize,
        columns: usize,
    },
    /// Orbital index is outside the supplied table.
    #[error("atomic orbital index {index} is outside table length {len}")]
    OrbitalIndexOutOfRange { index: usize, len: usize },
    /// Same-orbital `fdmocc` needs a valid relativistic kappa.
    #[error("same-orbital occupation product requires nonzero kappa")]
    ZeroKappa,
    /// FEFF `afgk` must be a square rank-3 table with nonempty orbital axes.
    #[error("atomic Coulomb coefficient table has invalid shape ({rows}, {columns}, {channels})")]
    CoefficientTableShape {
        rows: usize,
        columns: usize,
        channels: usize,
    },
    /// The requested `k/2` channel is outside the supplied table.
    #[error("atomic Coulomb rank {rank} maps to channel {channel}, but table has {channels}")]
    CoefficientChannelOutOfRange {
        rank: usize,
        channel: usize,
        channels: usize,
    },
}

/// Result of FEFF `cofcon` convergence acceleration.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AtomicConvergenceMix {
    /// Weight `a = 1 - b`.
    pub initial_weight: Real,
    /// Updated final-iteration weight `b`.
    pub final_weight: Real,
    /// Updated previous error `q`, set to the current error `p`.
    pub previous_error: Real,
}

/// FEFF `bkmrdf` angular coefficients for Breit interaction terms.
///
/// The three entries correspond to FEFF's `-1`, `0`, and `+1` magnetic-order
/// slots, respectively.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AtomicBreitAngularCoefficients {
    /// Magnetic-interaction coefficients, FEFF `cmag(1:3)`.
    pub magnetic: [Real; 3],
    /// Retarded-term coefficients, FEFF `cret(1:3)`.
    pub retarded: [Real; 3],
}

/// Inputs for FEFF `ATOM/muatco.f90` Coulomb angular coefficients.
#[derive(Debug, Clone, Copy)]
pub struct AtomicCoulombCoefficientInput<'a> {
    /// Relativistic kappa values for active orbitals, FEFF `kap`.
    pub kappas: &'a [i32],
    /// Active-orbital occupation counts, FEFF `xnel`.
    pub occupations: &'a [Real],
    /// Valence occupation flags, FEFF `xnval`; positive pairs skip exchange
    /// coefficients like FEFF.
    pub valence_occupations: &'a [Real],
}

/// Inputs for FEFF `ATOM/inmuat.f90` post-`getorb` orbital setup.
#[derive(Debug, Clone, Copy)]
pub struct AtomicOrbitalInitializationInput<'a> {
    /// Atomic number `nz`.
    pub atomic_number: usize,
    /// Requested ionicity `xionin`.
    pub ionicity: Real,
    /// Principal quantum numbers for compacted orbitals, FEFF `nq`.
    pub principal_quantum_numbers: &'a [usize],
    /// Relativistic kappa values for compacted orbitals, FEFF `kap`.
    pub kappas: &'a [i32],
    /// Electron occupations for compacted orbitals, FEFF `xnel`.
    pub occupations: &'a [Real],
}

/// Result of FEFF `ATOM/inmuat.f90` post-`getorb` orbital setup.
#[derive(Debug, Clone, PartialEq)]
pub struct AtomicOrbitalInitialization {
    /// Number of active orbitals, FEFF `norb`.
    pub orbital_count: usize,
    /// Number of self-consistent orbitals, FEFF `norbsc`.
    pub self_consistent_count: usize,
    /// Wavefunction convergence target, FEFF `testy`.
    pub wavefunction_precision: Real,
    /// Energy convergence target, FEFF `teste`.
    pub energy_precision: Real,
    /// FEFF matching precision ratios `rap`.
    pub precision_ratios: [Real; 2],
    /// First matching precision, FEFF `test1 = testy / rap(1)`.
    pub primary_matching_precision: Real,
    /// Second matching precision, FEFF `test2 = testy / rap(2)`.
    pub secondary_matching_precision: Real,
    /// Origin-development order, FEFF `ndor`.
    pub development_order: usize,
    /// Number of `soldir` attempts, FEFF `nes`.
    pub attempt_count: usize,
    /// Nuclear radius index, FEFF `nuc`.
    pub nucleus_index: usize,
    /// Odd radial grid length, FEFF `idim`.
    pub radial_count: usize,
    /// Initial one-electron energies, FEFF `en`.
    pub orbital_energies: Array1<Real>,
    /// Convergence accelerators, FEFF `scc`.
    pub convergence_acceleration: Array1<Real>,
    /// Initial wavefunction errors, FEFF `scw`.
    pub wavefunction_errors: Array1<Real>,
    /// Initial energy errors, FEFF `sce`.
    pub energy_errors: Array1<Real>,
    /// Active radial rows per orbital, FEFF `nmax`.
    pub active_lengths: Array1<usize>,
    /// Shell markers, FEFF `nre`; positive values mark open shells.
    pub shell_markers: Array1<i32>,
    /// Count of same-kappa pairs requiring Lagrange parameters, FEFF `ipl`.
    pub lagrange_pair_count: usize,
    /// Zero-initialized packed Lagrange storage, FEFF `eps`.
    pub lagrange_parameters: Array1<Real>,
}

/// Inputs for FEFF `ATOM/wfirdf.f90` starting-orbital generation.
///
/// `thomas_fermi_ionicity` is FEFF's `ch` argument. `scfdat` passes
/// `-xion - 1`, which is distinct from `inmuat`'s input ionicity.
#[derive(Debug, Clone, Copy)]
pub struct AtomicInitialOrbitalsInput<'a> {
    /// Nuclear charge `dz`.
    pub nuclear_charge: Real,
    /// FEFF `ch` for the Thomas-Fermi starting potential.
    pub thomas_fermi_ionicity: Real,
    /// Principal quantum numbers `nq`.
    pub principal_quantum_numbers: &'a [usize],
    /// Relativistic kappa values `kap`.
    pub kappas: &'a [i32],
    /// Initial per-orbital endpoint `nmax`.
    pub active_lengths: &'a [usize],
    /// FEFF speed of light `cl`.
    pub speed_of_light: Real,
    /// Exponential radial-grid step `hx`.
    pub step: Real,
    /// Requested nuclear-radius index `nuc`.
    pub requested_nucleus_index: isize,
    /// Number of radial rows `idim`.
    pub radial_count: usize,
    /// Number of origin-development coefficients `ndor`.
    pub coefficient_count: usize,
    /// FEFF `dr1`, the first radial point multiplied by `dz`.
    pub first_radius_times_charge: Real,
    /// FEFF `test1` matching precision.
    pub primary_matching_precision: Real,
    /// FEFF `test2` matching precision.
    pub secondary_matching_precision: Real,
    /// Maximum `soldir` attempts `nes`.
    pub max_attempt_count: usize,
}

/// Result of FEFF `ATOM/wfirdf.f90` starting-orbital generation.
#[derive(Debug, Clone, PartialEq)]
pub struct AtomicInitialOrbitals {
    /// Logarithmic radial grid `dr`.
    pub radii: Array1<Real>,
    /// Nuclear potential before speed-of-light scaling, FEFF `dvn`.
    pub nuclear_potential: Array1<Real>,
    /// Nuclear origin coefficients before speed-of-light scaling, FEFF `anoy`.
    pub nuclear_development_coefficients: Array1<Real>,
    /// Thomas-Fermi plus nuclear potential divided by `cl`, FEFF `dv`.
    pub potential: Array1<Real>,
    /// Potential origin coefficients divided by `cl`, FEFF `av`.
    pub potential_coefficients: Array1<Real>,
    /// Final one-based nuclear-radius index `nuc`.
    pub nucleus_index: usize,
    /// First origin powers per orbital, FEFF `fl`.
    pub orbital_powers: Array1<Real>,
    /// FEFF origin rescaling factors `fix`.
    pub origin_scales: Array1<Real>,
    /// Starting one-electron energies `en`.
    pub orbital_energies: Array1<Real>,
    /// Updated per-orbital endpoints `nmax`.
    pub active_lengths: Array1<usize>,
    /// Starting large components `cg`, indexed `(radial, orbital)`.
    pub large_components: Array2<Real>,
    /// Starting small components `cp`, indexed `(radial, orbital)`.
    pub small_components: Array2<Real>,
    /// Starting large origin coefficients `bg`, indexed `(coefficient, orbital)`.
    pub large_coefficients: Array2<Real>,
    /// Starting small origin coefficients `bp`, indexed `(coefficient, orbital)`.
    pub small_coefficients: Array2<Real>,
    /// Per-orbital `soldir` exhausted-attempt flag, FEFF `ifail != 0`.
    pub attempts_exhausted: Vec<bool>,
}

/// Inputs for one FEFF `ATOM/scfdat.f90` orbital iteration body.
///
/// This composes the main loop body after optional Schmidt/Lagrange setup:
/// `potrdf`, `vlda`, `soldir`, `cofcon`, and the `dsordf` normalization pass
/// for a single active orbital.
#[derive(Debug, Clone, Copy)]
pub struct AtomicScfOrbitalIterationInput<'a> {
    /// One-based active orbital `j`.
    pub active_orbital_1based: usize,
    /// FEFF exchange-correlation branch `idfock`.
    pub exchange_mode: AtomicLocalDensityExchangeMode,
    /// Whether to include non-diagonal Lagrange terms, equivalent to `ipl != 0`.
    pub include_lagrange: bool,
    /// Number of self-consistent orbitals `norbsc`.
    pub self_consistent_count: usize,
    /// FEFF speed of light `cl`.
    pub speed_of_light: Real,
    /// Logarithmic radial-grid step `hx`.
    pub step: Real,
    /// Positive radial grid `dr`.
    pub radii: ArrayView1<'a, Real>,
    /// Active radial row counts per orbital `nmax`.
    pub active_lengths: &'a [usize],
    /// Principal quantum numbers `nq`.
    pub principal_quantum_numbers: &'a [usize],
    /// Relativistic kappa values `kap`.
    pub kappas: &'a [i32],
    /// Origin powers `fl`.
    pub orbital_powers: &'a [Real],
    /// Active-orbital occupations `xnel`.
    pub occupations: &'a [Real],
    /// Valence occupations `xnval`.
    pub valence_occupations: &'a [Real],
    /// Shell markers `nre`.
    pub shell_markers: &'a [i32],
    /// Origin rescaling factors `fix`.
    pub origin_scales: &'a [Real],
    /// Coulomb angular coefficients `afgk`.
    pub coulomb_coefficients: ArrayView3<'a, Real>,
    /// Packed Lagrange parameters `eps`.
    pub lagrange_parameters: ArrayView1<'a, Real>,
    /// Nuclear potential `dvn`.
    pub nuclear_potential: ArrayView1<'a, Real>,
    /// Nuclear origin coefficients `anoy`.
    pub nuclear_development_coefficients: ArrayView1<'a, Real>,
    /// Current large components `cg`.
    pub large_components: ArrayView2<'a, Real>,
    /// Current small components `cp`.
    pub small_components: ArrayView2<'a, Real>,
    /// Current large origin coefficients `bg`.
    pub large_coefficients: ArrayView2<'a, Real>,
    /// Current small origin coefficients `bp`.
    pub small_coefficients: ArrayView2<'a, Real>,
    /// Current orbital energies `en`.
    pub orbital_energies: &'a [Real],
    /// Current convergence final weights `scc`.
    pub convergence_acceleration: &'a [Real],
    /// Previous wavefunction errors `scw`.
    pub wavefunction_errors: &'a [Real],
    /// FEFF `test1`.
    pub primary_matching_precision: Real,
    /// FEFF `test2`.
    pub secondary_matching_precision: Real,
    /// FEFF `nes`.
    pub max_attempt_count: usize,
}

/// Result of one FEFF `ATOM/scfdat.f90` orbital iteration body.
#[derive(Debug, Clone, PartialEq)]
pub struct AtomicScfOrbitalIteration {
    /// One-based active orbital `j`.
    pub active_orbital_1based: usize,
    /// Updated orbital energy `en(j)`.
    pub orbital_energy: Real,
    /// Updated active radial endpoint `nmax(j)`.
    pub active_len: usize,
    /// Updated normalized large component column `cg(:,j)`.
    pub large_component: Array1<Real>,
    /// Updated normalized small component column `cp(:,j)`.
    pub small_component: Array1<Real>,
    /// Updated normalized large origin coefficients `bg(:,j)`.
    pub large_coefficients: Array1<Real>,
    /// Updated normalized small origin coefficients `bp(:,j)`.
    pub small_coefficients: Array1<Real>,
    /// Updated convergence final weight `scc(j)`.
    pub convergence_acceleration: Real,
    /// Updated wavefunction error `scw(j)`.
    pub wavefunction_error: Real,
    /// Updated relative energy error `sce(j)`.
    pub energy_error: Real,
    /// FEFF `ifail != 0` from `soldir`.
    pub attempts_exhausted: bool,
    /// Total density from the `vlda` call, FEFF `srho`.
    pub total_density: Array1<Real>,
    /// Valence density from the `vlda` call, FEFF `srhovl`.
    pub valence_density: Array1<Real>,
    /// Potential supplied to `soldir`, FEFF `dv`.
    pub potential: Array1<Real>,
    /// Origin coefficients supplied to `soldir`, FEFF `av`.
    pub potential_coefficients: Array1<Real>,
    /// Normalization integral after convergence mixing.
    pub normalization: Real,
}

/// Inputs for FEFF `ATOM/scfdat.f90` self-consistent orbital iteration.
///
/// This owns the positive-`niter` production loop used by FEFF10 after
/// `wfirdf`: active-orbital selection, optional active `lagdat`, one
/// `potrdf -> vlda -> soldir` body, convergence mixing, and the final
/// density recomputation over the converged orbital tables.
#[derive(Debug, Clone, Copy)]
pub struct AtomicScfInput<'a> {
    /// Nuclear charge `dz`/`nz`.
    pub nuclear_charge: Real,
    /// FEFF exchange-correlation branch `idfock`.
    pub exchange_mode: AtomicLocalDensityExchangeMode,
    /// Whether to include non-diagonal Lagrange terms, equivalent to `ipl != 0`.
    pub include_lagrange: bool,
    /// Number of self-consistent orbitals `norbsc`.
    pub self_consistent_count: usize,
    /// Positive FEFF `niter` value, interpreted as iterations per orbital.
    pub max_orbital_iterations: usize,
    /// Wavefunction convergence target `testy`.
    pub wavefunction_precision: Real,
    /// Energy convergence target `teste`.
    pub energy_precision: Real,
    /// FEFF speed of light `cl`.
    pub speed_of_light: Real,
    /// Logarithmic radial-grid step `hx`.
    pub step: Real,
    /// Positive radial grid `dr`.
    pub radii: ArrayView1<'a, Real>,
    /// Active radial row counts per orbital `nmax`.
    pub active_lengths: &'a [usize],
    /// Principal quantum numbers `nq`.
    pub principal_quantum_numbers: &'a [usize],
    /// Relativistic kappa values `kap`.
    pub kappas: &'a [i32],
    /// Origin powers `fl`.
    pub orbital_powers: &'a [Real],
    /// Active-orbital occupations `xnel`.
    pub occupations: &'a [Real],
    /// Valence occupations `xnval` or FEFF's `xnvalp` exchange table.
    pub valence_occupations: &'a [Real],
    /// Shell markers `nre`.
    pub shell_markers: &'a [i32],
    /// Origin rescaling factors `fix`.
    pub origin_scales: &'a [Real],
    /// Coulomb angular coefficients `afgk`.
    pub coulomb_coefficients: ArrayView3<'a, Real>,
    /// Packed Lagrange parameters `eps`.
    pub lagrange_parameters: ArrayView1<'a, Real>,
    /// Nuclear potential `dvn`.
    pub nuclear_potential: ArrayView1<'a, Real>,
    /// Nuclear origin coefficients `anoy`.
    pub nuclear_development_coefficients: ArrayView1<'a, Real>,
    /// Current large components `cg`.
    pub large_components: ArrayView2<'a, Real>,
    /// Current small components `cp`.
    pub small_components: ArrayView2<'a, Real>,
    /// Current large origin coefficients `bg`.
    pub large_coefficients: ArrayView2<'a, Real>,
    /// Current small origin coefficients `bp`.
    pub small_coefficients: ArrayView2<'a, Real>,
    /// Current orbital energies `en`.
    pub orbital_energies: &'a [Real],
    /// Current convergence final weights `scc`.
    pub convergence_acceleration: &'a [Real],
    /// Previous wavefunction errors `scw`.
    pub wavefunction_errors: &'a [Real],
    /// Previous relative energy errors `sce`.
    pub energy_errors: &'a [Real],
    /// FEFF `test1`.
    pub primary_matching_precision: Real,
    /// FEFF `test2`.
    pub secondary_matching_precision: Real,
    /// FEFF `nes`.
    pub max_attempt_count: usize,
}

/// Result of FEFF `ATOM/scfdat.f90` self-consistent orbital iteration.
#[derive(Debug, Clone, PartialEq)]
pub struct AtomicScf {
    /// Number of active-orbital iterations executed, FEFF `nter`.
    pub iteration_count: usize,
    /// Updated one-electron energies `en`.
    pub orbital_energies: Array1<Real>,
    /// Updated per-orbital endpoints `nmax`.
    pub active_lengths: Array1<usize>,
    /// Updated large components `cg`, indexed `(radial, orbital)`.
    pub large_components: Array2<Real>,
    /// Updated small components `cp`, indexed `(radial, orbital)`.
    pub small_components: Array2<Real>,
    /// Updated large origin coefficients `bg`, indexed `(coefficient, orbital)`.
    pub large_coefficients: Array2<Real>,
    /// Updated small origin coefficients `bp`, indexed `(coefficient, orbital)`.
    pub small_coefficients: Array2<Real>,
    /// Updated convergence final weights `scc`.
    pub convergence_acceleration: Array1<Real>,
    /// Updated wavefunction errors `scw`.
    pub wavefunction_errors: Array1<Real>,
    /// Updated relative energy errors `sce`.
    pub energy_errors: Array1<Real>,
    /// Updated packed Lagrange parameters `eps`.
    pub lagrange_parameters: Array1<Real>,
    /// Per-orbital `soldir` exhausted-attempt flags.
    pub attempts_exhausted: Vec<bool>,
    /// Final total density `srho` before FEFF's `r**2` division.
    pub total_density: Array1<Real>,
    /// Final valence density `srhovl` before FEFF's `r**2` division.
    pub valence_density: Array1<Real>,
    /// Final exchange energy-density accumulator, FEFF `vtrho`.
    pub energy_density: Array1<Real>,
    /// Returned total density after FEFF divides by `dr**2`, FEFF `srho`.
    pub density_4pi: Array1<Real>,
    /// Returned valence density after FEFF divides by `dr**2`, FEFF `srhovl`.
    pub valence_density_4pi: Array1<Real>,
    /// Returned Coulomb potential, FEFF `vcoul = potslw(srho) - nz / dr`.
    pub coulomb_potential: Array1<Real>,
}

/// Inputs for composing FEFF `inmuat -> wfirdf -> scfdat` for one atomic state.
#[derive(Debug, Clone, Copy)]
pub struct AtomicScfStateInput<'a> {
    /// Atomic number `nz`/`dz`.
    pub atomic_number: usize,
    /// Requested ionicity `xion`.
    pub ionicity: Real,
    /// FEFF `wfirdf` Thomas-Fermi ionic charge argument, often `-xion - 1`.
    pub thomas_fermi_ionicity: Real,
    /// Compacted FEFF orbital configuration from `getorb`.
    pub configuration: &'a OrbitalConfiguration,
    /// FEFF exchange-correlation branch `idfock`.
    pub exchange_mode: AtomicLocalDensityExchangeMode,
    /// Positive FEFF `niter` value, interpreted as iterations per orbital.
    pub max_orbital_iterations: usize,
    /// FEFF speed of light `cl`.
    pub speed_of_light: Real,
    /// Logarithmic radial-grid step `hx`.
    pub step: Real,
    /// FEFF `nuc` request for `wfirdf`; negative values select the finite
    /// nucleus branch.
    pub requested_nucleus_index: isize,
    /// FEFF `dr1`, the first radial point multiplied by nuclear charge.
    pub first_radius_times_charge: Real,
}

/// Result of composing one source-backed FEFF ATOM SCF state.
#[derive(Debug, Clone, PartialEq)]
pub struct AtomicScfState {
    /// Principal quantum numbers converted from the compacted configuration.
    pub principal_quantum_numbers: Array1<usize>,
    /// Relativistic kappa values from the compacted configuration.
    pub kappas: Array1<i32>,
    /// Electron occupations from the compacted configuration.
    pub occupations: Array1<Real>,
    /// Valence occupations from the compacted configuration.
    pub valence_occupations: Array1<Real>,
    /// Spin magnetization weights from the compacted configuration, FEFF `xmag`.
    ///
    /// FEFF `ATOM/scfdat.f90` uses these weights with the converged orbital
    /// components to construct the free-atom spin density returned as `dmag`.
    pub spin_magnetization: Array1<Real>,
    /// FEFF `inmuat` deterministic orbital setup.
    pub orbital_initialization: AtomicOrbitalInitialization,
    /// FEFF `wfirdf` starting orbital state.
    pub initial_orbitals: AtomicInitialOrbitals,
    /// FEFF positive-`niter` `scfdat` result.
    pub scf: AtomicScf,
}

/// FEFF `ATOM/vlda.f90` local-density exchange mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AtomicLocalDensityExchangeMode {
    /// FEFF `idfock = 1`: pure Dirac-Fock branch, so no LDA correction is added.
    DiracFockOnly,
    /// FEFF `idfock = 2`: use total density for the Von Barth-Hedin potential.
    TotalDensity,
    /// FEFF `idfock = 5`: use valence density for the Von Barth-Hedin potential.
    ValenceDensity,
    /// FEFF `idfock = 6`: subtract the Dirac-Hara core-density contribution.
    CoreDensitySeparated,
}

impl TryFrom<i32> for AtomicLocalDensityExchangeMode {
    type Error = AtomMathError;

    fn try_from(idfock: i32) -> Result<Self, Self::Error> {
        match idfock {
            1 => Ok(Self::DiracFockOnly),
            2 => Ok(Self::TotalDensity),
            5 => Ok(Self::ValenceDensity),
            6 => Ok(Self::CoreDensitySeparated),
            _ => Err(AtomMathError::InvalidExchangeMode { idfock }),
        }
    }
}

/// Inputs for FEFF `ATOM/vlda.f90`.
///
/// Component matrices use `(radial, orbital)` layout. The three initial arrays
/// correspond to FEFF common-block `dv`, `av`, and caller-owned `vtrho`; the
/// result returns their updated values without mutating the input views.
#[derive(Debug, Clone, Copy)]
pub struct AtomicLocalDensityPotentialInput<'a> {
    /// FEFF `idfock` exchange-correlation branch.
    pub mode: AtomicLocalDensityExchangeMode,
    /// Whether to accumulate `vtrho`, equivalent to FEFF `ilast > 0`.
    pub accumulate_energy_density: bool,
    /// FEFF speed of light `cl` used to scale Hartree potentials into code units.
    pub speed_of_light: Real,
    /// Positive radial grid `dr`.
    pub radii: ArrayView1<'a, Real>,
    /// Active radial row counts per orbital, FEFF `nmax`.
    pub active_lengths: &'a [usize],
    /// Active-orbital occupations, FEFF `xnel`.
    pub occupations: &'a [Real],
    /// Valence occupations, FEFF `xnval`.
    pub valence_occupations: &'a [Real],
    /// Large radial components, indexed `(row, orbital)`, FEFF `cg`.
    pub large_components: ArrayView2<'a, Real>,
    /// Small radial components, indexed `(row, orbital)`, FEFF `cp`.
    pub small_components: ArrayView2<'a, Real>,
    /// Initial total potential, FEFF `dv`.
    pub initial_potential: ArrayView1<'a, Real>,
    /// Initial origin-development coefficients, FEFF `av`.
    pub initial_development_coefficients: ArrayView1<'a, Real>,
    /// Initial exchange-correlation energy-density accumulator, FEFF `vtrho`.
    pub initial_energy_density: ArrayView1<'a, Real>,
}

/// Result of FEFF `ATOM/vlda.f90`.
#[derive(Debug, Clone, PartialEq)]
pub struct AtomicLocalDensityPotential {
    /// Total spherical density accumulator, FEFF `srho`.
    pub total_density: Array1<Real>,
    /// Valence spherical density accumulator, FEFF `srhovl`.
    pub valence_density: Array1<Real>,
    /// Updated potential, FEFF `dv`.
    pub potential: Array1<Real>,
    /// Updated origin-development coefficients, FEFF `av`.
    pub development_coefficients: Array1<Real>,
    /// Updated exchange-correlation energy-density accumulator, FEFF `vtrho`.
    pub energy_density: Array1<Real>,
}

/// Inputs for FEFF `ATOM/potrdf.f90`.
///
/// This builds the central Coulomb potential and exchange/Lagrange source terms
/// for one active orbital. Component matrices use `(radial, orbital)` layout;
/// coefficient matrices use `(coefficient, orbital)` layout.
#[derive(Debug, Clone, Copy)]
pub struct AtomicOrbitalPotentialInput<'a> {
    /// One-based orbital index `ia`.
    pub active_orbital_1based: usize,
    /// Whether to include exchange terms, equivalent to FEFF `method != 0`.
    pub include_exchange: bool,
    /// Whether to include non-diagonal Lagrange terms, equivalent to FEFF `ipl != 0`.
    pub include_lagrange: bool,
    /// Number of self-consistent orbitals participating in Lagrange terms, FEFF `norbsc`.
    pub self_consistent_count: usize,
    /// FEFF speed of light `cl`.
    pub speed_of_light: Real,
    /// Logarithmic radial-grid step `hx`.
    pub step: Real,
    /// Positive radial grid `dr`.
    pub radii: ArrayView1<'a, Real>,
    /// Active radial row counts per orbital, FEFF `nmax`.
    pub active_lengths: &'a [usize],
    /// Relativistic kappa values, FEFF `kap`.
    pub kappas: &'a [i32],
    /// Origin powers per orbital, FEFF `fl`.
    pub orbital_powers: &'a [Real],
    /// Active-orbital occupations, FEFF `xnel`.
    pub occupations: &'a [Real],
    /// Shell markers used by FEFF's Lagrange branch, FEFF `nre`.
    pub shell_markers: &'a [i32],
    /// Origin rescaling factors, FEFF `fix`.
    pub origin_scales: &'a [Real],
    /// Coulomb angular coefficients, FEFF `afgk`.
    pub coulomb_coefficients: ArrayView3<'a, Real>,
    /// Packed Lagrange parameters, FEFF `eps`.
    pub lagrange_parameters: ArrayView1<'a, Real>,
    /// Nuclear radial potential, FEFF `dvn`.
    pub nuclear_potential: ArrayView1<'a, Real>,
    /// Nuclear origin-development coefficients, FEFF `anoy`.
    pub nuclear_development_coefficients: ArrayView1<'a, Real>,
    /// Large radial components, indexed `(row, orbital)`, FEFF `cg`.
    pub large_components: ArrayView2<'a, Real>,
    /// Small radial components, indexed `(row, orbital)`, FEFF `cp`.
    pub small_components: ArrayView2<'a, Real>,
    /// Large-component origin coefficients, indexed `(coefficient, orbital)`, FEFF `bg`.
    pub large_coefficients: ArrayView2<'a, Real>,
    /// Small-component origin coefficients, indexed `(coefficient, orbital)`, FEFF `bp`.
    pub small_coefficients: ArrayView2<'a, Real>,
}

/// Result of FEFF `ATOM/potrdf.f90`.
#[derive(Debug, Clone, PartialEq)]
pub struct AtomicOrbitalPotential {
    /// Updated central potential, FEFF `dv`.
    pub central_potential: Array1<Real>,
    /// Updated central-potential origin coefficients, FEFF `av`.
    pub central_development_coefficients: Array1<Real>,
    /// Large-component exchange/Lagrange source, FEFF `eg`.
    pub exchange_large: Array1<Real>,
    /// Small-component exchange/Lagrange source, FEFF `ep`.
    pub exchange_small: Array1<Real>,
    /// Large-source origin coefficients, FEFF `ceg`.
    pub exchange_large_coefficients: Array1<Real>,
    /// Small-source origin coefficients, FEFF `cep`.
    pub exchange_small_coefficients: Array1<Real>,
}

/// Inputs for FEFF `ATOM/lagdat.f90` non-diagonal Lagrange parameters.
#[derive(Debug, Clone, Copy)]
pub struct AtomicLagrangeParametersInput<'a> {
    /// Optional one-based orbital index `ia`; `None` computes every FEFF pair.
    pub active_orbital_1based: Option<usize>,
    /// Whether to include FEFF exchange terms, equivalent to `iex != 0`.
    pub include_exchange: bool,
    /// Relativistic kappa values for self-consistent orbitals, FEFF `kap`.
    pub kappas: &'a [i32],
    /// Active-orbital occupation counts, FEFF `xnel`.
    pub occupations: &'a [Real],
    /// FEFF `nre` shell markers; negative values are closed-shell markers.
    pub shell_markers: &'a [i32],
    /// Coulomb angular coefficients from [`super::atomic_coulomb_coefficients`].
    pub coulomb_coefficients: ArrayView3<'a, Real>,
}

/// Inputs for FEFF `ATOM/tabrat.f90` orbital moment and overlap tabulation.
#[derive(Debug, Clone, Copy)]
pub struct AtomicTabulationInput<'a> {
    /// Principal quantum numbers, FEFF `nq`.
    pub principal_quantum_numbers: &'a [usize],
    /// Relativistic kappa values for active orbitals, FEFF `kap`.
    pub kappas: &'a [i32],
    /// Active-orbital occupation counts, FEFF `xnel`.
    pub occupations: &'a [Real],
    /// One-electron orbital energies in Hartree, FEFF `en`.
    pub orbital_energies: &'a [Real],
}

/// FEFF `dsordf`-style request made by [`super::atomic_tabulation`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AtomicTabulationIntegralRequest {
    /// Zero-based left orbital index.
    pub left: usize,
    /// Zero-based right orbital index.
    pub right: usize,
    /// Power `n` in FEFF's average value of `r**n`.
    pub power: i32,
}

/// Result of FEFF `ATOM/tabrat.f90` tabulation.
#[derive(Debug, Clone, PartialEq)]
pub struct AtomicTabulation {
    /// Per-orbital electron counts, binding energies, and radial moments.
    pub orbitals: Vec<AtomicTabulatedOrbital>,
    /// Same-kappa overlap integrals for distinct orbital pairs.
    pub overlaps: Vec<AtomicTabulatedOverlap>,
}

/// One FEFF `tabrat` orbital row.
#[derive(Debug, Clone, PartialEq)]
pub struct AtomicTabulatedOrbital {
    /// Principal quantum number, FEFF `nq`.
    pub principal_quantum_number: usize,
    /// Spectroscopic label from FEFF's fixed `s`, `p*`, `p`, ... table.
    pub orbital_label: &'static str,
    /// Active-orbital occupation count, FEFF `xnel`.
    pub occupation: Real,
    /// Positive binding energy in eV, printed by FEFF as `-E`.
    pub binding_energy_ev: Real,
    /// Average values of `r**n` in FEFF's tabulation order.
    pub moments: Vec<AtomicTabulatedMoment>,
}

/// One average-value entry from FEFF `tabrat`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AtomicTabulatedMoment {
    /// Power `n` in `r**n`.
    pub power: i32,
    /// Average value returned by FEFF `dsordf`.
    pub value: Real,
}

/// One same-kappa overlap row from FEFF `tabrat`.
#[derive(Debug, Clone, PartialEq)]
pub struct AtomicTabulatedOverlap {
    /// Zero-based left orbital index.
    pub left: usize,
    /// Zero-based right orbital index.
    pub right: usize,
    /// Left principal quantum number.
    pub left_principal_quantum_number: usize,
    /// Left spectroscopic label.
    pub left_orbital_label: &'static str,
    /// Right principal quantum number.
    pub right_principal_quantum_number: usize,
    /// Right spectroscopic label.
    pub right_orbital_label: &'static str,
    /// Overlap integral returned by FEFF `dsordf`.
    pub value: Real,
}

/// Inputs for FEFF `ATOM/fpf0.f90` form-factor tabulation.
#[derive(Debug, Clone, Copy)]
pub struct AtomicFormFactorInput<'a> {
    /// Absorber atomic number, FEFF `iz`.
    pub atomic_number: usize,
    /// One-based core-hole orbital index, FEFF `iholep`.
    pub hole_orbital_1based: usize,
    /// Logarithmic radial-grid step, FEFF `hx`.
    pub radial_step: Real,
    /// Total atomic energy in FEFF atomic units, FEFF `eatom`.
    pub total_energy: Real,
    /// Radial grid in bohr, FEFF `dr`.
    pub radii: ArrayView1<'a, Real>,
    /// Spherical density table `4*pi*rho`, FEFF `srho`.
    pub density_4pi: ArrayView1<'a, Real>,
    /// Initial-state large Dirac component, FEFF `dgc0`.
    pub initial_large_component: ArrayView1<'a, Real>,
    /// Initial-state small Dirac component, FEFF `dpc0`.
    pub initial_small_component: ArrayView1<'a, Real>,
    /// Final-state large components indexed `(radial, orbital)`, FEFF `dgc(:,:,0)`.
    pub large_components: ArrayView2<'a, Real>,
    /// Final-state small components indexed `(radial, orbital)`, FEFF `dpc(:,:,0)`.
    pub small_components: ArrayView2<'a, Real>,
    /// Active-orbital occupations, FEFF `xnel`.
    pub occupations: &'a [Real],
    /// One-electron orbital energies, FEFF `eorb`.
    pub orbital_energies: &'a [Real],
    /// Relativistic kappa values, FEFF `kappa`.
    pub kappas: &'a [i32],
}

/// Result of FEFF `ATOM/fpf0.f90` without text-file emission.
#[derive(Debug, Clone, PartialEq)]
pub struct AtomicFormFactor {
    /// Absorber atomic number copied from the input.
    pub atomic_number: usize,
    /// Total-energy contribution to f-prime, FEFF `eatom*alphfs**2*5/3`.
    pub total_energy_fprime: Real,
    /// Empirical relativistic correction, FEFF `fpcorr`.
    pub relativistic_correction: Real,
    /// Dipole oscillator table in FEFF output order.
    pub oscillators: Vec<AtomicFormFactorOscillator>,
    /// Momentum-transfer grid `Q` in inverse Angstrom.
    pub form_factor_momentum: Array1<Real>,
    /// Nonresonant form-factor table `f0(Q)`.
    pub form_factor: Array1<Real>,
}

/// One oscillator-strength row from FEFF `fpf0.dat`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AtomicFormFactorOscillator {
    /// FEFF oscillator strength for this transition.
    pub oscillator_strength: Real,
    /// Bound-orbital energy in FEFF atomic units.
    pub excitation_energy: Real,
    /// One-based FEFF orbital index for this row.
    pub orbital_index_1based: usize,
}

/// Inputs for FEFF `ATOM/nucdev.f90` nuclear radial mesh construction.
#[derive(Debug, Clone, Copy)]
pub struct AtomicNuclearPotentialInput {
    /// Nuclear charge `dz`.
    pub nuclear_charge: Real,
    /// Exponential radial-grid step `hx`.
    pub step: Real,
    /// Requested nuclear-radius index `nuc`; negative values request FEFF's
    /// high-Z finite-nucleus branch and use the tabulated nuclear mass.
    pub requested_nucleus_index: isize,
    /// Number of radial tabulation points `np`.
    pub radial_count: usize,
    /// Number of origin development coefficients `ndor`.
    pub coefficient_count: usize,
    /// FEFF `dr1`, the first radial point multiplied by `dz`.
    pub first_radius_times_charge: Real,
}

/// Result of FEFF `ATOM/nucdev.f90`.
#[derive(Debug, Clone, PartialEq)]
pub struct AtomicNuclearPotential {
    /// Origin development coefficients `av`.
    pub development_coefficients: Array1<Real>,
    /// Radial grid `dr`.
    pub radii: Array1<Real>,
    /// Nuclear potential `dv`.
    pub potential: Array1<Real>,
    /// Final one-based nuclear-radius index `nuc`.
    pub nucleus_index: usize,
    /// Final FEFF `dr1`, possibly adjusted by the finite-nucleus branch.
    pub first_radius_times_charge: Real,
}

/// FEFF `dsordf` integrand family.
///
/// Orbital indices are one-based to match the Fortran interface. The
/// `multiply_by_derivative` variants correspond to negative `jnd` values in
/// FEFF, where the constructed orbital product is multiplied by the current
/// `dg/ag` development table before integration.
#[derive(Debug, Clone, Copy)]
pub enum AtomicDifferentialIntegralKind {
    /// `cg_i*cg_j + cp_i*cp_j`, FEFF `abs(jnd) == 1`.
    ComponentOverlap {
        left_orbital_1based: usize,
        right_orbital_1based: usize,
        multiply_by_derivative: bool,
    },
    /// `cg_i*cp_j`, FEFF `abs(jnd) == 2`.
    LargeSmallOverlap {
        left_orbital_1based: usize,
        right_orbital_1based: usize,
        multiply_by_derivative: bool,
    },
    /// `dg*cg_i + dp*cp_j`, FEFF `jnd == 3`.
    DerivativeProjection {
        large_orbital_1based: usize,
        small_orbital_1based: usize,
    },
    /// `dg*dg + dp*dp`, FEFF `jnd == 4`.
    DerivativeNorm { active_len: usize },
}

/// Inputs for FEFF `ATOM/dsordf.f90` radial integration.
///
/// `power` is FEFF's integer `n`, and `origin_power` is the `a` argument used
/// for the analytic origin correction. Component matrices use `(radial,
/// orbital)` layout; coefficient matrices use `(coefficient, orbital)` layout.
#[derive(Debug, Clone, Copy)]
pub struct AtomicDifferentialIntegralInput<'a> {
    /// Which FEFF `jnd` integrand to construct.
    pub kind: AtomicDifferentialIntegralKind,
    /// Power `n` in the radial factor `r**(n+1)`.
    pub power: i32,
    /// FEFF `a`, the origin power for the analytic first-interval correction.
    pub origin_power: Real,
    /// Logarithmic radial-grid step `hx`.
    pub step: Real,
    /// Positive radial grid `dr`.
    pub radii: ArrayView1<'a, Real>,
    /// Active radial row counts per orbital, FEFF `nmax`.
    pub active_lengths: &'a [usize],
    /// Origin powers per orbital, FEFF `fl`.
    pub orbital_powers: &'a [Real],
    /// Large radial components, indexed `(row, orbital)`, FEFF `cg`.
    pub large_components: ArrayView2<'a, Real>,
    /// Small radial components, indexed `(row, orbital)`, FEFF `cp`.
    pub small_components: ArrayView2<'a, Real>,
    /// Large-component origin coefficients, indexed `(coefficient, orbital)`, FEFF `bg`.
    pub large_coefficients: ArrayView2<'a, Real>,
    /// Small-component origin coefficients, indexed `(coefficient, orbital)`, FEFF `bp`.
    pub small_coefficients: ArrayView2<'a, Real>,
    /// Current large derivative/work function `dg`.
    pub derivative_large: ArrayView1<'a, Real>,
    /// Current small derivative/work function `dp`.
    pub derivative_small: ArrayView1<'a, Real>,
    /// Origin coefficients for [`Self::derivative_large`], FEFF `ag`.
    pub derivative_large_coefficients: ArrayView1<'a, Real>,
    /// Origin coefficients for [`Self::derivative_small`], FEFF `ap`.
    pub derivative_small_coefficients: ArrayView1<'a, Real>,
}

/// Inputs for FEFF `ATOM/yzkteg.f90`.
///
/// The source function is tabulated on the FEFF logarithmic radial grid. The
/// returned transform contains `yk` in FEFF's first work array and `zk` in the
/// second, along with both origin-development coefficient rows.
#[derive(Debug, Clone, Copy)]
pub struct AtomicYkZkTransformInput<'a> {
    /// Source function `f` before FEFF multiplies it by radius.
    pub source: ArrayView1<'a, Real>,
    /// Origin coefficients for `source`, FEFF `af`.
    pub source_coefficients: ArrayView1<'a, Real>,
    /// Positive radial grid `dr`.
    pub radii: ArrayView1<'a, Real>,
    /// First origin power for the source expansion, FEFF `ap`.
    pub initial_power: Real,
    /// Logarithmic radial-grid step `h`.
    pub step: Real,
    /// Coulomb rank `k`.
    pub angular_momentum: usize,
    /// Number of origin coefficients `nd`.
    pub coefficient_count: usize,
    /// Number of tabulated source rows `np`.
    pub source_len: usize,
    /// Work-array length `idim`.
    pub active_len: usize,
}

/// Result of FEFF `ATOM/yzkteg.f90`.
#[derive(Debug, Clone, PartialEq)]
pub struct AtomicYkZkTransform {
    /// FEFF `yk` output, stored in the transformed source array.
    pub yk: Array1<Real>,
    /// FEFF `zk` output.
    pub zk: Array1<Real>,
    /// Origin-development coefficients for `yk`, FEFF `af` after return.
    pub yk_coefficients: Array1<Real>,
    /// Origin-development coefficients for `zk`, FEFF `ag`.
    pub zk_coefficients: Array1<Real>,
    /// FEFF origin constant returned through `ap`.
    pub origin_constant: Real,
    /// Effective source length after FEFF clamps `np` to `idim - 2`.
    pub computed_source_len: usize,
}

/// Inputs for FEFF `ATOM/yzkrdf.f90` orbital source construction.
///
/// This covers the orbital branches used when FEFF calls `yzkrdf(i,j,k)` with
/// positive orbital indices. Set `large_small` for FEFF's `nem != 0` branch,
/// which builds `cg_i * cp_j`; otherwise the source is
/// `cg_i * cg_j + cp_i * cp_j`.
#[derive(Debug, Clone, Copy)]
pub struct AtomicYkZkExchangeInput<'a> {
    /// One-based left orbital index `i`.
    pub left_orbital_1based: usize,
    /// One-based right orbital index `j`.
    pub right_orbital_1based: usize,
    /// Whether to use the `nem != 0` large-small source branch.
    pub large_small: bool,
    /// Coulomb rank `k`.
    pub angular_momentum: usize,
    /// Logarithmic radial-grid step `hx`.
    pub step: Real,
    /// Positive radial grid `dr`.
    pub radii: ArrayView1<'a, Real>,
    /// Active radial row counts per orbital, FEFF `nmax`.
    pub active_lengths: &'a [usize],
    /// Origin powers per orbital, FEFF `fl`.
    pub orbital_powers: &'a [Real],
    /// Large radial components, indexed `(row, orbital)`, FEFF `cg`.
    pub large_components: ArrayView2<'a, Real>,
    /// Small radial components, indexed `(row, orbital)`, FEFF `cp`.
    pub small_components: ArrayView2<'a, Real>,
    /// Large-component origin coefficients, indexed `(coefficient, orbital)`, FEFF `bg`.
    pub large_coefficients: ArrayView2<'a, Real>,
    /// Small-component origin coefficients, indexed `(coefficient, orbital)`, FEFF `bp`.
    pub small_coefficients: ArrayView2<'a, Real>,
}

/// Inputs for FEFF `ATOM/yzkrdf.f90` prepared-source construction.
///
/// This covers the `i <= 0` branch where FEFF callers have already placed the
/// tabulated source in `dg` and its origin coefficients in `ag`. FEFF then sets
/// the source origin power to `k + 2` before calling `yzkteg`.
#[derive(Debug, Clone, Copy)]
pub struct AtomicYkZkPreparedSourceInput<'a> {
    /// Caller-provided source function `dg` before FEFF multiplies it by radius.
    pub source: ArrayView1<'a, Real>,
    /// Caller-provided origin coefficients for `source`, FEFF `ag`.
    pub source_coefficients: ArrayView1<'a, Real>,
    /// Positive radial grid `dr`.
    pub radii: ArrayView1<'a, Real>,
    /// Logarithmic radial-grid step `hx`.
    pub step: Real,
    /// Coulomb rank `k`.
    pub angular_momentum: usize,
    /// Number of origin coefficients `ndor`.
    pub coefficient_count: usize,
    /// Number of tabulated source rows, FEFF `j` in `yzkrdf(i,j,k)`.
    pub source_len: usize,
    /// Work-array length `idim`.
    pub active_len: usize,
}

/// Inputs for FEFF `ATOM/potslw.f90`.
///
/// `density` is FEFF `d`, the tabulated radial source before `potslw`
/// multiplies by `r`. `radii` is the logarithmic ATOM radial grid, `step` is
/// FEFF `dpas`, and `active_len` is the number of rows FEFF would pass as
/// `np`.
#[derive(Debug, Clone, Copy)]
pub struct AtomicFourPointCoulombPotentialInput<'a> {
    /// Source density `d`.
    pub density: ArrayView1<'a, Real>,
    /// Positive radial grid `dr`.
    pub radii: ArrayView1<'a, Real>,
    /// Logarithmic radial-grid step `dpas`.
    pub step: Real,
    /// Number of active radial rows `np`.
    pub active_len: usize,
}

/// Inputs for FEFF `ATOM/ortdat.f90` Schmidt orthogonalization.
#[derive(Debug, Clone, Copy)]
pub struct AtomicSchmidtOrthogonalizationInput<'a> {
    /// Optional one-based orbital index `ia`; `None` orthogonalizes all FEFF
    /// orbitals after the first against earlier same-kappa orbitals.
    pub active_orbital_1based: Option<usize>,
    /// Relativistic kappa values for active orbitals, FEFF `kap`.
    pub kappas: &'a [i32],
    /// Active radial row counts per orbital, FEFF `nmax`.
    pub active_lengths: &'a [usize],
    /// Origin powers, FEFF `fl`.
    pub orbital_powers: &'a [Real],
    /// Large radial components, indexed `(row, orbital)`, FEFF `cg`.
    pub large_components: ArrayView2<'a, Real>,
    /// Small radial components, indexed `(row, orbital)`, FEFF `cp`.
    pub small_components: ArrayView2<'a, Real>,
    /// Large-component origin coefficients, indexed `(coefficient, orbital)`, FEFF `bg`.
    pub large_coefficients: ArrayView2<'a, Real>,
    /// Small-component origin coefficients, indexed `(coefficient, orbital)`, FEFF `bp`.
    pub small_coefficients: ArrayView2<'a, Real>,
}

/// Result of FEFF `ATOM/ortdat.f90` Schmidt orthogonalization.
#[derive(Debug, Clone, PartialEq)]
pub struct AtomicSchmidtOrthogonalization {
    /// Updated large radial components, FEFF `cg`.
    pub large_components: Array2<Real>,
    /// Updated small radial components, FEFF `cp`.
    pub small_components: Array2<Real>,
    /// Updated large-component origin coefficients, FEFF `bg`.
    pub large_coefficients: Array2<Real>,
    /// Updated small-component origin coefficients, FEFF `bp`.
    pub small_coefficients: Array2<Real>,
    /// Updated active radial row counts, FEFF `nmax`.
    pub active_lengths: Vec<usize>,
}

/// FEFF `dsordf`-style integral request from `atomic_schmidt_orthogonalization`.
pub enum AtomicSchmidtIntegralRequest<'a> {
    /// Projection coefficient for subtracting a same-kappa reference orbital.
    Projection(AtomicSchmidtProjectionRequest<'a>),
    /// Norm of the current orthogonalized workspace before normalization.
    Norm(AtomicSchmidtNormRequest<'a>),
}

/// Projection request for FEFF `ortdat`'s `dsordf(j,j,0,3,fl(l))` call.
pub struct AtomicSchmidtProjectionRequest<'a> {
    /// Zero-based target orbital being orthogonalized, FEFF `l - 1`.
    pub target_orbital: usize,
    /// Zero-based same-kappa orbital being subtracted, FEFF `j - 1`.
    pub reference_orbital: usize,
    /// FEFF `fl(l)` passed through to `dsordf`.
    pub target_power: Real,
    /// Current target large-component workspace over the reference active rows.
    pub target_large: ArrayView1<'a, Real>,
    /// Current target small-component workspace over the reference active rows.
    pub target_small: ArrayView1<'a, Real>,
    /// Current target large origin coefficients.
    pub target_large_coefficients: ArrayView1<'a, Real>,
    /// Current target small origin coefficients.
    pub target_small_coefficients: ArrayView1<'a, Real>,
    /// Reference orbital large component over its active rows.
    pub reference_large: ArrayView1<'a, Real>,
    /// Reference orbital small component over its active rows.
    pub reference_small: ArrayView1<'a, Real>,
    /// Reference orbital large origin coefficients.
    pub reference_large_coefficients: ArrayView1<'a, Real>,
    /// Reference orbital small origin coefficients.
    pub reference_small_coefficients: ArrayView1<'a, Real>,
}

/// Norm request for FEFF `ortdat`'s `dsordf(l,max0,0,4,fl(l))` call.
pub struct AtomicSchmidtNormRequest<'a> {
    /// Zero-based target orbital being normalized, FEFF `l - 1`.
    pub target_orbital: usize,
    /// FEFF `max0`, the active length after all subtractions.
    pub active_len: usize,
    /// FEFF `fl(l)` passed through to `dsordf`.
    pub target_power: Real,
    /// Current target large-component workspace over `active_len` rows.
    pub target_large: ArrayView1<'a, Real>,
    /// Current target small-component workspace over `active_len` rows.
    pub target_small: ArrayView1<'a, Real>,
    /// Current target large origin coefficients.
    pub target_large_coefficients: ArrayView1<'a, Real>,
    /// Current target small origin coefficients.
    pub target_small_coefficients: ArrayView1<'a, Real>,
}

/// Inputs for FEFF `ATOM/etotal.f90` total-energy accumulation.
///
/// The radial integral solver is deliberately supplied as a callback to keep
/// this helper focused on FEFF's Coulomb/Breit energy algebra.
#[derive(Debug, Clone, Copy)]
pub struct AtomicTotalEnergyInput<'a> {
    /// Relativistic kappa values for active orbitals.
    pub kappas: &'a [i32],
    /// Occupation counts, FEFF `xnel`.
    pub occupations: &'a [Real],
    /// Valence occupation flags, FEFF `xnval`; positive values trigger FEFF's
    /// half-weight branch in exchange Coulomb accumulation.
    pub valence_occupations: &'a [Real],
    /// One-electron orbital energies, FEFF `en`.
    pub orbital_energies: &'a [Real],
    /// Coulomb angular coefficients, FEFF `afgk`, indexed as
    /// `(orbital, orbital, rank / 2)`.
    pub coulomb_coefficients: ArrayView3<'a, Real>,
}

/// Inputs for composing FEFF `ATOM/etotal.f90` with the ported
/// `ATOM/fdrirk.f90` radial integral driver.
#[derive(Debug, Clone, Copy)]
pub struct AtomicTotalEnergyRadialInput<'a> {
    /// Relativistic kappa values for active orbitals.
    pub kappas: &'a [i32],
    /// Occupation counts, FEFF `xnel`.
    pub occupations: &'a [Real],
    /// Valence occupation flags, FEFF `xnval`.
    pub valence_occupations: &'a [Real],
    /// One-electron orbital energies, FEFF `en`.
    pub orbital_energies: &'a [Real],
    /// Coulomb angular coefficients, FEFF `afgk`, indexed as
    /// `(orbital, orbital, rank / 2)`.
    pub coulomb_coefficients: ArrayView3<'a, Real>,
    /// Whether to use FEFF's `nem != 0` large-small radial source branch.
    pub large_small: bool,
    /// Logarithmic radial-grid step `hx`.
    pub step: Real,
    /// Positive radial grid `dr`.
    pub radii: ArrayView1<'a, Real>,
    /// Active radial row counts per orbital, FEFF `nmax`.
    pub active_lengths: &'a [usize],
    /// Origin powers per orbital, FEFF `fl`.
    pub orbital_powers: &'a [Real],
    /// Large radial components, indexed `(row, orbital)`, FEFF `cg`.
    pub large_components: ArrayView2<'a, Real>,
    /// Small radial components, indexed `(row, orbital)`, FEFF `cp`.
    pub small_components: ArrayView2<'a, Real>,
    /// Large-component origin coefficients, indexed `(coefficient, orbital)`, FEFF `bg`.
    pub large_coefficients: ArrayView2<'a, Real>,
    /// Small-component origin coefficients, indexed `(coefficient, orbital)`, FEFF `bp`.
    pub small_coefficients: ArrayView2<'a, Real>,
}

/// FEFF-style radial integral request passed to `atomic_total_energy`.
///
/// Orbital indices are one-based to mirror FEFF `fdrirk`. A value of `0`
/// preserves the sentinel cases used by FEFF for the already-tabulated first
/// radial factor.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AtomicRadialIntegralRequest {
    /// First left orbital index, FEFF `i`.
    pub first_left: usize,
    /// First right orbital index, FEFF `j`.
    pub first_right: usize,
    /// Second left orbital index, FEFF `l`.
    pub second_left: usize,
    /// Second right orbital index, FEFF `m`.
    pub second_right: usize,
    /// Radial integral rank, FEFF `k`.
    pub rank: usize,
}

/// Borrowed first radial factor for FEFF `ATOM/fdrirk.f90` sentinel calls.
///
/// FEFF stores this in common block `comdir` as `dg/ag` after a positive
/// `fdrirk(i,j,l,m,k)` call. Requests with `first_left == 0` or
/// `first_right == 0` reuse that state.
#[derive(Debug, Clone, Copy)]
pub struct AtomicRadialFirstFactorView<'a> {
    /// FEFF `dg`, the transformed `yk` radial factor.
    pub values: ArrayView1<'a, Real>,
    /// FEFF `ag`, shifted to origin power [`Self::origin_power`].
    pub coefficients: ArrayView1<'a, Real>,
    /// FEFF `a`, equal to `k + 1` for `fdrirk`.
    pub origin_power: Real,
}

/// Owned first radial factor produced by FEFF `ATOM/fdrirk.f90`.
#[derive(Debug, Clone, PartialEq)]
pub struct AtomicRadialFirstFactor {
    /// FEFF `dg`, the transformed `yk` radial factor.
    pub values: Array1<Real>,
    /// FEFF `ag`, shifted to origin power [`Self::origin_power`].
    pub coefficients: Array1<Real>,
    /// FEFF `a`, equal to `k + 1` for `fdrirk`.
    pub origin_power: Real,
}

impl AtomicRadialFirstFactor {
    /// Borrow this first factor for a later `fdrirk(0,0,...)` sentinel request.
    pub fn as_view(&self) -> AtomicRadialFirstFactorView<'_> {
        AtomicRadialFirstFactorView {
            values: self.values.view(),
            coefficients: self.coefficients.view(),
            origin_power: self.origin_power,
        }
    }
}

/// Inputs for FEFF `ATOM/fdrirk.f90`.
///
/// The first orbital pair constructs the Coulomb `yk` factor when both indices
/// are positive. A zero first index mirrors FEFF's common-block sentinel path
/// and requires [`Self::previous_first_factor`].
#[derive(Debug, Clone, Copy)]
pub struct AtomicRadialIntegralInput<'a> {
    /// FEFF-style `fdrirk(i,j,l,m,k)` request.
    pub request: AtomicRadialIntegralRequest,
    /// Whether to use FEFF's `nem != 0` large-small source branch.
    pub large_small: bool,
    /// Previous first factor for `fdrirk(0,0,l,m,k)`-style sentinel requests.
    pub previous_first_factor: Option<AtomicRadialFirstFactorView<'a>>,
    /// Relativistic kappa values, FEFF `kap`.
    pub kappas: &'a [i32],
    /// Logarithmic radial-grid step `hx`.
    pub step: Real,
    /// Positive radial grid `dr`.
    pub radii: ArrayView1<'a, Real>,
    /// Active radial row counts per orbital, FEFF `nmax`.
    pub active_lengths: &'a [usize],
    /// Origin powers per orbital, FEFF `fl`.
    pub orbital_powers: &'a [Real],
    /// Large radial components, indexed `(row, orbital)`, FEFF `cg`.
    pub large_components: ArrayView2<'a, Real>,
    /// Small radial components, indexed `(row, orbital)`, FEFF `cp`.
    pub small_components: ArrayView2<'a, Real>,
    /// Large-component origin coefficients, indexed `(coefficient, orbital)`, FEFF `bg`.
    pub large_coefficients: ArrayView2<'a, Real>,
    /// Small-component origin coefficients, indexed `(coefficient, orbital)`, FEFF `bp`.
    pub small_coefficients: ArrayView2<'a, Real>,
}

/// Result of FEFF `ATOM/fdrirk.f90`.
#[derive(Debug, Clone, PartialEq)]
pub struct AtomicRadialIntegral {
    /// FEFF radial integral value.
    pub value: Real,
    /// Newly computed first factor, present only when the first orbital pair
    /// in the request was positive.
    pub first_factor: Option<AtomicRadialFirstFactor>,
}

/// Result of FEFF `ATOM/etotal.f90` total-energy accumulation.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AtomicTotalEnergy {
    /// Total atomic energy in FEFF's internal energy units.
    pub total: Real,
    /// Direct Coulomb contribution, FEFF `ener(1)`.
    pub direct_coulomb: Real,
    /// Exchange Coulomb contribution, FEFF `ener(2)`.
    pub exchange_coulomb: Real,
    /// Magnetic Breit contribution, FEFF `ener(3)`.
    pub magnetic_breit: Real,
    /// Retarded Breit contribution, FEFF `ener(4)`.
    pub retarded_breit: Real,
}

/// Inputs for FEFF `ATOM/s02at.f90` relaxed-overlap amplitude reduction.
#[derive(Debug, Clone, Copy)]
pub struct AtomicOverlapAmplitudeReductionInput<'a> {
    /// Optional one-based orbital index containing the core hole, FEFF `ihole`.
    pub hole_orbital_1based: Option<usize>,
    /// Relativistic kappa values for active orbitals, FEFF `nk`.
    pub kappas: &'a [i32],
    /// Active-orbital occupation counts after valence subtraction, FEFF `xnel`.
    pub occupations: &'a [Real],
    /// Relaxed-orbital overlap integrals, FEFF `ovpint(i,j)`.
    pub overlap_integrals: ArrayView2<'a, Real>,
}
