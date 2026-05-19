//! Atomic lookup tables and small ATOM helper kernels ported from FEFF.
//!
//! This module ports `ATOM/nucmass.f90` and `COMMON/pertab.f90`. FEFF stores
//! many unsuffixed real literals in double-precision arrays, so those values
//! are rounded through single precision before use; the Rust tables keep that
//! behavior explicitly. It also includes compact helper routines from
//! `ATOM/aprdev.f90`, `ATOM/cofcon.f90`, `ATOM/dentfa.f90`,
//! `ATOM/fdmocc.f90`, `ATOM/akeato.f90`, `ATOM/muatco.f90`,
//! `ATOM/inmuat.f90`,
//! `ATOM/lagdat.f90`, `ATOM/ortdat.f90`, `ATOM/tabrat.f90`,
//! `ATOM/fpf0.f90`, `ATOM/nucdev.f90`, `ATOM/dsordf.f90`,
//! `ATOM/yzkteg.f90`, `ATOM/yzkrdf.f90`, `ATOM/fdrirk.f90`,
//! `ATOM/vlda.f90`, `ATOM/potrdf.f90`, `ATOM/intdir.f90`,
//! `ATOM/soldir.f90`,
//! `ATOM/bkmrdf.f90`, and
//! `ATOM/s02at.f90`.

use crate::angular::{AngularError, wigner_3j};
use crate::exchange::{ExchangeError, dirac_hara_exchange_potential, von_barth_hedin_potential};
use crate::grid::FEFF_FERMI_MOMENTUM_FACTOR;
use crate::quadrature::{QuadratureError, somm};
use ndarray::{
    Array1, Array2, Array3, ArrayView1, ArrayView2, ArrayView3, Axis, ShapeBuilder, Slice,
};
use thiserror::Error;

use crate::Real;

const ATOM_TABRAT_HARTREE_EV: Real = 27.211_396;
const ATOM_TABRAT_MOMENT_POWERS: [i32; 7] = [6, 4, 2, 1, -1, -2, -3];
const ATOM_TABRAT_LABELS: [&str; 9] = ["s", "p*", "p", "d*", "d", "f*", "f", "g*", "g"];
const ATOM_FPF0_BOHR_ANGSTROM: Real = 0.529_177_249;
const ATOM_FPF0_FINE_STRUCTURE: Real = 1.0 / 137.035_989_56;
const ATOM_FPF0_FORM_FACTOR_POINTS: usize = 81;
const ATOM_FPF0_MOMENTUM_STEP_INV_ANGSTROM: Real = 0.5;
const ATOM_NUCDEV_RADIUS_FACTOR: Real = 2.2677e-05;
const ATOM_INMUAT_WAVEFUNCTION_PRECISION: Real = 1.0e-5;
const ATOM_INMUAT_ENERGY_PRECISION: Real = 5.0e-6;
const ATOM_INMUAT_PRIMARY_RATIO: Real = 100.0;
const ATOM_INMUAT_SECONDARY_RATIO: Real = 10.0;
const ATOM_INMUAT_DEVELOPMENT_ORDER: usize = 10;
const ATOM_INMUAT_ATTEMPT_COUNT: usize = 50;
const ATOM_INMUAT_NUCLEUS_INDEX: usize = 11;
const ATOM_INMUAT_LAGRANGE_CAPACITY: usize = 820;
const ATOM_INMUAT_DEFAULT_RADIAL_COUNT: usize = 251;
const ATOM_INMUAT_ELECTRON_TOLERANCE: Real = 0.001;
const ATOM_INMUAT_DEFAULT_CONVERGENCE: Real = 0.3_f32 as Real;
const ATOM_INTDIR_HISTORY: usize = 5;
const ATOM_INTDIR_PREDICTOR: [Real; ATOM_INTDIR_HISTORY] =
    [251.0, -1274.0, 2616.0, -2774.0, 1901.0];
const ATOM_INTDIR_CORRECTOR_RAW: [Real; ATOM_INTDIR_HISTORY] = [-19.0, 106.0, -264.0, 646.0, 251.0];
const ATOM_INTDIR_MIX_NUMERATOR: Real = 473.0;
const ATOM_INTDIR_MIX_DENOMINATOR: Real = 502.0;
const ATOM_INTDIR_STEP_DIVISOR: Real = 720.0;
const ATOM_INTDIR_INWARD_THRESHOLD: Real = 700.0;
const ATOM_INTDIR_EXPONENT_FLOOR: Real = -170.0;
const ATOM_SOLDIR_MATCHING_POINT_TAIL_MARGIN: usize = 10;
const ATOM_SOLDIR_MATCHING_POINT_FALLBACK_OFFSET: usize = 12;
const ATOM_SOLDIR_FIXED_MATCHING_RELATIVE_ENERGY: Real = 1.0e-1;

/// Error returned by FEFF atomic lookup helpers.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Error)]
pub enum AtomicError {
    /// The requested FEFF atomic lookup table does not contain this atomic number.
    #[error("atomic number {z} is not present in the requested FEFF table")]
    InvalidAtomicNumber { z: usize },
}

/// Error returned by FEFF ATOM helper kernels.
#[derive(Debug, Clone, Copy, PartialEq, Error)]
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

/// Inputs for FEFF `ATOM/soldir.f90` `norm`.
#[derive(Debug, Clone, Copy)]
pub struct AtomicDiracNormalizationInput<'a> {
    /// Radial grid `dr`.
    pub radii: ArrayView1<'a, Real>,
    /// Large radial component `gg`.
    pub large_component: ArrayView1<'a, Real>,
    /// Small radial component `gp`.
    pub small_component: ArrayView1<'a, Real>,
    /// Large origin-development coefficients `ag`.
    pub large_coefficients: ArrayView1<'a, Real>,
    /// Small origin-development coefficients `ap`.
    pub small_coefficients: ArrayView1<'a, Real>,
    /// FEFF solution method selector; only `1` applies the matching correction.
    pub method: i32,
    /// Logarithmic radial-grid step `hx`.
    pub step: Real,
    /// Active origin-development coefficient count `ndor`.
    pub coefficient_count: usize,
    /// Inward small-component value at the matching point, FEFF `gpmat`.
    pub matching_small_component: Real,
    /// First origin-development power `fl`.
    pub origin_power: Real,
    /// Last active radial row `max0`.
    pub active_len: usize,
    /// One-based matching-point index `mat`.
    pub matching_index_1based: usize,
}

/// FEFF `soldir` normalization integral.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AtomicDiracNormalization {
    /// Normalization integral `b` before `soldir` takes its square root.
    pub norm: Real,
}

/// Inputs for FEFF `ATOM/soldir.f90` final wavefunction normalization.
#[derive(Debug, Clone, Copy)]
pub struct AtomicDiracSolutionNormalizationInput<'a> {
    /// Normalization integral `b` before FEFF takes its square root.
    pub norm: Real,
    /// Initial large origin coefficient `agi`, used only for FEFF's sign rule.
    pub initial_large_coefficient: Real,
    /// Initial small origin coefficient `api`, used only for FEFF's sign rule.
    pub initial_small_coefficient: Real,
    /// Large radial component `gg`.
    pub large_component: ArrayView1<'a, Real>,
    /// Small radial component `gp`.
    pub small_component: ArrayView1<'a, Real>,
    /// Large origin-development coefficients `ag`.
    pub large_coefficients: ArrayView1<'a, Real>,
    /// Small origin-development coefficients `ap`.
    pub small_coefficients: ArrayView1<'a, Real>,
    /// Active origin-development coefficient count `ndor`.
    pub coefficient_count: usize,
    /// Last active radial row `max0`.
    pub active_len: usize,
}

/// FEFF `soldir` normalized Dirac solution.
#[derive(Debug, Clone, PartialEq)]
pub struct AtomicDiracSolutionNormalization {
    /// Normalized large component `gg`; rows after `active_len` are zeroed.
    pub large_component: Array1<Real>,
    /// Normalized small component `gp`; rows after `active_len` are zeroed.
    pub small_component: Array1<Real>,
    /// Normalized large origin-development coefficients `ag`.
    pub large_coefficients: Array1<Real>,
    /// Normalized small origin-development coefficients `ap`.
    pub small_coefficients: Array1<Real>,
    /// Signed divisor used for radial components, FEFF `b` after sign adjustment.
    pub component_divisor: Real,
    /// Signed divisor used for origin coefficients, FEFF `c`.
    pub coefficient_divisor: Real,
}

/// Inputs for FEFF `ATOM/soldir.f90` radial node counting.
#[derive(Debug, Clone, Copy)]
pub struct AtomicDiracNodeCountInput<'a> {
    /// Large radial component `gg`.
    pub large_component: ArrayView1<'a, Real>,
    /// One-based matching-point index `mat`.
    pub matching_index_1based: usize,
    /// One-based candidate scan limit `j`; FEFF scans through `max(j, mat)`.
    pub scan_index_1based: usize,
}

/// FEFF `soldir` radial node count.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AtomicDiracNodeCount {
    /// FEFF `nd`, starting from one and counting sign changes through the scan.
    pub node_count: usize,
    /// Effective one-based scan limit, `max(j, mat)`.
    pub scan_index_1based: usize,
}

/// Inputs for FEFF `ATOM/soldir.f90` node-count energy search.
#[derive(Debug, Clone, Copy)]
pub struct AtomicDiracNodeEnergySearchInput {
    /// Current trial energy `en`.
    pub energy: Real,
    /// Current node count `nd`.
    pub node_count: usize,
    /// Target FEFF node count `node`.
    pub target_node_count: usize,
    /// FEFF upper bracket variable `esup`.
    pub energy_sup: Real,
    /// FEFF lower bracket variable `einf`.
    pub energy_inf: Real,
    /// Apparent-potential minimum `emin`.
    pub energy_floor: Real,
    /// Energy precision cutoff `test1`.
    pub energy_precision: Real,
    /// Current node-search attempt count `jes`.
    pub search_attempt_count: usize,
    /// Maximum node-search attempts `nes`.
    pub max_attempt_count: usize,
}

/// Result of FEFF `soldir` node-count energy search.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AtomicDiracNodeEnergySearch {
    /// Updated trial energy `en`.
    pub energy: Real,
    /// Updated FEFF upper bracket variable `esup`.
    pub energy_sup: Real,
    /// Updated FEFF lower bracket variable `einf`.
    pub energy_inf: Real,
    /// Updated attempt count `jes`.
    pub search_attempt_count: usize,
    /// Whether FEFF would return to the integration loop at label `106`.
    pub needs_reintegration: bool,
    /// Whether FEFF would set `ifail = 1` and continue with the current solution.
    pub attempts_exhausted: bool,
}

/// Inputs for FEFF `ATOM/soldir.f90` method-1 energy correction.
#[derive(Debug, Clone, Copy)]
pub struct AtomicDiracMethodOneEnergyCorrectionInput<'a> {
    /// FEFF speed of light `cl`.
    pub speed_of_light: Real,
    /// FEFF normalization integral `b` before the final square root.
    pub norm: Real,
    /// Large radial component `gg`.
    pub large_component: ArrayView1<'a, Real>,
    /// Small radial component `gp`.
    pub small_component: ArrayView1<'a, Real>,
    /// Inward small-component value at the matching point, FEFF `gpmat`.
    pub matching_small_component: Real,
    /// One-based matching-point index `mat`.
    pub matching_index_1based: usize,
}

/// FEFF `soldir` energy correction scalars.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AtomicDiracEnergyCorrection {
    /// Additive energy correction `f`.
    pub correction: Real,
    /// Small-component mismatch `c`, relative to `gpmat` when FEFF scales it.
    pub mismatch: Real,
}

/// Inputs for FEFF `ATOM/soldir.f90` energy-correction backtracking.
#[derive(Debug, Clone, Copy)]
pub struct AtomicDiracEnergyStepInput {
    /// Trial energy before applying `correction`, FEFF `en`.
    pub energy: Real,
    /// Additive energy correction `f`.
    pub correction: Real,
    /// Small-component or normalization mismatch `c`.
    pub mismatch: Real,
    /// FEFF `esup` search bracket value.
    pub energy_sup: Real,
    /// FEFF `einf` search bracket value.
    pub energy_inf: Real,
    /// Active mismatch tolerance, FEFF `test`.
    pub mismatch_precision: Real,
    /// Lower relative-step cutoff, FEFF `test1`.
    pub zero_energy_precision: Real,
}

/// Result of FEFF `soldir` energy-correction backtracking.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AtomicDiracEnergyStep {
    /// Updated trial energy `en`.
    pub energy: Real,
    /// Possibly halved correction `f`.
    pub correction: Real,
    /// FEFF relative step `g`.
    pub relative_step: Real,
    /// Whether FEFF would continue matching the small component.
    pub needs_rematch: bool,
}

/// Inputs for FEFF `ATOM/soldir.f90` small-component rematch attempt handling.
#[derive(Debug, Clone, Copy)]
pub struct AtomicDiracRematchAttemptInput {
    /// Small-component or normalization mismatch `c`.
    pub mismatch: Real,
    /// Active mismatch tolerance, FEFF `test`.
    pub mismatch_precision: Real,
    /// Current small-component matching attempt count `ies`.
    pub match_attempt_count: usize,
    /// Maximum matching attempts `nes`.
    pub max_attempt_count: usize,
}

/// FEFF `soldir` small-component rematch attempt result.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AtomicDiracRematchAttempt {
    /// Updated small-component matching attempt count `ies`.
    pub match_attempt_count: usize,
    /// Whether FEFF would jump back to label `105`.
    pub needs_rematch: bool,
    /// Whether FEFF would set `ifail = 1` and continue with the current solution.
    pub attempts_exhausted: bool,
}

/// Inputs for FEFF `ATOM/soldir.f90` homogeneous-system tail matching.
#[derive(Debug, Clone, Copy)]
pub struct AtomicDiracHomogeneousMatchInput<'a> {
    /// Homogeneous large radial component `gg`.
    pub large_component: ArrayView1<'a, Real>,
    /// Homogeneous small radial component `gp`.
    pub small_component: ArrayView1<'a, Real>,
    /// Inward large-component value at the matching point, FEFF `ggmat`.
    pub matching_large_component: Real,
    /// Last active radial row `max0`.
    pub active_len: usize,
    /// One-based matching-point index `mat`.
    pub matching_index_1based: usize,
}

/// FEFF `soldir` homogeneous-system tail matching result.
#[derive(Debug, Clone, PartialEq)]
pub struct AtomicDiracHomogeneousMatch {
    /// Matched large component `gg`.
    pub large_component: Array1<Real>,
    /// Matched small component `gp`.
    pub small_component: Array1<Real>,
    /// Scale applied from `mat` through `max0`, FEFF `a`.
    pub tail_scale: Real,
    /// One-based scan index set for later node counting, FEFF `j = mat`.
    pub scan_index_1based: usize,
}

/// Inputs for FEFF `ATOM/soldir.f90` method-1 large-component matching.
#[derive(Debug, Clone, Copy)]
pub struct AtomicDiracLargeComponentMatchInput<'a> {
    /// Inhomogeneous large radial component `gg`.
    pub large_component: ArrayView1<'a, Real>,
    /// Inhomogeneous small radial component `gp`.
    pub small_component: ArrayView1<'a, Real>,
    /// Homogeneous large radial component `hg`.
    pub homogeneous_large_component: ArrayView1<'a, Real>,
    /// Homogeneous small radial component `hp`.
    pub homogeneous_small_component: ArrayView1<'a, Real>,
    /// Inward large-component value at the matching point, FEFF `ggmat`.
    pub matching_large_component: Real,
    /// Last active radial row `max0`.
    pub active_len: usize,
    /// One-based matching-point index `mat`.
    pub matching_index_1based: usize,
}

/// FEFF `soldir` large-component matching result.
#[derive(Debug, Clone, PartialEq)]
pub struct AtomicDiracLargeComponentMatch {
    /// Matched large component `gg`.
    pub large_component: Array1<Real>,
    /// Matched small component `gp`.
    pub small_component: Array1<Real>,
    /// Homogeneous tail scale, FEFF `b`.
    pub tail_scale: Real,
    /// Large-component mismatch before matching, FEFF `a`.
    pub large_mismatch: Real,
}

/// Inputs for FEFF `ATOM/soldir.f90` method-2 two-component matching.
#[derive(Debug, Clone, Copy)]
pub struct AtomicDiracTwoComponentMatchInput<'a> {
    /// Inhomogeneous large radial component `gg`.
    pub large_component: ArrayView1<'a, Real>,
    /// Inhomogeneous small radial component `gp`.
    pub small_component: ArrayView1<'a, Real>,
    /// Inhomogeneous large origin-development coefficients `ag`.
    pub large_coefficients: ArrayView1<'a, Real>,
    /// Inhomogeneous small origin-development coefficients `ap`.
    pub small_coefficients: ArrayView1<'a, Real>,
    /// Homogeneous large radial component `hg`.
    pub homogeneous_large_component: ArrayView1<'a, Real>,
    /// Homogeneous small radial component `hp`.
    pub homogeneous_small_component: ArrayView1<'a, Real>,
    /// Homogeneous large origin-development coefficients `agh`.
    pub homogeneous_large_coefficients: ArrayView1<'a, Real>,
    /// Homogeneous small origin-development coefficients `aph`.
    pub homogeneous_small_coefficients: ArrayView1<'a, Real>,
    /// Inward large-component value at the matching point, FEFF `ggmat`.
    pub matching_large_component: Real,
    /// Inward small-component value at the matching point, FEFF `gpmat`.
    pub matching_small_component: Real,
    /// Homogeneous inward large-component value at the matching point, FEFF `hgmat`.
    pub homogeneous_matching_large_component: Real,
    /// Homogeneous inward small-component value at the matching point, FEFF `hpmat`.
    pub homogeneous_matching_small_component: Real,
    /// Active origin-development coefficient count `ndor`.
    pub coefficient_count: usize,
    /// Last active radial row `max0`.
    pub active_len: usize,
    /// One-based matching-point index `mat`.
    pub matching_index_1based: usize,
}

/// FEFF `soldir` two-component matching result.
#[derive(Debug, Clone, PartialEq)]
pub struct AtomicDiracTwoComponentMatch {
    /// Matched large component `gg`.
    pub large_component: Array1<Real>,
    /// Matched small component `gp`.
    pub small_component: Array1<Real>,
    /// Matched large origin-development coefficients `ag`.
    pub large_coefficients: Array1<Real>,
    /// Matched small origin-development coefficients `ap`.
    pub small_coefficients: Array1<Real>,
    /// Determinant of the matching system, FEFF `ah`.
    pub determinant: Real,
    /// Homogeneous scale applied before `mat`, FEFF `c`.
    pub prefix_scale: Real,
    /// Homogeneous scale applied from `mat` through `max0`, FEFF `b`.
    pub tail_scale: Real,
    /// Large-component mismatch before matching, FEFF `a`.
    pub large_mismatch: Real,
    /// Small-component mismatch before matching, FEFF `b` before reuse.
    pub small_mismatch: Real,
}

/// Inputs for FEFF `ATOM/soldir.f90` method-2 energy-disagreement matching.
#[derive(Debug, Clone, Copy)]
pub struct AtomicDiracEnergyDisagreementMatchInput<'a> {
    /// Energy-derivative large radial component `bg`.
    pub large_derivative: ArrayView1<'a, Real>,
    /// Energy-derivative small radial component `bp`.
    pub small_derivative: ArrayView1<'a, Real>,
    /// Energy-derivative large origin-development coefficients `bgh`.
    pub large_derivative_coefficients: ArrayView1<'a, Real>,
    /// Energy-derivative small origin-development coefficients `bph`.
    pub small_derivative_coefficients: ArrayView1<'a, Real>,
    /// Homogeneous large radial component `hg`.
    pub homogeneous_large_component: ArrayView1<'a, Real>,
    /// Homogeneous small radial component `hp`.
    pub homogeneous_small_component: ArrayView1<'a, Real>,
    /// Homogeneous large origin-development coefficients `agh`.
    pub homogeneous_large_coefficients: ArrayView1<'a, Real>,
    /// Homogeneous small origin-development coefficients `aph`.
    pub homogeneous_small_coefficients: ArrayView1<'a, Real>,
    /// Inward large derivative value at the matching point, FEFF `bgmat`.
    pub matching_large_derivative: Real,
    /// Inward small derivative value at the matching point, FEFF `bpmat`.
    pub matching_small_derivative: Real,
    /// Homogeneous inward large-component value at the matching point, FEFF `hgmat`.
    pub homogeneous_matching_large_component: Real,
    /// Homogeneous inward small-component value at the matching point, FEFF `hpmat`.
    pub homogeneous_matching_small_component: Real,
    /// Active origin-development coefficient count `ndor`.
    pub coefficient_count: usize,
    /// Last active radial row `max0`.
    pub active_len: usize,
    /// One-based matching-point index `mat`.
    pub matching_index_1based: usize,
}

/// FEFF `soldir` matched method-2 energy-derivative solution.
#[derive(Debug, Clone, PartialEq)]
pub struct AtomicDiracEnergyDisagreementMatch {
    /// Matched large derivative component `bg`.
    pub large_derivative: Array1<Real>,
    /// Matched small derivative component `bp`.
    pub small_derivative: Array1<Real>,
    /// Matched large derivative coefficients `bgh`.
    pub large_derivative_coefficients: Array1<Real>,
    /// Matched small derivative coefficients `bph`.
    pub small_derivative_coefficients: Array1<Real>,
    /// Determinant of the matching system, FEFF `ah`.
    pub determinant: Real,
    /// Homogeneous scale applied before `mat`, FEFF `a`.
    pub prefix_scale: Real,
    /// Homogeneous scale applied from `mat` through `max0`, FEFF reused `g`.
    pub tail_scale: Real,
    /// Large derivative mismatch before matching, FEFF `f`.
    pub large_mismatch: Real,
    /// Small derivative mismatch before matching, FEFF initial `g`.
    pub small_mismatch: Real,
}

/// Inputs for FEFF `ATOM/soldir.f90` method-2 energy-disagreement source.
#[derive(Debug, Clone, Copy)]
pub struct AtomicDiracEnergyDisagreementSourceInput<'a> {
    /// Current large radial component `gg`.
    pub large_component: ArrayView1<'a, Real>,
    /// Current small radial component `gp`.
    pub small_component: ArrayView1<'a, Real>,
    /// Current large origin-development coefficients `ag`.
    pub large_coefficients: ArrayView1<'a, Real>,
    /// Current small origin-development coefficients `ap`.
    pub small_coefficients: ArrayView1<'a, Real>,
    /// Radial grid `dr`.
    pub radii: ArrayView1<'a, Real>,
    /// FEFF speed of light `cl`.
    pub speed_of_light: Real,
    /// Active origin-development coefficient count `ndor`.
    pub coefficient_count: usize,
    /// Last active radial row `max0`.
    pub active_len: usize,
}

/// FEFF `soldir` method-2 energy-disagreement source arrays.
#[derive(Debug, Clone, PartialEq)]
pub struct AtomicDiracEnergyDisagreementSource {
    /// Large derivative source `bg`.
    pub large_source: Array1<Real>,
    /// Small derivative source `bp`.
    pub small_source: Array1<Real>,
    /// Large derivative source coefficients `bgh`.
    pub large_coefficients: Array1<Real>,
    /// Small derivative source coefficients `bph`.
    pub small_coefficients: Array1<Real>,
}

/// Inputs for FEFF `ATOM/soldir.f90` method-2 energy correction.
#[derive(Debug, Clone, Copy)]
pub struct AtomicDiracEnergyDisagreementCorrectionInput<'a> {
    /// Radial grid `dr`.
    pub radii: ArrayView1<'a, Real>,
    /// Current large radial component `gg`.
    pub large_component: ArrayView1<'a, Real>,
    /// Current small radial component `gp`.
    pub small_component: ArrayView1<'a, Real>,
    /// Matched large energy-derivative solution `bg`.
    pub large_derivative: ArrayView1<'a, Real>,
    /// Matched small energy-derivative solution `bp`.
    pub small_derivative: ArrayView1<'a, Real>,
    /// Current large origin-development coefficients `ag`.
    pub large_coefficients: ArrayView1<'a, Real>,
    /// Current small origin-development coefficients `ap`.
    pub small_coefficients: ArrayView1<'a, Real>,
    /// Matched large derivative coefficients `bgh`.
    pub large_derivative_coefficients: ArrayView1<'a, Real>,
    /// Matched small derivative coefficients `bph`.
    pub small_derivative_coefficients: ArrayView1<'a, Real>,
    /// FEFF normalization integral `b`.
    pub norm: Real,
    /// Logarithmic radial-grid step `hx`.
    pub step: Real,
    /// First origin-development power `fl`.
    pub origin_power: Real,
    /// Active origin-development coefficient count `ndor`.
    pub coefficient_count: usize,
    /// Last active radial row `max0`.
    pub active_len: usize,
}

/// FEFF `soldir` method-2 energy-corrected Dirac solution.
#[derive(Debug, Clone, PartialEq)]
pub struct AtomicDiracEnergyDisagreementCorrection {
    /// Corrected large component `gg`.
    pub large_component: Array1<Real>,
    /// Corrected small component `gp`.
    pub small_component: Array1<Real>,
    /// Corrected large origin-development coefficients `ag`.
    pub large_coefficients: Array1<Real>,
    /// Corrected small origin-development coefficients `ap`.
    pub small_coefficients: Array1<Real>,
    /// Cross integral `ah` used by FEFF's method-2 energy correction.
    pub overlap_integral: Real,
    /// Additive energy correction `f`.
    pub correction: Real,
    /// Normalization mismatch `c = 1 - b`.
    pub normalization_mismatch: Real,
}

/// Inputs for FEFF `ATOM/soldir.f90` matching-point relocation.
#[derive(Debug, Clone, Copy)]
pub struct AtomicDiracMatchingPointUpdateInput<'a> {
    /// Current large radial component `gg`.
    pub large_component: ArrayView1<'a, Real>,
    /// Last active radial row `max0`.
    pub active_len: usize,
    /// Current one-based matching-point index `mat`.
    pub matching_index_1based: usize,
    /// Whether FEFF has already relocated `mat`, FEFF `modmat != 0`.
    pub already_relocated: bool,
}

/// FEFF `soldir` matching-point relocation result.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AtomicDiracMatchingPointUpdate {
    /// Updated one-based matching-point index `mat`.
    pub matching_index_1based: usize,
    /// One-based peak index found by scanning `gg(i)^2`.
    pub peak_index_1based: usize,
    /// One-based scan limit that later node counting uses, FEFF `max(j, mat)`.
    pub scan_index_1based: usize,
    /// Final relocation flag, FEFF `modmat != 0`.
    pub relocated: bool,
    /// Whether FEFF would reintegrate with the updated matching point.
    pub needs_reintegration: bool,
}

/// Inputs for FEFF `ATOM/soldir.f90` inhomogeneous `intdir` seed setup.
#[derive(Debug, Clone, Copy)]
pub struct AtomicDiracInhomogeneousSeedInput<'a> {
    /// Large-component exchange/source rows, FEFF `eg`.
    pub large_source: ArrayView1<'a, Real>,
    /// Small-component exchange/source rows, FEFF `ep`.
    pub small_source: ArrayView1<'a, Real>,
    /// Large-source origin coefficients, FEFF `ceg`.
    pub large_source_coefficients: ArrayView1<'a, Real>,
    /// Small-source origin coefficients, FEFF `cep`.
    pub small_source_coefficients: ArrayView1<'a, Real>,
    /// Active origin-development coefficient count `ndor`.
    pub coefficient_count: usize,
}

/// Inputs for FEFF `ATOM/soldir.f90` homogeneous `intdir` seed setup.
#[derive(Debug, Clone, Copy)]
pub struct AtomicDiracHomogeneousSeedInput {
    /// Number of radial rows in `hg/hp`.
    pub radial_len: usize,
    /// Number of origin-development coefficient slots in `agh/aph`.
    pub coefficient_len: usize,
}

/// Seed arrays passed to FEFF `intdir` from `soldir`.
#[derive(Debug, Clone, PartialEq)]
pub struct AtomicDiracIntegrationSeed {
    /// Initial large component/source array.
    pub large_source: Array1<Real>,
    /// Initial small component/source array.
    pub small_source: Array1<Real>,
    /// Initial large origin-development/source coefficients.
    pub large_coefficients: Array1<Real>,
    /// Initial small origin-development/source coefficients.
    pub small_coefficients: Array1<Real>,
}

/// FEFF `ATOM/intdir.f90` integration branch.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AtomicDiracIntegrationMode {
    /// FEFF `imm = 0`: search the matching point, then integrate outward and inward.
    SearchMatchingPoint,
    /// FEFF `imm > 0`: use the supplied matching point and integrate outward and inward.
    FixedMatchingPoint,
    /// FEFF `imm < 0`: skip the outward pass and integrate inward only.
    InwardOnly,
}

/// Inputs for FEFF `ATOM/soldir.f90` shooting-pass setup at label `106`.
#[derive(Debug, Clone, Copy)]
pub struct AtomicDiracShootingPassSetupInput {
    /// Current trial energy `en`.
    pub energy: Real,
    /// Previous reference energy used by FEFF's `imm` heuristic, `enav`.
    pub previous_energy: Real,
}

/// FEFF `soldir` shooting-pass setup state.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AtomicDiracShootingPassSetup {
    /// Integration mode selected from FEFF `imm`.
    pub integration_mode: AtomicDiracIntegrationMode,
    /// Updated reference energy, FEFF `enav = en`.
    pub reference_energy: Real,
    /// Relative energy change used by FEFF's fixed-matching heuristic.
    pub relative_energy_change: Real,
    /// Matching-point relocation flag reset at label `106`, FEFF `modmat = 0`.
    pub relocated: bool,
}

/// Inputs for FEFF `ATOM/intdir.f90`.
///
/// Component and potential vectors use FEFF radial order. Only the first
/// [`AtomicDiracIntegrationInput::active_len`] rows participate in the
/// integration, but the returned arrays preserve the input capacity and update
/// the rows touched by FEFF's predictor-corrector sweep.
#[derive(Debug, Clone, Copy)]
pub struct AtomicDiracIntegrationInput<'a> {
    /// Initial large-component source `gg`; overwritten with the integrated solution.
    pub large_source: ArrayView1<'a, Real>,
    /// Initial small-component source `gp`; overwritten with the integrated solution.
    pub small_source: ArrayView1<'a, Real>,
    /// Initial large origin-development/source coefficients `ag`.
    pub large_coefficients: ArrayView1<'a, Real>,
    /// Initial small origin-development/source coefficients `ap`.
    pub small_coefficients: ArrayView1<'a, Real>,
    /// Radial grid `dr`.
    pub radii: ArrayView1<'a, Real>,
    /// Direct potential `dv` in FEFF `intdir` units.
    pub potential: ArrayView1<'a, Real>,
    /// Potential origin-development coefficients `av`.
    pub potential_coefficients: ArrayView1<'a, Real>,
    /// One-electron energy `en`.
    pub energy: Real,
    /// First origin-development power `fl`.
    pub origin_power: Real,
    /// Initial large origin coefficient `agi`.
    pub initial_large_coefficient: Real,
    /// Initial small origin coefficient `api`.
    pub initial_small_coefficient: Real,
    /// Initial large-component tail amplitude `ainf`.
    pub asymptotic_large_component: Real,
    /// Relativistic kappa `kap`; FEFF stores this as `fk`.
    pub kappa: i32,
    /// Speed of light `cl` in atomic units.
    pub speed_of_light: Real,
    /// Logarithmic radial-grid step `hx`.
    pub step: Real,
    /// Matching-point/tail precision `test1`.
    pub matching_precision: Real,
    /// Active origin-development coefficient count `ndor`.
    pub coefficient_count: usize,
    /// Active radial row count `np`.
    pub active_len: usize,
    /// Integration branch corresponding to FEFF `imm`.
    pub mode: AtomicDiracIntegrationMode,
    /// One-based matching point `mat` for fixed and inward-only modes.
    pub matching_index_1based: usize,
    /// One-based inward start `max0` for fixed and inward-only modes.
    pub max_index_1based: usize,
}

/// Result of FEFF `ATOM/intdir.f90`.
#[derive(Debug, Clone, PartialEq)]
pub struct AtomicDiracIntegration {
    /// Integrated large component `gg`.
    pub large_component: Array1<Real>,
    /// Integrated small component `gp`.
    pub small_component: Array1<Real>,
    /// Updated large origin-development coefficients `ag`.
    pub large_coefficients: Array1<Real>,
    /// Updated small origin-development coefficients `ap`.
    pub small_coefficients: Array1<Real>,
    /// Outward large-component value at the matching point, FEFF `ggmat`.
    pub matching_large_component: Option<Real>,
    /// Outward small-component value at the matching point, FEFF `gpmat`.
    pub matching_small_component: Option<Real>,
    /// Final one-based matching-point index `mat`.
    pub matching_index_1based: usize,
    /// Final one-based inward-start index `max0`.
    pub max_index_1based: usize,
}

/// Inputs for FEFF `ATOM/soldir.f90` setup before the first integration pass.
#[derive(Debug, Clone, Copy)]
pub struct AtomicDiracSolverSetupInput<'a> {
    /// Trial one-electron energy `en`.
    pub energy: Real,
    /// First origin-development power `fl`.
    pub origin_power: Real,
    /// Initial large origin coefficient `agi`.
    pub initial_large_coefficient: Real,
    /// Initial small origin coefficient `api`.
    pub initial_small_coefficient: Real,
    /// Principal quantum number `nq`.
    pub principal_quantum_number: usize,
    /// Relativistic kappa `kap`.
    pub kappa: i32,
    /// Speed of light `cl` in atomic units.
    pub speed_of_light: Real,
    /// Initial FEFF method selector; `<= 0` falls back to method `1`.
    pub method: i32,
    /// Radial grid `dr`.
    pub radii: ArrayView1<'a, Real>,
    /// Direct potential `dv`.
    pub potential: ArrayView1<'a, Real>,
    /// Potential origin coefficients `av`; only `av(1)` participates here.
    pub potential_coefficients: ArrayView1<'a, Real>,
    /// Number of active radial rows `np`.
    pub active_len: usize,
}

/// Deterministic `soldir` setup values shared by later integration passes.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AtomicDiracSolverSetup {
    /// FEFF `iex`: the originally requested method before fallback.
    pub requested_method: i32,
    /// Effective FEFF `method` after `method <= 0` fallback.
    pub method: i32,
    /// Trial energy after clamping below the apparent-potential floor.
    pub energy: Real,
    /// FEFF `emin`, the minimum apparent potential energy.
    pub energy_floor: Real,
    /// Initial small origin coefficient after point-nucleus correction.
    pub initial_small_coefficient: Real,
    /// Angular coefficient `ell = kappa * (kappa + 1) / (2 * cl)`.
    pub angular_term: Real,
    /// Target radial node count.
    pub target_nodes: i32,
    /// Twice the speed of light, FEFF `ccl`.
    pub doubled_speed_of_light: Real,
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
    /// Coulomb angular coefficients from [`atomic_coulomb_coefficients`].
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

/// FEFF `dsordf`-style request made by [`atomic_tabulation`].
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

/// Port of FEFF `COMMON/pertab.f90::atwtd`: return the periodic-table weight.
///
/// This is the table used by FEFF's Debye and Einstein-model mass calculations.
/// It covers `Z = 1..=139` and intentionally preserves FEFF's single-precision
/// rounding of the literal data statements.
pub fn atomic_weight(atomic_number: usize) -> Result<Real, AtomicError> {
    if atomic_number == 0 {
        return Err(AtomicError::InvalidAtomicNumber { z: atomic_number });
    }
    FEFF_ATOMIC_WEIGHTS
        .get(atomic_number - 1)
        .map(|&weight| Real::from(weight))
        .ok_or(AtomicError::InvalidAtomicNumber { z: atomic_number })
}

/// Port of FEFF `COMMON/pertab.f90::atsym`: return the element symbol.
///
/// FEFF's table is returned trimmed rather than padded to Fortran
/// `character*3` width. The values, including historical placeholder names
/// and table quirks, match FEFF's data statement.
pub fn atomic_symbol(atomic_number: usize) -> Result<&'static str, AtomicError> {
    if atomic_number == 0 {
        return Err(AtomicError::InvalidAtomicNumber { z: atomic_number });
    }
    FEFF_ATOMIC_SYMBOLS
        .get(atomic_number - 1)
        .copied()
        .ok_or(AtomicError::InvalidAtomicNumber { z: atomic_number })
}

/// Port of FEFF `nucmass`: return the tabulated standard atomic weight.
///
/// The value is returned in atomic mass units. FEFF uses this table when the
/// `HIGHZ` path requests a finite nuclear-radius model for heavy atoms.
pub fn nuclear_mass(atomic_number: usize) -> Result<Real, AtomicError> {
    if atomic_number == 0 {
        return Err(AtomicError::InvalidAtomicNumber { z: atomic_number });
    }
    FEFF_NUCLEAR_MASSES
        .get(atomic_number - 1)
        .map(|&mass| Real::from(mass))
        .ok_or(AtomicError::InvalidAtomicNumber { z: atomic_number })
}

/// Port of FEFF `ATOM/aprdev.f90`.
///
/// `term_count` is FEFF's 1-based `l` argument. The returned value is the
/// coefficient of power `l - 1` in the product of two coefficient rows.
pub fn atomic_polynomial_product_coefficient(
    left: &[Real],
    right: &[Real],
    term_count: usize,
) -> Result<Real, AtomMathError> {
    if term_count == 0 || term_count > left.len() || term_count > right.len() {
        return Err(AtomMathError::InvalidPolynomialTerm {
            term_count,
            left_len: left.len(),
            right_len: right.len(),
        });
    }
    validate_finite_slice("left coefficient", left)?;
    validate_finite_slice("right coefficient", right)?;

    Ok((0..term_count)
        .map(|index| left[index] * right[term_count - 1 - index])
        .sum())
}

/// Port of FEFF `ATOM/cofcon.f90` convergence acceleration.
///
/// FEFF adjusts the final-iteration weight by `0.1` when consecutive errors
/// have the same or opposite sign, then stores the current error as the next
/// previous error.
pub fn atomic_convergence_mix(
    final_weight: Real,
    current_error: Real,
    previous_error: Real,
) -> Result<AtomicConvergenceMix, AtomMathError> {
    validate_finite_scalar("final_weight", final_weight)?;
    validate_finite_scalar("current_error", current_error)?;
    validate_finite_scalar("previous_error", previous_error)?;

    let product = current_error * previous_error;
    validate_finite_scalar("error_product", product)?;

    let mut updated_final_weight = final_weight;
    if product < 0.0 {
        if updated_final_weight >= 0.2 {
            updated_final_weight -= 0.1;
        }
    } else if product > 0.0 && updated_final_weight <= 0.8 {
        updated_final_weight += 0.1;
    }

    Ok(AtomicConvergenceMix {
        initial_weight: 1.0 - updated_final_weight,
        final_weight: updated_final_weight,
        previous_error: current_error,
    })
}

/// Port of FEFF `ATOM/dentfa.f90`, the Thomas-Fermi density approximation.
///
/// `nuclear_charge + ionicity` is the effective electron count used by FEFF.
/// Values below `1e-4` return zero, matching the Fortran early exit.
pub fn thomas_fermi_density_potential(
    radius: Real,
    nuclear_charge: Real,
    ionicity: Real,
) -> Result<Real, AtomMathError> {
    validate_finite_scalar("radius", radius)?;
    validate_finite_scalar("nuclear_charge", nuclear_charge)?;
    validate_finite_scalar("ionicity", ionicity)?;
    if radius <= 0.0 {
        return Err(AtomMathError::NonPositiveRadius { radius });
    }

    let effective_charge = nuclear_charge + ionicity;
    if effective_charge < 1.0e-4 {
        return Ok(0.0);
    }

    let exponent = Real::from(1.0_f32 / 3.0_f32);
    let mut scaled = radius * effective_charge.powf(exponent);
    scaled = (scaled / Real::from(0.8853_f32)).sqrt();
    let numerator = scaled * (Real::from(0.60112_f32) * scaled + Real::from(1.81061_f32)) + 1.0;
    let denominator = scaled
        * (scaled
            * (scaled
                * (scaled * (Real::from(0.04793_f32) * scaled + Real::from(0.21465_f32))
                    + Real::from(0.77112_f32))
                + Real::from(1.39515_f32))
            + Real::from(1.81061_f32))
        + 1.0;
    let value = effective_charge * (1.0 - (numerator / denominator).powi(2)) / radius;
    validate_finite_scalar("dentfa", value)?;
    Ok(value)
}

/// Port of FEFF `ATOM/fdmocc.f90`, the occupation-number product.
///
/// `left` and `right` are zero-based Rust orbital indices. For equal orbitals,
/// FEFF applies the same degeneracy correction using the orbital kappa.
pub fn atomic_occupation_product(
    occupations: &[Real],
    kappas: &[i32],
    left: usize,
    right: usize,
) -> Result<Real, AtomMathError> {
    validate_occupation_tables(occupations, kappas)?;
    validate_orbital_index(left, occupations.len())?;
    validate_orbital_index(right, occupations.len())?;

    if left == right {
        let kappa_abs = kappas[left].unsigned_abs();
        if kappa_abs == 0 {
            return Err(AtomMathError::ZeroKappa);
        }
        let degeneracy = 2.0 * Real::from(kappa_abs);
        Ok(occupations[left] * (occupations[right] - 1.0) * degeneracy / (degeneracy - 1.0))
    } else {
        Ok(occupations[left] * occupations[right])
    }
}

/// Port of FEFF `ATOM/akeato.f90`, direct Coulomb angular coefficient lookup.
///
/// The orbital indices are zero-based. FEFF uses integer division `k / 2`;
/// odd ranks therefore map to the same channel as the preceding even rank.
pub fn atomic_direct_coulomb_coefficient(
    coefficients: ArrayView3<'_, Real>,
    left: usize,
    right: usize,
    rank: usize,
) -> Result<Real, AtomMathError> {
    validate_coefficient_table(coefficients, left, right, rank)?;
    direct_coulomb_coefficient_at(coefficients, left, right, rank)
}

/// Port of FEFF `ATOM/akeato.f90::bkeato`, exchange Coulomb coefficient lookup.
///
/// Equal orbitals return zero, matching FEFF's explicit same-index branch.
pub fn atomic_exchange_coulomb_coefficient(
    coefficients: ArrayView3<'_, Real>,
    left: usize,
    right: usize,
    rank: usize,
) -> Result<Real, AtomMathError> {
    validate_coefficient_table(coefficients, left, right, rank)?;
    exchange_coulomb_coefficient_at(coefficients, left, right, rank)
}

/// Port of FEFF `ATOM/muatco.f90`, Coulomb angular coefficient tabulation.
///
/// The returned `Array3` is indexed as `(orbital, orbital, rank / 2)` and keeps
/// FEFF's asymmetric storage convention: direct `F^k` coefficients occupy
/// `(min, max, rank / 2)`, while exchange `G^k` coefficients occupy
/// `(max, min, rank / 2)`.
pub fn atomic_coulomb_coefficients(
    input: AtomicCoulombCoefficientInput<'_>,
) -> Result<Array3<Real>, AtomMathError> {
    const CHANNELS: usize = 5;

    validate_coulomb_coefficient_input(&input)?;
    let orbital_count = input.kappas.len();
    let mut coefficients = Array3::<Real>::zeros((orbital_count, orbital_count, CHANNELS).f());

    for left in 0..orbital_count {
        let left_j2 = doubled_j_usize_from_kappa(input.kappas[left])?;
        for right in 0..=left {
            let right_j2 = doubled_j_usize_from_kappa(input.kappas[right])?;
            let max_rank = left_j2
                .checked_add(right_j2)
                .ok_or(AtomMathError::CoulombRankOutOfRange { rank: left_j2 })?
                / 2;
            let mut min_rank = left_j2.abs_diff(right_j2) / 2;
            if input.kappas[left].signum() != input.kappas[right].signum() {
                min_rank = min_rank
                    .checked_add(1)
                    .ok_or(AtomMathError::CoulombRankOutOfRange { rank: min_rank })?;
            }

            let same_closed_orbital = left == right && input.valence_occupations[left] <= 0.0;
            let same_orbital_correction = if same_closed_orbital { 1.0 } else { 0.0 };
            coefficients[(right, left, 0)] +=
                input.occupations[left] * (input.occupations[right] - same_orbital_correction);

            if input.valence_occupations[left] > 0.0 && input.valence_occupations[right] > 0.0 {
                continue;
            }

            let mut scale = coefficients[(right, left, 0)];
            if same_closed_orbital {
                let left_j2_real = Real::from(atom_usize_to_i32(left_j2)?);
                scale = -scale * (left_j2_real + 1.0) / left_j2_real;
                min_rank = min_rank
                    .checked_add(2)
                    .ok_or(AtomMathError::CoulombRankOutOfRange { rank: min_rank })?;
            }

            let mut rank = min_rank;
            while rank <= max_rank {
                let channel = rank / 2;
                if channel >= CHANNELS {
                    return Err(AtomMathError::CoefficientChannelOutOfRange {
                        rank,
                        channel,
                        channels: CHANNELS,
                    });
                }
                let doubled_rank = rank
                    .checked_mul(2)
                    .ok_or(AtomMathError::CoulombRankOutOfRange { rank })?;
                let wigner = wigner_3j(
                    atom_usize_to_i32(left_j2)?,
                    atom_usize_to_i32(doubled_rank)?,
                    atom_usize_to_i32(right_j2)?,
                    1,
                    0,
                    2,
                )
                .map_err(|source| AtomMathError::CoulombAngular { source })?;
                coefficients[(left, right, channel)] = scale * wigner * wigner;
                rank = rank
                    .checked_add(2)
                    .ok_or(AtomMathError::CoulombRankOutOfRange { rank })?;
            }
        }
    }

    Ok(coefficients)
}

/// Port of FEFF `ATOM/inmuat.f90` after the `getorb` occupation compaction.
///
/// The caller supplies compacted orbital quantum numbers and occupations, for
/// example from [`crate::orbital_configuration`]. This routine mirrors the
/// deterministic ATOM setup that follows `getorb`: electron-count checking,
/// convergence defaults, active lengths, open-shell flags, and the Lagrange
/// pair count.
pub fn atomic_orbital_initialization(
    input: AtomicOrbitalInitializationInput<'_>,
) -> Result<AtomicOrbitalInitialization, AtomMathError> {
    validate_orbital_initialization_input(&input)?;
    calculate_atomic_orbital_initialization(input)
}

/// Port of FEFF `ATOM/soldir.f90` `norm`.
///
/// This helper evaluates the radial and origin-development normalization term
/// used by `soldir` before scaling Dirac components. The `method == 1` branch
/// applies FEFF's matching-point correction to the small component.
pub fn atomic_dirac_normalization(
    input: AtomicDiracNormalizationInput<'_>,
) -> Result<AtomicDiracNormalization, AtomMathError> {
    validate_dirac_normalization_input(&input)?;
    calculate_atomic_dirac_normalization(input)
}

/// Port of FEFF `ATOM/soldir.f90` final normalization/sign-scaling block.
///
/// This helper takes the normalization integral produced by
/// [`atomic_dirac_normalization`], applies FEFF's sign conventions against the
/// initial origin coefficients, scales the active wavefunction prefix, and
/// clears inactive radial rows.
pub fn atomic_dirac_solution_normalization(
    input: AtomicDiracSolutionNormalizationInput<'_>,
) -> Result<AtomicDiracSolutionNormalization, AtomMathError> {
    validate_dirac_solution_normalization_input(&input)?;
    calculate_atomic_dirac_solution_normalization(input)
}

/// Port of FEFF `ATOM/soldir.f90` node counting after solution matching.
///
/// FEFF initializes `nd = 1`, scans through `max(j, mat)`, skips divisions
/// when the previous large-component sample is exactly zero, and counts both
/// sign changes and samples that land exactly on zero.
pub fn atomic_dirac_node_count(
    input: AtomicDiracNodeCountInput<'_>,
) -> Result<AtomicDiracNodeCount, AtomMathError> {
    validate_dirac_node_count_input(&input)?;
    calculate_atomic_dirac_node_count(input)
}

/// Port of FEFF `ATOM/soldir.f90` node-count energy search.
///
/// After counting nodes, FEFF adjusts the trial energy and search brackets before
/// reintegrating. Exhausting `nes` attempts is not a hard error in FEFF: it sets
/// `ifail = 1` and continues to normalization, so this helper reports that state
/// separately from the old `numerr` exits.
pub fn atomic_dirac_node_energy_search(
    input: AtomicDiracNodeEnergySearchInput,
) -> Result<AtomicDiracNodeEnergySearch, AtomMathError> {
    validate_dirac_node_energy_search_input(&input)?;
    calculate_atomic_dirac_node_energy_search(input)
}

/// Port of FEFF `ATOM/soldir.f90` method-1 energy correction.
///
/// FEFF uses the small-component disagreement at `mat` to form an additive
/// energy correction `f` and a relative mismatch `c`. When `gpmat` is exactly
/// zero, FEFF leaves `c` unscaled.
pub fn atomic_dirac_method_one_energy_correction(
    input: AtomicDiracMethodOneEnergyCorrectionInput<'_>,
) -> Result<AtomicDiracEnergyCorrection, AtomMathError> {
    validate_dirac_method_one_energy_correction_input(&input)?;
    calculate_atomic_dirac_method_one_energy_correction(input)
}

/// Port of FEFF `ATOM/soldir.f90` energy-correction backtracking.
///
/// This applies `en = en + f`, then repeatedly halves `f` while FEFF's
/// positivity, relative-step, or bracket checks reject the trial. The result
/// also reports whether `abs(c) > test`, which is the condition that makes
/// `soldir` run another small-component matching iteration.
pub fn atomic_dirac_energy_step(
    input: AtomicDiracEnergyStepInput,
) -> Result<AtomicDiracEnergyStep, AtomMathError> {
    validate_dirac_energy_step_input(&input)?;
    calculate_atomic_dirac_energy_step(input)
}

/// Port of FEFF `ATOM/soldir.f90` small-component rematch attempt handling.
///
/// After an accepted energy step, FEFF increments `ies` only when
/// `abs(c) > test`. Attempts up to `nes` jump back to label `105`; attempts
/// beyond `nes` set `ifail = 1` and continue normalization with the current
/// solution.
pub fn atomic_dirac_rematch_attempt(
    input: AtomicDiracRematchAttemptInput,
) -> Result<AtomicDiracRematchAttempt, AtomMathError> {
    validate_dirac_rematch_attempt_input(&input)?;
    calculate_atomic_dirac_rematch_attempt(input)
}

/// Port of FEFF `ATOM/soldir.f90` method-0 homogeneous tail matching.
///
/// FEFF scales only the tail rows `mat..=max0` so the outward solution matches
/// the inward large component at `mat`, then sets `j = mat` for node counting.
pub fn atomic_dirac_homogeneous_match(
    input: AtomicDiracHomogeneousMatchInput<'_>,
) -> Result<AtomicDiracHomogeneousMatch, AtomMathError> {
    validate_dirac_homogeneous_match_input(&input)?;
    calculate_atomic_dirac_homogeneous_match(input)
}

/// Port of FEFF `ATOM/soldir.f90` method-1 large-component matching.
///
/// FEFF computes the large-component mismatch at `mat`, derives a homogeneous
/// tail scale from `hg(mat)`, and adds the homogeneous solution from `mat`
/// through `max0` to match the inward large component.
pub fn atomic_dirac_large_component_match(
    input: AtomicDiracLargeComponentMatchInput<'_>,
) -> Result<AtomicDiracLargeComponentMatch, AtomMathError> {
    validate_dirac_large_component_match_input(&input)?;
    calculate_atomic_dirac_large_component_match(input)
}

/// Port of FEFF `ATOM/soldir.f90` method-2 two-component matching.
///
/// FEFF solves a two-by-two matching system using the homogeneous inward and
/// outward values. The prefix scale updates rows before `mat` plus origin
/// coefficients, while the tail scale updates rows from `mat` through `max0`.
pub fn atomic_dirac_two_component_match(
    input: AtomicDiracTwoComponentMatchInput<'_>,
) -> Result<AtomicDiracTwoComponentMatch, AtomMathError> {
    validate_dirac_two_component_match_input(&input)?;
    calculate_atomic_dirac_two_component_match(input)
}

/// Port of FEFF `ATOM/soldir.f90` method-2 energy-disagreement matching.
///
/// This names the second `method >= 2` two-component match in `soldir`, where
/// the matched system is the derivative solution `bg/bp/bgh/bph` produced by
/// the energy-disagreement integration.
pub fn atomic_dirac_energy_disagreement_match(
    input: AtomicDiracEnergyDisagreementMatchInput<'_>,
) -> Result<AtomicDiracEnergyDisagreementMatch, AtomMathError> {
    let match_input = dirac_energy_disagreement_match_as_two_component(&input);
    validate_dirac_two_component_match_input(&match_input)?;
    calculate_atomic_dirac_energy_disagreement_match(input)
}

/// Port of FEFF `ATOM/soldir.f90` method-2 energy-disagreement source setup.
///
/// This builds `bg/bp/bgh/bph` immediately before FEFF calls `intdir` for the
/// energy-disagreement system. The first coefficient slots are left zero here
/// because `intdir` overwrites them with `agi/api` before using the expansion.
pub fn atomic_dirac_energy_disagreement_source(
    input: AtomicDiracEnergyDisagreementSourceInput<'_>,
) -> Result<AtomicDiracEnergyDisagreementSource, AtomMathError> {
    validate_dirac_energy_disagreement_source_input(&input)?;
    calculate_atomic_dirac_energy_disagreement_source(input)
}

/// Port of FEFF `ATOM/soldir.f90` method-2 energy correction.
///
/// After FEFF integrates and matches the energy-disagreement system, it forms
/// the cross integral between the current solution and the derivative solution,
/// converts the normalization mismatch into an energy correction, and applies
/// that correction to the radial components and origin-development terms.
pub fn atomic_dirac_energy_disagreement_correction(
    input: AtomicDiracEnergyDisagreementCorrectionInput<'_>,
) -> Result<AtomicDiracEnergyDisagreementCorrection, AtomMathError> {
    validate_dirac_energy_disagreement_correction_input(&input)?;
    calculate_atomic_dirac_energy_disagreement_correction(input)
}

/// Port of FEFF `ATOM/soldir.f90` matching-point relocation.
///
/// FEFF searches the maximum `gg(i)^2`, relocates `mat` at most once, keeps
/// matching points odd, and falls back to `max0 - 12` when the peak is too
/// close to the tail for another integration pass.
pub fn atomic_dirac_matching_point_update(
    input: AtomicDiracMatchingPointUpdateInput<'_>,
) -> Result<AtomicDiracMatchingPointUpdate, AtomMathError> {
    validate_dirac_matching_point_update_input(&input)?;
    calculate_atomic_dirac_matching_point_update(input)
}

/// Port of FEFF `ATOM/soldir.f90` inhomogeneous `intdir` seed setup.
///
/// This copies `eg/ep` into the component work arrays and shifts `ceg/cep`
/// into coefficient slots `2..=ndor`. The first coefficient slot is left zero
/// here because `intdir` overwrites it with `agi/api` before use.
pub fn atomic_dirac_inhomogeneous_seed(
    input: AtomicDiracInhomogeneousSeedInput<'_>,
) -> Result<AtomicDiracIntegrationSeed, AtomMathError> {
    validate_dirac_inhomogeneous_seed_input(&input)?;
    calculate_atomic_dirac_inhomogeneous_seed(input)
}

/// Port of FEFF `ATOM/soldir.f90` homogeneous `intdir` seed setup.
///
/// FEFF zeros `hg/hp/agh/aph` before integrating the homogeneous system. This
/// helper returns those zeroed arrays in the same structure used by
/// [`atomic_dirac_inhomogeneous_seed`].
pub fn atomic_dirac_homogeneous_seed(
    input: AtomicDiracHomogeneousSeedInput,
) -> Result<AtomicDiracIntegrationSeed, AtomMathError> {
    validate_dirac_homogeneous_seed_input(&input)?;
    Ok(calculate_atomic_dirac_homogeneous_seed(input))
}

/// Port of FEFF `ATOM/soldir.f90` shooting-pass setup at label `106`.
///
/// FEFF resets `modmat`, chooses whether `intdir` should search a new matching
/// point or reuse the current one from the relative energy change, then updates
/// `enav` to the current trial energy.
pub fn atomic_dirac_shooting_pass_setup(
    input: AtomicDiracShootingPassSetupInput,
) -> Result<AtomicDiracShootingPassSetup, AtomMathError> {
    validate_dirac_shooting_pass_setup_input(&input)?;
    calculate_atomic_dirac_shooting_pass_setup(input)
}

/// Port of FEFF `ATOM/intdir.f90`, the real Dirac radial predictor-corrector.
///
/// This is the low-level integration step used by `soldir`: it can search for
/// a matching point, integrate through a supplied matching point, or generate
/// only the inward tail solution. FEFF reports failures through a global
/// `numerr`; this port returns structured errors instead.
pub fn atomic_dirac_integration(
    input: AtomicDiracIntegrationInput<'_>,
) -> Result<AtomicDiracIntegration, AtomMathError> {
    validate_dirac_integration_input(&input)?;
    calculate_atomic_dirac_integration(input)
}

/// Port of FEFF `ATOM/soldir.f90` setup before the first `intdir` call.
///
/// This evaluates the deterministic scalar state used by the shooting loop:
/// method fallback, point-nucleus small-component coefficient adjustment,
/// angular term, target node count, and the lower apparent-potential energy
/// bound.
pub fn atomic_dirac_solver_setup(
    input: AtomicDiracSolverSetupInput<'_>,
) -> Result<AtomicDiracSolverSetup, AtomMathError> {
    validate_dirac_solver_setup_input(&input)?;
    calculate_atomic_dirac_solver_setup(input)
}

/// Port of FEFF `ATOM/vlda.f90`, local-density exchange potential.
///
/// The routine builds FEFF's total and valence spherical densities from orbital
/// components, evaluates the selected `idfock` exchange branch, and returns the
/// updated potential/development/energy arrays.
pub fn atomic_local_density_potential(
    input: AtomicLocalDensityPotentialInput<'_>,
) -> Result<AtomicLocalDensityPotential, AtomMathError> {
    validate_local_density_potential_input(&input)?;
    calculate_atomic_local_density_potential(input)
}

/// Port of FEFF `ATOM/potrdf.f90`, orbital potential/source assembly.
pub fn atomic_orbital_potential(
    input: AtomicOrbitalPotentialInput<'_>,
) -> Result<AtomicOrbitalPotential, AtomMathError> {
    validate_orbital_potential_input(&input)?;
    calculate_atomic_orbital_potential(input)
}

/// Port of FEFF `ATOM/lagdat.f90`, non-diagonal Lagrange parameters.
///
/// The returned vector uses FEFF's packed triangular pair order. For zero-based
/// orbitals `i < j`, the packed index is `i + j * (j - 1) / 2`.
pub fn atomic_lagrange_parameters<F>(
    input: AtomicLagrangeParametersInput<'_>,
    radial_integral: F,
) -> Result<Array1<Real>, AtomMathError>
where
    F: FnMut(AtomicRadialIntegralRequest) -> Result<Real, AtomMathError>,
{
    validate_lagrange_parameters_input(&input)?;
    AtomicLagrangeContext {
        input,
        radial_integral,
    }
    .calculate()
}

/// Port of FEFF `ATOM/tabrat.f90`, excluding text emission.
///
/// The returned data mirrors the orbital moment rows and same-kappa overlap
/// rows that FEFF writes to the ATOM log. Radial integrals are supplied as a
/// callback so callers can plug in either the Rust `dsordf` port or a test
/// oracle while keeping `tabrat`'s bookkeeping explicit.
pub fn atomic_tabulation<F>(
    input: AtomicTabulationInput<'_>,
    radial_integral: F,
) -> Result<AtomicTabulation, AtomMathError>
where
    F: FnMut(AtomicTabulationIntegralRequest) -> Result<Real, AtomMathError>,
{
    validate_tabulation_input(&input)?;
    AtomicTabulationContext {
        input,
        radial_integral,
    }
    .calculate()
}

/// Port of FEFF `ATOM/fpf0.f90`, excluding direct file IO.
///
/// The returned structure mirrors the contents of `fpf0.dat`: scalar f-prime
/// corrections, dipole oscillator rows, and the fixed 81-point `f0(Q)` table on
/// FEFF's `0.5 Angstrom^-1` grid.
pub fn atomic_form_factor(
    input: AtomicFormFactorInput<'_>,
) -> Result<AtomicFormFactor, AtomMathError> {
    validate_form_factor_input(&input)?;
    AtomicFormFactorContext { input }.calculate()
}

/// Port of FEFF `ATOM/nucdev.f90`.
///
/// The point-nucleus branch returns the Coulomb potential `-dz/r`. Negative
/// `requested_nucleus_index` values select FEFF's finite uniform-nucleus branch
/// using the tabulated nuclear mass, matching the ATOM high-Z path.
pub fn atomic_nuclear_potential(
    input: AtomicNuclearPotentialInput,
) -> Result<AtomicNuclearPotential, AtomMathError> {
    validate_nuclear_potential_input(input)?;
    calculate_atomic_nuclear_potential(input)
}

/// Port of FEFF `ATOM/dsordf.f90`.
///
/// This evaluates FEFF's Simpson-rule radial integral and adds the analytic
/// origin-development correction from `aprdev`. The supported modes cover the
/// `jnd = -2, -1, 1, 2, 3, 4` branches used by FEFF10 `ATOM`.
pub fn atomic_differential_integral(
    input: AtomicDifferentialIntegralInput<'_>,
) -> Result<Real, AtomMathError> {
    validate_differential_integral_input(&input)?;
    calculate_atomic_differential_integral(input)
}

/// Port of FEFF `ATOM/yzkteg.f90`.
///
/// This builds the radial `yk` and `zk` Coulomb kernels using FEFF's four-point
/// point-to-point integration stencil and origin-development correction.
pub fn atomic_yk_zk_transform(
    input: AtomicYkZkTransformInput<'_>,
) -> Result<AtomicYkZkTransform, AtomMathError> {
    validate_yk_zk_transform_input(&input)?;
    calculate_atomic_yk_zk_transform(input)
}

/// Port of FEFF `ATOM/yzkrdf.f90` for positive orbital indices.
///
/// This constructs the source and origin coefficients from ATOM orbital
/// component tables, then delegates to [`atomic_yk_zk_transform`].
pub fn atomic_yk_zk_exchange(
    input: AtomicYkZkExchangeInput<'_>,
) -> Result<AtomicYkZkTransform, AtomMathError> {
    validate_yk_zk_exchange_input(&input)?;
    calculate_atomic_yk_zk_exchange(input)
}

/// Port of FEFF `ATOM/yzkrdf.f90` for caller-prepared sources.
///
/// This mirrors the `i <= 0` branch, where the caller supplies the source and
/// coefficients directly and FEFF uses `k + 2` as the first origin power before
/// delegating to `yzkteg`.
pub fn atomic_yk_zk_prepared_source(
    input: AtomicYkZkPreparedSourceInput<'_>,
) -> Result<AtomicYkZkTransform, AtomMathError> {
    let initial_power = input.angular_momentum.checked_add(2).ok_or(
        AtomMathError::YkZkAngularMomentumOutOfRange {
            angular_momentum: input.angular_momentum,
        },
    )? as Real;

    atomic_yk_zk_transform(AtomicYkZkTransformInput {
        source: input.source,
        source_coefficients: input.source_coefficients,
        radii: input.radii,
        initial_power,
        step: input.step,
        angular_momentum: input.angular_momentum,
        coefficient_count: input.coefficient_count,
        source_len: input.source_len,
        active_len: input.active_len,
    })
}

/// Port of FEFF `ATOM/fdrirk.f90`.
///
/// This composes the `yzkrdf` first-factor construction with the `dsordf`
/// radial integration branch. When the first pair in the request is zero, the
/// caller must pass the previous first factor to mirror FEFF's common-block
/// sentinel path.
pub fn atomic_radial_integral(
    input: AtomicRadialIntegralInput<'_>,
) -> Result<AtomicRadialIntegral, AtomMathError> {
    validate_radial_integral_input(&input)?;
    calculate_atomic_radial_integral(input)
}

/// Port of FEFF `ATOM/ortdat.f90`, Schmidt orthogonalization.
///
/// The supplied callback receives FEFF `dsordf`-style projection and norm
/// requests because the original routine depends on ATOM common-block radial
/// integration state. Returned matrices keep the caller's `(row, orbital)`
/// layout and update only FEFF's active rows for each orthogonalized orbital.
pub fn atomic_schmidt_orthogonalization<F>(
    input: AtomicSchmidtOrthogonalizationInput<'_>,
    overlap_integral: F,
) -> Result<AtomicSchmidtOrthogonalization, AtomMathError>
where
    F: for<'request> FnMut(AtomicSchmidtIntegralRequest<'request>) -> Result<Real, AtomMathError>,
{
    validate_schmidt_orthogonalization_input(&input)?;
    AtomicSchmidtContext {
        input,
        overlap_integral,
    }
    .calculate()
}

/// Port of FEFF `ATOM/bkmrdf.f90`, the Breit angular coefficients.
///
/// `left_kappa` and `right_kappa` are the relativistic kappa values for the
/// two orbitals. `rank` is FEFF's integer `k` for the Breit radial integral.
pub fn atomic_breit_angular_coefficients(
    left_kappa: i32,
    right_kappa: i32,
    rank: usize,
) -> Result<AtomicBreitAngularCoefficients, AtomMathError> {
    let left_j2 = doubled_j_from_kappa(left_kappa)?;
    let right_j2 = doubled_j_from_kappa(right_kappa)?;
    let rank_i32 = i32::try_from(rank).map_err(|_| AtomMathError::BreitRankOutOfRange { rank })?;
    let kappa_difference =
        right_kappa
            .checked_sub(left_kappa)
            .ok_or(AtomMathError::KappaDifferenceOutOfRange {
                left_kappa,
                right_kappa,
            })?;
    let kappa_sum =
        right_kappa
            .checked_add(left_kappa)
            .ok_or(AtomMathError::KappaDifferenceOutOfRange {
                left_kappa,
                right_kappa,
            })?;

    let mut coefficients = AtomicBreitAngularCoefficients {
        magnetic: [0.0; 3],
        retarded: [0.0; 3],
    };
    let mut angular_l = rank_i32
        .checked_sub(1)
        .ok_or(AtomMathError::BreitRankOutOfRange { rank })?;
    for order in 0..3 {
        if angular_l >= 0 {
            accumulate_breit_order(
                BreitOrderContext {
                    left_j2,
                    right_j2,
                    kappa_difference,
                    kappa_sum,
                    rank: rank_i32,
                    rank_usize: rank,
                    angular_l,
                    order,
                },
                &mut coefficients,
            )?;
        }
        angular_l = angular_l
            .checked_add(1)
            .ok_or(AtomMathError::BreitRankOutOfRange { rank })?;
    }

    Ok(coefficients)
}

/// Port of FEFF `ATOM/etotal.f90`, excluding the radial integral solver.
///
/// The supplied callback receives FEFF-style `fdrirk(i,j,l,m,k)` requests and
/// must return the corresponding radial integral. This function performs the
/// FEFF accumulation of direct Coulomb, exchange Coulomb, magnetic Breit,
/// retarded Breit, and one-electron energy terms.
pub fn atomic_total_energy<F>(
    input: AtomicTotalEnergyInput<'_>,
    radial_integral: F,
) -> Result<AtomicTotalEnergy, AtomMathError>
where
    F: FnMut(AtomicRadialIntegralRequest) -> Result<Real, AtomMathError>,
{
    validate_total_energy_input(&input)?;
    AtomicTotalEnergyContext {
        input,
        radial_integral,
    }
    .calculate()
}

/// Port of FEFF `ATOM/s02at.f90`, the relaxed-overlap amplitude reduction.
///
/// The overlap matrix is consumed in FEFF group order by kappa, using only the
/// upper triangular entries that FEFF copies into its symmetric work matrix.
pub fn atomic_overlap_amplitude_reduction(
    input: AtomicOverlapAmplitudeReductionInput<'_>,
) -> Result<Real, AtomMathError> {
    const KAPPA_MIN: i32 = -5;
    const KAPPA_MAX: i32 = 4;
    const GROUP_LIMIT: usize = 8;

    validate_overlap_amplitude_input(&input)?;
    let hole_zero_based = input.hole_orbital_1based.map(|index| index - 1);
    let mut amplitude = 1.0;

    for kappa in KAPPA_MIN..=KAPPA_MAX {
        if kappa == 0 {
            continue;
        }
        let group = input
            .kappas
            .iter()
            .enumerate()
            .filter_map(|(orbital, &orbital_kappa)| (orbital_kappa == kappa).then_some(orbital))
            .collect::<Vec<_>>();
        if group.is_empty() {
            continue;
        }
        if group.len() > GROUP_LIMIT {
            return Err(AtomMathError::KappaGroupTooLarge {
                kappa,
                count: group.len(),
                limit: GROUP_LIMIT,
            });
        }

        let mut matrix = s02at_overlap_matrix(&group, input.overlap_integrals);
        let determinant_all = s02at_squared_determinant_in_place(&mut matrix, group.len());
        let determinant_without_last =
            s02at_squared_determinant_in_place(&mut matrix, group.len() - 1);
        let last_orbital = *group.last().ok_or(AtomMathError::KappaGroupTooLarge {
            kappa,
            count: 0,
            limit: GROUP_LIMIT,
        })?;
        let occupation = input.occupations[last_orbital];
        let max_occupation = 2.0 * Real::from(kappa.unsigned_abs());
        let hole_vacancy = max_occupation - occupation;
        let hole_position = hole_zero_based.and_then(|hole| {
            group
                .iter()
                .position(|&orbital| orbital == hole)
                .map(|position| position + 1)
        });

        amplitude *= match hole_position {
            None => determinant_all.powf(occupation) * determinant_without_last.powf(hole_vacancy),
            Some(position) if position == group.len() => {
                determinant_all.powf(occupation - 1.0)
                    * determinant_without_last.powf(hole_vacancy + 1.0)
            }
            Some(position) => {
                let mut eliminated = s02at_eliminate_hole(matrix.view(), position - 1);
                let determinant_eliminated_all =
                    s02at_squared_determinant_in_place(&mut eliminated, group.len());
                let determinant_eliminated_without_last =
                    s02at_squared_determinant_in_place(&mut eliminated, group.len() - 1);
                let mixed = (determinant_eliminated_without_last * determinant_all * hole_vacancy
                    + determinant_eliminated_all * determinant_without_last * occupation)
                    / max_occupation;
                mixed
                    * determinant_all.powf(occupation - 1.0)
                    * determinant_without_last.powf(hole_vacancy - 1.0)
            }
        };
        validate_finite_scalar("s02", amplitude)?;
    }

    Ok(amplitude)
}

struct AtomicTotalEnergyContext<'a, F> {
    input: AtomicTotalEnergyInput<'a>,
    radial_integral: F,
}

struct AtomicLagrangeContext<'a, F> {
    input: AtomicLagrangeParametersInput<'a>,
    radial_integral: F,
}

struct AtomicTabulationContext<'a, F> {
    input: AtomicTabulationInput<'a>,
    radial_integral: F,
}

struct AtomicFormFactorContext<'a> {
    input: AtomicFormFactorInput<'a>,
}

struct AtomicSchmidtContext<'a, F> {
    input: AtomicSchmidtOrthogonalizationInput<'a>,
    overlap_integral: F,
}

struct AtomicSchmidtTables<'a> {
    large_components: &'a mut Array2<Real>,
    small_components: &'a mut Array2<Real>,
    large_coefficients: &'a mut Array2<Real>,
    small_coefficients: &'a mut Array2<Real>,
    active_lengths: &'a mut [usize],
}

struct AtomicSchmidtProjectionInput<'a> {
    target: usize,
    reference: usize,
    active_len: usize,
    work_large: &'a Array1<Real>,
    work_small: &'a Array1<Real>,
    work_large_coefficients: &'a Array1<Real>,
    work_small_coefficients: &'a Array1<Real>,
    large_components: ArrayView2<'a, Real>,
    small_components: ArrayView2<'a, Real>,
    large_coefficients: ArrayView2<'a, Real>,
    small_coefficients: ArrayView2<'a, Real>,
}

impl<F> AtomicLagrangeContext<'_, F>
where
    F: FnMut(AtomicRadialIntegralRequest) -> Result<Real, AtomMathError>,
{
    fn calculate(&mut self) -> Result<Array1<Real>, AtomMathError> {
        let orbital_count = self.input.kappas.len();
        let pair_count = orbital_pair_count(orbital_count)?;
        let mut parameters = Array1::<Real>::zeros(pair_count);

        if let Some(active_orbital_1based) = self.input.active_orbital_1based {
            let active = active_orbital_1based - 1;
            for other in 0..orbital_count {
                self.accumulate_pair(active, other, &mut parameters)?;
            }
        } else {
            for first in 0..orbital_count.saturating_sub(1) {
                for second in (first + 1)..orbital_count {
                    self.accumulate_pair(first, second, &mut parameters)?;
                }
            }
        }

        Ok(parameters)
    }

    fn accumulate_pair(
        &mut self,
        first: usize,
        second: usize,
        parameters: &mut Array1<Real>,
    ) -> Result<(), AtomMathError> {
        if first == second || self.input.kappas[first] != self.input.kappas[second] {
            return Ok(());
        }
        if self.input.shell_markers[first] < 0 && self.input.shell_markers[second] < 0 {
            return Ok(());
        }
        if self.input.occupations[first] == self.input.occupations[second] {
            return Ok(());
        }
        self.validate_pair_occupation(first)?;
        self.validate_pair_occupation(second)?;

        let mut value = self.direct_terms(first, second)?;
        if self.input.include_exchange {
            value += self.exchange_terms(first, second)?;
        }
        let packed = packed_orbital_pair_index(first, second)?;
        let parameter = value / (self.input.occupations[second] - self.input.occupations[first]);
        validate_finite_scalar("lagrange_parameter", parameter)?;
        let Some(slot) = parameters.get_mut(packed) else {
            return Err(AtomMathError::OrbitalPairTableTooLarge {
                orbital_count: self.input.kappas.len(),
            });
        };
        *slot = parameter;
        Ok(())
    }

    fn direct_terms(&mut self, first: usize, second: usize) -> Result<Real, AtomMathError> {
        let first_j2 = self.j2(first)?;
        let mut value = 0.0;
        for orbital in 0..self.input.kappas.len() {
            let orbital_j2 = self.j2(orbital)?;
            let max_rank = first_j2.min(orbital_j2);
            let mut rank = 0;
            while rank <= max_rank {
                let first_coefficient =
                    self.direct_coefficient(orbital, first, rank)? / self.input.occupations[first];
                let difference = first_coefficient
                    - self.direct_coefficient(orbital, second, rank)?
                        / self.input.occupations[second];
                if significant_relative_difference(difference, first_coefficient) {
                    value += difference
                        * self.radial(orbital + 1, orbital + 1, first + 1, second + 1, rank)?;
                }
                rank = rank
                    .checked_add(2)
                    .ok_or(AtomMathError::CoulombRankOutOfRange { rank })?;
            }
        }
        validate_finite_scalar("lagrange_direct_terms", value)?;
        Ok(value)
    }

    fn exchange_terms(&mut self, first: usize, second: usize) -> Result<Real, AtomMathError> {
        let first_j2 = self.j2(first)?;
        let mut value = 0.0;
        for orbital in 0..self.input.kappas.len() {
            let orbital_j2 = self.j2(orbital)?;
            let max_rank = first_j2
                .checked_add(orbital_j2)
                .ok_or(AtomMathError::CoulombRankOutOfRange { rank: first_j2 })?
                / 2;
            let mut rank = orbital_j2.abs_diff(max_rank);
            if self.input.kappas[first].signum() != self.input.kappas[orbital].signum() {
                rank = rank
                    .checked_add(1)
                    .ok_or(AtomMathError::CoulombRankOutOfRange { rank })?;
            }
            while rank <= max_rank {
                let first_coefficient = self.exchange_coefficient(orbital, second, rank)?
                    / self.input.occupations[second];
                let difference = first_coefficient
                    - self.exchange_coefficient(orbital, first, rank)?
                        / self.input.occupations[first];
                if significant_relative_difference(difference, first_coefficient) {
                    value += difference
                        * self.radial(first + 1, orbital + 1, second + 1, orbital + 1, rank)?;
                }
                rank = rank
                    .checked_add(2)
                    .ok_or(AtomMathError::CoulombRankOutOfRange { rank })?;
            }
        }
        validate_finite_scalar("lagrange_exchange_terms", value)?;
        Ok(value)
    }

    fn direct_coefficient(
        &self,
        left: usize,
        right: usize,
        rank: usize,
    ) -> Result<Real, AtomMathError> {
        direct_coulomb_coefficient_at(self.input.coulomb_coefficients, left, right, rank)
    }

    fn exchange_coefficient(
        &self,
        left: usize,
        right: usize,
        rank: usize,
    ) -> Result<Real, AtomMathError> {
        exchange_coulomb_coefficient_at(self.input.coulomb_coefficients, left, right, rank)
    }

    fn radial(
        &mut self,
        first_left: usize,
        first_right: usize,
        second_left: usize,
        second_right: usize,
        rank: usize,
    ) -> Result<Real, AtomMathError> {
        let value = (self.radial_integral)(AtomicRadialIntegralRequest {
            first_left,
            first_right,
            second_left,
            second_right,
            rank,
        })?;
        validate_finite_scalar("radial_integral", value)?;
        Ok(value)
    }

    fn j2(&self, orbital: usize) -> Result<usize, AtomMathError> {
        doubled_j_usize_from_kappa(self.input.kappas[orbital])
    }

    fn validate_pair_occupation(&self, orbital: usize) -> Result<(), AtomMathError> {
        let occupation = self.input.occupations[orbital];
        if occupation > 0.0 {
            Ok(())
        } else {
            Err(AtomMathError::NonPositiveOccupation {
                context: "lagdat",
                orbital_1based: orbital + 1,
                occupation,
            })
        }
    }
}

impl<F> AtomicTabulationContext<'_, F>
where
    F: FnMut(AtomicTabulationIntegralRequest) -> Result<Real, AtomMathError>,
{
    fn calculate(&mut self) -> Result<AtomicTabulation, AtomMathError> {
        let orbital_count = self.input.kappas.len();
        let mut orbitals = Vec::with_capacity(orbital_count);
        for orbital in 0..orbital_count {
            let orbital_label = atom_tabrat_orbital_label(self.input.kappas[orbital])?;
            let moments = self.orbital_moments(orbital)?;
            let binding_energy_ev = -self.input.orbital_energies[orbital] * ATOM_TABRAT_HARTREE_EV;
            validate_finite_scalar("tabrat_binding_energy_ev", binding_energy_ev)?;
            orbitals.push(AtomicTabulatedOrbital {
                principal_quantum_number: self.input.principal_quantum_numbers[orbital],
                orbital_label,
                occupation: self.input.occupations[orbital],
                binding_energy_ev,
                moments,
            });
        }

        let mut overlaps = Vec::new();
        for left in 0..orbital_count.saturating_sub(1) {
            for right in (left + 1)..orbital_count {
                if self.input.kappas[left] != self.input.kappas[right] {
                    continue;
                }
                let value = self.radial(left, right, 0)?;
                overlaps.push(AtomicTabulatedOverlap {
                    left,
                    right,
                    left_principal_quantum_number: self.input.principal_quantum_numbers[left],
                    left_orbital_label: atom_tabrat_orbital_label(self.input.kappas[left])?,
                    right_principal_quantum_number: self.input.principal_quantum_numbers[right],
                    right_orbital_label: atom_tabrat_orbital_label(self.input.kappas[right])?,
                    value,
                });
            }
        }

        Ok(AtomicTabulation { orbitals, overlaps })
    }

    fn orbital_moments(
        &mut self,
        orbital: usize,
    ) -> Result<Vec<AtomicTabulatedMoment>, AtomMathError> {
        let moment_count = if abs_kappa_i32(self.input.kappas[orbital])? - 1 <= 0 {
            ATOM_TABRAT_MOMENT_POWERS.len() - 1
        } else {
            ATOM_TABRAT_MOMENT_POWERS.len()
        };
        let mut moments = Vec::with_capacity(moment_count);
        for &power in ATOM_TABRAT_MOMENT_POWERS.iter().take(moment_count) {
            moments.push(AtomicTabulatedMoment {
                power,
                value: self.radial(orbital, orbital, power)?,
            });
        }
        Ok(moments)
    }

    fn radial(&mut self, left: usize, right: usize, power: i32) -> Result<Real, AtomMathError> {
        let value = (self.radial_integral)(AtomicTabulationIntegralRequest { left, right, power })?;
        validate_finite_scalar("tabrat_integral", value)?;
        Ok(value)
    }
}

impl AtomicFormFactorContext<'_> {
    fn calculate(&self) -> Result<AtomicFormFactor, AtomMathError> {
        let total_energy_fprime =
            self.input.total_energy * ATOM_FPF0_FINE_STRUCTURE.powi(2) * 5.0 / 3.0;
        let relativistic_correction = -((self.input.atomic_number as Real) / 82.5).powf(2.37);
        validate_finite_scalar("fpf0_total_energy_fprime", total_energy_fprime)?;
        validate_finite_scalar("fpf0_relativistic_correction", relativistic_correction)?;

        let radii = self.input.radii.iter().copied().collect::<Vec<_>>();
        let zeros = vec![0.0; radii.len()];
        let oscillators = self.oscillators(&radii, &zeros)?;
        let (form_factor_momentum, form_factor) = self.form_factor_table(&radii, &zeros)?;

        Ok(AtomicFormFactor {
            atomic_number: self.input.atomic_number,
            total_energy_fprime,
            relativistic_correction,
            oscillators,
            form_factor_momentum,
            form_factor,
        })
    }

    fn oscillators(
        &self,
        radii: &[Real],
        zeros: &[Real],
    ) -> Result<Vec<AtomicFormFactorOscillator>, AtomMathError> {
        let hole = self.input.hole_orbital_1based - 1;
        let initial_kappa = self.input.kappas[hole];
        let mut oscillators = vec![AtomicFormFactorOscillator {
            oscillator_strength: 2.0 * Real::from(abs_kappa_i32(initial_kappa)?),
            excitation_energy: self.input.orbital_energies[hole],
            orbital_index_1based: self.input.hole_orbital_1based,
        }];

        for orbital in 0..self.input.kappas.len() {
            if self.input.occupations[orbital] <= 0.0 {
                continue;
            }
            let Some((large_multiplier, small_multiplier)) =
                fpf0_dipole_multipliers(initial_kappa, self.input.kappas[orbital])?
            else {
                continue;
            };
            let wave_number =
                (self.input.orbital_energies[orbital] - self.input.orbital_energies[hole]).abs()
                    * ATOM_FPF0_FINE_STRUCTURE;
            let integrand = radii
                .iter()
                .enumerate()
                .map(|(radial, &radius)| {
                    let bessel = fpf0_spherical_bessel_j0(wave_number * radius);
                    (large_multiplier
                        * self.input.initial_large_component[radial]
                        * self.input.small_components[(radial, orbital)]
                        + small_multiplier
                            * self.input.initial_small_component[radial]
                            * self.input.large_components[(radial, orbital)])
                        * bessel
                })
                .collect::<Vec<_>>();
            validate_finite_slice("fpf0_oscillator_integrand", &integrand)?;
            let radial_integral = somm(radii, &integrand, zeros, self.input.radial_step, 2.0, 0)?;
            let oscillator_strength = radial_integral * radial_integral / 3.0;
            validate_finite_scalar("fpf0_oscillator_strength", oscillator_strength)?;
            oscillators.push(AtomicFormFactorOscillator {
                oscillator_strength,
                excitation_energy: self.input.orbital_energies[orbital],
                orbital_index_1based: orbital + 1,
            });
        }

        Ok(oscillators)
    }

    fn form_factor_table(
        &self,
        radii: &[Real],
        zeros: &[Real],
    ) -> Result<(Array1<Real>, Array1<Real>), AtomMathError> {
        let momentum = Array1::from_shape_fn(ATOM_FPF0_FORM_FACTOR_POINTS, |index| {
            ATOM_FPF0_MOMENTUM_STEP_INV_ANGSTROM * index as Real
        });
        let mut form_factor = Array1::<Real>::zeros(ATOM_FPF0_FORM_FACTOR_POINTS);

        for (index, value) in form_factor.iter_mut().enumerate() {
            let wave_number =
                ATOM_FPF0_MOMENTUM_STEP_INV_ANGSTROM * ATOM_FPF0_BOHR_ANGSTROM * index as Real;
            let integrand = radii
                .iter()
                .enumerate()
                .map(|(radial, &radius)| {
                    self.input.density_4pi[radial]
                        * radius
                        * radius
                        * fpf0_spherical_bessel_j0(wave_number * radius)
                })
                .collect::<Vec<_>>();
            validate_finite_slice("fpf0_form_factor_integrand", &integrand)?;
            *value = somm(radii, &integrand, zeros, self.input.radial_step, 2.0, 0)?;
            validate_finite_scalar("fpf0_form_factor", *value)?;
        }

        Ok((momentum, form_factor))
    }
}

fn calculate_atomic_nuclear_potential(
    input: AtomicNuclearPotentialInput,
) -> Result<AtomicNuclearPotential, AtomMathError> {
    let (nucleus_index, first_radius_times_charge) = atomic_nuclear_mesh_parameters(input)?;
    let first_radius = first_radius_times_charge / input.nuclear_charge;
    let radii = Array1::from_shape_fn(input.radial_count, |row| {
        first_radius * (input.step * row as Real).exp()
    });
    for radius in radii.iter().copied() {
        if radius <= 0.0 {
            return Err(AtomMathError::NonPositiveRadius { radius });
        }
        validate_finite_scalar("nucdev_radius", radius)?;
    }

    let mut development_coefficients = Array1::<Real>::zeros(input.coefficient_count);
    let mut potential = radii.mapv(|radius| -input.nuclear_charge / radius);
    if nucleus_index <= 1 {
        development_coefficients[0] = -input.nuclear_charge;
    } else {
        if nucleus_index > input.radial_count {
            return Err(AtomMathError::NuclearRadiusOutOfRange {
                nucleus_index,
                radial_count: input.radial_count,
            });
        }
        let nuclear_radius = radii[nucleus_index - 1];
        let quadratic = -3.0 * input.nuclear_charge / (nuclear_radius + nuclear_radius);
        let quartic = -quadratic / (3.0 * nuclear_radius * nuclear_radius);
        development_coefficients[1] = quadratic;
        development_coefficients[3] = quartic;
        for row in 0..(nucleus_index - 1) {
            potential[row] = quadratic + quartic * radii[row] * radii[row];
        }
    }
    for value in development_coefficients.iter().copied() {
        validate_finite_scalar("nucdev_coefficient", value)?;
    }
    for value in potential.iter().copied() {
        validate_finite_scalar("nucdev_potential", value)?;
    }

    Ok(AtomicNuclearPotential {
        development_coefficients,
        radii,
        potential,
        nucleus_index,
        first_radius_times_charge,
    })
}

fn calculate_atomic_orbital_initialization(
    input: AtomicOrbitalInitializationInput<'_>,
) -> Result<AtomicOrbitalInitialization, AtomMathError> {
    let orbital_count = input.occupations.len();
    let active_lengths =
        Array1::<usize>::from_elem(orbital_count, ATOM_INMUAT_DEFAULT_RADIAL_COUNT);
    let mut shell_markers = Array1::<i32>::from_elem(orbital_count, -1);
    let mut convergence_acceleration =
        Array1::<Real>::from_elem(orbital_count, ATOM_INMUAT_DEFAULT_CONVERGENCE);
    let mut lagrange_pair_count = 0usize;

    for orbital in 0..orbital_count {
        let kappa_abs = kappa_abs_usize(input.kappas[orbital])?;
        let angular_momentum = if input.kappas[orbital] < 0 {
            kappa_abs
                .checked_sub(1)
                .ok_or(AtomMathError::InvalidKappa {
                    kappa: input.kappas[orbital],
                })?
        } else {
            kappa_abs
        };
        let principal_quantum_number = input.principal_quantum_numbers[orbital];
        if angular_momentum >= principal_quantum_number || angular_momentum > 4 {
            return Err(AtomMathError::OrbitalAngularMomentumOutOfRange {
                orbital_1based: orbital + 1,
                principal_quantum_number,
                kappa: input.kappas[orbital],
                angular_momentum,
            });
        }

        let closed_shell_capacity = 2.0 * kappa_abs as Real;
        if input.occupations[orbital] < closed_shell_capacity {
            shell_markers[orbital] = 1;
        }
        if input.occupations[orbital] < 0.5 {
            convergence_acceleration[orbital] = 1.0;
        }
        for previous in 0..orbital {
            if input.kappas[previous] == input.kappas[orbital]
                && (shell_markers[previous] > 0 || shell_markers[orbital] > 0)
            {
                lagrange_pair_count = lagrange_pair_count
                    .checked_add(1)
                    .ok_or(AtomMathError::OrbitalPairTableTooLarge { orbital_count })?;
            }
        }
    }

    for value in convergence_acceleration.iter().copied() {
        validate_finite_scalar("inmuat_convergence_acceleration", value)?;
    }

    Ok(AtomicOrbitalInitialization {
        orbital_count,
        self_consistent_count: orbital_count,
        wavefunction_precision: ATOM_INMUAT_WAVEFUNCTION_PRECISION,
        energy_precision: ATOM_INMUAT_ENERGY_PRECISION,
        precision_ratios: [ATOM_INMUAT_PRIMARY_RATIO, ATOM_INMUAT_SECONDARY_RATIO],
        primary_matching_precision: ATOM_INMUAT_WAVEFUNCTION_PRECISION / ATOM_INMUAT_PRIMARY_RATIO,
        secondary_matching_precision: ATOM_INMUAT_WAVEFUNCTION_PRECISION
            / ATOM_INMUAT_SECONDARY_RATIO,
        development_order: ATOM_INMUAT_DEVELOPMENT_ORDER,
        attempt_count: ATOM_INMUAT_ATTEMPT_COUNT,
        nucleus_index: ATOM_INMUAT_NUCLEUS_INDEX,
        radial_count: ATOM_INMUAT_DEFAULT_RADIAL_COUNT,
        orbital_energies: Array1::<Real>::zeros(orbital_count),
        convergence_acceleration,
        wavefunction_errors: Array1::<Real>::zeros(orbital_count),
        energy_errors: Array1::<Real>::zeros(orbital_count),
        active_lengths,
        shell_markers,
        lagrange_pair_count,
        lagrange_parameters: Array1::<Real>::zeros(ATOM_INMUAT_LAGRANGE_CAPACITY),
    })
}

fn calculate_atomic_dirac_normalization(
    input: AtomicDiracNormalizationInput<'_>,
) -> Result<AtomicDiracNormalization, AtomMathError> {
    let mut radial_terms = input
        .radii
        .iter()
        .zip(input.large_component.iter())
        .zip(input.small_component.iter())
        .take(input.active_len)
        .map(|((&radius, &large), &small)| radius * (large * large + small * small))
        .collect::<Vec<_>>();

    if input.method == 1 {
        let matching = input.matching_index_1based - 1;
        let small = input.small_component[matching];
        radial_terms[matching] += input.radii[matching]
            * (input.matching_small_component * input.matching_small_component - small * small)
            / 2.0;
    }

    let mut norm = 0.0;
    for row in (1..input.active_len).step_by(2) {
        norm += radial_terms[row] + radial_terms[row] + radial_terms[row + 1];
    }
    norm = input.step * (norm + norm + radial_terms[0] - radial_terms[input.active_len - 1]) / 3.0;

    let first_radius = input.radii[0];
    for coefficient in 1..=input.coefficient_count {
        let exponent = input.origin_power + input.origin_power + coefficient as Real;
        if exponent == 0.0 {
            return Err(AtomMathError::ZeroDiracNormalizationOriginExponent);
        }
        let factor = first_radius.powf(exponent) / exponent;
        for left in 0..coefficient {
            let right = coefficient - 1 - left;
            norm += input.large_coefficients[left] * factor * input.large_coefficients[right]
                + input.small_coefficients[left] * factor * input.small_coefficients[right];
        }
    }
    validate_finite_scalar("soldir_norm", norm)?;

    Ok(AtomicDiracNormalization { norm })
}

fn calculate_atomic_dirac_solution_normalization(
    input: AtomicDiracSolutionNormalizationInput<'_>,
) -> Result<AtomicDiracSolutionNormalization, AtomMathError> {
    let mut large_component = input.large_component.to_owned();
    let mut small_component = input.small_component.to_owned();
    let mut large_coefficients = input.large_coefficients.to_owned();
    let mut small_coefficients = input.small_coefficients.to_owned();

    let norm_root = input.norm.sqrt();
    let mut coefficient_divisor = norm_root;
    if large_coefficients[0] * input.initial_large_coefficient < 0.0
        || small_coefficients[0] * input.initial_small_coefficient < 0.0
    {
        coefficient_divisor = -coefficient_divisor;
    }

    for (large, small) in large_coefficients
        .iter_mut()
        .zip(small_coefficients.iter_mut())
        .take(input.coefficient_count)
    {
        *large /= coefficient_divisor;
        *small /= coefficient_divisor;
    }

    let mut component_divisor = norm_root;
    if large_component[0] * input.initial_large_coefficient < 0.0
        || small_component[0] * input.initial_small_coefficient < 0.0
    {
        component_divisor = -component_divisor;
    }

    for (large, small) in large_component
        .iter_mut()
        .zip(small_component.iter_mut())
        .take(input.active_len)
    {
        *large /= component_divisor;
        *small /= component_divisor;
    }
    for (large, small) in large_component
        .iter_mut()
        .zip(small_component.iter_mut())
        .skip(input.active_len)
    {
        *large = 0.0;
        *small = 0.0;
    }

    Ok(AtomicDiracSolutionNormalization {
        large_component,
        small_component,
        large_coefficients,
        small_coefficients,
        component_divisor,
        coefficient_divisor,
    })
}

fn calculate_atomic_dirac_node_count(
    input: AtomicDiracNodeCountInput<'_>,
) -> Result<AtomicDiracNodeCount, AtomMathError> {
    let scan_index_1based = input.matching_index_1based.max(input.scan_index_1based);
    let mut node_count = 1;

    for row in 1..scan_index_1based {
        let previous = input.large_component[row - 1];
        if previous == 0.0 {
            continue;
        }
        if input.large_component[row] / previous <= 0.0 {
            node_count += 1;
        }
    }

    Ok(AtomicDiracNodeCount {
        node_count,
        scan_index_1based,
    })
}

fn calculate_atomic_dirac_node_energy_search(
    input: AtomicDiracNodeEnergySearchInput,
) -> Result<AtomicDiracNodeEnergySearch, AtomMathError> {
    let mut energy = input.energy;
    let mut energy_sup = input.energy_sup;
    let mut energy_inf = input.energy_inf;

    if input.node_count < input.target_node_count {
        energy_sup = energy;
        if energy_inf < 0.0 {
            energy = soldir_node_search_bisect(energy_inf, energy_sup, input.energy_precision)?;
        } else {
            energy *= 8.0e-1;
            if energy.abs() <= input.energy_precision {
                return Err(AtomMathError::DiracNodeEnergyTooSmall {
                    energy,
                    precision: input.energy_precision,
                });
            }
        }
    } else if input.node_count > input.target_node_count {
        energy_inf = energy;
        if energy_sup > input.energy_floor {
            energy = soldir_node_search_bisect(energy_inf, energy_sup, input.energy_precision)?;
        } else {
            energy *= 1.2;
            if energy <= input.energy_floor {
                return Err(AtomMathError::DiracNodeEnergyBelowPotentialFloor {
                    energy,
                    energy_floor: input.energy_floor,
                });
            }
        }
    } else {
        return Ok(AtomicDiracNodeEnergySearch {
            energy,
            energy_sup,
            energy_inf,
            search_attempt_count: input.search_attempt_count,
            needs_reintegration: false,
            attempts_exhausted: false,
        });
    }

    let search_attempt_count = input.search_attempt_count.checked_add(1).ok_or(
        AtomMathError::DiracNodeEnergyAttemptCountOutOfRange {
            search_attempt_count: input.search_attempt_count,
        },
    )?;
    let attempts_exhausted = search_attempt_count > input.max_attempt_count;

    validate_finite_scalar("soldir_node_energy", energy)?;
    validate_finite_scalar("soldir_node_energy_sup", energy_sup)?;
    validate_finite_scalar("soldir_node_energy_inf", energy_inf)?;

    Ok(AtomicDiracNodeEnergySearch {
        energy,
        energy_sup,
        energy_inf,
        search_attempt_count,
        needs_reintegration: !attempts_exhausted,
        attempts_exhausted,
    })
}

fn soldir_node_search_bisect(
    energy_inf: Real,
    energy_sup: Real,
    precision: Real,
) -> Result<Real, AtomMathError> {
    if (energy_inf - energy_sup).abs() > precision {
        Ok((energy_inf + energy_sup) / 2.0)
    } else {
        Err(AtomMathError::DiracNodeEnergyBracketCollapsed {
            energy_inf,
            energy_sup,
            precision,
        })
    }
}

fn calculate_atomic_dirac_method_one_energy_correction(
    input: AtomicDiracMethodOneEnergyCorrectionInput<'_>,
) -> Result<AtomicDiracEnergyCorrection, AtomMathError> {
    let matching = input.matching_index_1based - 1;
    let mut mismatch = input.matching_small_component - input.small_component[matching];
    let correction = input.large_component[matching] * mismatch * input.speed_of_light / input.norm;
    if input.matching_small_component != 0.0 {
        mismatch /= input.matching_small_component;
    }

    validate_finite_scalar("soldir_energy_correction", correction)?;
    validate_finite_scalar("soldir_energy_mismatch", mismatch)?;

    Ok(AtomicDiracEnergyCorrection {
        correction,
        mismatch,
    })
}

fn calculate_atomic_dirac_energy_step(
    input: AtomicDiracEnergyStepInput,
) -> Result<AtomicDiracEnergyStep, AtomMathError> {
    let mut correction = input.correction;
    let mut energy = input.energy + correction;
    let denominator = energy - correction;
    if denominator == 0.0 {
        return Err(AtomMathError::ZeroDiracEnergyCorrectionDenominator);
    }
    let mut relative_step = (correction / denominator).abs();
    let needs_rematch = input.mismatch.abs() > input.mismatch_precision;

    loop {
        let rejected = energy >= 0.0
            || relative_step > 2.0e-1
            || (needs_rematch && (energy < input.energy_sup || energy > input.energy_inf));
        if !rejected {
            break;
        }
        correction /= 2.0;
        relative_step /= 2.0;
        energy -= correction;
        if relative_step <= input.zero_energy_precision {
            return Err(AtomMathError::DiracEnergyCorrectionTooSmall { relative_step });
        }
    }

    validate_finite_scalar("soldir_energy_step_energy", energy)?;
    validate_finite_scalar("soldir_energy_step_correction", correction)?;
    validate_finite_scalar("soldir_energy_step_relative", relative_step)?;

    Ok(AtomicDiracEnergyStep {
        energy,
        correction,
        relative_step,
        needs_rematch,
    })
}

fn calculate_atomic_dirac_rematch_attempt(
    input: AtomicDiracRematchAttemptInput,
) -> Result<AtomicDiracRematchAttempt, AtomMathError> {
    if input.mismatch.abs() <= input.mismatch_precision {
        return Ok(AtomicDiracRematchAttempt {
            match_attempt_count: input.match_attempt_count,
            needs_rematch: false,
            attempts_exhausted: false,
        });
    }

    let match_attempt_count = input.match_attempt_count.checked_add(1).ok_or(
        AtomMathError::DiracRematchAttemptCountOutOfRange {
            match_attempt_count: input.match_attempt_count,
        },
    )?;
    let attempts_exhausted = match_attempt_count > input.max_attempt_count;

    Ok(AtomicDiracRematchAttempt {
        match_attempt_count,
        needs_rematch: !attempts_exhausted,
        attempts_exhausted,
    })
}

fn calculate_atomic_dirac_homogeneous_match(
    input: AtomicDiracHomogeneousMatchInput<'_>,
) -> Result<AtomicDiracHomogeneousMatch, AtomMathError> {
    let matching = input.matching_index_1based - 1;
    let denominator = input.large_component[matching];
    if denominator == 0.0 {
        return Err(AtomMathError::ZeroDiracMatchDenominator {
            field: "soldir_homogeneous_match_large_component",
        });
    }

    let mut large_component = input.large_component.to_owned();
    let mut small_component = input.small_component.to_owned();
    let tail_scale = input.matching_large_component / denominator;

    for row in matching..input.active_len {
        large_component[row] *= tail_scale;
        small_component[row] *= tail_scale;
    }

    validate_finite_scalar("soldir_homogeneous_match_scale", tail_scale)?;
    validate_finite_vector(
        "soldir_homogeneous_match_large_component",
        large_component.view(),
    )?;
    validate_finite_vector(
        "soldir_homogeneous_match_small_component",
        small_component.view(),
    )?;

    Ok(AtomicDiracHomogeneousMatch {
        large_component,
        small_component,
        tail_scale,
        scan_index_1based: input.matching_index_1based,
    })
}

fn calculate_atomic_dirac_large_component_match(
    input: AtomicDiracLargeComponentMatchInput<'_>,
) -> Result<AtomicDiracLargeComponentMatch, AtomMathError> {
    let matching = input.matching_index_1based - 1;
    let denominator = input.homogeneous_large_component[matching];
    if denominator == 0.0 {
        return Err(AtomMathError::ZeroDiracMatchDenominator {
            field: "soldir_match_homogeneous_large_component",
        });
    }

    let mut large_component = input.large_component.to_owned();
    let mut small_component = input.small_component.to_owned();
    let large_mismatch = large_component[matching] - input.matching_large_component;
    let tail_scale = -large_mismatch / denominator;

    for row in matching..input.active_len {
        large_component[row] += tail_scale * input.homogeneous_large_component[row];
        small_component[row] += tail_scale * input.homogeneous_small_component[row];
    }

    validate_finite_scalar("soldir_large_match_scale", tail_scale)?;
    validate_finite_vector("soldir_large_match_large_component", large_component.view())?;
    validate_finite_vector("soldir_large_match_small_component", small_component.view())?;

    Ok(AtomicDiracLargeComponentMatch {
        large_component,
        small_component,
        tail_scale,
        large_mismatch,
    })
}

fn calculate_atomic_dirac_two_component_match(
    input: AtomicDiracTwoComponentMatchInput<'_>,
) -> Result<AtomicDiracTwoComponentMatch, AtomMathError> {
    let matching = input.matching_index_1based - 1;
    let mut large_component = input.large_component.to_owned();
    let mut small_component = input.small_component.to_owned();
    let mut large_coefficients = input.large_coefficients.to_owned();
    let mut small_coefficients = input.small_coefficients.to_owned();

    let large_mismatch = large_component[matching] - input.matching_large_component;
    let small_mismatch = small_component[matching] - input.matching_small_component;
    let determinant = input.homogeneous_matching_small_component
        * input.homogeneous_large_component[matching]
        - input.homogeneous_matching_large_component * input.homogeneous_small_component[matching];
    if determinant == 0.0 {
        return Err(AtomMathError::ZeroDiracMatchDenominator {
            field: "soldir_match_determinant",
        });
    }

    let prefix_scale = (small_mismatch * input.homogeneous_large_component[matching]
        - large_mismatch * input.homogeneous_small_component[matching])
        / determinant;
    let tail_scale = (small_mismatch * input.homogeneous_matching_large_component
        - large_mismatch * input.homogeneous_matching_small_component)
        / determinant;

    for coefficient in 0..input.coefficient_count {
        large_coefficients[coefficient] +=
            prefix_scale * input.homogeneous_large_coefficients[coefficient];
        small_coefficients[coefficient] +=
            prefix_scale * input.homogeneous_small_coefficients[coefficient];
    }
    for row in 0..matching {
        large_component[row] += prefix_scale * input.homogeneous_large_component[row];
        small_component[row] += prefix_scale * input.homogeneous_small_component[row];
    }
    for row in matching..input.active_len {
        large_component[row] += tail_scale * input.homogeneous_large_component[row];
        small_component[row] += tail_scale * input.homogeneous_small_component[row];
    }

    validate_finite_scalar("soldir_two_match_determinant", determinant)?;
    validate_finite_scalar("soldir_two_match_prefix_scale", prefix_scale)?;
    validate_finite_scalar("soldir_two_match_tail_scale", tail_scale)?;
    validate_finite_vector("soldir_two_match_large_component", large_component.view())?;
    validate_finite_vector("soldir_two_match_small_component", small_component.view())?;
    validate_finite_vector(
        "soldir_two_match_large_coefficient",
        large_coefficients.view(),
    )?;
    validate_finite_vector(
        "soldir_two_match_small_coefficient",
        small_coefficients.view(),
    )?;

    Ok(AtomicDiracTwoComponentMatch {
        large_component,
        small_component,
        large_coefficients,
        small_coefficients,
        determinant,
        prefix_scale,
        tail_scale,
        large_mismatch,
        small_mismatch,
    })
}

fn calculate_atomic_dirac_energy_disagreement_match(
    input: AtomicDiracEnergyDisagreementMatchInput<'_>,
) -> Result<AtomicDiracEnergyDisagreementMatch, AtomMathError> {
    let matched = calculate_atomic_dirac_two_component_match(
        dirac_energy_disagreement_match_as_two_component(&input),
    )?;

    Ok(AtomicDiracEnergyDisagreementMatch {
        large_derivative: matched.large_component,
        small_derivative: matched.small_component,
        large_derivative_coefficients: matched.large_coefficients,
        small_derivative_coefficients: matched.small_coefficients,
        determinant: matched.determinant,
        prefix_scale: matched.prefix_scale,
        tail_scale: matched.tail_scale,
        large_mismatch: matched.large_mismatch,
        small_mismatch: matched.small_mismatch,
    })
}

fn dirac_energy_disagreement_match_as_two_component<'a>(
    input: &AtomicDiracEnergyDisagreementMatchInput<'a>,
) -> AtomicDiracTwoComponentMatchInput<'a> {
    AtomicDiracTwoComponentMatchInput {
        large_component: input.large_derivative,
        small_component: input.small_derivative,
        large_coefficients: input.large_derivative_coefficients,
        small_coefficients: input.small_derivative_coefficients,
        homogeneous_large_component: input.homogeneous_large_component,
        homogeneous_small_component: input.homogeneous_small_component,
        homogeneous_large_coefficients: input.homogeneous_large_coefficients,
        homogeneous_small_coefficients: input.homogeneous_small_coefficients,
        matching_large_component: input.matching_large_derivative,
        matching_small_component: input.matching_small_derivative,
        homogeneous_matching_large_component: input.homogeneous_matching_large_component,
        homogeneous_matching_small_component: input.homogeneous_matching_small_component,
        coefficient_count: input.coefficient_count,
        active_len: input.active_len,
        matching_index_1based: input.matching_index_1based,
    }
}

fn calculate_atomic_dirac_energy_disagreement_source(
    input: AtomicDiracEnergyDisagreementSourceInput<'_>,
) -> Result<AtomicDiracEnergyDisagreementSource, AtomMathError> {
    let mut large_source = Array1::<Real>::zeros(input.large_component.len());
    let mut small_source = Array1::<Real>::zeros(input.small_component.len());
    let mut large_coefficients = Array1::<Real>::zeros(input.large_coefficients.len());
    let mut small_coefficients = Array1::<Real>::zeros(input.small_coefficients.len());

    for coefficient in 1..input.coefficient_count {
        large_coefficients[coefficient] =
            input.large_coefficients[coefficient - 1] / input.speed_of_light;
        small_coefficients[coefficient] =
            input.small_coefficients[coefficient - 1] / input.speed_of_light;
    }
    for row in 0..input.active_len {
        let scale = input.radii[row] / input.speed_of_light;
        large_source[row] = input.large_component[row] * scale;
        small_source[row] = input.small_component[row] * scale;
    }

    validate_finite_vector(
        "soldir_energy_disagreement_large_source",
        large_source.view(),
    )?;
    validate_finite_vector(
        "soldir_energy_disagreement_small_source",
        small_source.view(),
    )?;
    validate_finite_vector(
        "soldir_energy_disagreement_large_coefficient",
        large_coefficients.view(),
    )?;
    validate_finite_vector(
        "soldir_energy_disagreement_small_coefficient",
        small_coefficients.view(),
    )?;

    Ok(AtomicDiracEnergyDisagreementSource {
        large_source,
        small_source,
        large_coefficients,
        small_coefficients,
    })
}

fn calculate_atomic_dirac_energy_disagreement_correction(
    input: AtomicDiracEnergyDisagreementCorrectionInput<'_>,
) -> Result<AtomicDiracEnergyDisagreementCorrection, AtomMathError> {
    let overlap_integral = atomic_dirac_energy_disagreement_overlap(&input)?;
    let denominator = overlap_integral + overlap_integral;
    if denominator == 0.0 {
        return Err(AtomMathError::ZeroDiracEnergyDisagreementCorrectionIntegral);
    }

    let normalization_mismatch = 1.0 - input.norm;
    let correction = normalization_mismatch / denominator;

    let mut large_component = input.large_component.to_owned();
    let mut small_component = input.small_component.to_owned();
    let mut large_coefficients = input.large_coefficients.to_owned();
    let mut small_coefficients = input.small_coefficients.to_owned();

    for row in 0..input.active_len {
        large_component[row] += correction * input.large_derivative[row];
        small_component[row] += correction * input.small_derivative[row];
    }
    for coefficient in 0..input.coefficient_count {
        large_coefficients[coefficient] +=
            correction * input.large_derivative_coefficients[coefficient];
        small_coefficients[coefficient] +=
            correction * input.small_derivative_coefficients[coefficient];
    }

    validate_finite_scalar(
        "soldir_energy_disagreement_correction_integral",
        overlap_integral,
    )?;
    validate_finite_scalar("soldir_energy_disagreement_correction", correction)?;
    validate_finite_scalar(
        "soldir_energy_disagreement_correction_mismatch",
        normalization_mismatch,
    )?;
    validate_finite_vector(
        "soldir_energy_disagreement_correction_large_component",
        large_component.view(),
    )?;
    validate_finite_vector(
        "soldir_energy_disagreement_correction_small_component",
        small_component.view(),
    )?;
    validate_finite_vector(
        "soldir_energy_disagreement_correction_large_coefficient",
        large_coefficients.view(),
    )?;
    validate_finite_vector(
        "soldir_energy_disagreement_correction_small_coefficient",
        small_coefficients.view(),
    )?;

    Ok(AtomicDiracEnergyDisagreementCorrection {
        large_component,
        small_component,
        large_coefficients,
        small_coefficients,
        overlap_integral,
        correction,
        normalization_mismatch,
    })
}

fn atomic_dirac_energy_disagreement_overlap(
    input: &AtomicDiracEnergyDisagreementCorrectionInput<'_>,
) -> Result<Real, AtomMathError> {
    let origin_denominator = input.origin_power + input.origin_power + 1.0;
    if origin_denominator == 0.0 {
        return Err(AtomMathError::ZeroDiracEnergyDisagreementCorrectionOriginExponent);
    }

    let overlap_at = |row: usize| {
        (input.large_component[row] * input.large_derivative[row]
            + input.small_component[row] * input.small_derivative[row])
            * input.radii[row]
    };
    let first = overlap_at(0);
    let last = overlap_at(input.active_len - 1);
    let middle = (1..input.active_len - 1)
        .map(|row| {
            let weight = if row % 2 == 1 { 4.0 } else { 2.0 };
            weight * overlap_at(row)
        })
        .sum::<Real>();
    let overlap = input.step * (first + middle + last) / 3.0 + first / origin_denominator;
    validate_finite_scalar("soldir_energy_disagreement_overlap", overlap)?;
    Ok(overlap)
}

fn calculate_atomic_dirac_matching_point_update(
    input: AtomicDiracMatchingPointUpdateInput<'_>,
) -> Result<AtomicDiracMatchingPointUpdate, AtomMathError> {
    let mut peak_index_1based = None;
    let mut peak_square = 0.0;
    for (row, &large) in input
        .large_component
        .iter()
        .take(input.active_len)
        .enumerate()
    {
        let square = large * large;
        validate_finite_scalar("soldir_matching_point_square", square)?;
        if square > peak_square {
            peak_square = square;
            peak_index_1based = Some(row + 1);
        }
    }
    let peak_index_1based = peak_index_1based.ok_or(AtomMathError::DiracMatchingPointNotFound {
        active_len: input.active_len,
    })?;

    let mut matching_index_1based = input.matching_index_1based;
    let mut scan_index_1based = peak_index_1based.max(matching_index_1based);
    let mut relocated = input.already_relocated;
    let mut needs_reintegration = false;

    if peak_index_1based > matching_index_1based && !input.already_relocated {
        relocated = true;
        matching_index_1based = odd_matching_index(peak_index_1based);
        if matching_index_1based < input.active_len - ATOM_SOLDIR_MATCHING_POINT_TAIL_MARGIN {
            needs_reintegration = true;
            scan_index_1based = peak_index_1based.max(matching_index_1based);
        } else {
            let fallback_index_1based =
                input.active_len - ATOM_SOLDIR_MATCHING_POINT_FALLBACK_OFFSET;
            matching_index_1based = odd_matching_index(fallback_index_1based);
            scan_index_1based = fallback_index_1based.max(matching_index_1based);
        }
    }

    Ok(AtomicDiracMatchingPointUpdate {
        matching_index_1based,
        peak_index_1based,
        scan_index_1based,
        relocated,
        needs_reintegration,
    })
}

fn odd_matching_index(index_1based: usize) -> usize {
    if index_1based.is_multiple_of(2) {
        index_1based + 1
    } else {
        index_1based
    }
}

fn calculate_atomic_dirac_inhomogeneous_seed(
    input: AtomicDiracInhomogeneousSeedInput<'_>,
) -> Result<AtomicDiracIntegrationSeed, AtomMathError> {
    let large_source = input.large_source.to_owned();
    let small_source = input.small_source.to_owned();
    let mut large_coefficients = Array1::<Real>::zeros(input.large_source_coefficients.len());
    let mut small_coefficients = Array1::<Real>::zeros(input.small_source_coefficients.len());

    for coefficient in 1..input.coefficient_count {
        large_coefficients[coefficient] = input.large_source_coefficients[coefficient - 1];
        small_coefficients[coefficient] = input.small_source_coefficients[coefficient - 1];
    }

    validate_finite_vector("soldir_seed_large_source", large_source.view())?;
    validate_finite_vector("soldir_seed_small_source", small_source.view())?;
    validate_finite_vector("soldir_seed_large_coefficient", large_coefficients.view())?;
    validate_finite_vector("soldir_seed_small_coefficient", small_coefficients.view())?;

    Ok(AtomicDiracIntegrationSeed {
        large_source,
        small_source,
        large_coefficients,
        small_coefficients,
    })
}

fn calculate_atomic_dirac_homogeneous_seed(
    input: AtomicDiracHomogeneousSeedInput,
) -> AtomicDiracIntegrationSeed {
    AtomicDiracIntegrationSeed {
        large_source: Array1::<Real>::zeros(input.radial_len),
        small_source: Array1::<Real>::zeros(input.radial_len),
        large_coefficients: Array1::<Real>::zeros(input.coefficient_len),
        small_coefficients: Array1::<Real>::zeros(input.coefficient_len),
    }
}

fn calculate_atomic_dirac_shooting_pass_setup(
    input: AtomicDiracShootingPassSetupInput,
) -> Result<AtomicDiracShootingPassSetup, AtomMathError> {
    if input.energy == 0.0 {
        return Err(AtomMathError::ZeroDiracShootingPassEnergy);
    }
    let relative_energy_change = ((input.previous_energy - input.energy) / input.energy).abs();
    validate_finite_scalar(
        "soldir_shooting_pass_relative_energy_change",
        relative_energy_change,
    )?;

    let integration_mode = if relative_energy_change < ATOM_SOLDIR_FIXED_MATCHING_RELATIVE_ENERGY {
        AtomicDiracIntegrationMode::FixedMatchingPoint
    } else {
        AtomicDiracIntegrationMode::SearchMatchingPoint
    };

    Ok(AtomicDiracShootingPassSetup {
        integration_mode,
        reference_energy: input.energy,
        relative_energy_change,
        relocated: false,
    })
}

fn calculate_atomic_dirac_integration(
    input: AtomicDiracIntegrationInput<'_>,
) -> Result<AtomicDiracIntegration, AtomMathError> {
    let mut large_component = input.large_source.to_owned();
    let mut small_component = input.small_source.to_owned();
    let mut large_coefficients = input.large_coefficients.to_owned();
    let mut small_coefficients = input.small_coefficients.to_owned();
    let predictor = ATOM_INTDIR_PREDICTOR;
    let (corrector, correction_mix) = atom_intdir_corrector_coefficients();
    let mut step = input.step / ATOM_INTDIR_STEP_DIVISOR;
    let energy_scaled = input.energy / input.speed_of_light;
    let doubled_speed = input.speed_of_light + input.speed_of_light;
    let kappa = input.kappa as Real;
    let angular_term = kappa * (kappa + 1.0) / doubled_speed;
    let mut matching_index_1based = input.matching_index_1based;
    let mut max_index_1based = input.max_index_1based;
    let mut tail_large = input.asymptotic_large_component;
    let mut matching_large_component = None;
    let mut matching_small_component = None;

    if input.mode != AtomicDiracIntegrationMode::InwardOnly {
        if input.mode == AtomicDiracIntegrationMode::SearchMatchingPoint {
            matching_index_1based = atom_intdir_search_matching_point(
                input.radii,
                input.potential,
                energy_scaled,
                angular_term,
                input.active_len,
            )?;
        }
        atom_intdir_origin_expansion(
            &mut large_coefficients,
            &mut small_coefficients,
            input,
            energy_scaled,
            doubled_speed,
            kappa,
        )?;

        let matching = matching_index_1based - 1;
        let mut large_derivative = [0.0; ATOM_INTDIR_HISTORY];
        let mut small_derivative = [0.0; ATOM_INTDIR_HISTORY];
        atom_intdir_initial_history(AtomIntdirInitialHistory {
            large_component: &mut large_component,
            small_component: &mut small_component,
            large_derivative: &mut large_derivative,
            small_derivative: &mut small_derivative,
            large_coefficients: &large_coefficients,
            small_coefficients: &small_coefficients,
            radii: input.radii,
            origin_power: input.origin_power,
            coefficient_count: input.coefficient_count,
            step,
        });

        let saved_large = large_component[matching];
        let saved_small = small_component[matching];
        atom_intdir_sweep(AtomIntdirSweep {
            large_component: &mut large_component,
            small_component: &mut small_component,
            large_derivative: &mut large_derivative,
            small_derivative: &mut small_derivative,
            radii: input.radii,
            potential: input.potential,
            predictor,
            corrector,
            correction_mix,
            energy_scaled,
            doubled_speed,
            kappa,
            step,
            start_index_1based: ATOM_INTDIR_HISTORY,
            target_index_1based: matching_index_1based,
            direction: 1,
        })?;
        matching_large_component = Some(large_component[matching]);
        matching_small_component = Some(small_component[matching]);
        large_component[matching] = saved_large;
        small_component[matching] = saved_small;

        if input.mode == AtomicDiracIntegrationMode::SearchMatchingPoint {
            if let Some(outward_large) = matching_large_component {
                let tail_limit = input.matching_precision * outward_large.abs();
                if tail_large > tail_limit {
                    tail_large = tail_limit;
                }
            }
            max_index_1based = input.active_len + 2;
            atom_intdir_adjust_inward_start(
                &mut max_index_1based,
                matching_index_1based,
                input.radii,
                input.potential,
                energy_scaled,
                input.speed_of_light,
            )?;
        }
    }

    let mut large_derivative = [0.0; ATOM_INTDIR_HISTORY];
    let mut small_derivative = [0.0; ATOM_INTDIR_HISTORY];
    loop {
        step = -step;
        let decay = atom_intdir_decay(input.energy, input.speed_of_light)?;
        if decay * input.radii[max_index_1based - 1] >= ATOM_INTDIR_EXPONENT_FLOOR {
            let ratio = decay / (doubled_speed + energy_scaled);
            let mut scale = tail_large / (decay * input.radii[max_index_1based - 1]).exp();
            if scale == 0.0 {
                scale = 1.0;
            }
            for history in 0..ATOM_INTDIR_HISTORY {
                let row_1based = max_index_1based - history;
                let row = row_1based - 1;
                large_component[row] = scale * (decay * input.radii[row]).exp();
                small_component[row] = ratio * large_component[row];
                large_derivative[history] = decay * input.radii[row] * large_component[row] * step;
                small_derivative[history] = ratio * large_derivative[history];
            }
            atom_intdir_sweep(AtomIntdirSweep {
                large_component: &mut large_component,
                small_component: &mut small_component,
                large_derivative: &mut large_derivative,
                small_derivative: &mut small_derivative,
                radii: input.radii,
                potential: input.potential,
                predictor,
                corrector,
                correction_mix,
                energy_scaled,
                doubled_speed,
                kappa,
                step,
                start_index_1based: max_index_1based + 1 - ATOM_INTDIR_HISTORY,
                target_index_1based: matching_index_1based,
                direction: -1,
            })?;
            break;
        }
        atom_intdir_adjust_inward_start(
            &mut max_index_1based,
            matching_index_1based,
            input.radii,
            input.potential,
            energy_scaled,
            input.speed_of_light,
        )?;
    }

    validate_finite_vector("intdir_large_component", large_component.view())?;
    validate_finite_vector("intdir_small_component", small_component.view())?;
    validate_finite_vector("intdir_large_coefficient", large_coefficients.view())?;
    validate_finite_vector("intdir_small_coefficient", small_coefficients.view())?;

    Ok(AtomicDiracIntegration {
        large_component,
        small_component,
        large_coefficients,
        small_coefficients,
        matching_large_component,
        matching_small_component,
        matching_index_1based,
        max_index_1based,
    })
}

fn atom_intdir_corrector_coefficients() -> ([Real; ATOM_INTDIR_HISTORY], Real) {
    let mix = ATOM_INTDIR_MIX_NUMERATOR / ATOM_INTDIR_MIX_DENOMINATOR;
    let complement = 1.0 - mix;
    let mut corrector = ATOM_INTDIR_CORRECTOR_RAW;
    let correction_mix = mix * corrector[ATOM_INTDIR_HISTORY - 1];
    let mut previous = corrector[0];
    for index in 1..ATOM_INTDIR_HISTORY {
        let current = corrector[index];
        corrector[index] = mix * previous + complement * ATOM_INTDIR_PREDICTOR[index];
        previous = current;
    }
    corrector[0] = mix * ATOM_INTDIR_PREDICTOR[0];
    (corrector, correction_mix)
}

fn atom_intdir_search_matching_point(
    radii: ArrayView1<'_, Real>,
    potential: ArrayView1<'_, Real>,
    energy_scaled: Real,
    angular_term: Real,
    active_len: usize,
) -> Result<usize, AtomMathError> {
    let mut matching = ATOM_INTDIR_HISTORY;
    let mut sign = 1.0;
    loop {
        matching += 2;
        if matching >= active_len {
            if energy_scaled > -0.0003 {
                matching = active_len - 12;
            } else {
                return Err(AtomMathError::DiracIntegrationMatchingPointNotFound { active_len });
            }
        }
        let row = matching - 1;
        let value =
            (potential[row] + angular_term / (radii[row] * radii[row]) - energy_scaled) * sign;
        if value <= 0.0 {
            sign = -sign;
            if sign < 0.0 {
                continue;
            }
            if matching >= active_len - ATOM_INTDIR_HISTORY {
                matching = active_len - 12;
            }
            return Ok(matching);
        }
    }
}

fn atom_intdir_origin_expansion(
    large_coefficients: &mut Array1<Real>,
    small_coefficients: &mut Array1<Real>,
    input: AtomicDiracIntegrationInput<'_>,
    energy_scaled: Real,
    doubled_speed: Real,
    kappa: Real,
) -> Result<(), AtomMathError> {
    large_coefficients[0] = input.initial_large_coefficient;
    small_coefficients[0] = input.initial_small_coefficient;
    for coefficient in 1..input.coefficient_count {
        let order = coefficient as Real;
        let large_power = input.origin_power + kappa + order;
        let small_power = input.origin_power - kappa + order;
        let denominator = large_power * small_power + input.potential_coefficients[0].powi(2);
        if denominator == 0.0 {
            return Err(AtomMathError::ZeroDiracIntegrationDevelopmentDenominator {
                coefficient_1based: coefficient + 1,
            });
        }
        let mut large_source = (energy_scaled + doubled_speed)
            * small_coefficients[coefficient - 1]
            + small_coefficients[coefficient];
        let mut small_source =
            energy_scaled * large_coefficients[coefficient - 1] + large_coefficients[coefficient];
        for previous in 0..coefficient {
            large_source -= input.potential_coefficients[previous + 1]
                * small_coefficients[coefficient - 1 - previous];
            small_source -= input.potential_coefficients[previous + 1]
                * large_coefficients[coefficient - 1 - previous];
        }
        large_coefficients[coefficient] = (small_power * large_source
            + input.potential_coefficients[0] * small_source)
            / denominator;
        small_coefficients[coefficient] = (input.potential_coefficients[0] * large_source
            - large_power * small_source)
            / denominator;
    }

    Ok(())
}

struct AtomIntdirInitialHistory<'a> {
    large_component: &'a mut Array1<Real>,
    small_component: &'a mut Array1<Real>,
    large_derivative: &'a mut [Real; ATOM_INTDIR_HISTORY],
    small_derivative: &'a mut [Real; ATOM_INTDIR_HISTORY],
    large_coefficients: &'a Array1<Real>,
    small_coefficients: &'a Array1<Real>,
    radii: ArrayView1<'a, Real>,
    origin_power: Real,
    coefficient_count: usize,
    step: Real,
}

fn atom_intdir_initial_history(input: AtomIntdirInitialHistory<'_>) {
    let AtomIntdirInitialHistory {
        large_component,
        small_component,
        large_derivative,
        small_derivative,
        large_coefficients,
        small_coefficients,
        radii,
        origin_power,
        coefficient_count,
        step,
    } = input;
    for row in 0..ATOM_INTDIR_HISTORY {
        large_component[row] = 0.0;
        small_component[row] = 0.0;
        large_derivative[row] = 0.0;
        small_derivative[row] = 0.0;
        for coefficient in 0..coefficient_count {
            let power = origin_power + coefficient as Real;
            let radial_power = radii[row].powf(power);
            let derivative_scale = power * radial_power * step;
            large_component[row] += radial_power * large_coefficients[coefficient];
            small_component[row] += radial_power * small_coefficients[coefficient];
            large_derivative[row] += derivative_scale * large_coefficients[coefficient];
            small_derivative[row] += derivative_scale * small_coefficients[coefficient];
        }
    }
}

struct AtomIntdirSweep<'a> {
    large_component: &'a mut Array1<Real>,
    small_component: &'a mut Array1<Real>,
    large_derivative: &'a mut [Real; ATOM_INTDIR_HISTORY],
    small_derivative: &'a mut [Real; ATOM_INTDIR_HISTORY],
    radii: ArrayView1<'a, Real>,
    potential: ArrayView1<'a, Real>,
    predictor: [Real; ATOM_INTDIR_HISTORY],
    corrector: [Real; ATOM_INTDIR_HISTORY],
    correction_mix: Real,
    energy_scaled: Real,
    doubled_speed: Real,
    kappa: Real,
    step: Real,
    start_index_1based: usize,
    target_index_1based: usize,
    direction: isize,
}

fn atom_intdir_sweep(input: AtomIntdirSweep<'_>) -> Result<(), AtomMathError> {
    let AtomIntdirSweep {
        large_component,
        small_component,
        large_derivative,
        small_derivative,
        radii,
        potential,
        predictor,
        corrector,
        correction_mix,
        energy_scaled,
        doubled_speed,
        kappa,
        step,
        start_index_1based,
        target_index_1based,
        direction,
    } = input;
    let mut row_1based = start_index_1based;
    let correction_step = correction_mix * step;
    loop {
        let current = row_1based - 1;
        let mut predicted_large = large_component[current] + large_derivative[0] * predictor[0];
        let mut predicted_small = small_component[current] + small_derivative[0] * predictor[0];
        row_1based = row_1based.saturating_add_signed(direction);
        let row = row_1based - 1;
        let previous_small = small_component[row];
        let previous_large = large_component[row];
        large_component[row] = predicted_large - large_derivative[0] * corrector[0];
        small_component[row] = predicted_small - small_derivative[0] * corrector[0];
        for history in 1..ATOM_INTDIR_HISTORY {
            predicted_large += large_derivative[history] * predictor[history];
            predicted_small += small_derivative[history] * predictor[history];
            large_component[row] += large_derivative[history] * corrector[history];
            small_component[row] += small_derivative[history] * corrector[history];
            large_derivative[history - 1] = large_derivative[history];
            small_derivative[history - 1] = small_derivative[history];
        }
        let scaled_potential = (energy_scaled - potential[row]) * radii[row];
        let shifted_potential = scaled_potential + doubled_speed * radii[row];
        large_component[row] += correction_step
            * (shifted_potential * predicted_small - kappa * predicted_large + previous_small);
        small_component[row] += correction_step
            * (kappa * predicted_small - scaled_potential * predicted_large - previous_large);
        large_derivative[ATOM_INTDIR_HISTORY - 1] = step
            * (shifted_potential * small_component[row] - kappa * large_component[row]
                + previous_small);
        small_derivative[ATOM_INTDIR_HISTORY - 1] = step
            * (kappa * small_component[row]
                - scaled_potential * large_component[row]
                - previous_large);
        if row_1based == target_index_1based {
            return Ok(());
        }
    }
}

fn atom_intdir_adjust_inward_start(
    max_index_1based: &mut usize,
    matching_index_1based: usize,
    radii: ArrayView1<'_, Real>,
    potential: ArrayView1<'_, Real>,
    energy_scaled: Real,
    speed_of_light: Real,
) -> Result<(), AtomMathError> {
    let threshold = ATOM_INTDIR_INWARD_THRESHOLD / speed_of_light;
    loop {
        *max_index_1based = max_index_1based.checked_sub(2).ok_or(
            AtomMathError::DiracIntegrationInwardStartTooClose {
                matching_index_1based,
            },
        )?;
        if (*max_index_1based + 1) <= (matching_index_1based + ATOM_INTDIR_HISTORY) {
            return Err(AtomMathError::DiracIntegrationInwardStartTooClose {
                matching_index_1based,
            });
        }
        let row = *max_index_1based - 1;
        if (potential[row] - energy_scaled) * radii[row] * radii[row] <= threshold {
            return Ok(());
        }
    }
}

fn atom_intdir_decay(energy: Real, speed_of_light: Real) -> Result<Real, AtomMathError> {
    let energy_scaled = energy / speed_of_light;
    let doubled_speed = speed_of_light + speed_of_light;
    let radicand = -energy_scaled * (doubled_speed + energy_scaled);
    if radicand.is_finite() && radicand > 0.0 {
        Ok(-radicand.sqrt())
    } else {
        Err(AtomMathError::InvalidDiracIntegrationEnergy {
            energy,
            speed_of_light,
        })
    }
}

fn calculate_atomic_dirac_solver_setup(
    input: AtomicDiracSolverSetupInput<'_>,
) -> Result<AtomicDiracSolverSetup, AtomMathError> {
    let requested_method = input.method;
    let method = if input.method <= 0 { 1 } else { input.method };
    let kappa = input.kappa as Real;
    let doubled_speed_of_light = input.speed_of_light + input.speed_of_light;
    let potential_origin = input.potential_coefficients[0];
    let mut initial_small_coefficient = input.initial_small_coefficient;

    if potential_origin < 0.0 {
        if input.kappa > 0 {
            initial_small_coefficient =
                -input.initial_large_coefficient * (kappa + input.origin_power) / potential_origin;
        } else if input.kappa < 0 {
            let denominator = kappa - input.origin_power;
            if denominator == 0.0 {
                return Err(AtomMathError::ZeroDiracSolverInitialCoefficientDenominator);
            }
            initial_small_coefficient =
                -input.initial_large_coefficient * potential_origin / denominator;
        }
    }

    let angular_term = kappa * (kappa + 1.0) / doubled_speed_of_light;
    let kappa_abs = input.kappa.unsigned_abs();
    let principal = i32::try_from(input.principal_quantum_number).map_err(|_| {
        AtomMathError::InvalidDiracSolverPrincipalQuantumNumber {
            principal_quantum_number: input.principal_quantum_number,
        }
    })?;
    let kappa_abs_i32 = i32::try_from(kappa_abs).map_err(|_| {
        AtomMathError::InvalidDiracSolverPrincipalQuantumNumber {
            principal_quantum_number: input.principal_quantum_number,
        }
    })?;
    let mut target_nodes = principal - kappa_abs_i32;
    if input.kappa < 0 {
        target_nodes += 1;
    }

    let mut energy_floor = 0.0;
    for row in 0..input.active_len {
        let radius = input.radii[row];
        let apparent =
            (angular_term / (radius * radius) + input.potential[row]) * input.speed_of_light;
        if apparent < energy_floor {
            energy_floor = apparent;
        }
    }
    if energy_floor >= 0.0 {
        return Err(AtomMathError::DiracSolverPotentialNotAttractive { energy_floor });
    }

    let energy = if input.energy < energy_floor {
        energy_floor * 0.9
    } else {
        input.energy
    };

    validate_finite_scalar("soldir_setup_energy", energy)?;
    validate_finite_scalar("soldir_setup_energy_floor", energy_floor)?;
    validate_finite_scalar(
        "soldir_setup_initial_small_coefficient",
        initial_small_coefficient,
    )?;
    validate_finite_scalar("soldir_setup_angular_term", angular_term)?;

    Ok(AtomicDiracSolverSetup {
        requested_method,
        method,
        energy,
        energy_floor,
        initial_small_coefficient,
        angular_term,
        target_nodes,
        doubled_speed_of_light,
    })
}

fn calculate_atomic_local_density_potential(
    input: AtomicLocalDensityPotentialInput<'_>,
) -> Result<AtomicLocalDensityPotential, AtomMathError> {
    let radial_count = input.radii.len();
    let orbital_count = input.active_lengths.len();
    let mut total_density = Array1::<Real>::zeros(radial_count);
    let mut valence_density = Array1::<Real>::zeros(radial_count);

    for orbital in 0..orbital_count {
        for row in 0..input.active_lengths[orbital] {
            let component_density = input.large_components[(row, orbital)].powi(2)
                + input.small_components[(row, orbital)].powi(2);
            total_density[row] += input.occupations[orbital] * component_density;
            valence_density[row] += input.valence_occupations[orbital] * component_density;
        }
    }

    let mut potential = input.initial_potential.to_owned();
    let mut development_coefficients = input.initial_development_coefficients.to_owned();
    let mut energy_density = input.initial_energy_density.to_owned();

    for row in 0..radial_count {
        let radius_squared = input.radii[row] * input.radii[row];
        let density = total_density[row] / radius_squared;
        if density <= 0.0 {
            continue;
        }

        let comparison_density = match input.mode {
            AtomicLocalDensityExchangeMode::ValenceDensity => valence_density[row] / radius_squared,
            AtomicLocalDensityExchangeMode::CoreDensitySeparated => {
                (total_density[row] - valence_density[row]) / radius_squared
            }
            AtomicLocalDensityExchangeMode::DiracFockOnly => 0.0,
            AtomicLocalDensityExchangeMode::TotalDensity => density,
        };
        let vxc = atomic_local_density_vxc(input.mode, density, comparison_density)?;
        if input.accumulate_energy_density {
            energy_density[row] += vxc * total_density[row];
        }
        let scaled_vxc = vxc / input.speed_of_light;
        if row == 0 {
            development_coefficients[1] += scaled_vxc;
        }
        potential[row] += scaled_vxc;
    }

    validate_finite_vector("vlda_total_density", total_density.view())?;
    validate_finite_vector("vlda_valence_density", valence_density.view())?;
    validate_finite_vector("vlda_potential", potential.view())?;
    validate_finite_vector(
        "vlda_development_coefficient",
        development_coefficients.view(),
    )?;
    validate_finite_vector("vlda_energy_density", energy_density.view())?;

    Ok(AtomicLocalDensityPotential {
        total_density,
        valence_density,
        potential,
        development_coefficients,
        energy_density,
    })
}

fn atomic_local_density_vxc(
    mode: AtomicLocalDensityExchangeMode,
    density: Real,
    comparison_density: Real,
) -> Result<Real, AtomMathError> {
    let density_parameter = (density / 3.0).powf(-1.0 / 3.0);
    let comparison_parameter = if comparison_density > 0.0 {
        (comparison_density / 3.0).powf(-1.0 / 3.0)
    } else {
        101.0
    };

    match mode {
        AtomicLocalDensityExchangeMode::DiracFockOnly => Ok(0.0),
        AtomicLocalDensityExchangeMode::TotalDensity
        | AtomicLocalDensityExchangeMode::ValenceDensity => {
            Ok(von_barth_hedin_potential(comparison_parameter, 1.0)?)
        }
        AtomicLocalDensityExchangeMode::CoreDensitySeparated => {
            let total_vbh = von_barth_hedin_potential(density_parameter, 1.0)?;
            let fermi_momentum = FEFF_FERMI_MOMENTUM_FACTOR / density_parameter;
            let dirac_hara = dirac_hara_exchange_potential(comparison_parameter, fermi_momentum)?;
            Ok(total_vbh - dirac_hara)
        }
    }
}

fn calculate_atomic_orbital_potential(
    input: AtomicOrbitalPotentialInput<'_>,
) -> Result<AtomicOrbitalPotential, AtomMathError> {
    let active_orbital =
        one_based_atomic_orbital_index(input.active_orbital_1based, input.active_lengths.len())?;
    let active_occupation =
        validate_positive_occupation("potrdf_active_orbital", active_orbital, input.occupations)?;
    let radial_count = input.radii.len();
    let coefficient_count = input.nuclear_development_coefficients.len();
    let active_j2 = kappa_angular_rank(input.kappas[active_orbital])?;

    let mut central_development_coefficients = input.nuclear_development_coefficients.to_owned();
    let mut central_work = Array1::<Real>::zeros(radial_count);
    let mut exchange_large_work = Array1::<Real>::zeros(radial_count);
    let mut exchange_small_work = Array1::<Real>::zeros(radial_count);
    let mut lagrange_large_work = Array1::<Real>::zeros(radial_count);
    let mut lagrange_small_work = Array1::<Real>::zeros(radial_count);
    let mut exchange_large_coefficients = Array1::<Real>::zeros(coefficient_count);
    let mut exchange_small_coefficients = Array1::<Real>::zeros(coefficient_count);

    let mut rank = 0usize;
    loop {
        let (source, source_coefficients, source_len) =
            direct_orbital_potential_source(input, active_orbital, active_occupation, rank)?;
        let transform = atomic_yk_zk_prepared_source(AtomicYkZkPreparedSourceInput {
            source: source.view(),
            source_coefficients: source_coefficients.view(),
            radii: input.radii,
            step: input.step,
            angular_momentum: rank,
            coefficient_count,
            source_len,
            active_len: radial_count,
        })?;

        for (coefficient, &value) in transform.yk_coefficients.iter().enumerate() {
            let target = rank
                .checked_add(coefficient)
                .and_then(|value| value.checked_add(3))
                .ok_or(AtomMathError::YkZkAngularMomentumOutOfRange {
                    angular_momentum: rank,
                })?;
            if target < coefficient_count {
                central_development_coefficients[target] -= value;
            }
        }
        for row in 0..radial_count {
            central_work[row] += transform.yk[row];
        }

        rank = rank
            .checked_add(2)
            .ok_or(AtomMathError::YkZkAngularMomentumOutOfRange {
                angular_momentum: rank,
            })?;
        if rank <= coefficient_count {
            central_development_coefficients[rank - 1] += transform.origin_constant;
        }
        if rank >= active_j2 {
            break;
        }
    }

    if input.include_exchange {
        accumulate_orbital_exchange_terms(
            input,
            active_orbital,
            active_occupation,
            active_j2,
            exchange_large_work.view_mut(),
            exchange_small_work.view_mut(),
            exchange_large_coefficients.view_mut(),
            exchange_small_coefficients.view_mut(),
        )?;
    }
    if input.include_lagrange {
        accumulate_orbital_lagrange_terms(
            input,
            active_orbital,
            lagrange_large_work.view_mut(),
            lagrange_small_work.view_mut(),
            exchange_large_coefficients.view_mut(),
            exchange_small_coefficients.view_mut(),
        )?;
    }

    for coefficient in 0..coefficient_count {
        central_development_coefficients[coefficient] /= input.speed_of_light;
        exchange_large_coefficients[coefficient] /= input.speed_of_light;
        exchange_small_coefficients[coefficient] /= input.speed_of_light;
    }

    let central_potential = Array1::from_shape_fn(radial_count, |row| {
        (central_work[row] / input.radii[row] + input.nuclear_potential[row]) / input.speed_of_light
    });
    let exchange_large = Array1::from_shape_fn(radial_count, |row| {
        (exchange_large_work[row] + lagrange_large_work[row] * input.radii[row])
            / input.speed_of_light
    });
    let exchange_small = Array1::from_shape_fn(radial_count, |row| {
        (exchange_small_work[row] + lagrange_small_work[row] * input.radii[row])
            / input.speed_of_light
    });

    validate_finite_vector("potrdf_central_potential", central_potential.view())?;
    validate_finite_vector(
        "potrdf_central_development_coefficient",
        central_development_coefficients.view(),
    )?;
    validate_finite_vector("potrdf_exchange_large", exchange_large.view())?;
    validate_finite_vector("potrdf_exchange_small", exchange_small.view())?;
    validate_finite_vector(
        "potrdf_exchange_large_coefficient",
        exchange_large_coefficients.view(),
    )?;
    validate_finite_vector(
        "potrdf_exchange_small_coefficient",
        exchange_small_coefficients.view(),
    )?;

    Ok(AtomicOrbitalPotential {
        central_potential,
        central_development_coefficients,
        exchange_large,
        exchange_small,
        exchange_large_coefficients,
        exchange_small_coefficients,
    })
}

fn direct_orbital_potential_source(
    input: AtomicOrbitalPotentialInput<'_>,
    active_orbital: usize,
    active_occupation: Real,
    rank: usize,
) -> Result<(Array1<Real>, Array1<Real>, usize), AtomMathError> {
    let radial_count = input.radii.len();
    let coefficient_count = input.nuclear_development_coefficients.len();
    let mut source = Array1::<Real>::zeros(radial_count);
    let mut source_coefficients = Array1::<Real>::zeros(coefficient_count);
    let mut source_len = 0usize;

    for orbital in 0..input.active_lengths.len() {
        let orbital_j2 = kappa_angular_rank(input.kappas[orbital])?;
        if rank > orbital_j2 {
            source_len = source_len.max(input.active_lengths[orbital]);
            continue;
        }

        let scale = atomic_direct_coulomb_coefficient(
            input.coulomb_coefficients,
            active_orbital,
            orbital,
            rank,
        )? / active_occupation;
        if scale != 0.0 {
            for row in 0..input.active_lengths[orbital] {
                source[row] += scale
                    * (input.large_components[(row, orbital)].powi(2)
                        + input.small_components[(row, orbital)].powi(2));
            }

            let origin_power_start = kappa_angular_rank(input.kappas[orbital])?
                .checked_add(1)
                .and_then(|value| value.checked_sub(rank))
                .ok_or(AtomMathError::YkZkAngularMomentumOutOfRange {
                    angular_momentum: rank,
                })?;
            let max_terms = coefficient_count
                .checked_add(2)
                .and_then(|value| value.checked_sub(origin_power_start))
                .unwrap_or(0);
            if max_terms > 0 {
                let coefficient_scale = scale * input.origin_scales[orbital].powi(2);
                let large_coefficients = input.large_coefficients.index_axis(Axis(1), orbital);
                let small_coefficients = input.small_coefficients.index_axis(Axis(1), orbital);
                for term in 1..=max_terms {
                    let target_1based = origin_power_start
                        .checked_add(term)
                        .and_then(|value| value.checked_sub(2))
                        .ok_or(AtomMathError::YkZkAngularMomentumOutOfRange {
                            angular_momentum: rank,
                        })?;
                    if target_1based == 0 || target_1based > coefficient_count {
                        continue;
                    }
                    source_coefficients[target_1based - 1] += coefficient_scale
                        * (polynomial_product_coefficient_view(
                            large_coefficients,
                            large_coefficients,
                            term,
                        )? + polynomial_product_coefficient_view(
                            small_coefficients,
                            small_coefficients,
                            term,
                        )?);
                }
            }
        }
        source_len = source_len.max(input.active_lengths[orbital]);
    }

    Ok((source, source_coefficients, source_len))
}

#[allow(clippy::too_many_arguments)]
fn accumulate_orbital_exchange_terms(
    input: AtomicOrbitalPotentialInput<'_>,
    active_orbital: usize,
    active_occupation: Real,
    active_j2: usize,
    mut exchange_large_work: ndarray::ArrayViewMut1<'_, Real>,
    mut exchange_small_work: ndarray::ArrayViewMut1<'_, Real>,
    mut exchange_large_coefficients: ndarray::ArrayViewMut1<'_, Real>,
    mut exchange_small_coefficients: ndarray::ArrayViewMut1<'_, Real>,
) -> Result<(), AtomMathError> {
    let coefficient_count = input.nuclear_development_coefficients.len();
    for orbital in 0..input.active_lengths.len() {
        if orbital == active_orbital {
            continue;
        }
        let orbital_j2 = kappa_angular_rank(input.kappas[orbital])?;
        let maximum_rank = (orbital_j2 + active_j2) / 2;
        let mut rank = orbital_j2.abs_diff(maximum_rank);
        if input.kappas[orbital].signum() * input.kappas[active_orbital].signum() < 0 {
            rank = rank
                .checked_add(1)
                .ok_or(AtomMathError::YkZkAngularMomentumOutOfRange {
                    angular_momentum: rank,
                })?;
        }

        while rank <= maximum_rank {
            let scale = atomic_exchange_coulomb_coefficient(
                input.coulomb_coefficients,
                orbital,
                active_orbital,
                rank,
            )? / active_occupation;
            if scale != 0.0 {
                let transform = atomic_yk_zk_exchange(AtomicYkZkExchangeInput {
                    left_orbital_1based: orbital + 1,
                    right_orbital_1based: active_orbital + 1,
                    large_small: false,
                    angular_momentum: rank,
                    step: input.step,
                    radii: input.radii,
                    active_lengths: input.active_lengths,
                    orbital_powers: input.orbital_powers,
                    large_components: input.large_components,
                    small_components: input.small_components,
                    large_coefficients: input.large_coefficients,
                    small_coefficients: input.small_coefficients,
                })?;
                for row in 0..input.active_lengths[orbital] {
                    exchange_large_work[row] +=
                        scale * transform.yk[row] * input.large_components[(row, orbital)];
                    exchange_small_work[row] +=
                        scale * transform.yk[row] * input.small_components[(row, orbital)];
                }

                let orbital_abs = kappa_abs_usize(input.kappas[orbital])?;
                let active_abs = kappa_abs_usize(input.kappas[active_orbital])?;
                let origin_shift = rank
                    .checked_add(1)
                    .and_then(|value| value.checked_add(orbital_abs))
                    .and_then(|value| value.checked_sub(active_abs))
                    .ok_or(AtomMathError::YkZkAngularMomentumOutOfRange {
                        angular_momentum: rank,
                    })?;
                if origin_shift <= coefficient_count {
                    let origin_scale =
                        scale * transform.origin_constant * input.origin_scales[orbital]
                            / input.origin_scales[active_orbital];
                    for coefficient_1based in origin_shift..=coefficient_count {
                        let source_index = coefficient_1based - origin_shift;
                        exchange_large_coefficients[coefficient_1based - 1] +=
                            input.large_coefficients[(source_index, orbital)] * origin_scale;
                        exchange_small_coefficients[coefficient_1based - 1] +=
                            input.small_coefficients[(source_index, orbital)] * origin_scale;
                    }
                }

                let exchange_start = kappa_angular_rank(input.kappas[orbital])?
                    .checked_add(2)
                    .ok_or(AtomMathError::YkZkAngularMomentumOutOfRange {
                        angular_momentum: rank,
                    })?;
                if exchange_start <= coefficient_count {
                    let large_coefficients = input.large_coefficients.index_axis(Axis(1), orbital);
                    let small_coefficients = input.small_coefficients.index_axis(Axis(1), orbital);
                    let source_scale = scale * input.origin_scales[orbital].powi(2);
                    for coefficient_1based in exchange_start..=coefficient_count {
                        let term = coefficient_1based + 1 - exchange_start;
                        exchange_large_coefficients[coefficient_1based - 1] -= source_scale
                            * polynomial_product_coefficient_view(
                                transform.yk_coefficients.view(),
                                large_coefficients,
                                term,
                            )?;
                        exchange_small_coefficients[coefficient_1based - 1] -= source_scale
                            * polynomial_product_coefficient_view(
                                transform.yk_coefficients.view(),
                                small_coefficients,
                                term,
                            )?;
                    }
                }
            }

            rank = rank
                .checked_add(2)
                .ok_or(AtomMathError::YkZkAngularMomentumOutOfRange {
                    angular_momentum: rank,
                })?;
        }
    }
    Ok(())
}

fn accumulate_orbital_lagrange_terms(
    input: AtomicOrbitalPotentialInput<'_>,
    active_orbital: usize,
    mut lagrange_large_work: ndarray::ArrayViewMut1<'_, Real>,
    mut lagrange_small_work: ndarray::ArrayViewMut1<'_, Real>,
    mut exchange_large_coefficients: ndarray::ArrayViewMut1<'_, Real>,
    mut exchange_small_coefficients: ndarray::ArrayViewMut1<'_, Real>,
) -> Result<(), AtomMathError> {
    for orbital in 0..input.self_consistent_count {
        if input.kappas[orbital] != input.kappas[active_orbital] || orbital == active_orbital {
            continue;
        }
        if input.shell_markers[orbital] < 0 && input.shell_markers[active_orbital] < 0 {
            continue;
        }

        let packed = packed_orbital_pair_index(orbital, active_orbital)?;
        let scale = input.lagrange_parameters[packed] * input.occupations[orbital];
        for row in 0..input.active_lengths[orbital] {
            lagrange_large_work[row] += scale * input.large_components[(row, orbital)];
            lagrange_small_work[row] += scale * input.small_components[(row, orbital)];
        }
        for coefficient in 0..input.nuclear_development_coefficients.len() {
            exchange_large_coefficients[coefficient] +=
                input.large_coefficients[(coefficient, orbital)] * scale;
            exchange_small_coefficients[coefficient] +=
                input.small_coefficients[(coefficient, orbital)] * scale;
        }
    }
    Ok(())
}

fn atomic_nuclear_mesh_parameters(
    input: AtomicNuclearPotentialInput,
) -> Result<(usize, Real), AtomMathError> {
    let mut nucleus_index = requested_nucleus_index_abs(input.requested_nucleus_index)?;
    let mut first_radius_times_charge = input.first_radius_times_charge;
    let mut nuclear_mass_amu = 0.0;
    if input.requested_nucleus_index < 0 {
        let atomic_number = atomic_number_from_charge(input.nuclear_charge)?;
        nuclear_mass_amu = nuclear_mass(atomic_number)?;
    }

    if nuclear_mass_amu <= 0.1 {
        return Ok((1, first_radius_times_charge));
    }

    if nucleus_index == 0 || nucleus_index > input.radial_count {
        return Err(AtomMathError::NuclearRadiusOutOfRange {
            nucleus_index,
            radial_count: input.radial_count,
        });
    }
    let mass_exponent = Real::from(1.0_f32 / 3.0_f32);
    let scaled_radius =
        input.nuclear_charge * nuclear_mass_amu.powf(mass_exponent) * ATOM_NUCDEV_RADIUS_FACTOR;
    let requested_first_radius = scaled_radius / (input.step * (nucleus_index as Real - 1.0)).exp();
    if requested_first_radius <= first_radius_times_charge {
        first_radius_times_charge = requested_first_radius;
    } else {
        let radius_steps = (scaled_radius / first_radius_times_charge).ln() / input.step;
        let half_steps = (radius_steps / 2.0).trunc();
        nucleus_index = 3 + 2 * half_steps as usize;
        if nucleus_index >= input.radial_count {
            return Err(AtomMathError::NuclearRadiusOutOfRange {
                nucleus_index,
                radial_count: input.radial_count,
            });
        }
        first_radius_times_charge =
            scaled_radius * (-(nucleus_index as Real - 1.0) * input.step).exp();
    }
    validate_finite_scalar(
        "nucdev_first_radius_times_charge",
        first_radius_times_charge,
    )?;
    Ok((nucleus_index, first_radius_times_charge))
}

fn requested_nucleus_index_abs(requested: isize) -> Result<usize, AtomMathError> {
    if requested == isize::MIN {
        return Err(AtomMathError::NuclearRadiusOutOfRange {
            nucleus_index: usize::MAX,
            radial_count: 0,
        });
    }
    Ok(requested.unsigned_abs())
}

fn atomic_number_from_charge(nuclear_charge: Real) -> Result<usize, AtomMathError> {
    if nuclear_charge < 1.0 || nuclear_charge > usize::MAX as Real {
        return Err(AtomMathError::InvalidNuclearPotentialScalar {
            field: "nuclear_charge",
            value: nuclear_charge,
        });
    }
    Ok(nuclear_charge.trunc() as usize)
}

struct AtomicDifferentialIntegralWork {
    values: Vec<Real>,
    coefficients: Vec<Real>,
    origin_power: Real,
}

fn calculate_atomic_differential_integral(
    input: AtomicDifferentialIntegralInput<'_>,
) -> Result<Real, AtomMathError> {
    let work = atomic_differential_integral_work(&input)?;
    integrate_atomic_differential_work(&input, work)
}

fn atomic_differential_integral_work(
    input: &AtomicDifferentialIntegralInput<'_>,
) -> Result<AtomicDifferentialIntegralWork, AtomMathError> {
    match input.kind {
        AtomicDifferentialIntegralKind::ComponentOverlap {
            left_orbital_1based,
            right_orbital_1based,
            multiply_by_derivative,
        } => atomic_component_overlap_integral_work(
            input,
            left_orbital_1based,
            right_orbital_1based,
            multiply_by_derivative,
            false,
        ),
        AtomicDifferentialIntegralKind::LargeSmallOverlap {
            left_orbital_1based,
            right_orbital_1based,
            multiply_by_derivative,
        } => atomic_component_overlap_integral_work(
            input,
            left_orbital_1based,
            right_orbital_1based,
            multiply_by_derivative,
            true,
        ),
        AtomicDifferentialIntegralKind::DerivativeProjection {
            large_orbital_1based,
            small_orbital_1based,
        } => atomic_derivative_projection_integral_work(
            input,
            large_orbital_1based,
            small_orbital_1based,
        ),
        AtomicDifferentialIntegralKind::DerivativeNorm { active_len } => {
            atomic_derivative_norm_integral_work(input, active_len)
        }
    }
}

fn atomic_component_overlap_integral_work(
    input: &AtomicDifferentialIntegralInput<'_>,
    left_orbital_1based: usize,
    right_orbital_1based: usize,
    multiply_by_derivative: bool,
    large_small: bool,
) -> Result<AtomicDifferentialIntegralWork, AtomMathError> {
    let left = one_based_atomic_orbital_index(left_orbital_1based, input.active_lengths.len())?;
    let right = one_based_atomic_orbital_index(right_orbital_1based, input.active_lengths.len())?;
    let active_len = input.active_lengths[left].min(input.active_lengths[right]);
    validate_differential_active_len(active_len, input.radii.len())?;

    let values = (0..active_len)
        .map(|row| {
            let base = if large_small {
                input.large_components[(row, left)] * input.small_components[(row, right)]
            } else {
                input.large_components[(row, left)] * input.large_components[(row, right)]
                    + input.small_components[(row, left)] * input.small_components[(row, right)]
            };
            if multiply_by_derivative {
                base * input.derivative_large[row]
            } else {
                base
            }
        })
        .collect::<Vec<_>>();

    let left_large_coefficients = input.large_coefficients.index_axis(Axis(1), left);
    let right_large_coefficients = input.large_coefficients.index_axis(Axis(1), right);
    let left_small_coefficients = input.small_coefficients.index_axis(Axis(1), left);
    let right_small_coefficients = input.small_coefficients.index_axis(Axis(1), right);
    let coefficient_count = input.large_coefficients.nrows();
    let mut coefficients = (1..=coefficient_count)
        .map(|term| {
            if large_small {
                polynomial_product_coefficient_view(
                    left_large_coefficients,
                    right_small_coefficients,
                    term,
                )
            } else {
                Ok(polynomial_product_coefficient_view(
                    left_large_coefficients,
                    right_large_coefficients,
                    term,
                )? + polynomial_product_coefficient_view(
                    left_small_coefficients,
                    right_small_coefficients,
                    term,
                )?)
            }
        })
        .collect::<Result<Vec<_>, _>>()?;

    let mut origin_power = input.orbital_powers[left] + input.orbital_powers[right];
    if multiply_by_derivative {
        origin_power += input.origin_power;
        coefficients = (1..=coefficient_count)
            .map(|term| {
                polynomial_product_coefficient_slice_view(
                    &coefficients,
                    input.derivative_large_coefficients,
                    term,
                )
            })
            .collect::<Result<Vec<_>, _>>()?;
    }
    validate_finite_scalar("dsordf_origin_power", origin_power)?;

    Ok(AtomicDifferentialIntegralWork {
        values,
        coefficients,
        origin_power,
    })
}

fn atomic_derivative_projection_integral_work(
    input: &AtomicDifferentialIntegralInput<'_>,
    large_orbital_1based: usize,
    small_orbital_1based: usize,
) -> Result<AtomicDifferentialIntegralWork, AtomMathError> {
    let large_orbital =
        one_based_atomic_orbital_index(large_orbital_1based, input.active_lengths.len())?;
    let small_orbital =
        one_based_atomic_orbital_index(small_orbital_1based, input.active_lengths.len())?;
    let active_len = input.active_lengths[large_orbital].min(input.active_lengths[small_orbital]);
    validate_differential_active_len(active_len, input.radii.len())?;

    let values = (0..active_len)
        .map(|row| {
            input.derivative_large[row] * input.large_components[(row, large_orbital)]
                + input.derivative_small[row] * input.small_components[(row, small_orbital)]
        })
        .collect::<Vec<_>>();
    let large_coefficients = input.large_coefficients.index_axis(Axis(1), large_orbital);
    let small_coefficients = input.small_coefficients.index_axis(Axis(1), small_orbital);
    let coefficient_count = input.large_coefficients.nrows();
    let coefficients = (1..=coefficient_count)
        .map(|term| -> Result<Real, AtomMathError> {
            Ok(polynomial_product_coefficient_view(
                large_coefficients,
                input.derivative_large_coefficients,
                term,
            )? + polynomial_product_coefficient_view(
                small_coefficients,
                input.derivative_small_coefficients,
                term,
            )?)
        })
        .collect::<Result<Vec<_>, _>>()?;
    let origin_power = input.origin_power + input.orbital_powers[large_orbital];
    validate_finite_scalar("dsordf_origin_power", origin_power)?;

    Ok(AtomicDifferentialIntegralWork {
        values,
        coefficients,
        origin_power,
    })
}

fn atomic_derivative_norm_integral_work(
    input: &AtomicDifferentialIntegralInput<'_>,
    active_len: usize,
) -> Result<AtomicDifferentialIntegralWork, AtomMathError> {
    validate_differential_active_len(active_len, input.radii.len())?;

    let values = (0..active_len)
        .map(|row| {
            input.derivative_large[row] * input.derivative_large[row]
                + input.derivative_small[row] * input.derivative_small[row]
        })
        .collect::<Vec<_>>();
    let coefficient_count = input.derivative_large_coefficients.len();
    let coefficients = (1..=coefficient_count)
        .map(|term| -> Result<Real, AtomMathError> {
            Ok(polynomial_product_coefficient_view(
                input.derivative_large_coefficients,
                input.derivative_large_coefficients,
                term,
            )? + polynomial_product_coefficient_view(
                input.derivative_small_coefficients,
                input.derivative_small_coefficients,
                term,
            )?)
        })
        .collect::<Result<Vec<_>, _>>()?;
    let origin_power = input.origin_power + input.origin_power;
    validate_finite_scalar("dsordf_origin_power", origin_power)?;

    Ok(AtomicDifferentialIntegralWork {
        values,
        coefficients,
        origin_power,
    })
}

fn integrate_atomic_differential_work(
    input: &AtomicDifferentialIntegralInput<'_>,
    work: AtomicDifferentialIntegralWork,
) -> Result<Real, AtomMathError> {
    let radial_power = input
        .power
        .checked_add(1)
        .ok_or(AtomMathError::DifferentialIntegralPowerOutOfRange { power: input.power })?;
    let scaled = work
        .values
        .iter()
        .enumerate()
        .map(|(row, &value)| value * input.radii[row].powi(radial_power))
        .collect::<Vec<_>>();
    validate_finite_slice("dsordf_integrand", &scaled)?;

    let active_len = scaled.len();
    let mut integral = 0.0;
    let mut row_1based = 2;
    while row_1based < active_len {
        let row = row_1based - 1;
        integral += scaled[row] + scaled[row] + scaled[row + 1];
        row_1based += 2;
    }
    integral = input.step * (integral + integral + scaled[0] - scaled[active_len - 1]) / 3.0;

    let mut origin_exponent = work.origin_power + Real::from(input.power);
    validate_finite_scalar("dsordf_origin_exponent", origin_exponent)?;
    for coefficient in work.coefficients {
        origin_exponent += 1.0;
        if origin_exponent == 0.0 {
            return Err(AtomMathError::ZeroDifferentialIntegralOriginExponent);
        }
        let correction = coefficient * input.radii[0].powf(origin_exponent) / origin_exponent;
        validate_finite_scalar("dsordf_origin_correction", correction)?;
        integral += correction;
    }
    validate_finite_scalar("dsordf_integral", integral)?;
    Ok(integral)
}

fn polynomial_product_coefficient_view(
    left: ArrayView1<'_, Real>,
    right: ArrayView1<'_, Real>,
    term_count: usize,
) -> Result<Real, AtomMathError> {
    polynomial_product_coefficient_indexed(
        term_count,
        left.len(),
        right.len(),
        |index| left[index],
        |index| right[index],
    )
}

fn polynomial_product_coefficient_slice_view(
    left: &[Real],
    right: ArrayView1<'_, Real>,
    term_count: usize,
) -> Result<Real, AtomMathError> {
    polynomial_product_coefficient_indexed(
        term_count,
        left.len(),
        right.len(),
        |index| left[index],
        |index| right[index],
    )
}

fn polynomial_product_coefficient_indexed(
    term_count: usize,
    left_len: usize,
    right_len: usize,
    mut left_at: impl FnMut(usize) -> Real,
    mut right_at: impl FnMut(usize) -> Real,
) -> Result<Real, AtomMathError> {
    if term_count == 0 || term_count > left_len || term_count > right_len {
        return Err(AtomMathError::InvalidPolynomialTerm {
            term_count,
            left_len,
            right_len,
        });
    }
    Ok((0..term_count)
        .map(|index| left_at(index) * right_at(term_count - 1 - index))
        .sum())
}

fn one_based_atomic_orbital_index(
    orbital_1based: usize,
    orbital_count: usize,
) -> Result<usize, AtomMathError> {
    if (1..=orbital_count).contains(&orbital_1based) {
        Ok(orbital_1based - 1)
    } else {
        Err(AtomMathError::ActiveOrbitalOutOfRange {
            active_orbital_1based: orbital_1based,
            orbital_count,
        })
    }
}

fn calculate_atomic_yk_zk_transform(
    input: AtomicYkZkTransformInput<'_>,
) -> Result<AtomicYkZkTransform, AtomMathError> {
    let source_len = input.source_len.min(input.active_len - 2);
    let k_i32 = i32::try_from(input.angular_momentum).map_err(|_| {
        AtomMathError::YkZkAngularMomentumOutOfRange {
            angular_momentum: input.angular_momentum,
        }
    })?;
    let k_plus_one = input.angular_momentum.checked_add(1).ok_or(
        AtomMathError::YkZkAngularMomentumOutOfRange {
            angular_momentum: input.angular_momentum,
        },
    )?;
    let k_plus_one_i32 =
        i32::try_from(k_plus_one).map_err(|_| AtomMathError::YkZkAngularMomentumOutOfRange {
            angular_momentum: input.angular_momentum,
        })?;
    let k_real = input.angular_momentum as Real;
    let order = (input.angular_momentum + input.angular_momentum + 1) as Real;

    let mut yk = Array1::<Real>::zeros(input.active_len);
    let mut zk = Array1::<Real>::zeros(input.active_len);
    let mut yk_coefficients = Array1::from_iter(
        (0..input.coefficient_count).map(|coefficient| input.source_coefficients[coefficient]),
    );
    let mut zk_coefficients = Array1::<Real>::zeros(input.coefficient_count);

    let mut power = input.initial_power;
    let mut origin_constant = 0.0;
    for coefficient in 0..input.coefficient_count {
        power += 1.0;
        let zk_denominator = power + k_real;
        validate_yk_zk_denominator("zk_origin", zk_denominator)?;
        zk_coefficients[coefficient] = yk_coefficients[coefficient] / zk_denominator;

        if yk_coefficients[coefficient] != 0.0 {
            let radius_power = input.radii[0].powf(power);
            zk[0] += zk_coefficients[coefficient] * radius_power;
            zk[1] += zk_coefficients[coefficient] * input.radii[1].powf(power);

            let yk_denominator = power - k_real - 1.0;
            validate_yk_zk_denominator("yk_origin", yk_denominator)?;
            yk_coefficients[coefficient] = order * zk_coefficients[coefficient] / yk_denominator;
            origin_constant += yk_coefficients[coefficient] * radius_power;
        }
    }

    for row in 0..source_len {
        yk[row] = input.source[row] * input.radii[row];
    }
    yk[source_len] = 0.0;
    yk[source_len + 1] = 0.0;

    let step_exp = input.step.exp();
    let attenuation = step_exp.powi(-k_i32);
    let base_weight = input.step / 24.0;
    let middle_weight = 13.0 * base_weight;
    let leading_weight = attenuation * attenuation * base_weight;
    let trailing_weight = base_weight / attenuation;

    for row_1based in 3..=(source_len + 1) {
        let row = row_1based - 1;
        zk[row] = zk[row - 1] * attenuation
            + (middle_weight * (yk[row] + yk[row - 1] * attenuation)
                - (yk[row - 2] * leading_weight + yk[row + 1] * trailing_weight));
    }

    yk[source_len - 1] = zk[source_len - 1];
    for row in source_len..input.active_len {
        yk[row] = yk[row - 1] * attenuation;
    }

    let backward_trailing_weight = order * trailing_weight * step_exp;
    let backward_leading_weight = order * leading_weight / (step_exp * step_exp);
    let backward_attenuation = attenuation / step_exp;
    let backward_middle_weight = order * middle_weight;
    for row_1based in (2..source_len).rev() {
        let row = row_1based - 1;
        yk[row] = yk[row + 1] * backward_attenuation
            + (backward_middle_weight * (zk[row] + zk[row + 1] * backward_attenuation)
                - (zk[row + 2] * backward_leading_weight + zk[row - 1] * backward_trailing_weight));
    }

    let attenuation_squared = backward_attenuation * backward_attenuation;
    let first_weight = 8.0 * backward_middle_weight / 13.0;
    yk[0] = yk[2] * attenuation_squared
        + first_weight * (zk[2] * attenuation_squared + 4.0 * backward_attenuation * zk[1] + zk[0]);
    origin_constant = (origin_constant + yk[0]) / input.radii[0].powi(k_plus_one_i32);

    validate_finite_scalar("yk_zk_origin_constant", origin_constant)?;
    for row in 0..input.active_len {
        validate_finite_scalar("yk", yk[row])?;
        validate_finite_scalar("zk", zk[row])?;
    }
    for coefficient in 0..input.coefficient_count {
        validate_finite_scalar("yk_coefficient", yk_coefficients[coefficient])?;
        validate_finite_scalar("zk_coefficient", zk_coefficients[coefficient])?;
    }

    Ok(AtomicYkZkTransform {
        yk,
        zk,
        yk_coefficients,
        zk_coefficients,
        origin_constant,
        computed_source_len: source_len,
    })
}

fn validate_yk_zk_denominator(field: &'static str, value: Real) -> Result<(), AtomMathError> {
    if value == 0.0 {
        Err(AtomMathError::ZeroYkZkDenominator { field })
    } else {
        validate_finite_scalar(field, value)
    }
}

fn calculate_atomic_yk_zk_exchange(
    input: AtomicYkZkExchangeInput<'_>,
) -> Result<AtomicYkZkTransform, AtomMathError> {
    let left =
        one_based_atomic_orbital_index(input.left_orbital_1based, input.active_lengths.len())?;
    let right =
        one_based_atomic_orbital_index(input.right_orbital_1based, input.active_lengths.len())?;
    let source_len = input.active_lengths[left].min(input.active_lengths[right]);
    let source = Array1::from_shape_fn(input.radii.len(), |row| {
        if row < source_len {
            if input.large_small {
                input.large_components[(row, left)] * input.small_components[(row, right)]
            } else {
                input.large_components[(row, left)] * input.large_components[(row, right)]
                    + input.small_components[(row, left)] * input.small_components[(row, right)]
            }
        } else {
            0.0
        }
    });

    let left_large_coefficients = input.large_coefficients.index_axis(Axis(1), left);
    let right_large_coefficients = input.large_coefficients.index_axis(Axis(1), right);
    let left_small_coefficients = input.small_coefficients.index_axis(Axis(1), left);
    let right_small_coefficients = input.small_coefficients.index_axis(Axis(1), right);
    let coefficient_count = input.large_coefficients.nrows();
    let coefficients = (1..=coefficient_count)
        .map(|term| -> Result<Real, AtomMathError> {
            if input.large_small {
                polynomial_product_coefficient_view(
                    left_large_coefficients,
                    right_small_coefficients,
                    term,
                )
            } else {
                Ok(polynomial_product_coefficient_view(
                    left_large_coefficients,
                    right_large_coefficients,
                    term,
                )? + polynomial_product_coefficient_view(
                    left_small_coefficients,
                    right_small_coefficients,
                    term,
                )?)
            }
        })
        .collect::<Result<Vec<_>, _>>()?;
    let source_coefficients = Array1::from_vec(coefficients);

    atomic_yk_zk_transform(AtomicYkZkTransformInput {
        source: source.view(),
        source_coefficients: source_coefficients.view(),
        radii: input.radii,
        initial_power: input.orbital_powers[left] + input.orbital_powers[right],
        step: input.step,
        angular_momentum: input.angular_momentum,
        coefficient_count,
        source_len,
        active_len: input.radii.len(),
    })
}

fn calculate_atomic_radial_integral(
    input: AtomicRadialIntegralInput<'_>,
) -> Result<AtomicRadialIntegral, AtomMathError> {
    let first_factor = if input.request.first_left > 0 && input.request.first_right > 0 {
        Some(calculate_atomic_radial_first_factor(&input)?)
    } else {
        None
    };

    if input.request.second_left == 0 || input.request.second_right == 0 {
        return Ok(AtomicRadialIntegral {
            value: 0.0,
            first_factor,
        });
    }

    let value = match first_factor.as_ref() {
        Some(first_factor) => {
            calculate_atomic_radial_second_integral(&input, first_factor.as_view())?
        }
        None => calculate_atomic_radial_second_integral(
            &input,
            input
                .previous_first_factor
                .ok_or(AtomMathError::MissingRadialFirstFactor)?,
        )?,
    };

    Ok(AtomicRadialIntegral {
        value,
        first_factor,
    })
}

fn calculate_atomic_radial_second_integral(
    input: &AtomicRadialIntegralInput<'_>,
    first_factor: AtomicRadialFirstFactorView<'_>,
) -> Result<Real, AtomMathError> {
    let zero_derivative = Array1::<Real>::zeros(input.radii.len());
    let zero_coefficients = Array1::<Real>::zeros(input.large_coefficients.nrows());
    let kind = if input.large_small {
        AtomicDifferentialIntegralKind::LargeSmallOverlap {
            left_orbital_1based: input.request.second_left,
            right_orbital_1based: input.request.second_right,
            multiply_by_derivative: true,
        }
    } else {
        AtomicDifferentialIntegralKind::ComponentOverlap {
            left_orbital_1based: input.request.second_left,
            right_orbital_1based: input.request.second_right,
            multiply_by_derivative: true,
        }
    };
    atomic_differential_integral(AtomicDifferentialIntegralInput {
        kind,
        power: -1,
        origin_power: first_factor.origin_power,
        step: input.step,
        radii: input.radii,
        active_lengths: input.active_lengths,
        orbital_powers: input.orbital_powers,
        large_components: input.large_components,
        small_components: input.small_components,
        large_coefficients: input.large_coefficients,
        small_coefficients: input.small_coefficients,
        derivative_large: first_factor.values,
        derivative_small: zero_derivative.view(),
        derivative_large_coefficients: first_factor.coefficients,
        derivative_small_coefficients: zero_coefficients.view(),
    })
}

fn calculate_atomic_radial_first_factor(
    input: &AtomicRadialIntegralInput<'_>,
) -> Result<AtomicRadialFirstFactor, AtomMathError> {
    let transform = atomic_yk_zk_exchange(AtomicYkZkExchangeInput {
        left_orbital_1based: input.request.first_left,
        right_orbital_1based: input.request.first_right,
        large_small: input.large_small,
        angular_momentum: input.request.rank,
        step: input.step,
        radii: input.radii,
        active_lengths: input.active_lengths,
        orbital_powers: input.orbital_powers,
        large_components: input.large_components,
        small_components: input.small_components,
        large_coefficients: input.large_coefficients,
        small_coefficients: input.small_coefficients,
    })?;

    let left = one_based_atomic_orbital_index(input.request.first_left, input.kappas.len())?;
    let right = one_based_atomic_orbital_index(input.request.first_right, input.kappas.len())?;
    let left_abs = abs_kappa_i32(input.kappas[left])? as usize;
    let right_abs = abs_kappa_i32(input.kappas[right])? as usize;
    let abs_sum = left_abs
        .checked_add(right_abs)
        .ok_or(AtomMathError::RadialIntegralIndexOutOfRange)?;
    let shifted_start = abs_sum.saturating_sub(input.request.rank).max(1);

    let coefficient_count = transform.yk_coefficients.len();
    let mut coefficients = Array1::<Real>::zeros(coefficient_count);
    for (source, coefficient) in transform.yk_coefficients.iter().copied().enumerate() {
        let target_1based = shifted_start
            .checked_add(source)
            .ok_or(AtomMathError::RadialIntegralIndexOutOfRange)?;
        if target_1based <= coefficient_count {
            coefficients[target_1based - 1] = -coefficient;
        }
    }
    coefficients[0] += transform.origin_constant;

    let origin_power = (input.request.rank as Real) + 1.0;
    validate_finite_scalar("radial_first_factor_origin_power", origin_power)?;
    validate_finite_vector("radial_first_factor", transform.yk.view())?;
    validate_finite_vector("radial_first_factor_coefficient", coefficients.view())?;

    Ok(AtomicRadialFirstFactor {
        values: transform.yk,
        coefficients,
        origin_power,
    })
}

impl<F> AtomicSchmidtContext<'_, F>
where
    F: for<'request> FnMut(AtomicSchmidtIntegralRequest<'request>) -> Result<Real, AtomMathError>,
{
    fn calculate(&mut self) -> Result<AtomicSchmidtOrthogonalization, AtomMathError> {
        let mut large_components = self.input.large_components.to_owned();
        let mut small_components = self.input.small_components.to_owned();
        let mut large_coefficients = self.input.large_coefficients.to_owned();
        let mut small_coefficients = self.input.small_coefficients.to_owned();
        let mut active_lengths = self.input.active_lengths.to_vec();

        {
            let mut tables = AtomicSchmidtTables {
                large_components: &mut large_components,
                small_components: &mut small_components,
                large_coefficients: &mut large_coefficients,
                small_coefficients: &mut small_coefficients,
                active_lengths: &mut active_lengths,
            };

            if let Some(active_orbital_1based) = self.input.active_orbital_1based {
                let target = active_orbital_1based - 1;
                self.orthogonalize_orbital(target, self.input.kappas.len(), &mut tables)?;
            } else {
                for target in 1..self.input.kappas.len() {
                    self.orthogonalize_orbital(target, target, &mut tables)?;
                }
            }
        }

        Ok(AtomicSchmidtOrthogonalization {
            large_components,
            small_components,
            large_coefficients,
            small_coefficients,
            active_lengths,
        })
    }

    fn orthogonalize_orbital(
        &mut self,
        target: usize,
        reference_limit: usize,
        tables: &mut AtomicSchmidtTables<'_>,
    ) -> Result<(), AtomMathError> {
        let radial_rows = tables.large_components.nrows();
        let coefficient_rows = tables.large_coefficients.nrows();
        let mut active_len = tables.active_lengths[target];
        let mut work_large = Array1::<Real>::zeros(radial_rows);
        let mut work_small = Array1::<Real>::zeros(radial_rows);
        let mut work_large_coefficients = tables
            .large_coefficients
            .index_axis(Axis(1), target)
            .to_owned();
        let mut work_small_coefficients = tables
            .small_coefficients
            .index_axis(Axis(1), target)
            .to_owned();

        for row in 0..active_len {
            work_large[row] = tables.large_components[(row, target)];
            work_small[row] = tables.small_components[(row, target)];
        }

        for reference in 0..reference_limit {
            if reference == target || self.input.kappas[reference] != self.input.kappas[target] {
                continue;
            }
            let reference_len = tables.active_lengths[reference];
            let projection = self.projection(AtomicSchmidtProjectionInput {
                target,
                reference,
                active_len: reference_len,
                work_large: &work_large,
                work_small: &work_small,
                work_large_coefficients: &work_large_coefficients,
                work_small_coefficients: &work_small_coefficients,
                large_components: tables.large_components.view(),
                small_components: tables.small_components.view(),
                large_coefficients: tables.large_coefficients.view(),
                small_coefficients: tables.small_coefficients.view(),
            })?;

            for row in 0..reference_len {
                work_large[row] -= projection * tables.large_components[(row, reference)];
                work_small[row] -= projection * tables.small_components[(row, reference)];
            }
            for coefficient in 0..coefficient_rows {
                work_large_coefficients[coefficient] -=
                    projection * tables.large_coefficients[(coefficient, reference)];
                work_small_coefficients[coefficient] -=
                    projection * tables.small_coefficients[(coefficient, reference)];
            }
            active_len = active_len.max(reference_len);
        }

        tables.active_lengths[target] = active_len;
        let norm = self.norm(
            target,
            active_len,
            &work_large,
            &work_small,
            &work_large_coefficients,
            &work_small_coefficients,
        )?;
        if !norm.is_finite() || norm <= 0.0 {
            return Err(AtomMathError::NonPositiveNorm {
                orbital_1based: target + 1,
                norm,
            });
        }
        let scale = norm.sqrt();
        validate_finite_scalar("schmidt_norm_scale", scale)?;

        for row in 0..active_len {
            tables.large_components[(row, target)] = work_large[row] / scale;
            tables.small_components[(row, target)] = work_small[row] / scale;
            validate_finite_scalar(
                "schmidt_large_component",
                tables.large_components[(row, target)],
            )?;
            validate_finite_scalar(
                "schmidt_small_component",
                tables.small_components[(row, target)],
            )?;
        }
        for coefficient in 0..coefficient_rows {
            tables.large_coefficients[(coefficient, target)] =
                work_large_coefficients[coefficient] / scale;
            tables.small_coefficients[(coefficient, target)] =
                work_small_coefficients[coefficient] / scale;
            validate_finite_scalar(
                "schmidt_large_coefficient",
                tables.large_coefficients[(coefficient, target)],
            )?;
            validate_finite_scalar(
                "schmidt_small_coefficient",
                tables.small_coefficients[(coefficient, target)],
            )?;
        }
        Ok(())
    }

    fn projection(
        &mut self,
        input: AtomicSchmidtProjectionInput<'_>,
    ) -> Result<Real, AtomMathError> {
        let value = {
            let work_large_view = input.work_large.view();
            let work_small_view = input.work_small.view();
            let reference_large_column =
                input.large_components.index_axis(Axis(1), input.reference);
            let reference_small_column =
                input.small_components.index_axis(Axis(1), input.reference);
            let target_large = work_large_view.slice_axis(Axis(0), Slice::from(..input.active_len));
            let target_small = work_small_view.slice_axis(Axis(0), Slice::from(..input.active_len));
            let reference_large =
                reference_large_column.slice_axis(Axis(0), Slice::from(..input.active_len));
            let reference_small =
                reference_small_column.slice_axis(Axis(0), Slice::from(..input.active_len));
            let request = AtomicSchmidtProjectionRequest {
                target_orbital: input.target,
                reference_orbital: input.reference,
                target_power: self.input.orbital_powers[input.target],
                target_large,
                target_small,
                target_large_coefficients: input.work_large_coefficients.view(),
                target_small_coefficients: input.work_small_coefficients.view(),
                reference_large,
                reference_small,
                reference_large_coefficients: input
                    .large_coefficients
                    .index_axis(Axis(1), input.reference),
                reference_small_coefficients: input
                    .small_coefficients
                    .index_axis(Axis(1), input.reference),
            };
            (self.overlap_integral)(AtomicSchmidtIntegralRequest::Projection(request))?
        };
        validate_finite_scalar("schmidt_projection", value)?;
        Ok(value)
    }

    fn norm(
        &mut self,
        target: usize,
        active_len: usize,
        work_large: &Array1<Real>,
        work_small: &Array1<Real>,
        work_large_coefficients: &Array1<Real>,
        work_small_coefficients: &Array1<Real>,
    ) -> Result<Real, AtomMathError> {
        let value = {
            let work_large_view = work_large.view();
            let work_small_view = work_small.view();
            let target_large = work_large_view.slice_axis(Axis(0), Slice::from(..active_len));
            let target_small = work_small_view.slice_axis(Axis(0), Slice::from(..active_len));
            let request = AtomicSchmidtNormRequest {
                target_orbital: target,
                active_len,
                target_power: self.input.orbital_powers[target],
                target_large,
                target_small,
                target_large_coefficients: work_large_coefficients.view(),
                target_small_coefficients: work_small_coefficients.view(),
            };
            (self.overlap_integral)(AtomicSchmidtIntegralRequest::Norm(request))?
        };
        validate_finite_scalar("schmidt_norm", value)?;
        Ok(value)
    }
}

impl<F> AtomicTotalEnergyContext<'_, F>
where
    F: FnMut(AtomicRadialIntegralRequest) -> Result<Real, AtomMathError>,
{
    fn calculate(&mut self) -> Result<AtomicTotalEnergy, AtomMathError> {
        let direct_coulomb = self.direct_coulomb_energy()?;
        let exchange_coulomb = self.exchange_coulomb_energy()?;
        let (magnetic_breit, retarded_breit) = self.breit_energies()?;
        let orbital_energy = self
            .input
            .orbital_energies
            .iter()
            .zip(self.input.occupations)
            .map(|(&energy, &occupation)| energy * occupation)
            .sum::<Real>();
        let total =
            -(direct_coulomb + exchange_coulomb) + magnetic_breit + retarded_breit + orbital_energy;

        Ok(AtomicTotalEnergy {
            total,
            direct_coulomb,
            exchange_coulomb,
            magnetic_breit,
            retarded_breit,
        })
    }

    fn direct_coulomb_energy(&mut self) -> Result<Real, AtomMathError> {
        let mut energy = 0.0;
        for left in 0..self.orbital_count() {
            let left_l = self.abs_kappa(left)? - 1;
            for right in 0..=left {
                let symmetry_weight = if right == left { 2.0 } else { 1.0 };
                let right_l = self.abs_kappa(right)? - 1;
                let max_rank = 2 * left_l.min(right_l);
                let mut rank = 0;
                while rank <= max_rank {
                    let radial = self.radial(left + 1, left + 1, right + 1, right + 1, rank)?;
                    energy +=
                        radial * self.direct_coefficient(left, right, rank)? / symmetry_weight;
                    rank += 2;
                }
            }
        }
        Ok(energy)
    }

    fn exchange_coulomb_energy(&mut self) -> Result<Real, AtomMathError> {
        let mut energy = 0.0;
        for left in 1..self.orbital_count() {
            let valence_weight = if self.input.valence_occupations[left] > 0.0 {
                0.5
            } else {
                1.0
            };
            for right in 0..left {
                if self.input.valence_occupations[right] > 0.0 {
                    continue;
                }
                let left_abs = self.abs_kappa(left)?;
                let right_abs = self.abs_kappa(right)?;
                let mut rank = left_abs.abs_diff(right_abs);
                if self.kappa(left).signum() != self.kappa(right).signum() {
                    rank += 1;
                }
                let max_rank = left_abs + right_abs - 1;
                while rank <= max_rank {
                    let radial = self.radial(left + 1, right + 1, left + 1, right + 1, rank)?;
                    energy -=
                        radial * self.exchange_coefficient(left, right, rank)? * valence_weight;
                    rank += 2;
                }
            }
        }
        Ok(energy)
    }

    fn breit_energies(&mut self) -> Result<(Real, Real), AtomMathError> {
        let mut magnetic = 0.0;
        let mut retarded = 0.0;
        for right in 0..self.orbital_count() {
            let right_j2 = self.j2(right)?;
            for left in 0..=right {
                let left_j2 = self.j2(left)?;
                let max_rank = left_j2.min(right_j2);
                let mut rank = 1;
                while rank <= max_rank {
                    let radial = self.radial(right + 1, right + 1, left + 1, left + 1, rank)?;
                    if left == right {
                        let coefficients = atomic_breit_angular_coefficients(
                            self.kappa(right),
                            self.kappa(right),
                            rank,
                        )?;
                        let occupation = atomic_occupation_product(
                            self.input.occupations,
                            self.input.kappas,
                            right,
                            right,
                        )?;
                        magnetic +=
                            coefficients.magnetic.iter().sum::<Real>() * radial * occupation / 2.0;
                    }
                    rank += 2;
                }
            }
        }

        for right in 1..self.orbital_count() {
            let right_branch = self.exchange_breit_branch(right)?;
            for left in 0..right {
                let left_branch = self.exchange_breit_branch(left)?;
                let occupation = atomic_occupation_product(
                    self.input.occupations,
                    self.input.kappas,
                    right,
                    left,
                )?;
                let mut rank = left_branch.minimum_rank(right_branch)?;
                let max_rank = left_branch.maximum_rank(right_branch)?;
                let parity_sum = i32::try_from(rank)
                    .map_err(|_| AtomMathError::BreitRankOutOfRange { rank })?
                    .checked_add(left_branch.angular_l)
                    .and_then(|value| value.checked_add(right_branch.angular_l))
                    .ok_or(AtomMathError::BreitBranchOutOfRange)?;
                if parity_sum % 2 == 0 {
                    rank += 1;
                }
                let kappa_sum = self.abs_kappa(right)? + self.abs_kappa(left)?;
                while rank <= max_rank {
                    let coefficients = atomic_breit_angular_coefficients(
                        self.kappa(right),
                        self.kappa(left),
                        rank,
                    )?;
                    let radials = self.exchange_breit_radials(left, right, rank, kappa_sum)?;
                    magnetic += coefficients
                        .magnetic
                        .iter()
                        .zip(radials)
                        .map(|(&coefficient, radial)| coefficient * radial * occupation)
                        .sum::<Real>();
                    retarded += coefficients
                        .retarded
                        .iter()
                        .zip(radials)
                        .map(|(&coefficient, radial)| coefficient * radial * occupation)
                        .sum::<Real>();
                    rank += 2;
                }
            }
        }

        Ok((magnetic, retarded))
    }

    fn exchange_breit_radials(
        &mut self,
        left: usize,
        right: usize,
        rank: usize,
        kappa_sum: usize,
    ) -> Result<[Real; 3], AtomMathError> {
        let mut radials = [0.0; 3];
        if !(kappa_sum <= rank && self.kappa(left) < 0 && self.kappa(right) > 0) {
            radials[0] = self.radial(left + 1, right + 1, left + 1, right + 1, rank)?;
            radials[1] = self.radial(0, 0, right + 1, left + 1, rank)?;
        }
        if !(kappa_sum <= rank && self.kappa(left) > 0 && self.kappa(right) < 0) {
            radials[2] = self.radial(right + 1, left + 1, right + 1, left + 1, rank)?;
            if radials[1] == 0.0 {
                radials[1] = self.radial(0, 0, left + 1, right + 1, rank)?;
            }
        }
        Ok(radials)
    }

    fn radial(
        &mut self,
        first_left: usize,
        first_right: usize,
        second_left: usize,
        second_right: usize,
        rank: usize,
    ) -> Result<Real, AtomMathError> {
        let request = AtomicRadialIntegralRequest {
            first_left,
            first_right,
            second_left,
            second_right,
            rank,
        };
        let value = (self.radial_integral)(request)?;
        validate_finite_scalar("radial_integral", value)?;
        Ok(value)
    }

    fn direct_coefficient(
        &self,
        left: usize,
        right: usize,
        rank: usize,
    ) -> Result<Real, AtomMathError> {
        let channel = self.coefficient_channel(rank)?;
        if left <= right {
            Ok(self.input.coulomb_coefficients[(left, right, channel)])
        } else {
            Ok(self.input.coulomb_coefficients[(right, left, channel)])
        }
    }

    fn exchange_coefficient(
        &self,
        left: usize,
        right: usize,
        rank: usize,
    ) -> Result<Real, AtomMathError> {
        let channel = self.coefficient_channel(rank)?;
        if left < right {
            Ok(self.input.coulomb_coefficients[(right, left, channel)])
        } else if left > right {
            Ok(self.input.coulomb_coefficients[(left, right, channel)])
        } else {
            Ok(0.0)
        }
    }

    fn coefficient_channel(&self, rank: usize) -> Result<usize, AtomMathError> {
        let channel = rank / 2;
        let channels = self.input.coulomb_coefficients.shape()[2];
        if channel >= channels {
            Err(AtomMathError::CoefficientChannelOutOfRange {
                rank,
                channel,
                channels,
            })
        } else {
            Ok(channel)
        }
    }

    fn exchange_breit_branch(&self, orbital: usize) -> Result<BreitExchangeBranch, AtomMathError> {
        let mut angular_l = abs_kappa_i32(self.kappa(orbital))?;
        let mut sign_shift = -1;
        if self.kappa(orbital) < 0 {
            sign_shift = 1;
            angular_l -= 1;
        }
        Ok(BreitExchangeBranch {
            angular_l,
            sign_shift,
        })
    }

    fn orbital_count(&self) -> usize {
        self.input.kappas.len()
    }

    fn kappa(&self, orbital: usize) -> i32 {
        self.input.kappas[orbital]
    }

    fn abs_kappa(&self, orbital: usize) -> Result<usize, AtomMathError> {
        usize::try_from(abs_kappa_i32(self.kappa(orbital))?).map_err(|_| {
            AtomMathError::InvalidKappa {
                kappa: self.kappa(orbital),
            }
        })
    }

    fn j2(&self, orbital: usize) -> Result<usize, AtomMathError> {
        usize::try_from(doubled_j_from_kappa(self.kappa(orbital))?).map_err(|_| {
            AtomMathError::InvalidKappa {
                kappa: self.kappa(orbital),
            }
        })
    }
}

#[derive(Debug, Clone, Copy)]
struct BreitExchangeBranch {
    angular_l: i32,
    sign_shift: i32,
}

impl BreitExchangeBranch {
    fn minimum_rank(self, other: Self) -> Result<usize, AtomMathError> {
        let first = other
            .angular_l
            .checked_add(other.sign_shift)
            .and_then(|value| value.checked_sub(self.angular_l))
            .ok_or(AtomMathError::BreitBranchOutOfRange)?
            .unsigned_abs();
        let second = self
            .angular_l
            .checked_add(self.sign_shift)
            .and_then(|value| value.checked_sub(other.angular_l))
            .ok_or(AtomMathError::BreitBranchOutOfRange)?
            .unsigned_abs();
        usize::try_from(first.min(second)).map_err(|_| AtomMathError::BreitBranchOutOfRange)
    }

    fn maximum_rank(self, other: Self) -> Result<usize, AtomMathError> {
        let first = self
            .angular_l
            .checked_add(other.angular_l)
            .and_then(|value| value.checked_add(other.sign_shift))
            .ok_or(AtomMathError::BreitBranchOutOfRange)?;
        let second = self
            .angular_l
            .checked_add(other.angular_l)
            .and_then(|value| value.checked_add(self.sign_shift))
            .ok_or(AtomMathError::BreitBranchOutOfRange)?;
        usize::try_from(first.max(second)).map_err(|_| AtomMathError::BreitBranchOutOfRange)
    }
}

#[derive(Debug, Clone, Copy)]
struct BreitOrderContext {
    left_j2: i32,
    right_j2: i32,
    kappa_difference: i32,
    kappa_sum: i32,
    rank: i32,
    rank_usize: usize,
    angular_l: i32,
    order: usize,
}

#[derive(Debug, Clone, Copy)]
struct BreitOrderTerms {
    cm: Real,
    cz: Real,
    cp: Real,
    d: Real,
    retardation: Option<BreitRetardationTerms>,
}

#[derive(Debug, Clone, Copy)]
struct BreitRetardationTerms {
    am: Real,
    az: Real,
    ap: Real,
    scale: Real,
}

fn accumulate_breit_order(
    context: BreitOrderContext,
    coefficients: &mut AtomicBreitAngularCoefficients,
) -> Result<(), AtomMathError> {
    let wigner_j3 = context
        .angular_l
        .checked_mul(2)
        .ok_or(AtomMathError::BreitRankOutOfRange {
            rank: context.rank_usize,
        })?;
    let wigner = wigner_3j(context.left_j2, context.right_j2, wigner_j3, -1, 1, 2)?;
    if wigner == 0.0 {
        return Ok(());
    }

    let angular_denominator = context
        .angular_l
        .checked_mul(2)
        .and_then(|value| value.checked_add(1))
        .ok_or(AtomMathError::BreitRankOutOfRange {
            rank: context.rank_usize,
        })?;
    let squared_wigner = wigner * wigner;
    let terms = breit_order_terms(context, angular_denominator);

    if let Some(retardation) = terms.retardation {
        let mut retardation_scale = retardation.scale;
        let denominator = Real::from(angular_denominator.abs()) * terms.d;
        if denominator != 0.0 {
            retardation_scale /= denominator;
        }
        coefficients.retarded[0] +=
            squared_wigner * (retardation.am - retardation_scale * terms.cm);
        coefficients.retarded[1] +=
            (squared_wigner + squared_wigner) * (retardation.az - retardation_scale * terms.cz);
        coefficients.retarded[2] +=
            squared_wigner * (retardation.ap - retardation_scale * terms.cp);
    }

    if terms.d != 0.0 {
        let magnetic_scale = squared_wigner / terms.d;
        coefficients.magnetic[0] += terms.cm * magnetic_scale;
        coefficients.magnetic[1] += terms.cz * (magnetic_scale + magnetic_scale);
        coefficients.magnetic[2] += terms.cp * magnetic_scale;
    }

    Ok(())
}

fn breit_order_terms(context: BreitOrderContext, angular_denominator: i32) -> BreitOrderTerms {
    match context.order {
        0 => {
            let cm = square(context.kappa_difference + context.rank);
            let cz = square(context.kappa_difference) - square(context.rank);
            let cp = square(context.rank - context.kappa_difference);
            let scale = Real::from(context.rank);
            let retardation = breit_retardation_shape(
                context.kappa_difference,
                context.angular_l,
                angular_denominator,
                scale,
            );
            BreitOrderTerms {
                cm,
                cz,
                cp,
                d: scale * Real::from(context.rank + context.rank + 1),
                retardation: Some(retardation),
            }
        }
        1 => {
            let cm = square(context.kappa_sum);
            BreitOrderTerms {
                cm,
                cz: cm,
                cp: cm,
                d: Real::from(context.rank) * Real::from(context.rank + 1),
                retardation: None,
            }
        }
        _ => {
            let cm = square(context.kappa_difference - context.angular_l);
            let cz = square(context.kappa_difference) - square(context.angular_l);
            let cp = square(context.kappa_difference + context.angular_l);
            let scale = Real::from(context.angular_l);
            let retardation = breit_retardation_shape(
                context.kappa_difference,
                context.angular_l,
                -angular_denominator,
                scale,
            );
            BreitOrderTerms {
                cm,
                cz,
                cp,
                d: scale * Real::from(context.rank + context.rank + 1),
                retardation: Some(retardation),
            }
        }
    }
}

fn breit_retardation_shape(
    kappa_difference: i32,
    angular_l: i32,
    denominator: i32,
    scale: Real,
) -> BreitRetardationTerms {
    let next_l = angular_l + 1;
    let denominator = Real::from(denominator);
    BreitRetardationTerms {
        am: Real::from((kappa_difference - angular_l) * (kappa_difference + next_l)) / denominator,
        az: Real::from(kappa_difference * kappa_difference + angular_l * next_l) / denominator,
        ap: Real::from((angular_l + kappa_difference) * (kappa_difference - next_l)) / denominator,
        scale,
    }
}

fn square(value: i32) -> Real {
    let value = Real::from(value);
    value * value
}

fn doubled_j_from_kappa(kappa: i32) -> Result<i32, AtomMathError> {
    abs_kappa_i32(kappa)?
        .checked_mul(2)
        .and_then(|value| value.checked_sub(1))
        .ok_or(AtomMathError::InvalidKappa { kappa })
}

fn doubled_j_usize_from_kappa(kappa: i32) -> Result<usize, AtomMathError> {
    usize::try_from(doubled_j_from_kappa(kappa)?).map_err(|_| AtomMathError::InvalidKappa { kappa })
}

fn kappa_angular_rank(kappa: i32) -> Result<usize, AtomMathError> {
    doubled_j_usize_from_kappa(kappa)
}

fn kappa_abs_usize(kappa: i32) -> Result<usize, AtomMathError> {
    usize::try_from(abs_kappa_i32(kappa)?).map_err(|_| AtomMathError::InvalidKappa { kappa })
}

fn atom_usize_to_i32(value: usize) -> Result<i32, AtomMathError> {
    i32::try_from(value).map_err(|_| AtomMathError::CoulombRankOutOfRange { rank: value })
}

fn direct_coulomb_coefficient_at(
    coefficients: ArrayView3<'_, Real>,
    left: usize,
    right: usize,
    rank: usize,
) -> Result<Real, AtomMathError> {
    let channel = coefficient_channel(coefficients, rank)?;
    if left <= right {
        Ok(coefficients[(left, right, channel)])
    } else {
        Ok(coefficients[(right, left, channel)])
    }
}

fn exchange_coulomb_coefficient_at(
    coefficients: ArrayView3<'_, Real>,
    left: usize,
    right: usize,
    rank: usize,
) -> Result<Real, AtomMathError> {
    let channel = coefficient_channel(coefficients, rank)?;
    if left < right {
        Ok(coefficients[(right, left, channel)])
    } else if left > right {
        Ok(coefficients[(left, right, channel)])
    } else {
        Ok(0.0)
    }
}

fn coefficient_channel(
    coefficients: ArrayView3<'_, Real>,
    rank: usize,
) -> Result<usize, AtomMathError> {
    let channel = rank / 2;
    let channels = coefficients.shape()[2];
    if channel >= channels {
        Err(AtomMathError::CoefficientChannelOutOfRange {
            rank,
            channel,
            channels,
        })
    } else {
        Ok(channel)
    }
}

fn orbital_pair_count(orbital_count: usize) -> Result<usize, AtomMathError> {
    orbital_count
        .checked_mul(orbital_count.saturating_sub(1))
        .map(|count| count / 2)
        .ok_or(AtomMathError::OrbitalPairTableTooLarge { orbital_count })
}

fn packed_orbital_pair_index(first: usize, second: usize) -> Result<usize, AtomMathError> {
    let lower = first.min(second);
    let upper = first.max(second);
    upper
        .checked_mul(upper.saturating_sub(1))
        .map(|value| value / 2)
        .and_then(|value| value.checked_add(lower))
        .ok_or(AtomMathError::OrbitalPairTableTooLarge {
            orbital_count: upper.saturating_add(1),
        })
}

fn significant_relative_difference(difference: Real, reference: Real) -> bool {
    let relative = if reference == 0.0 {
        difference
    } else {
        difference / reference
    };
    relative.abs() >= 1.0e-7
}

fn atom_tabrat_orbital_label(kappa: i32) -> Result<&'static str, AtomMathError> {
    abs_kappa_i32(kappa)?;
    let title_index = if kappa > 0 {
        kappa.checked_mul(2)
    } else {
        kappa.checked_mul(-2).and_then(|value| value.checked_sub(1))
    }
    .ok_or(AtomMathError::OrbitalLabelKappaOutOfRange { kappa })?;
    let label_index = usize::try_from(title_index - 1)
        .map_err(|_| AtomMathError::OrbitalLabelKappaOutOfRange { kappa })?;
    ATOM_TABRAT_LABELS
        .get(label_index)
        .copied()
        .ok_or(AtomMathError::OrbitalLabelKappaOutOfRange { kappa })
}

fn fpf0_dipole_multipliers(
    initial_kappa: i32,
    final_kappa: i32,
) -> Result<Option<(Real, Real)>, AtomMathError> {
    abs_kappa_i32(initial_kappa)?;
    abs_kappa_i32(final_kappa)?;
    let kappa_sum =
        initial_kappa
            .checked_add(final_kappa)
            .ok_or(AtomMathError::KappaDifferenceOutOfRange {
                left_kappa: initial_kappa,
                right_kappa: final_kappa,
            })?;
    let mut kappa_difference =
        final_kappa
            .checked_sub(initial_kappa)
            .ok_or(AtomMathError::KappaDifferenceOutOfRange {
                left_kappa: initial_kappa,
                right_kappa: final_kappa,
            })?;
    let difference_abs =
        kappa_difference
            .checked_abs()
            .ok_or(AtomMathError::KappaDifferenceOutOfRange {
                left_kappa: initial_kappa,
                right_kappa: final_kappa,
            })?;
    if kappa_sum != 0 && difference_abs != 1 {
        return Ok(None);
    }
    if difference_abs > 1 {
        kappa_difference = 0;
    }

    let two_j = 2.0 * Real::from(abs_kappa_i32(initial_kappa)?) - 1.0;
    let multipliers = match (kappa_difference, initial_kappa.is_positive()) {
        (-1, true) => (0.0, (2.0 * (two_j + 1.0) * (two_j - 1.0) / two_j).sqrt()),
        (-1, false) => (
            0.0,
            -(2.0 * (two_j + 1.0) * (two_j + 3.0) / (two_j + 2.0)).sqrt(),
        ),
        (0, true) => (
            -((two_j + 1.0) * two_j / (two_j + 2.0)).sqrt(),
            -((two_j + 1.0) * (two_j + 2.0) / two_j).sqrt(),
        ),
        (0, false) => (
            ((two_j + 1.0) * (two_j + 2.0) / two_j).sqrt(),
            ((two_j + 1.0) * two_j / (two_j + 2.0)).sqrt(),
        ),
        (1, true) => (
            (2.0 * (two_j + 1.0) * (two_j + 3.0) / (two_j + 2.0)).sqrt(),
            0.0,
        ),
        (1, false) => (-(2.0 * (two_j + 1.0) * (two_j - 1.0) / two_j).sqrt(), 0.0),
        _ => return Ok(None),
    };
    validate_finite_scalar("fpf0_large_multiplier", multipliers.0)?;
    validate_finite_scalar("fpf0_small_multiplier", multipliers.1)?;
    Ok(Some(multipliers))
}

fn fpf0_spherical_bessel_j0(argument: Real) -> Real {
    if argument == 0.0 {
        1.0
    } else {
        argument.sin() / argument
    }
}

fn abs_kappa_i32(kappa: i32) -> Result<i32, AtomMathError> {
    if kappa == 0 {
        return Err(AtomMathError::InvalidKappa { kappa });
    }
    kappa
        .checked_abs()
        .ok_or(AtomMathError::InvalidKappa { kappa })
}

fn s02at_overlap_matrix(group: &[usize], overlaps: ArrayView2<'_, Real>) -> Array2<Real> {
    let order = group.len();
    let mut matrix = Array2::zeros((order, order).f());
    for column in 0..order {
        for row in 0..=column {
            let value = overlaps[(group[row], group[column])];
            matrix[(row, column)] = value;
            matrix[(column, row)] = value;
        }
    }
    matrix
}

fn s02at_eliminate_hole(matrix: ArrayView2<'_, Real>, hole: usize) -> Array2<Real> {
    Array2::from_shape_fn((matrix.nrows(), matrix.ncols()).f(), |(row, column)| {
        if row == hole && column == hole {
            1.0
        } else if row == hole || column == hole {
            0.0
        } else {
            matrix[(row, column)]
        }
    })
}

fn s02at_squared_determinant_in_place(matrix: &mut Array2<Real>, order: usize) -> Real {
    let determinant = s02at_determinant_in_place(matrix, order);
    determinant * determinant
}

fn s02at_determinant_in_place(matrix: &mut Array2<Real>, order: usize) -> Real {
    let mut determinant = 1.0;
    for pivot in 0..order {
        if matrix[(pivot, pivot)] == 0.0 {
            let Some(swap_column) = (pivot..order).find(|&column| matrix[(pivot, column)] != 0.0)
            else {
                return 0.0;
            };
            for row in pivot..order {
                let saved = matrix[(row, swap_column)];
                matrix[(row, swap_column)] = matrix[(row, pivot)];
                matrix[(row, pivot)] = saved;
            }
            determinant = -determinant;
        }

        determinant *= matrix[(pivot, pivot)];
        if pivot + 1 < order {
            for row in (pivot + 1)..order {
                for column in (pivot + 1)..order {
                    matrix[(row, column)] -=
                        matrix[(row, pivot)] * matrix[(pivot, column)] / matrix[(pivot, pivot)];
                }
            }
        }
    }
    determinant
}

fn validate_overlap_amplitude_input(
    input: &AtomicOverlapAmplitudeReductionInput<'_>,
) -> Result<(), AtomMathError> {
    let orbital_count = input.kappas.len();
    validate_orbital_table_len("occupations", orbital_count, input.occupations.len())?;
    if let Some(hole_orbital_1based) = input.hole_orbital_1based
        && !(1..=orbital_count).contains(&hole_orbital_1based)
    {
        return Err(AtomMathError::HoleOrbitalOutOfRange {
            hole_orbital_1based,
            orbital_count,
        });
    }
    let [rows, columns] = input.overlap_integrals.shape().try_into().map_err(|_| {
        AtomMathError::OverlapMatrixShape {
            expected: orbital_count,
            rows: input.overlap_integrals.nrows(),
            columns: input.overlap_integrals.ncols(),
        }
    })?;
    if rows != orbital_count || columns != orbital_count {
        return Err(AtomMathError::OverlapMatrixShape {
            expected: orbital_count,
            rows,
            columns,
        });
    }
    for &kappa in input.kappas {
        abs_kappa_i32(kappa)?;
    }
    validate_finite_slice("occupation", input.occupations)?;
    for &value in &input.overlap_integrals {
        validate_finite_scalar("overlap_integral", value)?;
    }
    Ok(())
}

fn validate_total_energy_input(input: &AtomicTotalEnergyInput<'_>) -> Result<(), AtomMathError> {
    let orbital_count = input.kappas.len();
    if orbital_count == 0 {
        return Err(AtomMathError::EmptyOrbitalTable);
    }
    validate_orbital_table_len("occupations", orbital_count, input.occupations.len())?;
    validate_orbital_table_len(
        "valence_occupations",
        orbital_count,
        input.valence_occupations.len(),
    )?;
    validate_orbital_table_len(
        "orbital_energies",
        orbital_count,
        input.orbital_energies.len(),
    )?;
    validate_finite_slice("occupation", input.occupations)?;
    validate_finite_slice("valence_occupation", input.valence_occupations)?;
    validate_finite_slice("orbital_energy", input.orbital_energies)?;
    for &kappa in input.kappas {
        doubled_j_from_kappa(kappa)?;
    }
    validate_coefficient_table(
        input.coulomb_coefficients,
        orbital_count - 1,
        orbital_count - 1,
        0,
    )
}

fn validate_lagrange_parameters_input(
    input: &AtomicLagrangeParametersInput<'_>,
) -> Result<(), AtomMathError> {
    let orbital_count = input.kappas.len();
    if orbital_count == 0 {
        return Err(AtomMathError::EmptyOrbitalTable);
    }
    validate_orbital_table_len("occupations", orbital_count, input.occupations.len())?;
    validate_orbital_table_len("shell_markers", orbital_count, input.shell_markers.len())?;
    if let Some(active_orbital_1based) = input.active_orbital_1based
        && !(1..=orbital_count).contains(&active_orbital_1based)
    {
        return Err(AtomMathError::ActiveOrbitalOutOfRange {
            active_orbital_1based,
            orbital_count,
        });
    }
    validate_finite_slice("occupation", input.occupations)?;
    for &kappa in input.kappas {
        doubled_j_from_kappa(kappa)?;
    }
    validate_coefficient_table(
        input.coulomb_coefficients,
        orbital_count - 1,
        orbital_count - 1,
        0,
    )?;
    orbital_pair_count(orbital_count)?;
    Ok(())
}

fn validate_form_factor_input(input: &AtomicFormFactorInput<'_>) -> Result<(), AtomMathError> {
    if input.atomic_number == 0 {
        return Err(AtomMathError::InvalidFormFactorAtomicNumber {
            atomic_number: input.atomic_number,
        });
    }
    validate_finite_scalar("radial_step", input.radial_step)?;
    validate_finite_scalar("total_energy", input.total_energy)?;

    let radial_count = input.radii.len();
    validate_radial_table_len("density_4pi", radial_count, input.density_4pi.len())?;
    validate_radial_table_len(
        "initial_large_component",
        radial_count,
        input.initial_large_component.len(),
    )?;
    validate_radial_table_len(
        "initial_small_component",
        radial_count,
        input.initial_small_component.len(),
    )?;

    let orbital_count = input.kappas.len();
    if orbital_count == 0 {
        return Err(AtomMathError::EmptyOrbitalTable);
    }
    validate_orbital_table_len("occupations", orbital_count, input.occupations.len())?;
    validate_orbital_table_len(
        "orbital_energies",
        orbital_count,
        input.orbital_energies.len(),
    )?;
    if !(1..=orbital_count).contains(&input.hole_orbital_1based) {
        return Err(AtomMathError::HoleOrbitalOutOfRange {
            hole_orbital_1based: input.hole_orbital_1based,
            orbital_count,
        });
    }
    validate_matrix_shape(
        "large_components",
        input.large_components,
        radial_count,
        orbital_count,
    )?;
    validate_matrix_shape(
        "small_components",
        input.small_components,
        radial_count,
        orbital_count,
    )?;

    for &kappa in input.kappas {
        abs_kappa_i32(kappa)?;
    }
    validate_finite_vector("radius", input.radii)?;
    validate_finite_vector("density_4pi", input.density_4pi)?;
    validate_finite_vector("initial_large_component", input.initial_large_component)?;
    validate_finite_vector("initial_small_component", input.initial_small_component)?;
    validate_finite_slice("occupation", input.occupations)?;
    validate_finite_slice("orbital_energy", input.orbital_energies)?;
    validate_finite_matrix("large_component", input.large_components)?;
    validate_finite_matrix("small_component", input.small_components)?;
    Ok(())
}

fn validate_nuclear_potential_input(
    input: AtomicNuclearPotentialInput,
) -> Result<(), AtomMathError> {
    validate_positive_finite_nuclear_scalar("nuclear_charge", input.nuclear_charge)?;
    validate_positive_finite_nuclear_scalar("step", input.step)?;
    validate_positive_finite_nuclear_scalar(
        "first_radius_times_charge",
        input.first_radius_times_charge,
    )?;
    validate_nuclear_count("radial_count", input.radial_count, 1)?;
    validate_nuclear_count("coefficient_count", input.coefficient_count, 5)?;
    Ok(())
}

fn validate_differential_integral_input(
    input: &AtomicDifferentialIntegralInput<'_>,
) -> Result<(), AtomMathError> {
    validate_finite_scalar("dsordf_step", input.step)?;
    validate_finite_scalar("dsordf_origin_power", input.origin_power)?;
    if input.radii.is_empty() {
        return Err(AtomMathError::InvalidDifferentialIntegralActiveLength {
            active_len: 0,
            radial_count: 0,
        });
    }
    validate_positive_finite_radii(input.radii)?;

    let orbital_count = input.active_lengths.len();
    if orbital_count == 0 {
        return Err(AtomMathError::EmptyOrbitalTable);
    }
    validate_orbital_table_len("orbital_powers", orbital_count, input.orbital_powers.len())?;
    validate_finite_slice("orbital_power", input.orbital_powers)?;
    validate_matrix_shape(
        "large_components",
        input.large_components,
        input.radii.len(),
        orbital_count,
    )?;
    validate_matrix_shape(
        "small_components",
        input.small_components,
        input.radii.len(),
        orbital_count,
    )?;

    let coefficient_count = input.large_coefficients.nrows();
    validate_coefficient_count("large_coefficients", coefficient_count)?;
    validate_matrix_shape(
        "large_coefficients",
        input.large_coefficients,
        coefficient_count,
        orbital_count,
    )?;
    validate_matrix_shape(
        "small_coefficients",
        input.small_coefficients,
        coefficient_count,
        orbital_count,
    )?;
    validate_radial_table_len(
        "derivative_large",
        input.radii.len(),
        input.derivative_large.len(),
    )?;
    validate_radial_table_len(
        "derivative_small",
        input.radii.len(),
        input.derivative_small.len(),
    )?;
    validate_coefficient_vector_len(
        "derivative_large_coefficients",
        coefficient_count,
        input.derivative_large_coefficients.len(),
    )?;
    validate_coefficient_vector_len(
        "derivative_small_coefficients",
        coefficient_count,
        input.derivative_small_coefficients.len(),
    )?;

    validate_finite_matrix("large_component", input.large_components)?;
    validate_finite_matrix("small_component", input.small_components)?;
    validate_finite_matrix("large_coefficient", input.large_coefficients)?;
    validate_finite_matrix("small_coefficient", input.small_coefficients)?;
    validate_finite_vector("derivative_large", input.derivative_large)?;
    validate_finite_vector("derivative_small", input.derivative_small)?;
    validate_finite_vector(
        "derivative_large_coefficient",
        input.derivative_large_coefficients,
    )?;
    validate_finite_vector(
        "derivative_small_coefficient",
        input.derivative_small_coefficients,
    )?;

    match input.kind {
        AtomicDifferentialIntegralKind::ComponentOverlap {
            left_orbital_1based,
            right_orbital_1based,
            ..
        }
        | AtomicDifferentialIntegralKind::LargeSmallOverlap {
            left_orbital_1based,
            right_orbital_1based,
            ..
        } => {
            let left = one_based_atomic_orbital_index(left_orbital_1based, orbital_count)?;
            let right = one_based_atomic_orbital_index(right_orbital_1based, orbital_count)?;
            validate_differential_active_len(
                input.active_lengths[left].min(input.active_lengths[right]),
                input.radii.len(),
            )?;
        }
        AtomicDifferentialIntegralKind::DerivativeProjection {
            large_orbital_1based,
            small_orbital_1based,
        } => {
            let large = one_based_atomic_orbital_index(large_orbital_1based, orbital_count)?;
            let small = one_based_atomic_orbital_index(small_orbital_1based, orbital_count)?;
            validate_differential_active_len(
                input.active_lengths[large].min(input.active_lengths[small]),
                input.radii.len(),
            )?;
        }
        AtomicDifferentialIntegralKind::DerivativeNorm { active_len } => {
            validate_differential_active_len(active_len, input.radii.len())?;
        }
    }
    Ok(())
}

fn validate_yk_zk_transform_input(
    input: &AtomicYkZkTransformInput<'_>,
) -> Result<(), AtomMathError> {
    if input.active_len < 4 {
        return Err(AtomMathError::InvalidDifferentialIntegralActiveLength {
            active_len: input.active_len,
            radial_count: input.active_len,
        });
    }
    if input.source_len < 2 {
        return Err(AtomMathError::InvalidDifferentialIntegralActiveLength {
            active_len: input.source_len,
            radial_count: input.active_len,
        });
    }
    validate_coefficient_count("source_coefficients", input.coefficient_count)?;
    validate_radial_table_len("source", input.active_len, input.source.len())?;
    validate_radial_table_len("radii", input.active_len, input.radii.len())?;
    validate_coefficient_vector_len(
        "source_coefficients",
        input.coefficient_count,
        input.source_coefficients.len(),
    )?;
    validate_finite_scalar("yk_zk_initial_power", input.initial_power)?;
    validate_finite_scalar("yk_zk_step", input.step)?;
    if input.step == 0.0 {
        return Err(AtomMathError::ZeroYkZkDenominator { field: "step" });
    }
    if input.angular_momentum > i32::MAX as usize {
        return Err(AtomMathError::YkZkAngularMomentumOutOfRange {
            angular_momentum: input.angular_momentum,
        });
    }
    validate_positive_finite_radii(input.radii)?;
    validate_finite_vector("yk_zk_source", input.source)?;
    validate_finite_vector("yk_zk_source_coefficient", input.source_coefficients)?;
    Ok(())
}

fn validate_yk_zk_exchange_input(input: &AtomicYkZkExchangeInput<'_>) -> Result<(), AtomMathError> {
    validate_finite_scalar("yk_zk_step", input.step)?;
    let orbital_count = input.active_lengths.len();
    if orbital_count == 0 {
        return Err(AtomMathError::EmptyOrbitalTable);
    }
    validate_orbital_table_len("orbital_powers", orbital_count, input.orbital_powers.len())?;
    let left = one_based_atomic_orbital_index(input.left_orbital_1based, orbital_count)?;
    let right = one_based_atomic_orbital_index(input.right_orbital_1based, orbital_count)?;
    validate_positive_finite_radii(input.radii)?;
    validate_matrix_shape(
        "large_components",
        input.large_components,
        input.radii.len(),
        orbital_count,
    )?;
    validate_matrix_shape(
        "small_components",
        input.small_components,
        input.radii.len(),
        orbital_count,
    )?;
    let coefficient_count = input.large_coefficients.nrows();
    validate_coefficient_count("large_coefficients", coefficient_count)?;
    validate_matrix_shape(
        "large_coefficients",
        input.large_coefficients,
        coefficient_count,
        orbital_count,
    )?;
    validate_matrix_shape(
        "small_coefficients",
        input.small_coefficients,
        coefficient_count,
        orbital_count,
    )?;
    validate_differential_active_len(
        input.active_lengths[left].min(input.active_lengths[right]),
        input.radii.len(),
    )?;
    validate_finite_slice("orbital_power", input.orbital_powers)?;
    validate_finite_matrix("large_component", input.large_components)?;
    validate_finite_matrix("small_component", input.small_components)?;
    validate_finite_matrix("large_coefficient", input.large_coefficients)?;
    validate_finite_matrix("small_coefficient", input.small_coefficients)?;
    Ok(())
}

fn validate_radial_integral_input(
    input: &AtomicRadialIntegralInput<'_>,
) -> Result<(), AtomMathError> {
    validate_finite_scalar("fdrirk_step", input.step)?;
    let orbital_count = input.active_lengths.len();
    if orbital_count == 0 {
        return Err(AtomMathError::EmptyOrbitalTable);
    }
    validate_orbital_table_len("kappas", orbital_count, input.kappas.len())?;
    validate_orbital_table_len("orbital_powers", orbital_count, input.orbital_powers.len())?;
    validate_positive_finite_radii(input.radii)?;
    validate_matrix_shape(
        "large_components",
        input.large_components,
        input.radii.len(),
        orbital_count,
    )?;
    validate_matrix_shape(
        "small_components",
        input.small_components,
        input.radii.len(),
        orbital_count,
    )?;
    let coefficient_count = input.large_coefficients.nrows();
    validate_coefficient_count("large_coefficients", coefficient_count)?;
    validate_matrix_shape(
        "large_coefficients",
        input.large_coefficients,
        coefficient_count,
        orbital_count,
    )?;
    validate_matrix_shape(
        "small_coefficients",
        input.small_coefficients,
        coefficient_count,
        orbital_count,
    )?;
    for &kappa in input.kappas {
        abs_kappa_i32(kappa)?;
    }
    validate_finite_slice("orbital_power", input.orbital_powers)?;
    validate_finite_matrix("large_component", input.large_components)?;
    validate_finite_matrix("small_component", input.small_components)?;
    validate_finite_matrix("large_coefficient", input.large_coefficients)?;
    validate_finite_matrix("small_coefficient", input.small_coefficients)?;

    if input.request.first_left > 0 && input.request.first_right > 0 {
        let left = one_based_atomic_orbital_index(input.request.first_left, orbital_count)?;
        let right = one_based_atomic_orbital_index(input.request.first_right, orbital_count)?;
        validate_differential_active_len(
            input.active_lengths[left].min(input.active_lengths[right]),
            input.radii.len(),
        )?;
    }
    if input.request.second_left > 0 && input.request.second_right > 0 {
        let left = one_based_atomic_orbital_index(input.request.second_left, orbital_count)?;
        let right = one_based_atomic_orbital_index(input.request.second_right, orbital_count)?;
        validate_differential_active_len(
            input.active_lengths[left].min(input.active_lengths[right]),
            input.radii.len(),
        )?;
        if (input.request.first_left == 0 || input.request.first_right == 0)
            && input.previous_first_factor.is_none()
        {
            return Err(AtomMathError::MissingRadialFirstFactor);
        }
    }
    if let Some(first_factor) = input.previous_first_factor {
        validate_radial_table_len(
            "previous_first_factor",
            input.radii.len(),
            first_factor.values.len(),
        )?;
        validate_coefficient_vector_len(
            "previous_first_factor_coefficients",
            coefficient_count,
            first_factor.coefficients.len(),
        )?;
        validate_finite_vector("previous_first_factor", first_factor.values)?;
        validate_finite_vector(
            "previous_first_factor_coefficient",
            first_factor.coefficients,
        )?;
        validate_finite_scalar(
            "previous_first_factor_origin_power",
            first_factor.origin_power,
        )?;
    }
    Ok(())
}

fn validate_tabulation_input(input: &AtomicTabulationInput<'_>) -> Result<(), AtomMathError> {
    let orbital_count = input.kappas.len();
    if orbital_count == 0 {
        return Err(AtomMathError::EmptyOrbitalTable);
    }
    validate_orbital_table_len(
        "principal_quantum_numbers",
        orbital_count,
        input.principal_quantum_numbers.len(),
    )?;
    validate_orbital_table_len("occupations", orbital_count, input.occupations.len())?;
    validate_orbital_table_len(
        "orbital_energies",
        orbital_count,
        input.orbital_energies.len(),
    )?;
    for (orbital, &principal_quantum_number) in input.principal_quantum_numbers.iter().enumerate() {
        if principal_quantum_number == 0 {
            return Err(AtomMathError::InvalidPrincipalQuantumNumber {
                orbital_1based: orbital + 1,
                principal_quantum_number,
            });
        }
    }
    for &kappa in input.kappas {
        atom_tabrat_orbital_label(kappa)?;
    }
    validate_finite_slice("occupation", input.occupations)?;
    validate_finite_slice("orbital_energy", input.orbital_energies)?;
    Ok(())
}

fn validate_schmidt_orthogonalization_input(
    input: &AtomicSchmidtOrthogonalizationInput<'_>,
) -> Result<(), AtomMathError> {
    let orbital_count = input.kappas.len();
    if orbital_count == 0 {
        return Err(AtomMathError::EmptyOrbitalTable);
    }
    validate_orbital_table_len("active_lengths", orbital_count, input.active_lengths.len())?;
    validate_orbital_table_len("orbital_powers", orbital_count, input.orbital_powers.len())?;
    if let Some(active_orbital_1based) = input.active_orbital_1based
        && !(1..=orbital_count).contains(&active_orbital_1based)
    {
        return Err(AtomMathError::ActiveOrbitalOutOfRange {
            active_orbital_1based,
            orbital_count,
        });
    }

    let radial_rows = input.large_components.nrows();
    let coefficient_rows = input.large_coefficients.nrows();
    validate_matrix_shape(
        "large_components",
        input.large_components,
        radial_rows,
        orbital_count,
    )?;
    validate_matrix_shape(
        "small_components",
        input.small_components,
        radial_rows,
        orbital_count,
    )?;
    validate_matrix_shape(
        "large_coefficients",
        input.large_coefficients,
        coefficient_rows,
        orbital_count,
    )?;
    validate_matrix_shape(
        "small_coefficients",
        input.small_coefficients,
        coefficient_rows,
        orbital_count,
    )?;

    for (orbital, &active_len) in input.active_lengths.iter().enumerate() {
        if active_len > radial_rows {
            return Err(AtomMathError::ActiveLengthOutOfRange {
                orbital_1based: orbital + 1,
                active_len,
                row_count: radial_rows,
            });
        }
    }
    for &kappa in input.kappas {
        doubled_j_from_kappa(kappa)?;
    }
    validate_finite_slice("orbital_power", input.orbital_powers)?;
    validate_finite_matrix("large_component", input.large_components)?;
    validate_finite_matrix("small_component", input.small_components)?;
    validate_finite_matrix("large_coefficient", input.large_coefficients)?;
    validate_finite_matrix("small_coefficient", input.small_coefficients)?;
    Ok(())
}

fn validate_coulomb_coefficient_input(
    input: &AtomicCoulombCoefficientInput<'_>,
) -> Result<(), AtomMathError> {
    let orbital_count = input.kappas.len();
    validate_orbital_table_len("occupations", orbital_count, input.occupations.len())?;
    validate_orbital_table_len(
        "valence_occupations",
        orbital_count,
        input.valence_occupations.len(),
    )?;
    validate_finite_slice("occupation", input.occupations)?;
    validate_finite_slice("valence_occupation", input.valence_occupations)?;
    for &kappa in input.kappas {
        doubled_j_from_kappa(kappa)?;
    }
    Ok(())
}

fn validate_orbital_initialization_input(
    input: &AtomicOrbitalInitializationInput<'_>,
) -> Result<(), AtomMathError> {
    if input.atomic_number == 0 {
        return Err(AtomMathError::InvalidAtomicNumber {
            atomic_number: input.atomic_number,
        });
    }
    validate_finite_scalar("inmuat_ionicity", input.ionicity)?;
    let orbital_count = input.occupations.len();
    if orbital_count == 0 {
        return Err(AtomMathError::EmptyOrbitalTable);
    }
    validate_orbital_table_len(
        "principal_quantum_numbers",
        orbital_count,
        input.principal_quantum_numbers.len(),
    )?;
    validate_orbital_table_len("kappas", orbital_count, input.kappas.len())?;
    validate_finite_slice("occupation", input.occupations)?;
    for (orbital, &principal_quantum_number) in input.principal_quantum_numbers.iter().enumerate() {
        if principal_quantum_number == 0 {
            return Err(AtomMathError::InvalidPrincipalQuantumNumber {
                orbital_1based: orbital + 1,
                principal_quantum_number,
            });
        }
    }
    for &kappa in input.kappas {
        kappa_abs_usize(kappa)?;
    }
    let actual = input.occupations.iter().copied().sum::<Real>();
    let expected = input.atomic_number as Real - input.ionicity;
    if (expected - actual).abs() > ATOM_INMUAT_ELECTRON_TOLERANCE {
        return Err(AtomMathError::ElectronCountMismatch {
            atomic_number: input.atomic_number,
            ionicity: input.ionicity,
            expected,
            actual,
            tolerance: ATOM_INMUAT_ELECTRON_TOLERANCE,
        });
    }
    Ok(())
}

fn validate_dirac_normalization_input(
    input: &AtomicDiracNormalizationInput<'_>,
) -> Result<(), AtomMathError> {
    validate_positive_finite_scalar("soldir_step", input.step)?;
    validate_finite_scalar(
        "soldir_matching_small_component",
        input.matching_small_component,
    )?;
    validate_finite_scalar("soldir_origin_power", input.origin_power)?;
    validate_dirac_normalization_active_len(input.active_len, input.radii.len())?;
    validate_radial_table_len(
        "soldir_large_component",
        input.radii.len(),
        input.large_component.len(),
    )?;
    validate_radial_table_len(
        "soldir_small_component",
        input.radii.len(),
        input.small_component.len(),
    )?;
    validate_positive_finite_radii(input.radii)?;
    validate_finite_vector("soldir_large_component", input.large_component)?;
    validate_finite_vector("soldir_small_component", input.small_component)?;
    validate_coefficient_count("soldir_large_coefficients", input.coefficient_count)?;
    validate_coefficient_vector_capacity(
        "soldir_large_coefficients",
        input.coefficient_count,
        input.large_coefficients.len(),
    )?;
    validate_coefficient_vector_capacity(
        "soldir_small_coefficients",
        input.coefficient_count,
        input.small_coefficients.len(),
    )?;
    validate_finite_vector("soldir_large_coefficient", input.large_coefficients)?;
    validate_finite_vector("soldir_small_coefficient", input.small_coefficients)?;
    if input.method == 1
        && (input.matching_index_1based == 0 || input.matching_index_1based > input.active_len)
    {
        return Err(AtomMathError::DiracNormalizationMatchingIndexOutOfRange {
            matching_index_1based: input.matching_index_1based,
            active_len: input.active_len,
        });
    }
    Ok(())
}

fn validate_dirac_solution_normalization_input(
    input: &AtomicDiracSolutionNormalizationInput<'_>,
) -> Result<(), AtomMathError> {
    validate_positive_finite_scalar("soldir_solution_norm", input.norm)?;
    validate_finite_scalar(
        "soldir_solution_initial_large_coefficient",
        input.initial_large_coefficient,
    )?;
    validate_finite_scalar(
        "soldir_solution_initial_small_coefficient",
        input.initial_small_coefficient,
    )?;
    validate_dirac_solution_normalization_active_len(
        input.active_len,
        input.large_component.len(),
    )?;
    validate_radial_table_len(
        "soldir_solution_small_component",
        input.large_component.len(),
        input.small_component.len(),
    )?;
    validate_finite_vector("soldir_solution_large_component", input.large_component)?;
    validate_finite_vector("soldir_solution_small_component", input.small_component)?;
    validate_coefficient_count(
        "soldir_solution_large_coefficients",
        input.coefficient_count,
    )?;
    validate_coefficient_vector_capacity(
        "soldir_solution_large_coefficients",
        input.coefficient_count,
        input.large_coefficients.len(),
    )?;
    validate_coefficient_vector_capacity(
        "soldir_solution_small_coefficients",
        input.coefficient_count,
        input.small_coefficients.len(),
    )?;
    validate_finite_vector(
        "soldir_solution_large_coefficient",
        input.large_coefficients,
    )?;
    validate_finite_vector(
        "soldir_solution_small_coefficient",
        input.small_coefficients,
    )?;
    Ok(())
}

fn validate_dirac_node_count_input(
    input: &AtomicDiracNodeCountInput<'_>,
) -> Result<(), AtomMathError> {
    validate_finite_vector("soldir_node_large_component", input.large_component)?;
    validate_dirac_node_count_index(
        "matching",
        input.matching_index_1based,
        input.large_component.len(),
    )?;
    validate_dirac_node_count_index("scan", input.scan_index_1based, input.large_component.len())?;
    Ok(())
}

fn validate_dirac_node_energy_search_input(
    input: &AtomicDiracNodeEnergySearchInput,
) -> Result<(), AtomMathError> {
    validate_finite_scalar("soldir_node_search_energy", input.energy)?;
    validate_finite_scalar("soldir_node_search_esup", input.energy_sup)?;
    validate_finite_scalar("soldir_node_search_einf", input.energy_inf)?;
    validate_finite_scalar("soldir_node_search_emin", input.energy_floor)?;
    validate_positive_finite_scalar("soldir_node_search_precision", input.energy_precision)?;
    Ok(())
}

fn validate_dirac_method_one_energy_correction_input(
    input: &AtomicDiracMethodOneEnergyCorrectionInput<'_>,
) -> Result<(), AtomMathError> {
    validate_positive_finite_scalar("soldir_energy_speed_of_light", input.speed_of_light)?;
    validate_positive_finite_scalar("soldir_energy_norm", input.norm)?;
    validate_finite_scalar(
        "soldir_energy_matching_small_component",
        input.matching_small_component,
    )?;
    validate_radial_table_len(
        "soldir_energy_small_component",
        input.large_component.len(),
        input.small_component.len(),
    )?;
    validate_finite_vector("soldir_energy_large_component", input.large_component)?;
    validate_finite_vector("soldir_energy_small_component", input.small_component)?;
    validate_dirac_energy_correction_matching_index(
        input.matching_index_1based,
        input.large_component.len(),
    )?;
    Ok(())
}

fn validate_dirac_energy_step_input(
    input: &AtomicDiracEnergyStepInput,
) -> Result<(), AtomMathError> {
    validate_finite_scalar("soldir_energy_step_energy", input.energy)?;
    validate_finite_scalar("soldir_energy_step_correction", input.correction)?;
    validate_finite_scalar("soldir_energy_step_mismatch", input.mismatch)?;
    validate_finite_scalar("soldir_energy_step_esup", input.energy_sup)?;
    validate_finite_scalar("soldir_energy_step_einf", input.energy_inf)?;
    validate_positive_finite_scalar(
        "soldir_energy_step_mismatch_precision",
        input.mismatch_precision,
    )?;
    validate_positive_finite_scalar(
        "soldir_energy_step_zero_precision",
        input.zero_energy_precision,
    )?;
    Ok(())
}

fn validate_dirac_rematch_attempt_input(
    input: &AtomicDiracRematchAttemptInput,
) -> Result<(), AtomMathError> {
    validate_finite_scalar("soldir_rematch_mismatch", input.mismatch)?;
    validate_positive_finite_scalar(
        "soldir_rematch_mismatch_precision",
        input.mismatch_precision,
    )?;
    Ok(())
}

fn validate_dirac_large_component_match_input(
    input: &AtomicDiracLargeComponentMatchInput<'_>,
) -> Result<(), AtomMathError> {
    validate_finite_scalar(
        "soldir_match_matching_large_component",
        input.matching_large_component,
    )?;
    validate_dirac_match_active_len(input.active_len, input.large_component.len())?;
    validate_dirac_match_matching_index(input.matching_index_1based, input.active_len)?;
    validate_radial_table_len(
        "soldir_match_small_component",
        input.large_component.len(),
        input.small_component.len(),
    )?;
    validate_radial_table_len(
        "soldir_match_homogeneous_large_component",
        input.large_component.len(),
        input.homogeneous_large_component.len(),
    )?;
    validate_radial_table_len(
        "soldir_match_homogeneous_small_component",
        input.large_component.len(),
        input.homogeneous_small_component.len(),
    )?;
    validate_finite_vector("soldir_match_large_component", input.large_component)?;
    validate_finite_vector("soldir_match_small_component", input.small_component)?;
    validate_finite_vector(
        "soldir_match_homogeneous_large_component",
        input.homogeneous_large_component,
    )?;
    validate_finite_vector(
        "soldir_match_homogeneous_small_component",
        input.homogeneous_small_component,
    )?;
    Ok(())
}

fn validate_dirac_homogeneous_match_input(
    input: &AtomicDiracHomogeneousMatchInput<'_>,
) -> Result<(), AtomMathError> {
    validate_finite_scalar(
        "soldir_homogeneous_match_matching_large_component",
        input.matching_large_component,
    )?;
    validate_dirac_match_active_len(input.active_len, input.large_component.len())?;
    validate_dirac_match_matching_index(input.matching_index_1based, input.active_len)?;
    validate_radial_table_len(
        "soldir_homogeneous_match_small_component",
        input.large_component.len(),
        input.small_component.len(),
    )?;
    validate_finite_vector(
        "soldir_homogeneous_match_large_component",
        input.large_component,
    )?;
    validate_finite_vector(
        "soldir_homogeneous_match_small_component",
        input.small_component,
    )?;
    Ok(())
}

fn validate_dirac_two_component_match_input(
    input: &AtomicDiracTwoComponentMatchInput<'_>,
) -> Result<(), AtomMathError> {
    validate_finite_scalar(
        "soldir_two_match_matching_large_component",
        input.matching_large_component,
    )?;
    validate_finite_scalar(
        "soldir_two_match_matching_small_component",
        input.matching_small_component,
    )?;
    validate_finite_scalar(
        "soldir_two_match_homogeneous_matching_large_component",
        input.homogeneous_matching_large_component,
    )?;
    validate_finite_scalar(
        "soldir_two_match_homogeneous_matching_small_component",
        input.homogeneous_matching_small_component,
    )?;
    validate_dirac_match_active_len(input.active_len, input.large_component.len())?;
    validate_dirac_match_matching_index(input.matching_index_1based, input.active_len)?;
    validate_radial_table_len(
        "soldir_two_match_small_component",
        input.large_component.len(),
        input.small_component.len(),
    )?;
    validate_radial_table_len(
        "soldir_two_match_homogeneous_large_component",
        input.large_component.len(),
        input.homogeneous_large_component.len(),
    )?;
    validate_radial_table_len(
        "soldir_two_match_homogeneous_small_component",
        input.large_component.len(),
        input.homogeneous_small_component.len(),
    )?;
    validate_coefficient_count(
        "soldir_two_match_large_coefficients",
        input.coefficient_count,
    )?;
    validate_coefficient_vector_capacity(
        "soldir_two_match_large_coefficients",
        input.coefficient_count,
        input.large_coefficients.len(),
    )?;
    validate_coefficient_vector_capacity(
        "soldir_two_match_small_coefficients",
        input.coefficient_count,
        input.small_coefficients.len(),
    )?;
    validate_coefficient_vector_capacity(
        "soldir_two_match_homogeneous_large_coefficients",
        input.coefficient_count,
        input.homogeneous_large_coefficients.len(),
    )?;
    validate_coefficient_vector_capacity(
        "soldir_two_match_homogeneous_small_coefficients",
        input.coefficient_count,
        input.homogeneous_small_coefficients.len(),
    )?;
    validate_finite_vector("soldir_two_match_large_component", input.large_component)?;
    validate_finite_vector("soldir_two_match_small_component", input.small_component)?;
    validate_finite_vector(
        "soldir_two_match_homogeneous_large_component",
        input.homogeneous_large_component,
    )?;
    validate_finite_vector(
        "soldir_two_match_homogeneous_small_component",
        input.homogeneous_small_component,
    )?;
    validate_finite_vector(
        "soldir_two_match_large_coefficient",
        input.large_coefficients,
    )?;
    validate_finite_vector(
        "soldir_two_match_small_coefficient",
        input.small_coefficients,
    )?;
    validate_finite_vector(
        "soldir_two_match_homogeneous_large_coefficient",
        input.homogeneous_large_coefficients,
    )?;
    validate_finite_vector(
        "soldir_two_match_homogeneous_small_coefficient",
        input.homogeneous_small_coefficients,
    )?;
    Ok(())
}

fn validate_dirac_energy_disagreement_source_input(
    input: &AtomicDiracEnergyDisagreementSourceInput<'_>,
) -> Result<(), AtomMathError> {
    validate_positive_finite_scalar(
        "soldir_energy_disagreement_speed_of_light",
        input.speed_of_light,
    )?;
    validate_dirac_energy_disagreement_active_len(input.active_len, input.radii.len())?;
    validate_radial_table_len(
        "soldir_energy_disagreement_large_component",
        input.radii.len(),
        input.large_component.len(),
    )?;
    validate_radial_table_len(
        "soldir_energy_disagreement_small_component",
        input.radii.len(),
        input.small_component.len(),
    )?;
    validate_positive_finite_radii(input.radii)?;
    validate_coefficient_count(
        "soldir_energy_disagreement_large_coefficients",
        input.coefficient_count,
    )?;
    validate_coefficient_vector_capacity(
        "soldir_energy_disagreement_large_coefficients",
        input.coefficient_count,
        input.large_coefficients.len(),
    )?;
    validate_coefficient_vector_capacity(
        "soldir_energy_disagreement_small_coefficients",
        input.coefficient_count,
        input.small_coefficients.len(),
    )?;
    validate_finite_vector(
        "soldir_energy_disagreement_large_component",
        input.large_component,
    )?;
    validate_finite_vector(
        "soldir_energy_disagreement_small_component",
        input.small_component,
    )?;
    validate_finite_vector(
        "soldir_energy_disagreement_large_coefficient",
        input.large_coefficients,
    )?;
    validate_finite_vector(
        "soldir_energy_disagreement_small_coefficient",
        input.small_coefficients,
    )?;
    Ok(())
}

fn validate_dirac_energy_disagreement_correction_input(
    input: &AtomicDiracEnergyDisagreementCorrectionInput<'_>,
) -> Result<(), AtomMathError> {
    validate_positive_finite_scalar("soldir_energy_disagreement_correction_step", input.step)?;
    validate_finite_scalar("soldir_energy_disagreement_correction_norm", input.norm)?;
    validate_finite_scalar(
        "soldir_energy_disagreement_correction_origin_power",
        input.origin_power,
    )?;
    validate_dirac_energy_disagreement_correction_active_len(input.active_len, input.radii.len())?;
    validate_radial_table_len(
        "soldir_energy_disagreement_correction_large_component",
        input.radii.len(),
        input.large_component.len(),
    )?;
    validate_radial_table_len(
        "soldir_energy_disagreement_correction_small_component",
        input.radii.len(),
        input.small_component.len(),
    )?;
    validate_radial_table_len(
        "soldir_energy_disagreement_correction_large_derivative",
        input.radii.len(),
        input.large_derivative.len(),
    )?;
    validate_radial_table_len(
        "soldir_energy_disagreement_correction_small_derivative",
        input.radii.len(),
        input.small_derivative.len(),
    )?;
    validate_positive_finite_radii(input.radii)?;
    validate_coefficient_count(
        "soldir_energy_disagreement_correction_large_coefficients",
        input.coefficient_count,
    )?;
    validate_coefficient_vector_capacity(
        "soldir_energy_disagreement_correction_large_coefficients",
        input.coefficient_count,
        input.large_coefficients.len(),
    )?;
    validate_coefficient_vector_capacity(
        "soldir_energy_disagreement_correction_small_coefficients",
        input.coefficient_count,
        input.small_coefficients.len(),
    )?;
    validate_coefficient_vector_capacity(
        "soldir_energy_disagreement_correction_large_derivative_coefficients",
        input.coefficient_count,
        input.large_derivative_coefficients.len(),
    )?;
    validate_coefficient_vector_capacity(
        "soldir_energy_disagreement_correction_small_derivative_coefficients",
        input.coefficient_count,
        input.small_derivative_coefficients.len(),
    )?;
    validate_finite_vector(
        "soldir_energy_disagreement_correction_large_component",
        input.large_component,
    )?;
    validate_finite_vector(
        "soldir_energy_disagreement_correction_small_component",
        input.small_component,
    )?;
    validate_finite_vector(
        "soldir_energy_disagreement_correction_large_derivative",
        input.large_derivative,
    )?;
    validate_finite_vector(
        "soldir_energy_disagreement_correction_small_derivative",
        input.small_derivative,
    )?;
    validate_finite_vector(
        "soldir_energy_disagreement_correction_large_coefficient",
        input.large_coefficients,
    )?;
    validate_finite_vector(
        "soldir_energy_disagreement_correction_small_coefficient",
        input.small_coefficients,
    )?;
    validate_finite_vector(
        "soldir_energy_disagreement_correction_large_derivative_coefficient",
        input.large_derivative_coefficients,
    )?;
    validate_finite_vector(
        "soldir_energy_disagreement_correction_small_derivative_coefficient",
        input.small_derivative_coefficients,
    )?;
    Ok(())
}

fn validate_dirac_matching_point_update_input(
    input: &AtomicDiracMatchingPointUpdateInput<'_>,
) -> Result<(), AtomMathError> {
    validate_finite_vector(
        "soldir_matching_point_large_component",
        input.large_component,
    )?;
    if input.active_len > ATOM_SOLDIR_MATCHING_POINT_FALLBACK_OFFSET
        && input.active_len <= input.large_component.len()
    {
        validate_dirac_match_matching_index(input.matching_index_1based, input.active_len)
    } else {
        Err(AtomMathError::InvalidDiracMatchActiveLength {
            active_len: input.active_len,
            radial_count: input.large_component.len(),
        })
    }
}

fn validate_dirac_inhomogeneous_seed_input(
    input: &AtomicDiracInhomogeneousSeedInput<'_>,
) -> Result<(), AtomMathError> {
    validate_radial_table_len(
        "soldir_seed_small_source",
        input.large_source.len(),
        input.small_source.len(),
    )?;
    validate_coefficient_count("soldir_seed_coefficients", input.coefficient_count)?;
    validate_coefficient_vector_capacity(
        "soldir_seed_large_coefficients",
        input.coefficient_count,
        input.large_source_coefficients.len(),
    )?;
    validate_coefficient_vector_capacity(
        "soldir_seed_small_coefficients",
        input.coefficient_count,
        input.small_source_coefficients.len(),
    )?;
    validate_finite_vector("soldir_seed_large_source", input.large_source)?;
    validate_finite_vector("soldir_seed_small_source", input.small_source)?;
    validate_finite_vector(
        "soldir_seed_large_source_coefficient",
        input.large_source_coefficients,
    )?;
    validate_finite_vector(
        "soldir_seed_small_source_coefficient",
        input.small_source_coefficients,
    )?;
    Ok(())
}

fn validate_dirac_homogeneous_seed_input(
    input: &AtomicDiracHomogeneousSeedInput,
) -> Result<(), AtomMathError> {
    if input.radial_len == 0 {
        return Err(AtomMathError::InvalidCount {
            field: "soldir_homogeneous_seed_radial_len",
            minimum: 1,
            actual: input.radial_len,
        });
    }
    if input.coefficient_len == 0 {
        return Err(AtomMathError::InvalidCount {
            field: "soldir_homogeneous_seed_coefficient_len",
            minimum: 1,
            actual: input.coefficient_len,
        });
    }
    Ok(())
}

fn validate_dirac_shooting_pass_setup_input(
    input: &AtomicDiracShootingPassSetupInput,
) -> Result<(), AtomMathError> {
    validate_finite_scalar("soldir_shooting_pass_energy", input.energy)?;
    validate_finite_scalar(
        "soldir_shooting_pass_previous_energy",
        input.previous_energy,
    )?;
    Ok(())
}

fn validate_dirac_integration_input(
    input: &AtomicDiracIntegrationInput<'_>,
) -> Result<(), AtomMathError> {
    validate_positive_finite_scalar("intdir_speed_of_light", input.speed_of_light)?;
    validate_positive_finite_scalar("intdir_step", input.step)?;
    validate_positive_finite_scalar("intdir_matching_precision", input.matching_precision)?;
    validate_finite_scalar("intdir_energy", input.energy)?;
    validate_finite_scalar("intdir_origin_power", input.origin_power)?;
    validate_finite_scalar(
        "intdir_initial_large_coefficient",
        input.initial_large_coefficient,
    )?;
    validate_finite_scalar(
        "intdir_initial_small_coefficient",
        input.initial_small_coefficient,
    )?;
    validate_finite_scalar(
        "intdir_asymptotic_large_component",
        input.asymptotic_large_component,
    )?;
    if input.kappa == 0 {
        return Err(AtomMathError::InvalidKappa { kappa: input.kappa });
    }
    validate_dirac_integration_active_len(input.active_len, input.radii.len())?;
    validate_radial_table_len(
        "intdir_large_source",
        input.radii.len(),
        input.large_source.len(),
    )?;
    validate_radial_table_len(
        "intdir_small_source",
        input.radii.len(),
        input.small_source.len(),
    )?;
    validate_radial_table_len("intdir_potential", input.radii.len(), input.potential.len())?;
    validate_positive_finite_radii(input.radii)?;
    validate_finite_vector("intdir_large_source", input.large_source)?;
    validate_finite_vector("intdir_small_source", input.small_source)?;
    validate_finite_vector("intdir_potential", input.potential)?;
    validate_coefficient_count("intdir_large_coefficients", input.coefficient_count)?;
    validate_coefficient_vector_capacity(
        "intdir_large_coefficients",
        input.coefficient_count,
        input.large_coefficients.len(),
    )?;
    validate_coefficient_vector_capacity(
        "intdir_small_coefficients",
        input.coefficient_count,
        input.small_coefficients.len(),
    )?;
    validate_coefficient_vector_capacity(
        "intdir_potential_coefficients",
        input.coefficient_count,
        input.potential_coefficients.len(),
    )?;
    validate_finite_vector("intdir_large_coefficient", input.large_coefficients)?;
    validate_finite_vector("intdir_small_coefficient", input.small_coefficients)?;
    validate_finite_vector("intdir_potential_coefficient", input.potential_coefficients)?;
    atom_intdir_decay(input.energy, input.speed_of_light)?;

    match input.mode {
        AtomicDiracIntegrationMode::SearchMatchingPoint => Ok(()),
        AtomicDiracIntegrationMode::FixedMatchingPoint | AtomicDiracIntegrationMode::InwardOnly => {
            validate_dirac_integration_matching_index(
                input.matching_index_1based,
                input.active_len,
            )?;
            validate_dirac_integration_max_index(
                input.max_index_1based,
                input.matching_index_1based,
                input.active_len,
            )
        }
    }
}

fn validate_dirac_solver_setup_input(
    input: &AtomicDiracSolverSetupInput<'_>,
) -> Result<(), AtomMathError> {
    validate_finite_scalar("soldir_setup_energy", input.energy)?;
    validate_finite_scalar("soldir_setup_origin_power", input.origin_power)?;
    validate_finite_scalar(
        "soldir_setup_initial_large_coefficient",
        input.initial_large_coefficient,
    )?;
    validate_finite_scalar(
        "soldir_setup_initial_small_coefficient",
        input.initial_small_coefficient,
    )?;
    validate_positive_finite_scalar("soldir_setup_speed_of_light", input.speed_of_light)?;
    if input.kappa == 0 {
        return Err(AtomMathError::InvalidKappa { kappa: input.kappa });
    }
    if input.principal_quantum_number == 0 {
        return Err(AtomMathError::InvalidDiracSolverPrincipalQuantumNumber {
            principal_quantum_number: input.principal_quantum_number,
        });
    }
    validate_dirac_solver_setup_active_len(input.active_len, input.radii.len())?;
    validate_radial_table_len(
        "soldir_setup_potential",
        input.radii.len(),
        input.potential.len(),
    )?;
    validate_coefficient_vector_capacity(
        "soldir_setup_potential_coefficients",
        1,
        input.potential_coefficients.len(),
    )?;
    validate_positive_finite_radii(input.radii)?;
    validate_finite_vector("soldir_setup_potential", input.potential)?;
    validate_finite_vector(
        "soldir_setup_potential_coefficient",
        input.potential_coefficients,
    )?;
    Ok(())
}

fn validate_local_density_potential_input(
    input: &AtomicLocalDensityPotentialInput<'_>,
) -> Result<(), AtomMathError> {
    let radial_count = input.radii.len();
    if radial_count == 0 {
        return Err(AtomMathError::InvalidCount {
            field: "vlda_radii",
            minimum: 1,
            actual: radial_count,
        });
    }
    validate_positive_finite_scalar("vlda_speed_of_light", input.speed_of_light)?;
    validate_positive_finite_radii(input.radii)?;
    validate_radial_table_len(
        "initial_potential",
        radial_count,
        input.initial_potential.len(),
    )?;
    validate_radial_table_len(
        "initial_energy_density",
        radial_count,
        input.initial_energy_density.len(),
    )?;
    if input.initial_development_coefficients.len() < 2 {
        return Err(AtomMathError::InvalidCount {
            field: "initial_development_coefficients",
            minimum: 2,
            actual: input.initial_development_coefficients.len(),
        });
    }

    let orbital_count = input.active_lengths.len();
    if orbital_count == 0 {
        return Err(AtomMathError::EmptyOrbitalTable);
    }
    validate_orbital_table_len("occupations", orbital_count, input.occupations.len())?;
    validate_orbital_table_len(
        "valence_occupations",
        orbital_count,
        input.valence_occupations.len(),
    )?;
    validate_matrix_shape(
        "large_components",
        input.large_components,
        radial_count,
        orbital_count,
    )?;
    validate_matrix_shape(
        "small_components",
        input.small_components,
        radial_count,
        orbital_count,
    )?;
    for (orbital, &active_len) in input.active_lengths.iter().enumerate() {
        if active_len > radial_count {
            return Err(AtomMathError::ActiveLengthOutOfRange {
                orbital_1based: orbital + 1,
                active_len,
                row_count: radial_count,
            });
        }
    }

    validate_finite_slice("occupation", input.occupations)?;
    validate_finite_slice("valence_occupation", input.valence_occupations)?;
    validate_finite_matrix("large_component", input.large_components)?;
    validate_finite_matrix("small_component", input.small_components)?;
    validate_finite_vector("initial_potential", input.initial_potential)?;
    validate_finite_vector(
        "initial_development_coefficient",
        input.initial_development_coefficients,
    )?;
    validate_finite_vector("initial_energy_density", input.initial_energy_density)?;
    Ok(())
}

fn validate_orbital_potential_input(
    input: &AtomicOrbitalPotentialInput<'_>,
) -> Result<(), AtomMathError> {
    let radial_count = input.radii.len();
    if radial_count < 4 {
        return Err(AtomMathError::InvalidDifferentialIntegralActiveLength {
            active_len: radial_count,
            radial_count,
        });
    }
    validate_positive_finite_scalar("potrdf_speed_of_light", input.speed_of_light)?;
    validate_finite_scalar("potrdf_step", input.step)?;
    validate_positive_finite_radii(input.radii)?;
    validate_radial_table_len(
        "nuclear_potential",
        radial_count,
        input.nuclear_potential.len(),
    )?;

    let coefficient_count = input.nuclear_development_coefficients.len();
    validate_coefficient_count("nuclear_development_coefficients", coefficient_count)?;

    let orbital_count = input.active_lengths.len();
    if orbital_count == 0 {
        return Err(AtomMathError::EmptyOrbitalTable);
    }
    if input.self_consistent_count > orbital_count {
        return Err(AtomMathError::ActiveOrbitalOutOfRange {
            active_orbital_1based: input.self_consistent_count,
            orbital_count,
        });
    }
    one_based_atomic_orbital_index(input.active_orbital_1based, orbital_count)?;
    validate_orbital_table_len("kappas", orbital_count, input.kappas.len())?;
    validate_orbital_table_len("orbital_powers", orbital_count, input.orbital_powers.len())?;
    validate_orbital_table_len("occupations", orbital_count, input.occupations.len())?;
    validate_orbital_table_len("shell_markers", orbital_count, input.shell_markers.len())?;
    validate_orbital_table_len("origin_scales", orbital_count, input.origin_scales.len())?;
    validate_matrix_shape(
        "large_components",
        input.large_components,
        radial_count,
        orbital_count,
    )?;
    validate_matrix_shape(
        "small_components",
        input.small_components,
        radial_count,
        orbital_count,
    )?;
    validate_matrix_shape(
        "large_coefficients",
        input.large_coefficients,
        coefficient_count,
        orbital_count,
    )?;
    validate_matrix_shape(
        "small_coefficients",
        input.small_coefficients,
        coefficient_count,
        orbital_count,
    )?;

    for (orbital, &active_len) in input.active_lengths.iter().enumerate() {
        if active_len > radial_count {
            return Err(AtomMathError::ActiveLengthOutOfRange {
                orbital_1based: orbital + 1,
                active_len,
                row_count: radial_count,
            });
        }
    }
    for &kappa in input.kappas {
        kappa_angular_rank(kappa)?;
    }
    for &origin_scale in input.origin_scales {
        validate_positive_finite_scalar("origin_scale", origin_scale)?;
    }
    validate_positive_occupation(
        "potrdf_active_orbital",
        input.active_orbital_1based - 1,
        input.occupations,
    )?;

    if input.include_lagrange {
        let expected_pairs = orbital_pair_count(orbital_count)?;
        validate_coefficient_vector_len(
            "lagrange_parameters",
            expected_pairs,
            input.lagrange_parameters.len(),
        )?;
        validate_finite_vector("lagrange_parameter", input.lagrange_parameters)?;
    }

    validate_finite_slice("orbital_power", input.orbital_powers)?;
    validate_finite_slice("occupation", input.occupations)?;
    validate_finite_matrix("large_component", input.large_components)?;
    validate_finite_matrix("small_component", input.small_components)?;
    validate_finite_matrix("large_coefficient", input.large_coefficients)?;
    validate_finite_matrix("small_coefficient", input.small_coefficients)?;
    validate_finite_vector("nuclear_potential", input.nuclear_potential)?;
    validate_finite_vector(
        "nuclear_development_coefficient",
        input.nuclear_development_coefficients,
    )?;
    Ok(())
}

fn validate_positive_finite_scalar(field: &'static str, value: Real) -> Result<(), AtomMathError> {
    validate_finite_scalar(field, value)?;
    if value > 0.0 {
        Ok(())
    } else {
        Err(AtomMathError::NonPositiveScalar { field, value })
    }
}

fn validate_positive_finite_nuclear_scalar(
    field: &'static str,
    value: Real,
) -> Result<(), AtomMathError> {
    if value.is_finite() && value > 0.0 {
        Ok(())
    } else {
        Err(AtomMathError::InvalidNuclearPotentialScalar { field, value })
    }
}

fn validate_nuclear_count(
    field: &'static str,
    actual: usize,
    minimum: usize,
) -> Result<(), AtomMathError> {
    if actual >= minimum {
        Ok(())
    } else {
        Err(AtomMathError::InvalidNuclearPotentialCount {
            field,
            minimum,
            actual,
        })
    }
}

fn validate_differential_active_len(
    active_len: usize,
    radial_count: usize,
) -> Result<(), AtomMathError> {
    if active_len > 0 && active_len <= radial_count && active_len % 2 == 1 {
        Ok(())
    } else {
        Err(AtomMathError::InvalidDifferentialIntegralActiveLength {
            active_len,
            radial_count,
        })
    }
}

fn validate_dirac_normalization_active_len(
    active_len: usize,
    radial_count: usize,
) -> Result<(), AtomMathError> {
    if active_len > 0 && active_len <= radial_count && active_len % 2 == 1 {
        Ok(())
    } else {
        Err(AtomMathError::InvalidDiracNormalizationActiveLength {
            active_len,
            radial_count,
        })
    }
}

fn validate_dirac_integration_active_len(
    active_len: usize,
    radial_count: usize,
) -> Result<(), AtomMathError> {
    if active_len > ATOM_INTDIR_HISTORY + 12 && active_len <= radial_count {
        Ok(())
    } else {
        Err(AtomMathError::InvalidDiracIntegrationActiveLength {
            active_len,
            radial_count,
        })
    }
}

fn validate_dirac_solver_setup_active_len(
    active_len: usize,
    radial_count: usize,
) -> Result<(), AtomMathError> {
    if active_len > 0 && active_len <= radial_count {
        Ok(())
    } else {
        Err(AtomMathError::InvalidDiracSolverSetupActiveLength {
            active_len,
            radial_count,
        })
    }
}

fn validate_dirac_solution_normalization_active_len(
    active_len: usize,
    radial_count: usize,
) -> Result<(), AtomMathError> {
    if active_len > 0 && active_len <= radial_count {
        Ok(())
    } else {
        Err(
            AtomMathError::InvalidDiracSolutionNormalizationActiveLength {
                active_len,
                radial_count,
            },
        )
    }
}

fn validate_dirac_match_active_len(
    active_len: usize,
    radial_count: usize,
) -> Result<(), AtomMathError> {
    if active_len > 0 && active_len <= radial_count {
        Ok(())
    } else {
        Err(AtomMathError::InvalidDiracMatchActiveLength {
            active_len,
            radial_count,
        })
    }
}

fn validate_dirac_energy_disagreement_active_len(
    active_len: usize,
    radial_count: usize,
) -> Result<(), AtomMathError> {
    if active_len > 0 && active_len <= radial_count {
        Ok(())
    } else {
        Err(AtomMathError::InvalidDiracEnergyDisagreementActiveLength {
            active_len,
            radial_count,
        })
    }
}

fn validate_dirac_energy_disagreement_correction_active_len(
    active_len: usize,
    radial_count: usize,
) -> Result<(), AtomMathError> {
    if active_len > 0 && active_len % 2 == 1 && active_len <= radial_count {
        Ok(())
    } else {
        Err(
            AtomMathError::InvalidDiracEnergyDisagreementCorrectionActiveLength {
                active_len,
                radial_count,
            },
        )
    }
}

fn validate_dirac_match_matching_index(
    matching_index_1based: usize,
    active_len: usize,
) -> Result<(), AtomMathError> {
    if matching_index_1based > 0 && matching_index_1based <= active_len {
        Ok(())
    } else {
        Err(AtomMathError::DiracMatchMatchingIndexOutOfRange {
            matching_index_1based,
            active_len,
        })
    }
}

fn validate_dirac_node_count_index(
    field: &'static str,
    index_1based: usize,
    radial_count: usize,
) -> Result<(), AtomMathError> {
    if index_1based > 0 && index_1based <= radial_count {
        Ok(())
    } else {
        Err(AtomMathError::InvalidDiracNodeCountIndex {
            field,
            index_1based,
            radial_count,
        })
    }
}

fn validate_dirac_energy_correction_matching_index(
    matching_index_1based: usize,
    radial_count: usize,
) -> Result<(), AtomMathError> {
    if matching_index_1based > 0 && matching_index_1based <= radial_count {
        Ok(())
    } else {
        Err(
            AtomMathError::DiracEnergyCorrectionMatchingIndexOutOfRange {
                matching_index_1based,
                radial_count,
            },
        )
    }
}

fn validate_dirac_integration_matching_index(
    matching_index_1based: usize,
    active_len: usize,
) -> Result<(), AtomMathError> {
    if matching_index_1based > ATOM_INTDIR_HISTORY && matching_index_1based <= active_len {
        Ok(())
    } else {
        Err(AtomMathError::DiracIntegrationMatchingIndexOutOfRange {
            matching_index_1based,
            active_len,
        })
    }
}

fn validate_dirac_integration_max_index(
    max_index_1based: usize,
    matching_index_1based: usize,
    active_len: usize,
) -> Result<(), AtomMathError> {
    if max_index_1based <= active_len
        && max_index_1based > matching_index_1based + ATOM_INTDIR_HISTORY
    {
        Ok(())
    } else {
        Err(AtomMathError::DiracIntegrationMaxIndexOutOfRange {
            max_index_1based,
            matching_index_1based,
            active_len,
        })
    }
}

fn validate_coefficient_count(table: &'static str, actual_len: usize) -> Result<(), AtomMathError> {
    if actual_len > 0 {
        Ok(())
    } else {
        Err(AtomMathError::CoefficientTableLengthMismatch {
            table,
            expected_len: 1,
            actual_len,
        })
    }
}

fn validate_coefficient_vector_len(
    table: &'static str,
    expected_len: usize,
    actual_len: usize,
) -> Result<(), AtomMathError> {
    if actual_len == expected_len {
        Ok(())
    } else {
        Err(AtomMathError::CoefficientTableLengthMismatch {
            table,
            expected_len,
            actual_len,
        })
    }
}

fn validate_coefficient_vector_capacity(
    table: &'static str,
    required_len: usize,
    actual_len: usize,
) -> Result<(), AtomMathError> {
    if actual_len >= required_len {
        Ok(())
    } else {
        Err(AtomMathError::CoefficientTableLengthMismatch {
            table,
            expected_len: required_len,
            actual_len,
        })
    }
}

fn validate_matrix_shape(
    table: &'static str,
    matrix: ArrayView2<'_, Real>,
    expected_rows: usize,
    expected_columns: usize,
) -> Result<(), AtomMathError> {
    let rows = matrix.nrows();
    let columns = matrix.ncols();
    if rows == expected_rows && columns == expected_columns {
        Ok(())
    } else {
        Err(AtomMathError::MatrixShape {
            table,
            expected_rows,
            expected_columns,
            rows,
            columns,
        })
    }
}

fn validate_orbital_table_len(
    table: &'static str,
    expected_len: usize,
    actual_len: usize,
) -> Result<(), AtomMathError> {
    if actual_len == expected_len {
        Ok(())
    } else {
        Err(AtomMathError::OrbitalTableLengthMismatch {
            table,
            expected_len,
            actual_len,
        })
    }
}

fn validate_radial_table_len(
    table: &'static str,
    expected_len: usize,
    actual_len: usize,
) -> Result<(), AtomMathError> {
    if actual_len == expected_len {
        Ok(())
    } else {
        Err(AtomMathError::RadialTableLengthMismatch {
            table,
            expected_len,
            actual_len,
        })
    }
}

fn validate_occupation_tables(occupations: &[Real], kappas: &[i32]) -> Result<(), AtomMathError> {
    if occupations.len() != kappas.len() {
        return Err(AtomMathError::OccupationKappaLengthMismatch {
            occupation_len: occupations.len(),
            kappa_len: kappas.len(),
        });
    }
    validate_finite_slice("occupation", occupations)
}

fn validate_positive_occupation(
    context: &'static str,
    orbital: usize,
    occupations: &[Real],
) -> Result<Real, AtomMathError> {
    let occupation = occupations[orbital];
    if occupation > 0.0 {
        Ok(occupation)
    } else {
        Err(AtomMathError::NonPositiveOccupation {
            context,
            orbital_1based: orbital + 1,
            occupation,
        })
    }
}

fn validate_orbital_index(index: usize, len: usize) -> Result<(), AtomMathError> {
    if index < len {
        Ok(())
    } else {
        Err(AtomMathError::OrbitalIndexOutOfRange { index, len })
    }
}

fn validate_coefficient_table(
    coefficients: ArrayView3<'_, Real>,
    left: usize,
    right: usize,
    rank: usize,
) -> Result<(), AtomMathError> {
    let shape = coefficients.shape();
    let rows = shape[0];
    let columns = shape[1];
    let channels = shape[2];
    if rows == 0 || columns == 0 || rows != columns || channels == 0 {
        return Err(AtomMathError::CoefficientTableShape {
            rows,
            columns,
            channels,
        });
    }
    if left >= rows {
        return Err(AtomMathError::OrbitalIndexOutOfRange {
            index: left,
            len: rows,
        });
    }
    if right >= columns {
        return Err(AtomMathError::OrbitalIndexOutOfRange {
            index: right,
            len: columns,
        });
    }
    let channel = rank / 2;
    if channel >= channels {
        return Err(AtomMathError::CoefficientChannelOutOfRange {
            rank,
            channel,
            channels,
        });
    }
    for value in coefficients.iter().copied() {
        if !value.is_finite() {
            return Err(AtomMathError::NonFiniteScalar {
                field: "coefficient",
                value,
            });
        }
    }
    Ok(())
}

fn validate_finite_slice(field: &'static str, values: &[Real]) -> Result<(), AtomMathError> {
    for &value in values {
        validate_finite_scalar(field, value)?;
    }
    Ok(())
}

fn validate_finite_vector(
    field: &'static str,
    values: ArrayView1<'_, Real>,
) -> Result<(), AtomMathError> {
    for value in values.iter().copied() {
        validate_finite_scalar(field, value)?;
    }
    Ok(())
}

fn validate_finite_matrix(
    field: &'static str,
    matrix: ArrayView2<'_, Real>,
) -> Result<(), AtomMathError> {
    for value in matrix.iter().copied() {
        validate_finite_scalar(field, value)?;
    }
    Ok(())
}

fn validate_positive_finite_radii(values: ArrayView1<'_, Real>) -> Result<(), AtomMathError> {
    for &radius in values {
        validate_finite_scalar("radius", radius)?;
        if radius <= 0.0 {
            return Err(AtomMathError::NonPositiveRadius { radius });
        }
    }
    Ok(())
}

fn validate_finite_scalar(field: &'static str, value: Real) -> Result<(), AtomMathError> {
    if value.is_finite() {
        Ok(())
    } else {
        Err(AtomMathError::NonFiniteScalar { field, value })
    }
}

const FEFF_ATOMIC_WEIGHTS: [f32; 139] = [
    1.0079, 4.0026, 6.941, 9.0122, 10.81, 12.01, 14.007, 15.999, 18.998, 20.18, 22.9898, 24.305,
    26.982, 28.086, 30.974, 32.064, 35.453, 39.948, 39.09, 40.08, 44.956, 47.90, 50.942, 52.00,
    54.938, 55.85, 58.93, 58.71, 63.55, 65.38, 69.72, 72.59, 74.922, 78.96, 79.91, 83.80, 85.47,
    87.62, 88.91, 91.22, 92.91, 95.94, 98.91, 101.07, 102.90, 106.40, 107.87, 112.40, 114.82,
    118.69, 121.75, 127.60, 126.90, 131.30, 132.91, 137.34, 138.91, 140.12, 140.91, 144.24, 145.0,
    150.35, 151.96, 157.25, 158.92, 162.50, 164.93, 167.26, 168.93, 173.04, 174.97, 178.49, 180.95,
    183.85, 186.2, 190.20, 192.22, 195.09, 196.97, 200.59, 204.37, 207.19, 208.98, 210.0, 210.0,
    222.0, 223.0, 226.0, 227.0, 232.04, 231.0, 238.03, 237.05, 244.0, 243.0, 247.0, 247.0, 251.0,
    252.0, 257.0, 258.0, 259.0, 266.0, 267.0, 268.0, 269.0, 270.0, 269.0, 278.0, 281.0, 282.0,
    285.0, 286.0, 289.0, 289.0, 293.0, 294.0, 294.0, 315.0, 320.0, 330.0, 334.0, 337.0, 340.0,
    344.0, 347.0, 350.0, 354.0, 357.0, 361.0, 364.0, 367.0, 371.0, 374.0, 378.0, 381.0, 385.0,
    388.0, 392.0,
];

const FEFF_ATOMIC_SYMBOLS: [&str; 139] = [
    "H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne", "Na", "Mg", "Al", "Si", "P", "S", "Cl",
    "Ar", "K", "Ca", "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn", "Ga", "Ge", "As",
    "Se", "Br", "Kr", "Rb", "Sr", "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In",
    "Sn", "Sb", "Te", "I", "Xe", "Cs", "Ba", "La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd", "Tb",
    "Dy", "Ho", "Er", "Tm", "Yb", "Lu", "Hf", "Ta", "W", "Te", "Os", "Ir", "Pt", "Au", "Hg", "Tl",
    "Pb", "Bi", "Po", "At", "Rn", "Fr", "Ra", "Ac", "Th", "Pa", "U", "Np", "Pu", "Am", "Cm", "Bk",
    "Cf", "Es", "Fm", "Md", "No", "Lr", "Rf", "Db", "Sg", "Bh", "Hs", "Mt", "Ds", "Rg", "Cn",
    "Uut", "Fl", "Uup", "Lv", "Uus", "Uuo", "Uue", "Ubn", "Ubu", "Ubb", "Ubt", "Ubq", "Ubp", "Ubh",
    "Ubs", "Ubo", "Ube", "Utn", "Utu", "Utb", "Utt", "Utq", "Utp", "Uth", "Uts", "Uto", "Ute",
];

#[allow(clippy::excessive_precision)]
const FEFF_NUCLEAR_MASSES: [f32; 138] = [
    1.00794,
    4.002602,
    6.941,
    9.012182,
    10.811,
    12.0107,
    14.0067,
    15.9994,
    18.9984032,
    20.1797,
    22.98976928,
    24.305,
    26.9815386,
    28.0855,
    30.973762,
    32.065,
    35.453,
    39.948,
    39.0983,
    40.078,
    44.955912,
    47.867,
    50.9415,
    51.9961,
    54.938045,
    55.845,
    58.933195,
    58.6934,
    63.546,
    65.38,
    69.723,
    72.64,
    74.9216,
    78.96,
    79.904,
    83.798,
    85.4678,
    87.62,
    88.90585,
    91.224,
    92.90638,
    95.96,
    98.0,
    101.07,
    102.9055,
    106.42,
    107.8682,
    112.411,
    114.818,
    118.71,
    121.76,
    127.6,
    126.90447,
    131.293,
    132.9054519,
    137.327,
    138.90547,
    140.116,
    140.90765,
    144.242,
    145.0,
    150.36,
    151.964,
    157.25,
    158.92535,
    162.5,
    164.93032,
    167.259,
    168.93421,
    173.054,
    174.9668,
    178.49,
    180.94788,
    183.84,
    186.207,
    190.23,
    192.217,
    195.084,
    196.966569,
    200.59,
    204.3833,
    207.2,
    208.9804,
    209.0,
    210.0,
    222.0,
    223.0,
    226.0,
    227.0,
    232.03806,
    231.03588,
    238.02891,
    237.0,
    244.0,
    243.0,
    247.0,
    247.0,
    251.0,
    252.0,
    257.0,
    258.0,
    259.0,
    262.0,
    265.0,
    268.0,
    271.0,
    272.0,
    277.0,
    276.0,
    281.0,
    280.0,
    285.0,
    284.0,
    289.0,
    288.0,
    293.0,
    294.0,
    294.0,
    315.0,
    320.0,
    330.0,
    334.0,
    337.0,
    340.0,
    344.0,
    347.0,
    350.0,
    354.0,
    357.0,
    361.0,
    364.0,
    367.0,
    371.0,
    374.0,
    378.0,
    381.0,
    385.0,
    388.0,
];

#[cfg(test)]
mod tests {
    use ndarray::{Array1, Array2, Array3};

    use super::*;

    fn assert_close(actual: Real, expected: Real) {
        assert_close_with(actual, expected, 1.0e-12);
    }

    fn assert_close_with(actual: Real, expected: Real, tolerance: Real) {
        assert!(
            (actual - expected).abs() < tolerance,
            "actual={actual}, expected={expected}, diff={}",
            (actual - expected).abs()
        );
    }

    fn assert_some_close(actual: Option<Real>, expected: Real, tolerance: Real) {
        match actual {
            Some(value) => assert_close_with(value, expected, tolerance),
            None => assert_eq!(actual, Some(expected)),
        }
    }

    #[test]
    fn atomic_weight_matches_feff_pertab_reference() -> Result<(), AtomicError> {
        assert_close(atomic_weight(1)?, 1.007_899_999_618_530_3);
        assert_close(atomic_weight(2)?, 4.002_600_193_023_682);
        assert_close(atomic_weight(26)?, 55.849_998_474_121_094);
        assert_close(atomic_weight(75)?, 186.199_996_948_242_2);
        assert_close(atomic_weight(92)?, 238.029_998_779_296_88);
        assert_close(atomic_weight(118)?, 294.0);
        assert_close(atomic_weight(121)?, 330.0);
        assert_close(atomic_weight(139)?, 392.0);
        Ok(())
    }

    #[test]
    fn atomic_symbol_matches_feff_pertab_reference() -> Result<(), AtomicError> {
        assert_eq!(atomic_symbol(1)?, "H");
        assert_eq!(atomic_symbol(2)?, "He");
        assert_eq!(atomic_symbol(26)?, "Fe");
        assert_eq!(atomic_symbol(75)?, "Te");
        assert_eq!(atomic_symbol(92)?, "U");
        assert_eq!(atomic_symbol(118)?, "Uuo");
        assert_eq!(atomic_symbol(121)?, "Ubu");
        assert_eq!(atomic_symbol(139)?, "Ute");
        Ok(())
    }

    #[test]
    fn nuclear_mass_matches_feff_reference() -> Result<(), AtomicError> {
        assert_close(nuclear_mass(1)?, 1.007_940_053_939_819_3);
        assert_close(nuclear_mass(6)?, 12.010_700_225_830_078);
        assert_close(nuclear_mass(29)?, 63.546_001_434_326_17);
        assert_close(nuclear_mass(57)?, 138.905_471_801_757_8);
        assert_close(nuclear_mass(92)?, 238.028_915_405_273_44);
        assert_close(nuclear_mass(118)?, 294.0);
        assert_close(nuclear_mass(121)?, 330.0);
        assert_close(nuclear_mass(138)?, 388.0);
        Ok(())
    }

    #[test]
    fn nuclear_mass_rejects_invalid_atomic_numbers() {
        assert_eq!(
            nuclear_mass(0),
            Err(AtomicError::InvalidAtomicNumber { z: 0 })
        );
        assert_eq!(
            nuclear_mass(139),
            Err(AtomicError::InvalidAtomicNumber { z: 139 })
        );
        assert_eq!(
            atomic_weight(140),
            Err(AtomicError::InvalidAtomicNumber { z: 140 })
        );
        assert_eq!(
            atomic_symbol(0),
            Err(AtomicError::InvalidAtomicNumber { z: 0 })
        );
    }

    #[test]
    fn atom_nuclear_potential_matches_feff_nucdev_reference() -> Result<(), AtomMathError> {
        let point = atomic_nuclear_potential(AtomicNuclearPotentialInput {
            nuclear_charge: 26.0,
            step: 0.05,
            requested_nucleus_index: 1,
            radial_count: 251,
            coefficient_count: 10,
            first_radius_times_charge: 26.0 * (-8.8_f64).exp(),
        })?;
        assert_eq!(point.nucleus_index, 1);
        assert_close_with(
            point.first_radius_times_charge,
            3.919_059_952_482e-3,
            5.0e-16,
        );
        assert_close(point.development_coefficients[0], -26.0);
        assert_close_with(point.radii[0], 1.507_330_750_955e-4, 5.0e-16);
        assert_close_with(point.potential[0], -1.724_903_441_632e5, 5.0e-8);
        assert_close_with(point.radii[4], 1.841_057_936_676e-4, 5.0e-16);
        assert_close_with(point.potential[4], -1.412_231_493_754e5, 5.0e-8);

        let finite = atomic_nuclear_potential(AtomicNuclearPotentialInput {
            nuclear_charge: 92.0,
            step: 0.05,
            requested_nucleus_index: -11,
            radial_count: 251,
            coefficient_count: 10,
            first_radius_times_charge: 92.0 * (-8.8_f64).exp(),
        })?;
        assert_eq!(finite.nucleus_index, 11);
        assert_close_with(
            finite.first_radius_times_charge,
            7.842_167_533_588e-3,
            5.0e-15,
        );
        assert_close_with(
            finite.development_coefficients[1],
            -9.819_368_462_521e5,
            5.0e-7,
        );
        assert_close_with(
            finite.development_coefficients[3],
            1.657_185_951_350e13,
            10.0,
        );
        assert_close_with(finite.radii[0], 8.524_095_145_204e-5, 5.0e-16);
        assert_close_with(finite.potential[0], -8.615_253_868_304e5, 5.0e-7);
        assert_close_with(finite.radii[10], 1.405_385_697_937e-4, 5.0e-16);
        assert_close_with(finite.potential[10], -6.546_245_641_680e5, 5.0e-7);
        Ok(())
    }

    #[test]
    fn atom_helper_kernels_match_feff_reference() -> Result<(), AtomMathError> {
        let left = (1..=10)
            .map(|index| 0.1 * index as Real + 0.03)
            .collect::<Vec<_>>();
        let right = (1..=10)
            .map(|index| -0.04 * index as Real + 0.25)
            .collect::<Vec<_>>();
        assert_close(
            atomic_polynomial_product_coefficient(&left, &right, 3)?,
            0.125_300_000_000_000_02,
        );
        assert_close(
            atomic_polynomial_product_coefficient(&left, &right, 7)?,
            0.382_9,
        );

        let mixed = atomic_convergence_mix(0.5, 0.3, 0.2)?;
        assert_close(mixed.initial_weight, 0.4);
        assert_close(mixed.final_weight, 0.6);
        assert_close(mixed.previous_error, 0.3);

        let mixed = atomic_convergence_mix(0.2, 0.5, -0.4)?;
        assert_close(mixed.initial_weight, 0.9);
        assert_close(mixed.final_weight, 0.1);
        assert_close(mixed.previous_error, 0.5);

        let mixed = atomic_convergence_mix(0.9, 0.5, 0.4)?;
        assert_close(mixed.initial_weight, 0.099_999_999_999_999_98);
        assert_close(mixed.final_weight, 0.9);
        assert_close(mixed.previous_error, 0.5);

        assert_close(
            thomas_fermi_density_potential(0.45, 29.0, -1.0)?,
            43.097_863_212_551_05,
        );
        assert_close(
            thomas_fermi_density_potential(1.25, 8.0, -2.5)?,
            3.548_014_948_104_207,
        );
        assert_close(thomas_fermi_density_potential(1.25, 0.0, 0.0)?, 0.0);

        let mut occupations = vec![0.0; 41];
        let mut kappas = vec![1; 41];
        occupations[1] = 1.5;
        occupations[4] = 3.0;
        kappas[1] = -1;
        kappas[4] = -3;
        assert_close(atomic_occupation_product(&occupations, &kappas, 4, 4)?, 7.2);
        assert_close(atomic_occupation_product(&occupations, &kappas, 1, 4)?, 4.5);
        Ok(())
    }

    #[test]
    fn atom_coulomb_coefficient_lookups_match_feff_reference() -> Result<(), AtomMathError> {
        let coefficients = Array3::from_shape_fn((41, 41, 5), |(row, column, channel)| {
            1000.0 * (row + 1) as Real + 10.0 * (column + 1) as Real + channel as Real
        });

        assert_close(
            atomic_direct_coulomb_coefficient(coefficients.view(), 1, 4, 4)?,
            2052.0,
        );
        assert_close(
            atomic_direct_coulomb_coefficient(coefficients.view(), 4, 1, 4)?,
            2052.0,
        );
        assert_close(
            atomic_exchange_coulomb_coefficient(coefficients.view(), 1, 4, 4)?,
            5022.0,
        );
        assert_close(
            atomic_exchange_coulomb_coefficient(coefficients.view(), 4, 1, 4)?,
            5022.0,
        );
        assert_close(
            atomic_exchange_coulomb_coefficient(coefficients.view(), 4, 4, 4)?,
            0.0,
        );
        Ok(())
    }

    #[test]
    fn atom_coulomb_coefficients_match_feff_muatco_reference() -> Result<(), AtomMathError> {
        let kappas = [-1, 1, -2, 2, -3];
        let occupations = [2.0, 1.5, 3.0, 0.5, 4.0];
        let valence_occupations = [0.0, 0.5, 0.0, 0.25, 0.0];
        let coefficients = atomic_coulomb_coefficients(AtomicCoulombCoefficientInput {
            kappas: &kappas,
            occupations: &occupations,
            valence_occupations: &valence_occupations,
        })?;

        let expected = [
            [
                [2.0, 3.0, 6.0, 1.0, 8.0],
                [0.5, 2.25, 4.5, 0.75, 6.0],
                [1.000_000_000_000_000_7, 0.0, 6.0, 1.5, 12.0],
                [0.0, 0.0, 0.025_000_000_000_000_026, 0.25, 2.0],
                [0.0, 0.0, 1.199_999_999_999_999_3, 0.0, 12.0],
            ],
            [
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [
                    0.0,
                    0.450_000_000_000_000_2,
                    -0.400_000_000_000_000_3,
                    0.0,
                    0.0,
                ],
                [
                    0.100_000_000_000_000_03,
                    0.0,
                    0.096_428_571_428_571_31,
                    0.0,
                    0.0,
                ],
                [
                    0.799_999_999_999_999_5,
                    0.428_571_428_571_428_2,
                    0.342_857_142_857_142_47,
                    0.028_571_428_571_428_536,
                    -0.548_571_428_571_427_9,
                ],
            ],
            [
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [
                    0.0,
                    0.0,
                    0.0,
                    0.095_238_095_238_094_86,
                    -0.228_571_428_571_427_8,
                ],
            ],
            [
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
            ],
            [
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
            ],
        ];

        for (channel, rows) in expected.iter().enumerate() {
            for (row, columns) in rows.iter().enumerate() {
                for (column, &expected) in columns.iter().enumerate() {
                    assert_close_with(coefficients[(row, column, channel)], expected, 1.0e-12);
                }
            }
        }

        Ok(())
    }

    #[test]
    fn atom_breit_coefficients_match_feff_bkmrdf_reference() -> Result<(), AtomMathError> {
        let cases = [
            (
                -1,
                -1,
                1,
                [0.5, 0.333_333_333_333_333_2, 0.5],
                [
                    -0.166_666_666_666_666_69,
                    0.333_333_333_333_333_37,
                    -0.166_666_666_666_666_69,
                ],
            ),
            (
                -1,
                1,
                1,
                [
                    1.500_000_000_000_000_4,
                    1.000_000_000_000_000_2,
                    0.166_666_666_666_666_7,
                ],
                [
                    1.500_000_000_000_000_4,
                    3.000_000_000_000_001,
                    0.833_333_333_333_333_6,
                ],
            ),
            (
                1,
                -2,
                1,
                [
                    0.500_000_000_000_000_2,
                    0.333_333_333_333_334_8,
                    0.100_000_000_000_000_06,
                ],
                [
                    -0.166_666_666_666_667_4,
                    -0.666_666_666_666_669_6,
                    -0.126_666_666_666_667_1,
                ],
            ),
            (
                -2,
                2,
                3,
                [
                    0.116_666_666_666_666_78,
                    0.033_333_333_333_333_36,
                    0.002_380_952_380_952_383,
                ],
                [
                    0.070_000_000_000_000_05,
                    0.420_000_000_000_000_3,
                    0.058_571_428_571_428_62,
                ],
            ),
            (
                -3,
                -3,
                5,
                [
                    0.050_505_050_505_050_37,
                    0.072_150_072_150_071_99,
                    0.050_505_050_505_050_37,
                ],
                [
                    -0.039_281_705_948_372_45,
                    0.078_563_411_896_744_9,
                    -0.039_281_705_948_372_45,
                ],
            ),
            (
                2,
                -4,
                3,
                [
                    0.102_380_952_380_952_13,
                    0.201_587_301_587_301_2,
                    0.254_761_904_761_904_3,
                ],
                [
                    0.238_500_881_834_214_8,
                    0.721_305_114_638_447_2,
                    0.264_320_987_654_320_66,
                ],
            ),
        ];

        for (left, right, rank, magnetic, retarded) in cases {
            let actual = atomic_breit_angular_coefficients(left, right, rank)?;
            for index in 0..3 {
                assert_close(actual.magnetic[index], magnetic[index]);
                assert_close(actual.retarded[index], retarded[index]);
            }
        }
        Ok(())
    }

    #[allow(clippy::excessive_precision)]
    #[test]
    fn atom_orbital_initialization_matches_feff_inmuat_reference() -> Result<(), AtomMathError> {
        let open_shell = atomic_orbital_initialization(AtomicOrbitalInitializationInput {
            atomic_number: 4,
            ionicity: 0.0,
            principal_quantum_numbers: &[2, 3, 1],
            kappas: &[1, 1, -1],
            occupations: &[0.4, 1.6, 2.0],
        })?;

        assert_eq!(open_shell.orbital_count, 3);
        assert_eq!(open_shell.self_consistent_count, 3);
        assert_eq!(open_shell.lagrange_pair_count, 1);
        assert_eq!(open_shell.radial_count, 251);
        assert_eq!(open_shell.development_order, 10);
        assert_eq!(open_shell.attempt_count, 50);
        assert_eq!(open_shell.nucleus_index, 11);
        assert_close_with(
            open_shell.wavefunction_precision,
            1.000_000_000_000_000_08e-5,
            1.0e-20,
        );
        assert_close_with(
            open_shell.energy_precision,
            5.000_000_000_000_000_41e-6,
            1.0e-20,
        );
        assert_close(open_shell.precision_ratios[0], 100.0);
        assert_close(open_shell.precision_ratios[1], 10.0);
        assert_close_with(open_shell.primary_matching_precision, 1.0e-7, 1.0e-20);
        assert_close_with(open_shell.secondary_matching_precision, 1.0e-6, 1.0e-20);
        assert_eq!(open_shell.shell_markers.to_vec(), vec![1, 1, -1]);
        assert_eq!(open_shell.active_lengths.to_vec(), vec![251, 251, 251]);
        assert_close_with(open_shell.convergence_acceleration[0], 1.0, 1.0e-16);
        assert_close_with(
            open_shell.convergence_acceleration[1],
            3.000_000_119_209_289_55e-1,
            1.0e-16,
        );
        assert_close_with(
            open_shell.convergence_acceleration[2],
            3.000_000_119_209_289_55e-1,
            1.0e-16,
        );
        assert!(
            open_shell
                .orbital_energies
                .iter()
                .all(|&value| value == 0.0)
        );
        assert!(
            open_shell
                .wavefunction_errors
                .iter()
                .all(|&value| value == 0.0)
        );
        assert!(open_shell.energy_errors.iter().all(|&value| value == 0.0));
        assert_eq!(open_shell.lagrange_parameters.len(), 820);
        assert!(
            open_shell
                .lagrange_parameters
                .iter()
                .all(|&value| value == 0.0)
        );

        let closed_shell = atomic_orbital_initialization(AtomicOrbitalInitializationInput {
            atomic_number: 10,
            ionicity: 0.0,
            principal_quantum_numbers: &[1, 2, 2, 2],
            kappas: &[-1, -1, 1, -2],
            occupations: &[2.0, 2.0, 2.0, 4.0],
        })?;
        assert_eq!(closed_shell.orbital_count, 4);
        assert_eq!(closed_shell.self_consistent_count, 4);
        assert_eq!(closed_shell.lagrange_pair_count, 0);
        assert_eq!(closed_shell.shell_markers.to_vec(), vec![-1, -1, -1, -1]);
        assert_eq!(
            closed_shell.active_lengths.to_vec(),
            vec![251, 251, 251, 251]
        );
        for value in closed_shell.convergence_acceleration {
            assert_close_with(value, 3.000_000_119_209_289_55e-1, 1.0e-16);
        }
        Ok(())
    }

    #[test]
    fn atom_dirac_normalization_matches_feff_soldir_norm_reference() -> Result<(), AtomMathError> {
        let fixture = sample_soldir_norm_fixture();

        let method_one = atomic_dirac_normalization(fixture.input(1, 6, 0.177, 0.82, 11, 5))?;
        assert_close_with(method_one.norm, 5.408_474_263_575_392e-6, 1.0e-18);

        let method_two = atomic_dirac_normalization(fixture.input(2, 8, 0.0, 1.35, 13, 7))?;
        assert_close_with(method_two.norm, 9.499_334_208_495_336e-6, 1.0e-18);
        Ok(())
    }

    #[allow(clippy::excessive_precision)]
    #[test]
    fn atom_dirac_solution_normalization_matches_feff_soldir_reference() -> Result<(), AtomMathError>
    {
        let kept_fixture = sample_soldir_solution_normalization_fixture(false, false);
        let kept = atomic_dirac_solution_normalization(kept_fixture.input(6.25, 0.8, -0.4))?;
        assert_close_with(kept.component_divisor, 2.5, 1.0e-18);
        assert_close_with(kept.coefficient_divisor, 2.5, 1.0e-18);
        assert_close_with(kept.large_coefficients[0], 8.4e-2, 1.0e-18);
        assert_close_with(kept.small_coefficients[0], -4.28e-2, 1.0e-18);
        assert_close_with(kept.large_coefficients[3], 3.84e-1, 1.0e-18);
        assert_close_with(kept.small_coefficients[3], -1.568e-1, 1.0e-18);
        assert_close_with(kept.large_component[0], 1.64e-2, 1.0e-18);
        assert_close_with(kept.small_component[0], -1.18e-2, 1.0e-18);
        assert_close_with(kept.large_component[6], 1.316e-1, 1.0e-18);
        assert_close_with(kept.large_component[7], 0.0, 1.0e-18);
        assert_close_with(kept.small_component[8], 0.0, 1.0e-18);

        let flipped_fixture = sample_soldir_solution_normalization_fixture(true, true);
        let flipped =
            atomic_dirac_solution_normalization(flipped_fixture.input(1.44, 0.75, -0.25))?;
        assert_close_with(flipped.component_divisor, -1.2, 1.0e-18);
        assert_close_with(flipped.coefficient_divisor, -1.2, 1.0e-18);
        assert_close_with(
            flipped.large_coefficients[0],
            1.750_000_000_000_000_2e-1,
            1.0e-18,
        );
        assert_close_with(
            flipped.small_coefficients[0],
            8.916_666_666_666_667_2e-2,
            1.0e-18,
        );
        assert_close_with(
            flipped.large_coefficients[3],
            -8.000_000_000_000_000_4e-1,
            1.0e-18,
        );
        assert_close_with(
            flipped.small_coefficients[3],
            3.266_666_666_666_667_2e-1,
            1.0e-18,
        );
        assert_close_with(
            flipped.large_component[0],
            3.416_666_666_666_667_2e-2,
            1.0e-18,
        );
        assert_close_with(
            flipped.small_component[0],
            2.458_333_333_333_333_2e-2,
            1.0e-18,
        );
        assert_close_with(
            flipped.large_component[6],
            -2.741_666_666_666_666_7e-1,
            1.0e-18,
        );
        assert_close_with(flipped.large_component[7], 0.0, 1.0e-18);
        assert_close_with(flipped.small_component[8], 0.0, 1.0e-18);
        Ok(())
    }

    #[test]
    fn atom_dirac_node_count_matches_feff_soldir_reference() -> Result<(), AtomMathError> {
        let large_component = sample_soldir_node_count_component();

        let limited = atomic_dirac_node_count(AtomicDiracNodeCountInput {
            large_component: large_component.view(),
            matching_index_1based: 4,
            scan_index_1based: 7,
        })?;
        assert_eq!(limited.scan_index_1based, 7);
        assert_eq!(limited.node_count, 4);

        let matching_extends = atomic_dirac_node_count(AtomicDiracNodeCountInput {
            large_component: large_component.view(),
            matching_index_1based: 8,
            scan_index_1based: 3,
        })?;
        assert_eq!(matching_extends.scan_index_1based, 8);
        assert_eq!(matching_extends.node_count, 4);

        let full = atomic_dirac_node_count(AtomicDiracNodeCountInput {
            large_component: large_component.view(),
            matching_index_1based: 1,
            scan_index_1based: 9,
        })?;
        assert_eq!(full.scan_index_1based, 9);
        assert_eq!(full.node_count, 5);
        Ok(())
    }

    #[allow(clippy::excessive_precision)]
    #[test]
    fn atom_dirac_node_energy_search_matches_feff_soldir_reference() -> Result<(), AtomMathError> {
        let too_few_scale = atomic_dirac_node_energy_search(AtomicDiracNodeEnergySearchInput {
            energy: -0.5,
            node_count: 2,
            target_node_count: 4,
            energy_sup: -5.0,
            energy_inf: 1.0,
            energy_floor: -5.0,
            energy_precision: 1.0e-7,
            search_attempt_count: 0,
            max_attempt_count: 50,
        })?;
        assert_close_with(too_few_scale.energy, -4.000_000_000_000_000_2e-1, 1.0e-18);
        assert_close_with(too_few_scale.energy_sup, -5.0e-1, 1.0e-18);
        assert_close_with(too_few_scale.energy_inf, 1.0, 1.0e-18);
        assert_eq!(too_few_scale.search_attempt_count, 1);
        assert!(too_few_scale.needs_reintegration);
        assert!(!too_few_scale.attempts_exhausted);

        let too_few_bisect = atomic_dirac_node_energy_search(AtomicDiracNodeEnergySearchInput {
            energy: -0.6,
            node_count: 2,
            target_node_count: 4,
            energy_sup: -5.0,
            energy_inf: -0.2,
            energy_floor: -5.0,
            energy_precision: 1.0e-7,
            search_attempt_count: 4,
            max_attempt_count: 50,
        })?;
        assert_close_with(too_few_bisect.energy, -4.000_000_000_000_000_2e-1, 1.0e-18);
        assert_close_with(
            too_few_bisect.energy_sup,
            -5.999_999_999_999_999_8e-1,
            1.0e-18,
        );
        assert_close_with(
            too_few_bisect.energy_inf,
            -2.000_000_000_000_000_1e-1,
            1.0e-18,
        );
        assert_eq!(too_few_bisect.search_attempt_count, 5);
        assert!(too_few_bisect.needs_reintegration);
        assert!(!too_few_bisect.attempts_exhausted);

        let too_many_scale = atomic_dirac_node_energy_search(AtomicDiracNodeEnergySearchInput {
            energy: -0.5,
            node_count: 5,
            target_node_count: 3,
            energy_sup: -5.0,
            energy_inf: 1.0,
            energy_floor: -5.0,
            energy_precision: 1.0e-7,
            search_attempt_count: 7,
            max_attempt_count: 50,
        })?;
        assert_close_with(too_many_scale.energy, -5.999_999_999_999_999_8e-1, 1.0e-18);
        assert_close_with(too_many_scale.energy_sup, -5.0, 1.0e-18);
        assert_close_with(too_many_scale.energy_inf, -5.0e-1, 1.0e-18);
        assert_eq!(too_many_scale.search_attempt_count, 8);
        assert!(too_many_scale.needs_reintegration);
        assert!(!too_many_scale.attempts_exhausted);

        let too_many_bisect = atomic_dirac_node_energy_search(AtomicDiracNodeEnergySearchInput {
            energy: -0.4,
            node_count: 5,
            target_node_count: 3,
            energy_sup: -0.7,
            energy_inf: 1.0,
            energy_floor: -5.0,
            energy_precision: 1.0e-7,
            search_attempt_count: 2,
            max_attempt_count: 50,
        })?;
        assert_close_with(too_many_bisect.energy, -5.500_000_000_000_000_4e-1, 1.0e-18);
        assert_close_with(
            too_many_bisect.energy_sup,
            -6.999_999_999_999_999_6e-1,
            1.0e-18,
        );
        assert_close_with(
            too_many_bisect.energy_inf,
            -4.000_000_000_000_000_2e-1,
            1.0e-18,
        );
        assert_eq!(too_many_bisect.search_attempt_count, 3);
        assert!(too_many_bisect.needs_reintegration);
        assert!(!too_many_bisect.attempts_exhausted);

        let matched = atomic_dirac_node_energy_search(AtomicDiracNodeEnergySearchInput {
            energy: -0.4,
            node_count: 3,
            target_node_count: 3,
            energy_sup: -0.7,
            energy_inf: -0.2,
            energy_floor: -5.0,
            energy_precision: 1.0e-7,
            search_attempt_count: 2,
            max_attempt_count: 50,
        })?;
        assert_close_with(matched.energy, -4.000_000_000_000_000_2e-1, 1.0e-18);
        assert_close_with(matched.energy_sup, -6.999_999_999_999_999_6e-1, 1.0e-18);
        assert_close_with(matched.energy_inf, -2.000_000_000_000_000_1e-1, 1.0e-18);
        assert_eq!(matched.search_attempt_count, 2);
        assert!(!matched.needs_reintegration);
        assert!(!matched.attempts_exhausted);

        let exhausted = atomic_dirac_node_energy_search(AtomicDiracNodeEnergySearchInput {
            energy: -0.5,
            node_count: 2,
            target_node_count: 4,
            energy_sup: -5.0,
            energy_inf: 1.0,
            energy_floor: -5.0,
            energy_precision: 1.0e-7,
            search_attempt_count: 1,
            max_attempt_count: 1,
        })?;
        assert_close_with(exhausted.energy, -4.000_000_000_000_000_2e-1, 1.0e-18);
        assert_close_with(exhausted.energy_sup, -5.0e-1, 1.0e-18);
        assert_close_with(exhausted.energy_inf, 1.0, 1.0e-18);
        assert_eq!(exhausted.search_attempt_count, 2);
        assert!(!exhausted.needs_reintegration);
        assert!(exhausted.attempts_exhausted);

        assert!(matches!(
            atomic_dirac_node_energy_search(AtomicDiracNodeEnergySearchInput {
                energy: -1.0e-8,
                node_count: 2,
                target_node_count: 4,
                energy_sup: -5.0,
                energy_inf: 1.0,
                energy_floor: -5.0,
                energy_precision: 1.0e-7,
                search_attempt_count: 0,
                max_attempt_count: 50,
            }),
            Err(AtomMathError::DiracNodeEnergyTooSmall { .. })
        ));
        assert!(matches!(
            atomic_dirac_node_energy_search(AtomicDiracNodeEnergySearchInput {
                energy: -5.0,
                node_count: 5,
                target_node_count: 3,
                energy_sup: -5.5,
                energy_inf: 1.0,
                energy_floor: -5.5,
                energy_precision: 1.0e-7,
                search_attempt_count: 0,
                max_attempt_count: 50,
            }),
            Err(AtomMathError::DiracNodeEnergyBelowPotentialFloor { .. })
        ));
        assert!(matches!(
            atomic_dirac_node_energy_search(AtomicDiracNodeEnergySearchInput {
                energy: -0.5,
                node_count: 2,
                target_node_count: 4,
                energy_sup: -5.0,
                energy_inf: -0.500_000_05,
                energy_floor: -5.0,
                energy_precision: 1.0e-6,
                search_attempt_count: 0,
                max_attempt_count: 50,
            }),
            Err(AtomMathError::DiracNodeEnergyBracketCollapsed { .. })
        ));
        Ok(())
    }

    #[allow(clippy::excessive_precision)]
    #[test]
    fn atom_dirac_energy_correction_matches_feff_soldir_reference() -> Result<(), AtomMathError> {
        let large_component = Array1::from_vec(vec![0.12, -0.22, 0.31, 0.27, -0.18]);
        let small_component = Array1::from_vec(vec![-0.011, 0.024, 0.047, -0.018, 0.009]);

        let scaled =
            atomic_dirac_method_one_energy_correction(AtomicDiracMethodOneEnergyCorrectionInput {
                speed_of_light: 137.0373,
                norm: 2.6,
                large_component: large_component.view(),
                small_component: small_component.view(),
                matching_small_component: 0.052,
                matching_index_1based: 3,
            })?;
        assert_close_with(scaled.correction, 8.169_531_346_153_841_0e-2, 1.0e-16);
        assert_close_with(scaled.mismatch, 9.615_384_615_384_610_4e-2, 1.0e-16);

        let zero_matching =
            atomic_dirac_method_one_energy_correction(AtomicDiracMethodOneEnergyCorrectionInput {
                speed_of_light: 137.0373,
                norm: 1.9,
                large_component: large_component.view(),
                small_component: small_component.view(),
                matching_small_component: 0.0,
                matching_index_1based: 4,
            })?;
        assert_close_with(
            zero_matching.correction,
            3.505_269_884_210_525_7e-1,
            1.0e-16,
        );
        assert_close_with(zero_matching.mismatch, 1.8e-2, 1.0e-18);

        let accepted = atomic_dirac_energy_step(AtomicDiracEnergyStepInput {
            energy: -0.5,
            correction: -0.02,
            mismatch: 0.001,
            energy_sup: -0.8,
            energy_inf: -0.2,
            mismatch_precision: 0.01,
            zero_energy_precision: 1.0e-7,
        })?;
        assert_close_with(accepted.energy, -5.2e-1, 1.0e-18);
        assert_close_with(accepted.correction, -2.0e-2, 1.0e-18);
        assert_close_with(accepted.relative_step, 4.0e-2, 1.0e-18);
        assert!(!accepted.needs_rematch);

        let positive_halved = atomic_dirac_energy_step(AtomicDiracEnergyStepInput {
            energy: -0.05,
            correction: 0.08,
            mismatch: 0.001,
            energy_sup: -0.8,
            energy_inf: -0.02,
            mismatch_precision: 0.01,
            zero_energy_precision: 1.0e-7,
        })?;
        assert_close_with(positive_halved.energy, -4.0e-2, 1.0e-18);
        assert_close_with(positive_halved.correction, 1.0e-2, 1.0e-18);
        assert_close_with(
            positive_halved.relative_step,
            1.999_999_999_999_999_8e-1,
            1.0e-18,
        );
        assert!(!positive_halved.needs_rematch);

        let bound_halved = atomic_dirac_energy_step(AtomicDiracEnergyStepInput {
            energy: -1.0,
            correction: 0.30,
            mismatch: 0.4,
            energy_sup: -1.2,
            energy_inf: -0.8,
            mismatch_precision: 0.1,
            zero_energy_precision: 1.0e-7,
        })?;
        assert_close_with(bound_halved.energy, -8.5e-1, 1.0e-18);
        assert_close_with(bound_halved.correction, 1.5e-1, 1.0e-18);
        assert_close_with(bound_halved.relative_step, 1.5e-1, 1.0e-18);
        assert!(bound_halved.needs_rematch);

        let too_small = atomic_dirac_energy_step(AtomicDiracEnergyStepInput {
            energy: -1.0,
            correction: 1.0e-9,
            mismatch: 1.0,
            energy_sup: -0.5,
            energy_inf: -0.6,
            mismatch_precision: 0.1,
            zero_energy_precision: 1.0e-7,
        });
        let Err(AtomMathError::DiracEnergyCorrectionTooSmall { relative_step }) = too_small else {
            return Err(AtomMathError::NonFiniteScalar {
                field: "soldir_energy_too_small_reference",
                value: 0.0,
            });
        };
        assert_close_with(relative_step, 5.0e-10, 1.0e-24);
        Ok(())
    }

    #[allow(clippy::excessive_precision)]
    #[test]
    fn atom_dirac_loop_state_matches_feff_soldir_reference() -> Result<(), AtomMathError> {
        let far_energy = atomic_dirac_shooting_pass_setup(AtomicDiracShootingPassSetupInput {
            energy: -0.5,
            previous_energy: 1.0,
        })?;
        assert_eq!(
            far_energy.integration_mode,
            AtomicDiracIntegrationMode::SearchMatchingPoint
        );
        assert!(!far_energy.relocated);
        assert_close_with(far_energy.reference_energy, -5.0e-1, 1.0e-18);
        assert_close_with(far_energy.relative_energy_change, 3.0, 1.0e-18);

        let near_energy = atomic_dirac_shooting_pass_setup(AtomicDiracShootingPassSetupInput {
            energy: -0.5,
            previous_energy: -0.54,
        })?;
        assert_eq!(
            near_energy.integration_mode,
            AtomicDiracIntegrationMode::FixedMatchingPoint
        );
        assert!(!near_energy.relocated);
        assert_close_with(near_energy.reference_energy, -5.0e-1, 1.0e-18);
        assert_close_with(
            near_energy.relative_energy_change,
            8.000_000_000_000_007_1e-2,
            1.0e-17,
        );

        let far_negative = atomic_dirac_shooting_pass_setup(AtomicDiracShootingPassSetupInput {
            energy: -0.5,
            previous_energy: -0.42,
        })?;
        assert_eq!(
            far_negative.integration_mode,
            AtomicDiracIntegrationMode::SearchMatchingPoint
        );
        assert_close_with(
            far_negative.relative_energy_change,
            1.600_000_000_000_000_3e-1,
            1.0e-17,
        );

        let below_test = atomic_dirac_rematch_attempt(AtomicDiracRematchAttemptInput {
            mismatch: 0.005,
            mismatch_precision: 0.01,
            match_attempt_count: 3,
            max_attempt_count: 5,
        })?;
        assert_eq!(below_test.match_attempt_count, 3);
        assert!(!below_test.needs_rematch);
        assert!(!below_test.attempts_exhausted);

        let retry_left = atomic_dirac_rematch_attempt(AtomicDiracRematchAttemptInput {
            mismatch: 0.02,
            mismatch_precision: 0.01,
            match_attempt_count: 4,
            max_attempt_count: 5,
        })?;
        assert_eq!(retry_left.match_attempt_count, 5);
        assert!(retry_left.needs_rematch);
        assert!(!retry_left.attempts_exhausted);

        let exhausted = atomic_dirac_rematch_attempt(AtomicDiracRematchAttemptInput {
            mismatch: 0.02,
            mismatch_precision: 0.01,
            match_attempt_count: 5,
            max_attempt_count: 5,
        })?;
        assert_eq!(exhausted.match_attempt_count, 6);
        assert!(!exhausted.needs_rematch);
        assert!(exhausted.attempts_exhausted);
        Ok(())
    }

    #[allow(clippy::excessive_precision)]
    #[test]
    fn atom_dirac_integration_seeds_match_feff_soldir_reference() -> Result<(), AtomMathError> {
        let radial_count = 8;
        let coefficient_count = 5;
        let large_source = Array1::from_shape_fn(radial_count, |row| {
            let index = (row + 1) as Real;
            0.05 * index + 0.003 * index * index
        });
        let small_source = Array1::from_shape_fn(radial_count, |row| {
            let index = (row + 1) as Real;
            -0.04 * index + 0.002 * index * index
        });
        let large_source_coefficients = Array1::from_shape_fn(coefficient_count, |row| {
            let index = (row + 1) as Real;
            0.11 * index - 0.004 * index * index
        });
        let small_source_coefficients = Array1::from_shape_fn(coefficient_count, |row| {
            let index = (row + 1) as Real;
            -0.09 * index + 0.005 * index * index
        });

        let inhomogeneous = atomic_dirac_inhomogeneous_seed(AtomicDiracInhomogeneousSeedInput {
            large_source: large_source.view(),
            small_source: small_source.view(),
            large_source_coefficients: large_source_coefficients.view(),
            small_source_coefficients: small_source_coefficients.view(),
            coefficient_count,
        })?;
        assert_close_with(
            inhomogeneous.large_source[0],
            5.300_000_000_000_000_5e-2,
            1.0e-18,
        );
        assert_close_with(
            inhomogeneous.large_source[7],
            5.920_000_000_000_000_8e-1,
            1.0e-17,
        );
        assert_close_with(
            inhomogeneous.small_source[4],
            -1.500_000_000_000_000_2e-1,
            1.0e-18,
        );
        assert_close_with(inhomogeneous.large_coefficients[0], 0.0, 1.0e-18);
        assert_close_with(
            inhomogeneous.large_coefficients[1],
            1.060_000_000_000_000_0e-1,
            1.0e-18,
        );
        assert_close_with(
            inhomogeneous.large_coefficients[4],
            3.760_000_000_000_000_0e-1,
            1.0e-18,
        );
        assert_close_with(inhomogeneous.small_coefficients[0], 0.0, 1.0e-18);
        assert_close_with(
            inhomogeneous.small_coefficients[1],
            -8.499_999_999_999_999_2e-2,
            1.0e-18,
        );
        assert_close_with(
            inhomogeneous.small_coefficients[4],
            -2.799_999_999_999_999_7e-1,
            1.0e-18,
        );

        let homogeneous = atomic_dirac_homogeneous_seed(AtomicDiracHomogeneousSeedInput {
            radial_len: radial_count,
            coefficient_len: coefficient_count,
        })?;
        assert_eq!(homogeneous.large_source.len(), radial_count);
        assert_eq!(homogeneous.small_source.len(), radial_count);
        assert_eq!(homogeneous.large_coefficients.len(), coefficient_count);
        assert_eq!(homogeneous.small_coefficients.len(), coefficient_count);
        assert!(homogeneous.large_source.iter().all(|&value| value == 0.0));
        assert!(homogeneous.small_source.iter().all(|&value| value == 0.0));
        assert!(
            homogeneous
                .large_coefficients
                .iter()
                .all(|&value| value == 0.0)
        );
        assert!(
            homogeneous
                .small_coefficients
                .iter()
                .all(|&value| value == 0.0)
        );
        Ok(())
    }

    #[allow(clippy::excessive_precision)]
    #[test]
    fn atom_dirac_matching_matches_feff_soldir_reference() -> Result<(), AtomMathError> {
        let large_component = Array1::from_shape_fn(8, |row| {
            let index = (row + 1) as Real;
            0.08 * index - 0.006 * index * index
        });
        let small_component = Array1::from_shape_fn(8, |row| {
            let index = (row + 1) as Real;
            -0.025 * index + 0.0015 * index * index
        });
        let homogeneous_large_component = Array1::from_shape_fn(8, |row| {
            let index = (row + 1) as Real;
            0.018 * index + 0.0007 * index * index
        });
        let homogeneous_small_component = Array1::from_shape_fn(8, |row| {
            let index = (row + 1) as Real;
            -0.012 * index + 0.0004 * index * index
        });

        let homogeneous_match = atomic_dirac_homogeneous_match(AtomicDiracHomogeneousMatchInput {
            large_component: large_component.view(),
            small_component: small_component.view(),
            matching_large_component: 0.240,
            active_len: 7,
            matching_index_1based: 4,
        })?;
        assert_close_with(
            homogeneous_match.tail_scale,
            1.071_428_571_428_571_4,
            1.0e-16,
        );
        assert_eq!(homogeneous_match.scan_index_1based, 4);
        assert_close_with(homogeneous_match.large_component[0], 7.4e-2, 1.0e-18);
        assert_close_with(
            homogeneous_match.large_component[3],
            2.399_999_999_999_999_9e-1,
            1.0e-18,
        );
        assert_close_with(
            homogeneous_match.large_component[6],
            2.850_000_000_000_000_3e-1,
            1.0e-16,
        );
        assert_close_with(
            homogeneous_match.large_component[7],
            large_component[7],
            1.0e-18,
        );
        assert_close_with(
            homogeneous_match.small_component[3],
            -8.142_857_142_857_143_3e-2,
            1.0e-17,
        );
        assert_close_with(
            homogeneous_match.small_component[6],
            -1.087_500_000_000_000_0e-1,
            1.0e-17,
        );

        let large_match =
            atomic_dirac_large_component_match(AtomicDiracLargeComponentMatchInput {
                large_component: large_component.view(),
                small_component: small_component.view(),
                homogeneous_large_component: homogeneous_large_component.view(),
                homogeneous_small_component: homogeneous_small_component.view(),
                matching_large_component: 0.240,
                active_len: 7,
                matching_index_1based: 4,
            })?;
        assert_close_with(large_match.tail_scale, 1.923_076_923_076_921_5e-1, 1.0e-16);
        assert_close_with(large_match.large_mismatch, -1.6e-2, 1.0e-16);
        assert_close_with(large_match.large_component[3], 2.4e-1, 1.0e-18);
        assert_close_with(
            large_match.large_component[6],
            2.968_269_230_769_230_4e-1,
            1.0e-16,
        );
        assert_close_with(
            large_match.small_component[6],
            -1.138_846_153_846_153_9e-1,
            1.0e-16,
        );
        assert_close_with(large_match.large_component[7], large_component[7], 1.0e-18);

        let large_coefficients = Array1::from_shape_fn(4, |row| {
            let index = (row + 1) as Real;
            0.11 * index - 0.004 * index * index
        });
        let small_coefficients = Array1::from_shape_fn(4, |row| {
            let index = (row + 1) as Real;
            -0.07 * index + 0.003 * index * index
        });
        let homogeneous_large_coefficients = Array1::from_shape_fn(4, |row| {
            let index = (row + 1) as Real;
            0.012 * index + 0.0005 * index * index
        });
        let homogeneous_small_coefficients = Array1::from_shape_fn(4, |row| {
            let index = (row + 1) as Real;
            -0.009 * index + 0.0003 * index * index
        });

        let two_match = atomic_dirac_two_component_match(AtomicDiracTwoComponentMatchInput {
            large_component: large_component.view(),
            small_component: small_component.view(),
            large_coefficients: large_coefficients.view(),
            small_coefficients: small_coefficients.view(),
            homogeneous_large_component: homogeneous_large_component.view(),
            homogeneous_small_component: homogeneous_small_component.view(),
            homogeneous_large_coefficients: homogeneous_large_coefficients.view(),
            homogeneous_small_coefficients: homogeneous_small_coefficients.view(),
            matching_large_component: 0.285,
            matching_small_component: -0.068,
            homogeneous_matching_large_component: 0.087,
            homogeneous_matching_small_component: -0.047,
            coefficient_count: 4,
            active_len: 8,
            matching_index_1based: 5,
        })?;
        assert_close_with(two_match.determinant, -7.025e-4, 1.0e-18);
        assert_close_with(two_match.tail_scale, 4.756_583_629_893_235_4, 1.0e-15);
        assert_close_with(two_match.prefix_scale, 5.475_088_967_971_526_4, 1.0e-15);
        assert_close_with(two_match.large_mismatch, -3.5e-2, 1.0e-16);
        assert_close_with(two_match.small_mismatch, -1.95e-2, 1.0e-16);
        assert_close_with(
            two_match.large_component[0],
            1.763_841_637_010_675_2e-1,
            1.0e-16,
        );
        assert_close_with(
            two_match.large_component[4],
            7.613_327_402_135_228_2e-1,
            1.0e-15,
        );
        assert_close_with(
            two_match.small_component[4],
            -3.253_291_814_946_617_2e-1,
            1.0e-15,
        );
        assert_close_with(
            two_match.large_coefficients[3],
            6.826_049_822_064_055_3e-1,
            1.0e-15,
        );
        Ok(())
    }

    #[allow(clippy::excessive_precision)]
    #[test]
    fn atom_dirac_energy_disagreement_match_matches_feff_soldir_reference()
    -> Result<(), AtomMathError> {
        let radial_count = 8;
        let coefficient_count = 5;
        let large_derivative = Array1::from_shape_fn(radial_count, |row| {
            let index = (row + 1) as Real;
            0.004 * index + 0.0005 * index * index
        });
        let small_derivative = Array1::from_shape_fn(radial_count, |row| {
            let index = (row + 1) as Real;
            -0.003 * index + 0.0002 * index * index
        });
        let homogeneous_large_component = Array1::from_shape_fn(radial_count, |row| {
            let index = (row + 1) as Real;
            0.018 * index + 0.0007 * index * index
        });
        let homogeneous_small_component = Array1::from_shape_fn(radial_count, |row| {
            let index = (row + 1) as Real;
            -0.012 * index + 0.0004 * index * index
        });
        let large_derivative_coefficients = Array1::from_shape_fn(coefficient_count, |row| {
            let index = (row + 1) as Real;
            0.0008 * index + 0.00007 * index * index
        });
        let small_derivative_coefficients = Array1::from_shape_fn(coefficient_count, |row| {
            let index = (row + 1) as Real;
            -0.0006 * index + 0.00005 * index * index
        });
        let homogeneous_large_coefficients = Array1::from_shape_fn(coefficient_count, |row| {
            let index = (row + 1) as Real;
            0.012 * index + 0.0005 * index * index
        });
        let homogeneous_small_coefficients = Array1::from_shape_fn(coefficient_count, |row| {
            let index = (row + 1) as Real;
            -0.009 * index + 0.0003 * index * index
        });

        let matched =
            atomic_dirac_energy_disagreement_match(AtomicDiracEnergyDisagreementMatchInput {
                large_derivative: large_derivative.view(),
                small_derivative: small_derivative.view(),
                large_derivative_coefficients: large_derivative_coefficients.view(),
                small_derivative_coefficients: small_derivative_coefficients.view(),
                homogeneous_large_component: homogeneous_large_component.view(),
                homogeneous_small_component: homogeneous_small_component.view(),
                homogeneous_large_coefficients: homogeneous_large_coefficients.view(),
                homogeneous_small_coefficients: homogeneous_small_coefficients.view(),
                matching_large_derivative: 0.037,
                matching_small_derivative: -0.011,
                homogeneous_matching_large_component: 0.087,
                homogeneous_matching_small_component: -0.047,
                coefficient_count,
                active_len: radial_count,
                matching_index_1based: 5,
            })?;

        assert_close_with(matched.determinant, -7.025e-4, 1.0e-18);
        assert_close_with(matched.prefix_scale, 1.672_597_864_768_679_6e-1, 1.0e-16);
        assert_close_with(matched.tail_scale, 1.772_241_992_882_559_2e-1, 1.0e-16);
        assert_close_with(matched.large_mismatch, -4.499_999_999_999_997_1e-3, 1.0e-18);
        assert_close_with(matched.small_mismatch, 1.000_000_000_000_000_9e-3, 1.0e-18);
        assert_close_with(
            matched.large_derivative[0],
            7.627_758_007_117_430_9e-3,
            1.0e-18,
        );
        assert_close_with(
            matched.large_derivative[4],
            5.155_160_142_348_751_1e-2,
            1.0e-17,
        );
        assert_close_with(
            matched.small_derivative[4],
            -1.886_120_996_441_279_3e-2,
            1.0e-17,
        );
        assert_close_with(
            matched.large_derivative_coefficients[4],
            1.787_633_451_957_292_7e-2,
            1.0e-17,
        );
        assert_close_with(
            matched.small_derivative_coefficients[4],
            -8.022_241_992_882_548_1e-3,
            1.0e-18,
        );
        Ok(())
    }

    #[allow(clippy::excessive_precision)]
    #[test]
    fn atom_dirac_energy_disagreement_source_matches_feff_soldir_reference()
    -> Result<(), AtomMathError> {
        let radial_count = 8;
        let coefficient_count = 5;
        let radii = Array1::from_shape_fn(radial_count, |row| 0.08 * (0.11 * row as Real).exp());
        let large_component = Array1::from_shape_fn(radial_count, |row| {
            let index = (row + 1) as Real;
            0.06 * index - 0.002 * index * index
        });
        let small_component = Array1::from_shape_fn(radial_count, |row| {
            let index = (row + 1) as Real;
            -0.015 * index + 0.0007 * index * index
        });
        let large_coefficients = Array1::from_shape_fn(coefficient_count, |row| {
            let index = (row + 1) as Real;
            0.12 * index - 0.004 * index * index
        });
        let small_coefficients = Array1::from_shape_fn(coefficient_count, |row| {
            let index = (row + 1) as Real;
            -0.08 * index + 0.003 * index * index
        });

        let source =
            atomic_dirac_energy_disagreement_source(AtomicDiracEnergyDisagreementSourceInput {
                large_component: large_component.view(),
                small_component: small_component.view(),
                large_coefficients: large_coefficients.view(),
                small_coefficients: small_coefficients.view(),
                radii: radii.view(),
                speed_of_light: 137.0373,
                coefficient_count,
                active_len: 7,
            })?;

        assert_close_with(source.large_coefficients[0], 0.0, 1.0e-18);
        assert_close_with(
            source.large_coefficients[1],
            8.464_848_621_506_699_7e-4,
            1.0e-18,
        );
        assert_close_with(
            source.large_coefficients[2],
            1.634_591_457_946_121_3e-3,
            1.0e-18,
        );
        assert_close_with(
            source.large_coefficients[3],
            2.364_319_787_386_353_7e-3,
            1.0e-18,
        );
        assert_close_with(
            source.large_coefficients[4],
            3.035_669_850_471_368_2e-3,
            1.0e-18,
        );
        assert_close_with(source.small_coefficients[0], 0.0, 1.0e-18);
        assert_close_with(
            source.small_coefficients[1],
            -5.618_908_136_689_792_0e-4,
            1.0e-18,
        );
        assert_close_with(
            source.small_coefficients[2],
            -1.079_997_927_571_544_4e-3,
            1.0e-18,
        );
        assert_close_with(
            source.small_coefficients[3],
            -1.554_321_341_707_695_8e-3,
            1.0e-18,
        );
        assert_close_with(
            source.small_coefficients[4],
            -1.984_861_056_077_433_2e-3,
            1.0e-18,
        );
        assert_close_with(source.large_source[0], 3.385_939_448_602_680_0e-5, 1.0e-19);
        assert_close_with(source.large_source[3], 1.689_008_004_217_633_1e-4, 1.0e-18);
        assert_close_with(source.large_source[6], 3.636_984_276_120_176_0e-4, 1.0e-18);
        assert_close_with(source.large_source[7], 0.0, 1.0e-18);
        assert_close_with(source.small_source[0], -8.348_092_088_796_263_3e-6, 1.0e-20);
        assert_close_with(source.small_source[3], -3.962_672_625_279_831_4e-5, 1.0e-19);
        assert_close_with(source.small_source[6], -7.985_552_432_350_822_2e-5, 1.0e-19);
        assert_close_with(source.small_source[7], 0.0, 1.0e-18);
        Ok(())
    }

    #[allow(clippy::excessive_precision)]
    #[test]
    fn atom_dirac_energy_disagreement_correction_matches_feff_soldir_reference()
    -> Result<(), AtomMathError> {
        let radial_count = 8;
        let coefficient_count = 5;
        let radii = Array1::from_shape_fn(radial_count, |row| 0.08 * (0.11 * row as Real).exp());
        let large_component = Array1::from_shape_fn(radial_count, |row| {
            let index = (row + 1) as Real;
            0.08 * index - 0.003 * index * index
        });
        let small_component = Array1::from_shape_fn(radial_count, |row| {
            let index = (row + 1) as Real;
            -0.018 * index + 0.0008 * index * index
        });
        let large_derivative = Array1::from_shape_fn(radial_count, |row| {
            let index = (row + 1) as Real;
            0.002 * index + 0.0003 * index * index
        });
        let small_derivative = Array1::from_shape_fn(radial_count, |row| {
            let index = (row + 1) as Real;
            -0.0014 * index + 0.0001 * index * index
        });
        let large_coefficients = Array1::from_shape_fn(coefficient_count, |row| {
            let index = (row + 1) as Real;
            0.13 * index - 0.005 * index * index
        });
        let small_coefficients = Array1::from_shape_fn(coefficient_count, |row| {
            let index = (row + 1) as Real;
            -0.09 * index + 0.0035 * index * index
        });
        let large_derivative_coefficients = Array1::from_shape_fn(coefficient_count, |row| {
            let index = (row + 1) as Real;
            0.0007 * index + 0.00004 * index * index
        });
        let small_derivative_coefficients = Array1::from_shape_fn(coefficient_count, |row| {
            let index = (row + 1) as Real;
            -0.0005 * index + 0.00003 * index * index
        });

        let correction = atomic_dirac_energy_disagreement_correction(
            AtomicDiracEnergyDisagreementCorrectionInput {
                radii: radii.view(),
                large_component: large_component.view(),
                small_component: small_component.view(),
                large_derivative: large_derivative.view(),
                small_derivative: small_derivative.view(),
                large_coefficients: large_coefficients.view(),
                small_coefficients: small_coefficients.view(),
                large_derivative_coefficients: large_derivative_coefficients.view(),
                small_derivative_coefficients: small_derivative_coefficients.view(),
                norm: 0.913,
                step: 0.11,
                origin_power: 1.30,
                coefficient_count,
                active_len: 7,
            },
        )?;

        assert_close_with(
            correction.overlap_integral,
            3.960_742_076_990_347_3e-4,
            1.0e-18,
        );
        assert_close_with(correction.correction, 1.098_279_038_483_979_4e2, 1.0e-12);
        assert_close_with(
            correction.normalization_mismatch,
            8.699_999_999_999_996_6e-2,
            1.0e-18,
        );
        assert_close_with(
            correction.large_component[0],
            3.296_041_788_513_152_7e-1,
            1.0e-16,
        );
        assert_close_with(
            correction.large_component[3],
            1.677_797_169_259_493_5,
            1.0e-15,
        );
        assert_close_with(
            correction.large_component[6],
            3.565_060_840_449_020_5,
            1.0e-15,
        );
        assert_close_with(correction.large_component[7], 4.48e-1, 1.0e-18);
        assert_close_with(
            correction.small_component[0],
            -1.599_762_750_029_173_1e-1,
            1.0e-16,
        );
        assert_close_with(
            correction.small_component[6],
            -6.249_567_288_571_498_1e-1,
            1.0e-16,
        );
        assert_close_with(correction.small_component[7], -9.28e-2, 1.0e-18);
        assert_close_with(
            correction.large_coefficients[4],
            1.019_225_567_317_790_8,
            1.0e-15,
        );
        assert_close_with(
            correction.small_coefficients[4],
            -5.546_988_317_346_963_6e-1,
            1.0e-16,
        );
        Ok(())
    }

    #[test]
    fn atom_dirac_matching_point_update_matches_feff_soldir_reference() -> Result<(), AtomMathError>
    {
        let mut large_component = Array1::<Real>::zeros(25);
        large_component[2] = 0.60;
        large_component[4] = 0.40;
        let no_update = atomic_dirac_matching_point_update(AtomicDiracMatchingPointUpdateInput {
            large_component: large_component.view(),
            active_len: 13,
            matching_index_1based: 5,
            already_relocated: false,
        })?;
        assert_eq!(no_update.matching_index_1based, 5);
        assert_eq!(no_update.peak_index_1based, 3);
        assert_eq!(no_update.scan_index_1based, 5);
        assert!(!no_update.relocated);
        assert!(!no_update.needs_reintegration);

        large_component.fill(0.0);
        large_component[5] = 0.90;
        let reintegrate_even =
            atomic_dirac_matching_point_update(AtomicDiracMatchingPointUpdateInput {
                large_component: large_component.view(),
                active_len: 21,
                matching_index_1based: 3,
                already_relocated: false,
            })?;
        assert_eq!(reintegrate_even.matching_index_1based, 7);
        assert_eq!(reintegrate_even.peak_index_1based, 6);
        assert_eq!(reintegrate_even.scan_index_1based, 7);
        assert!(reintegrate_even.relocated);
        assert!(reintegrate_even.needs_reintegration);

        large_component.fill(0.0);
        large_component[17] = 0.90;
        let fallback_tail =
            atomic_dirac_matching_point_update(AtomicDiracMatchingPointUpdateInput {
                large_component: large_component.view(),
                active_len: 21,
                matching_index_1based: 5,
                already_relocated: false,
            })?;
        assert_eq!(fallback_tail.matching_index_1based, 9);
        assert_eq!(fallback_tail.peak_index_1based, 18);
        assert_eq!(fallback_tail.scan_index_1based, 9);
        assert!(fallback_tail.relocated);
        assert!(!fallback_tail.needs_reintegration);

        let already_moved =
            atomic_dirac_matching_point_update(AtomicDiracMatchingPointUpdateInput {
                large_component: large_component.view(),
                active_len: 21,
                matching_index_1based: 5,
                already_relocated: true,
            })?;
        assert_eq!(already_moved.matching_index_1based, 5);
        assert_eq!(already_moved.peak_index_1based, 18);
        assert_eq!(already_moved.scan_index_1based, 18);
        assert!(already_moved.relocated);
        assert!(!already_moved.needs_reintegration);
        Ok(())
    }

    #[allow(clippy::excessive_precision)]
    #[test]
    fn atom_dirac_solver_setup_matches_feff_soldir_reference() -> Result<(), AtomMathError> {
        let fixture = sample_soldir_setup_fixture();

        let clamped = atomic_dirac_solver_setup(fixture.input(-8.0, 0, -2, 2, true))?;
        assert_eq!(clamped.requested_method, 0);
        assert_eq!(clamped.method, 1);
        assert_eq!(clamped.target_nodes, 1);
        assert_close_with(clamped.energy, -5.963_839_259_330_666_4, 1.0e-14);
        assert_close_with(clamped.energy_floor, -6.626_488_065_922_962_4, 1.0e-14);
        assert_close_with(
            clamped.initial_small_coefficient,
            -1.472_928_410_311_296_5e-2,
            1.0e-16,
        );
        assert_close_with(clamped.angular_term, 7.297_283_294_402_327_9e-3, 1.0e-18);
        assert_close_with(clamped.doubled_speed_of_light, 274.0746, 1.0e-12);

        let positive_kappa = atomic_dirac_solver_setup(fixture.input(-0.2, 2, 1, 3, true))?;
        assert_eq!(positive_kappa.requested_method, 2);
        assert_eq!(positive_kappa.method, 2);
        assert_eq!(positive_kappa.target_nodes, 2);
        assert_close_with(positive_kappa.energy, -0.2, 1.0e-18);
        assert_close_with(
            positive_kappa.energy_floor,
            -6.626_488_065_922_962_4,
            1.0e-14,
        );
        assert_close_with(
            positive_kappa.initial_small_coefficient,
            3.160_423_066_381_816_8e1,
            1.0e-13,
        );
        assert_close_with(
            positive_kappa.angular_term,
            7.297_283_294_402_327_9e-3,
            1.0e-18,
        );

        let no_adjust = atomic_dirac_solver_setup(fixture.input(-0.1, -1, -1, 1, false))?;
        assert_eq!(no_adjust.requested_method, -1);
        assert_eq!(no_adjust.method, 1);
        assert_eq!(no_adjust.target_nodes, 1);
        assert_close_with(no_adjust.energy, -0.1, 1.0e-18);
        assert_close_with(no_adjust.energy_floor, -5.619_077_423_139_916_0e1, 1.0e-13);
        assert_close_with(no_adjust.initial_small_coefficient, -6.0e-3, 1.0e-18);
        assert_close_with(no_adjust.angular_term, 0.0, 1.0e-18);
        Ok(())
    }

    #[allow(clippy::excessive_precision)]
    #[test]
    fn atom_dirac_integration_matches_feff_intdir_reference() -> Result<(), AtomMathError> {
        let fixture = sample_intdir_fixture();

        let searched = atomic_dirac_integration(fixture.input(
            AtomicDiracIntegrationMode::SearchMatchingPoint,
            0,
            0,
        ))?;
        assert_eq!(searched.matching_index_1based, 127);
        assert_eq!(searched.max_index_1based, 151);
        assert_some_close(
            searched.matching_large_component,
            7.844_180_279_031_651_7e-1,
            1.0e-12,
        );
        assert_some_close(
            searched.matching_small_component,
            6.433_852_518_326_962_0e-4,
            1.0e-15,
        );
        assert_close_with(
            searched.large_component[126],
            3.946_584_591_497_206_1e2,
            1.0e-9,
        );
        assert_close_with(
            searched.small_component[126],
            -5.380_100_169_329_787_9e-1,
            1.0e-12,
        );
        assert_close_with(
            searched.large_coefficients[1],
            -1.096_438_489_149_803_4,
            1.0e-12,
        );
        assert_close_with(
            searched.small_coefficients[1],
            2.146_028_457_009_671_9e-2,
            1.0e-14,
        );
        assert_close_with(
            searched.large_component[150],
            7.844_180_279_031_651_3e-8,
            1.0e-20,
        );
        assert_close_with(
            searched.small_component[150],
            -1.144_825_333_416_651_0e-10,
            1.0e-22,
        );

        let fixed = atomic_dirac_integration(fixture.input(
            AtomicDiracIntegrationMode::FixedMatchingPoint,
            65,
            139,
        ))?;
        assert_eq!(fixed.matching_index_1based, 65);
        assert_eq!(fixed.max_index_1based, 139);
        assert_some_close(
            fixed.matching_large_component,
            -4.787_017_896_869_409_0e-2,
            1.0e-13,
        );
        assert_some_close(
            fixed.matching_small_component,
            2.893_471_976_931_037_7e-3,
            1.0e-15,
        );
        assert_close_with(fixed.large_component[64], 2.250_038_459_307_619_5, 1.0e-13);
        assert_close_with(
            fixed.small_component[64],
            1.444_514_204_264_709_7e-2,
            1.0e-15,
        );
        assert_close_with(
            fixed.large_coefficients[1],
            -1.096_438_489_149_803_4,
            1.0e-12,
        );
        assert_close_with(
            fixed.small_coefficients[1],
            2.146_028_457_009_671_9e-2,
            1.0e-14,
        );
        assert_close_with(fixed.large_component[138], 2.0e-2, 1.0e-20);
        assert_close_with(
            fixed.small_component[138],
            -2.918_916_426_428_632_8e-5,
            1.0e-22,
        );

        let inward = atomic_dirac_integration(fixture.input(
            AtomicDiracIntegrationMode::InwardOnly,
            65,
            139,
        ))?;
        assert_eq!(inward.matching_large_component, None);
        assert_eq!(inward.matching_small_component, None);
        assert_eq!(inward.matching_index_1based, 65);
        assert_eq!(inward.max_index_1based, 139);
        assert_close_with(inward.large_component[64], 2.250_038_459_307_619_5, 1.0e-13);
        assert_close_with(
            inward.small_component[64],
            1.444_514_204_264_709_7e-2,
            1.0e-15,
        );
        assert_close_with(inward.large_coefficients[1], 4.0e-4, 1.0e-18);
        assert_close_with(inward.small_coefficients[1], -3.0e-4, 1.0e-18);
        assert_close_with(inward.large_component[138], 2.0e-2, 1.0e-20);
        assert_close_with(
            inward.small_component[138],
            -2.918_916_426_428_632_8e-5,
            1.0e-22,
        );
        Ok(())
    }

    #[test]
    fn atom_total_energy_matches_feff_etotal_reference() -> Result<(), AtomMathError> {
        let kappas = [-1, 1, -2, 2];
        let occupations = [2.0, 1.5, 3.0, 0.5];
        let valence_occupations = [0.0, 0.0, 1.0, 0.0];
        let orbital_energies = [-0.7, -0.3, -0.12, -0.05];
        let coefficients = Array3::from_shape_fn((4, 4, 6), |(row, column, channel)| {
            0.01 * (100 * (row + 1) + 10 * (column + 1) + channel + 1) as Real
        });

        let energy = atomic_total_energy(
            AtomicTotalEnergyInput {
                kappas: &kappas,
                occupations: &occupations,
                valence_occupations: &valence_occupations,
                orbital_energies: &orbital_energies,
                coulomb_coefficients: coefficients.view(),
            },
            |request| {
                Ok(0.0001 * (request.rank + 1) as Real
                    + 0.001 * request.first_left as Real
                    + 0.0002 * request.first_right as Real
                    + 0.00003 * request.second_left as Real
                    + 0.000004 * request.second_right as Real)
            },
        )?;

        assert_close(energy.total, -2.230_065_144_829_932);
        assert_close_with(energy.direct_coulomb, 0.109_629, 1.0e-6);
        assert_close_with(energy.exchange_coulomb, -0.055_702_8, 1.0e-6);
        assert_close_with(energy.magnetic_breit, 0.075_902_3, 1.0e-6);
        assert_close_with(energy.retarded_breit, -0.017_041_4, 1.0e-6);
        Ok(())
    }

    #[test]
    fn atom_lagrange_parameters_match_feff_lagdat_reference() -> Result<(), AtomMathError> {
        let kappas = [-1, -1, 1, 1, -2];
        let occupations = [2.0, 1.0, 1.5, 0.5, 3.0];
        let valence_occupations = [0.0, 0.0, 0.25, 0.0, 0.0];
        let shell_markers = [-1, 1, 1, 1, -1];
        let coefficients = atomic_coulomb_coefficients(AtomicCoulombCoefficientInput {
            kappas: &kappas,
            occupations: &occupations,
            valence_occupations: &valence_occupations,
        })?;

        let all_parameters = atomic_lagrange_parameters(
            AtomicLagrangeParametersInput {
                active_orbital_1based: None,
                include_exchange: true,
                kappas: &kappas,
                occupations: &occupations,
                shell_markers: &shell_markers,
                coulomb_coefficients: coefficients.view(),
            },
            sample_atomic_radial_integral,
        )?;
        let expected_all = [
            -1.780_000_000_000_000_1e-3,
            0.0,
            0.0,
            0.0,
            0.0,
            -6.871_000_000_000_001e-3,
            0.0,
            0.0,
            0.0,
            0.0,
        ];
        for (&actual, expected) in all_parameters.iter().zip(expected_all) {
            assert_close_with(actual, expected, 1.0e-12);
        }

        let active_parameters = atomic_lagrange_parameters(
            AtomicLagrangeParametersInput {
                active_orbital_1based: Some(2),
                include_exchange: false,
                kappas: &kappas,
                occupations: &occupations,
                shell_markers: &shell_markers,
                coulomb_coefficients: coefficients.view(),
            },
            sample_atomic_radial_integral,
        )?;
        let expected_active = [
            -1.200_000_000_000_000_1e-3,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ];
        for (&actual, expected) in active_parameters.iter().zip(expected_active) {
            assert_close_with(actual, expected, 1.0e-12);
        }

        Ok(())
    }

    #[test]
    fn atom_tabulation_matches_feff_tabrat_reference() -> Result<(), AtomMathError> {
        let principal_quantum_numbers = [1, 2, 2, 3, 3];
        let kappas = [-1, -1, 1, -2, 1];
        let occupations = [2.0, 1.5, 0.5, 3.0, 0.25];
        let orbital_energies = [-0.70, -0.25, -0.18, -0.09, -0.04];
        let tabulation = atomic_tabulation(
            AtomicTabulationInput {
                principal_quantum_numbers: &principal_quantum_numbers,
                kappas: &kappas,
                occupations: &occupations,
                orbital_energies: &orbital_energies,
            },
            sample_atomic_tabrat_integral,
        )?;

        let expected = [
            (
                1,
                "s",
                2.0,
                19.047_977_2,
                [0.136, 0.134, 0.132, 0.131, 0.129, 0.128, 0.0],
                6,
            ),
            (
                2,
                "s",
                1.5,
                6.802_849,
                [0.166, 0.164, 0.162, 0.161, 0.159, 0.158, 0.0],
                6,
            ),
            (
                2,
                "p*",
                0.5,
                4.898_051_28,
                [0.196, 0.194, 0.192, 0.191, 0.189, 0.188, 0.0],
                6,
            ),
            (
                3,
                "p",
                3.0,
                2.449_025_64,
                [0.226, 0.224, 0.222, 0.221, 0.219, 0.218, 0.217],
                7,
            ),
            (
                3,
                "p*",
                0.25,
                1.088_455_84,
                [0.256, 0.254, 0.252, 0.251, 0.249, 0.248, 0.0],
                6,
            ),
        ];
        for (orbital, (nq, label, occupation, binding_energy_ev, moments, moment_count)) in
            tabulation.orbitals.iter().zip(expected)
        {
            assert_eq!(orbital.principal_quantum_number, nq);
            assert_eq!(orbital.orbital_label, label);
            assert_close(orbital.occupation, occupation);
            assert_close_with(orbital.binding_energy_ev, binding_energy_ev, 1.0e-10);
            assert_eq!(orbital.moments.len(), moment_count);
            for ((moment, &expected_value), &expected_power) in orbital
                .moments
                .iter()
                .zip(moments.iter())
                .zip(ATOM_TABRAT_MOMENT_POWERS.iter())
            {
                assert_eq!(moment.power, expected_power);
                assert_close(moment.value, expected_value);
            }
        }
        assert_eq!(tabulation.overlaps.len(), 2);
        assert_eq!(tabulation.overlaps[0].left, 0);
        assert_eq!(tabulation.overlaps[0].right, 1);
        assert_eq!(tabulation.overlaps[0].left_orbital_label, "s");
        assert_eq!(tabulation.overlaps[0].right_orbital_label, "s");
        assert_close(tabulation.overlaps[0].value, 0.15);
        assert_eq!(tabulation.overlaps[1].left, 2);
        assert_eq!(tabulation.overlaps[1].right, 4);
        assert_eq!(tabulation.overlaps[1].left_orbital_label, "p*");
        assert_eq!(tabulation.overlaps[1].right_orbital_label, "p*");
        assert_close(tabulation.overlaps[1].value, 0.23);
        Ok(())
    }

    #[allow(clippy::excessive_precision)]
    #[test]
    fn atom_differential_integral_matches_feff_dsordf_reference() -> Result<(), AtomMathError> {
        let fixture = sample_dsordf_fixture();
        let cases = [
            (
                AtomicDifferentialIntegralKind::ComponentOverlap {
                    left_orbital_1based: 1,
                    right_orbital_1based: 2,
                    multiply_by_derivative: false,
                },
                2,
                0.0,
                4.983_995_991_889_760_16e-9,
            ),
            (
                AtomicDifferentialIntegralKind::ComponentOverlap {
                    left_orbital_1based: 1,
                    right_orbital_1based: 3,
                    multiply_by_derivative: true,
                },
                -1,
                0.4,
                4.174_834_158_519_188_87e-5,
            ),
            (
                AtomicDifferentialIntegralKind::LargeSmallOverlap {
                    left_orbital_1based: 2,
                    right_orbital_1based: 3,
                    multiply_by_derivative: false,
                },
                1,
                0.0,
                -5.798_475_020_316_198_31e-8,
            ),
            (
                AtomicDifferentialIntegralKind::LargeSmallOverlap {
                    left_orbital_1based: 2,
                    right_orbital_1based: 1,
                    multiply_by_derivative: true,
                },
                0,
                0.3,
                -4.232_100_062_570_746_56e-8,
            ),
            (
                AtomicDifferentialIntegralKind::DerivativeProjection {
                    large_orbital_1based: 2,
                    small_orbital_1based: 3,
                },
                0,
                0.45,
                1.816_237_327_192_537_93e-5,
            ),
            (
                AtomicDifferentialIntegralKind::DerivativeNorm { active_len: 9 },
                0,
                0.45,
                5.411_954_636_180_096_36e-5,
            ),
        ];

        for (kind, power, origin_power, expected) in cases {
            let actual = atomic_differential_integral(fixture.input(kind, power, origin_power))?;
            assert_close_with(actual, expected, 1.0e-17);
        }
        Ok(())
    }

    #[allow(clippy::excessive_precision)]
    #[test]
    fn atom_local_density_potential_matches_feff_vlda_reference() -> Result<(), AtomMathError> {
        let fixture = sample_vlda_fixture();

        let valence = atomic_local_density_potential(
            fixture.input(AtomicLocalDensityExchangeMode::ValenceDensity, true),
        )?;
        assert_close_with(
            valence.total_density[0],
            6.809_505_899_999_999_42e-3,
            1.0e-17,
        );
        assert_close_with(
            valence.total_density[4],
            8.670_367_500_000_001_48e-3,
            1.0e-17,
        );
        assert_close_with(
            valence.total_density[9],
            4.974_400_000_000_001_22e-3,
            1.0e-17,
        );
        assert_close_with(
            valence.valence_density[0],
            2.049_973_999_999_999_98e-3,
            1.0e-17,
        );
        assert_close_with(
            valence.valence_density[4],
            2.672_390_000_000_000_62e-3,
            1.0e-17,
        );
        assert_close_with(
            valence.valence_density[9],
            1.243_600_000_000_000_30e-3,
            1.0e-17,
        );
        assert_close_with(valence.potential[0], -7.054_707_605_385_910_14e-3, 2.0e-10);
        assert_close_with(valence.potential[4], -6.362_810_615_972_094_51e-3, 2.0e-10);
        assert_close_with(valence.potential[9], -3.663_730_681_720_983_07e-3, 2.0e-10);
        assert_close_with(valence.potential[12], 1.300_000_000_000_000_16e-3, 1.0e-18);
        assert_close_with(
            valence.development_coefficients[1],
            1.284_529_239_461_408_91e-2,
            2.0e-10,
        );
        assert_close_with(
            valence.energy_density[0],
            -4.676_397_112_407_516_99e-3,
            2.0e-10,
        );
        assert_close_with(
            valence.energy_density[4],
            1.845_934_601_355_665_38e-3,
            2.0e-10,
        );
        assert_close_with(
            valence.energy_density[9],
            1.682_086_596_903_880_49e-2,
            2.0e-10,
        );
        assert_close_with(
            valence.energy_density[12],
            2.600_000_000_000_000_23e-2,
            1.0e-18,
        );

        let core = atomic_local_density_potential(
            fixture.input(AtomicLocalDensityExchangeMode::CoreDensitySeparated, true),
        )?;
        assert_close_with(core.potential[0], -4.639_483_986_312_321_55e-3, 2.0e-10);
        assert_close_with(core.potential[4], -4.094_974_008_849_363_44e-3, 2.0e-10);
        assert_close_with(core.potential[9], -1.989_064_683_335_639_83e-3, 2.0e-10);
        assert_close_with(
            core.development_coefficients[1],
            1.526_051_601_368_767_77e-2,
            2.0e-10,
        );
        assert_close_with(
            core.energy_density[0],
            -2.422_637_366_298_145_52e-3,
            2.0e-10,
        );
        assert_close_with(core.energy_density[4], 4.540_470_272_335_868_71e-3, 2.0e-10);
        assert_close_with(core.energy_density[9], 1.796_243_867_752_029_75e-2, 2.0e-10);

        let total = atomic_local_density_potential(
            fixture.input(AtomicLocalDensityExchangeMode::TotalDensity, false),
        )?;
        assert_close_with(total.potential[0], -1.030_418_779_316_292_88e-2, 2.0e-10);
        assert_close_with(total.potential[4], -9.399_113_926_789_406_24e-3, 2.0e-10);
        assert_close_with(total.potential[9], -6.124_858_580_930_082_20e-3, 2.0e-10);
        assert_close_with(
            total.energy_density[0],
            2.000_000_000_000_000_04e-3,
            1.0e-18,
        );
        assert_close_with(
            total.energy_density[4],
            1.000_000_000_000_000_02e-2,
            1.0e-18,
        );
        assert_close_with(
            total.energy_density[9],
            2.000_000_000_000_000_04e-2,
            1.0e-18,
        );

        let dirac = atomic_local_density_potential(
            fixture.input(AtomicLocalDensityExchangeMode::DiracFockOnly, true),
        )?;
        assert_close_with(dirac.potential[0], 1.000_000_000_000_000_05e-4, 1.0e-19);
        assert_close_with(dirac.potential[4], 5.000_000_000_000_000_10e-4, 1.0e-19);
        assert_close_with(dirac.potential[9], 1.000_000_000_000_000_02e-3, 1.0e-18);
        assert_close_with(
            dirac.development_coefficients[1],
            2.000_000_000_000_000_04e-2,
            1.0e-18,
        );
        assert_close_with(
            dirac.energy_density[0],
            2.000_000_000_000_000_04e-3,
            1.0e-18,
        );
        assert_close_with(
            dirac.energy_density[4],
            1.000_000_000_000_000_02e-2,
            1.0e-18,
        );
        assert_close_with(
            dirac.energy_density[9],
            2.000_000_000_000_000_04e-2,
            1.0e-18,
        );
        Ok(())
    }

    #[allow(clippy::excessive_precision)]
    #[test]
    fn atom_orbital_potential_matches_feff_potrdf_reference() -> Result<(), AtomMathError> {
        let fixture = sample_potrdf_fixture();

        let full = atomic_orbital_potential(fixture.input(true, true))?;
        for (index, expected) in [
            (0, -1.451_464_734_879_546_50e-3),
            (4, -1.422_294_851_220_632_99e-3),
            (9, -1.385_920_785_309_911_19e-3),
            (12, -1.364_108_165_051_381_58e-3),
        ] {
            assert_close_with(full.central_potential[index], expected, 1.0e-15);
        }
        for (index, expected) in [
            (0, -2.189_205_772_127_074_25e-4),
            (1, -4.371_323_520_763_144_61e-4),
            (3, -8.773_080_991_906_762_18e-4),
            (5, -1.317_263_492_825_318_69e-3),
        ] {
            assert_close_with(
                full.central_development_coefficients[index],
                expected,
                1.0e-15,
            );
        }
        for (index, expected) in [
            (0, 1.702_743_222_291_228_72e-7),
            (4, 2.294_031_531_020_954_80e-7),
            (9, 0.0),
        ] {
            assert_close_with(full.exchange_large[index], expected, 1.0e-16);
        }
        for (index, expected) in [
            (0, -4.763_258_868_894_551_20e-8),
            (4, -4.776_069_610_481_555_18e-8),
            (9, 0.0),
        ] {
            assert_close_with(full.exchange_small[index], expected, 1.0e-16);
        }
        for (index, expected) in [
            (0, 2.307_477_389_651_008_40e-5),
            (2, 4.794_137_619_410_912_88e-5),
            (5, 7.832_202_463_049_932_73e-5),
        ] {
            assert_close_with(full.exchange_large_coefficients[index], expected, 1.0e-16);
        }
        for (index, expected) in [
            (0, 1.845_981_911_720_806_31e-6),
            (2, -4.841_519_661_027_267_90e-6),
            (5, -1.331_336_809_940_218_72e-5),
        ] {
            assert_close_with(full.exchange_small_coefficients[index], expected, 1.0e-16);
        }

        let direct = atomic_orbital_potential(fixture.input(false, false))?;
        for (actual, expected) in direct
            .central_potential
            .iter()
            .zip(full.central_potential.iter())
        {
            assert_close_with(*actual, *expected, 1.0e-16);
        }
        for (actual, expected) in direct
            .central_development_coefficients
            .iter()
            .zip(full.central_development_coefficients.iter())
        {
            assert_close_with(*actual, *expected, 1.0e-16);
        }
        for value in direct
            .exchange_large
            .iter()
            .chain(direct.exchange_small.iter())
            .chain(direct.exchange_large_coefficients.iter())
            .chain(direct.exchange_small_coefficients.iter())
        {
            assert_close_with(*value, 0.0, 1.0e-20);
        }
        Ok(())
    }

    #[allow(clippy::excessive_precision)]
    #[test]
    fn atom_yk_zk_transform_matches_feff_yzkteg_reference() -> Result<(), AtomMathError> {
        let fixture = sample_yzkteg_fixture();
        let transform = atomic_yk_zk_transform(fixture.input())?;

        assert_eq!(transform.computed_source_len, 9);
        assert_close_with(
            transform.origin_constant,
            1.024_939_588_738_283_48e2,
            1.0e-11,
        );
        assert_close_with(transform.yk[0], 3.871_202_667_947_041_34e-4, 1.0e-16);
        assert_close_with(transform.yk[1], 4.476_978_947_879_065_22e-4, 1.0e-16);
        assert_close_with(transform.yk[4], 6.350_731_526_853_801_77e-4, 1.0e-16);
        assert_close_with(transform.yk[8], 6.665_230_606_586_294_07e-4, 1.0e-16);
        assert_close_with(transform.yk[12], 4.467_837_687_045_075_67e-4, 1.0e-16);
        assert_close_with(transform.zk[0], 1.055_350_291_449_006_03e-5, 1.0e-17);
        assert_close_with(transform.zk[1], 1.147_457_094_885_342_41e-5, 1.0e-17);
        assert_close_with(transform.zk[4], 1.675_242_796_907_188_86e-4, 1.0e-16);
        assert_close_with(transform.zk[9], 7.118_915_805_710_559_43e-4, 1.0e-16);
        assert_close_with(
            transform.yk_coefficients[0],
            -3.906_646_372_399_797_53e-2,
            1.0e-16,
        );
        assert_close_with(
            transform.yk_coefficients[3],
            6.197_311_460_469_354_11e-2,
            1.0e-16,
        );
        assert_close_with(
            transform.zk_coefficients[0],
            1.054_794_520_547_945_24e-2,
            1.0e-17,
        );
        assert_close_with(
            transform.zk_coefficients[3],
            2.045_112_781_954_887_27e-2,
            1.0e-17,
        );
        Ok(())
    }

    #[allow(clippy::excessive_precision)]
    #[test]
    fn atom_yk_zk_prepared_source_matches_feff_yzkrdf_reference() -> Result<(), AtomMathError> {
        let fixture = sample_yzkteg_fixture();
        let rank_two = atomic_yk_zk_prepared_source(fixture.prepared_input(9, 2))?;
        assert_eq!(rank_two.computed_source_len, 9);
        assert_close_with(
            rank_two.origin_constant,
            1.110_957_296_725_969_88e2,
            1.0e-11,
        );
        assert_close_with(rank_two.yk[0], 3.746_164_822_999_324_47e-4, 1.0e-16);
        assert_close_with(rank_two.yk[1], 4.361_981_443_957_904_09e-4, 1.0e-16);
        assert_close_with(rank_two.yk[4], 6.265_729_070_725_439_66e-4, 1.0e-16);
        assert_close_with(rank_two.yk[8], 6.608_249_600_892_370_22e-4, 1.0e-16);
        assert_close_with(rank_two.yk[12], 4.429_642_176_685_166_84e-4, 1.0e-16);
        assert_close_with(rank_two.zk[0], 4.277_638_252_436_042_60e-12, 1.0e-22);
        assert_close_with(rank_two.zk[1], 5.499_800_258_296_022_76e-12, 1.0e-22);
        assert_close_with(rank_two.zk[4], 1.590_237_125_316_554_21e-4, 1.0e-16);
        assert_close_with(rank_two.zk[9], 7.067_357_259_641_375_48e-4, 1.0e-16);
        assert_close_with(
            rank_two.yk_coefficients[0],
            1.374_999_999_999_999_83e-2,
            1.0e-17,
        );
        assert_close_with(
            rank_two.yk_coefficients[3],
            1.360_000_000_000_000_10e-2,
            1.0e-17,
        );

        let rank_one = atomic_yk_zk_prepared_source(fixture.prepared_input(7, 1))?;
        assert_eq!(rank_one.computed_source_len, 7);
        assert_close_with(rank_one.origin_constant, 1.293_492_132_385_440_25, 1.0e-13);
        assert_close_with(rank_one.yk[0], 2.908_635_211_432_032_27e-4, 1.0e-16);
        assert_close_with(rank_one.yk[1], 3.220_388_501_435_997_46e-4, 1.0e-16);
        assert_close_with(rank_one.yk[4], 4.003_521_683_966_694_17e-4, 1.0e-16);
        assert_close_with(rank_one.yk[8], 3.610_570_331_017_010_91e-4, 1.0e-16);
        assert_close_with(rank_one.yk[12], 2.956_084_966_154_574_63e-4, 1.0e-16);
        assert_close_with(rank_one.zk[0], 3.988_806_776_811_954_55e-10, 1.0e-20);
        assert_close_with(rank_one.zk[1], 4.878_024_038_015_732_55e-10, 1.0e-20);
        assert_close_with(rank_one.zk[4], 1.686_537_565_491_518_30e-4, 1.0e-16);
        assert_close_with(rank_one.zk[9], 0.0, 1.0e-20);
        assert_close_with(
            rank_one.yk_coefficients[0],
            1.155_000_000_000_000_12e-2,
            1.0e-17,
        );
        assert_close_with(
            rank_one.yk_coefficients[3],
            1.020_000_000_000_000_07e-2,
            1.0e-17,
        );
        Ok(())
    }

    #[allow(clippy::excessive_precision)]
    #[test]
    fn atom_yk_zk_exchange_matches_feff_yzkrdf_reference() -> Result<(), AtomMathError> {
        let fixture = sample_yzkrdf_fixture();
        let overlap = atomic_yk_zk_exchange(fixture.yzkrdf_input(1, 2, 2, false))?;
        assert_eq!(overlap.computed_source_len, 9);
        assert_close_with(overlap.origin_constant, -2.571_240_643_442_588_96, 1.0e-12);
        assert_close_with(overlap.yk[0], 1.109_878_400_538_443_00e-5, 1.0e-17);
        assert_close_with(overlap.yk[1], 1.135_633_080_766_094_54e-5, 1.0e-17);
        assert_close_with(overlap.yk[4], 1.178_867_152_957_986_59e-5, 1.0e-17);
        assert_close_with(overlap.yk[8], 1.017_973_162_090_520_64e-5, 1.0e-17);
        assert_close_with(overlap.yk[12], 6.823_678_168_755_628_77e-6, 1.0e-18);
        assert_close_with(overlap.zk[0], 5.468_221_372_334_369_25e-6, 1.0e-18);
        assert_close_with(overlap.zk[1], 5.909_940_448_128_294_29e-6, 1.0e-18);
        assert_close_with(overlap.zk[4], 7.024_129_238_136_815_07e-6, 1.0e-18);
        assert_close_with(overlap.zk[9], 1.014_708_699_883_866_62e-5, 1.0e-17);
        assert_close_with(
            overlap.yk_coefficients[0],
            -9.990_630_795_999_924_30e-3,
            1.0e-17,
        );
        assert_close_with(
            overlap.yk_coefficients[3],
            8.575_701_162_755_210_16e-2,
            1.0e-16,
        );

        let large_small = atomic_yk_zk_exchange(fixture.yzkrdf_input(2, 3, 1, true))?;
        assert_eq!(large_small.computed_source_len, 7);
        assert_close_with(
            large_small.origin_constant,
            -2.237_401_842_533_894_71e-2,
            1.0e-14,
        );
        assert_close_with(large_small.yk[0], -1.770_958_131_971_287_30e-6, 1.0e-18);
        assert_close_with(large_small.yk[1], -2.024_241_049_179_754_12e-6, 1.0e-18);
        assert_close_with(large_small.yk[4], -2.505_938_578_653_440_58e-6, 1.0e-18);
        assert_close_with(large_small.yk[8], -2.208_316_861_767_755_49e-6, 1.0e-18);
        assert_close_with(large_small.yk[12], -1.808_016_927_269_919_53e-6, 1.0e-18);
        assert_close_with(large_small.zk[0], 3.406_624_777_460_352_47e-7, 1.0e-19);
        assert_close_with(large_small.zk[1], 3.708_373_404_554_750_70e-7, 1.0e-19);
        assert_close_with(large_small.zk[4], -1.328_125_640_689_300_04e-6, 1.0e-18);
        assert_close_with(large_small.zk[9], 0.0, 1.0e-19);
        assert_close_with(
            large_small.yk_coefficients[0],
            -3.957_309_029_859_694_58e-3,
            1.0e-17,
        );
        assert_close_with(
            large_small.yk_coefficients[3],
            -2.038_402_989_657_719_41e-3,
            1.0e-17,
        );
        Ok(())
    }

    #[allow(clippy::excessive_precision)]
    #[test]
    fn atom_radial_integral_matches_feff_fdrirk_reference() -> Result<(), AtomMathError> {
        let fixture = sample_yzkrdf_fixture();
        let kappas = [-1, 1, -2];

        let overlap = atomic_radial_integral(fixture.fdrirk_input(
            AtomicRadialIntegralRequest {
                first_left: 1,
                first_right: 2,
                second_left: 1,
                second_right: 3,
                rank: 2,
            },
            &kappas,
            false,
            None,
        ))?;
        assert_close_with(overlap.value, 3.844_030_024_958_072_30e-9, 1.0e-20);
        let overlap_factor = overlap
            .first_factor
            .as_ref()
            .ok_or(AtomMathError::MissingRadialFirstFactor)?;
        assert_close_with(
            overlap_factor.values[0],
            1.109_878_400_538_443_00e-5,
            1.0e-17,
        );
        assert_close_with(
            overlap_factor.values[3],
            1.171_927_755_618_356_82e-5,
            1.0e-17,
        );
        assert_close_with(
            overlap_factor.coefficients[0],
            -2.561_250_012_646_588_91,
            1.0e-12,
        );
        assert_close_with(
            overlap_factor.coefficients[3],
            -8.575_701_162_755_210_16e-2,
            1.0e-16,
        );

        let large_small = atomic_radial_integral(fixture.fdrirk_input(
            AtomicRadialIntegralRequest {
                first_left: 2,
                first_right: 3,
                second_left: 1,
                second_right: 2,
                rank: 1,
            },
            &kappas,
            true,
            None,
        ))?;
        assert_close_with(large_small.value, 2.056_815_682_976_472_25e-10, 1.0e-21);
        let large_small_factor = large_small
            .first_factor
            .as_ref()
            .ok_or(AtomMathError::MissingRadialFirstFactor)?;
        assert_close_with(
            large_small_factor.coefficients[0],
            -2.237_401_842_533_894_71e-2,
            1.0e-14,
        );
        assert_close_with(
            large_small_factor.coefficients[3],
            9.462_409_003_166_756_97e-4,
            1.0e-17,
        );

        let first = atomic_radial_integral(fixture.fdrirk_input(
            AtomicRadialIntegralRequest {
                first_left: 2,
                first_right: 1,
                second_left: 2,
                second_right: 1,
                rank: 1,
            },
            &kappas,
            false,
            None,
        ))?;
        assert_close_with(first.value, -3.712_970_151_907_870_88e-9, 1.0e-20);
        let previous = first
            .first_factor
            .as_ref()
            .ok_or(AtomMathError::MissingRadialFirstFactor)?;
        let sentinel = atomic_radial_integral(fixture.fdrirk_input(
            AtomicRadialIntegralRequest {
                first_left: 0,
                first_right: 0,
                second_left: 1,
                second_right: 2,
                rank: 1,
            },
            &kappas,
            false,
            Some(previous.as_view()),
        ))?;
        assert_close_with(sentinel.value, -3.712_970_151_907_870_88e-9, 1.0e-20);
        assert!(sentinel.first_factor.is_none());

        let no_second = atomic_radial_integral(fixture.fdrirk_input(
            AtomicRadialIntegralRequest {
                first_left: 1,
                first_right: 2,
                second_left: 0,
                second_right: 0,
                rank: 2,
            },
            &kappas,
            false,
            None,
        ))?;
        assert_close(no_second.value, 0.0);
        assert!(no_second.first_factor.is_some());
        Ok(())
    }

    #[test]
    fn atom_form_factor_matches_feff_fpf0_reference() -> Result<(), AtomMathError> {
        let radial_count = 251;
        let orbital_count = 5;
        let radial_step = 0.05;
        let radii = Array1::from_shape_fn(radial_count, |index| {
            (-8.8 + radial_step * index as Real).exp()
        });
        let density_4pi = Array1::from_shape_fn(radial_count, |index| {
            0.3 * (-0.7 * radii[index]).exp() + 0.01 * (index + 1).rem_euclid(7) as Real
        });
        let initial_large_component = Array1::from_shape_fn(radial_count, |index| {
            0.2 * (-0.4 * radii[index]).exp() + 0.001 * (index + 1) as Real
        });
        let initial_small_component = Array1::from_shape_fn(radial_count, |index| {
            -0.05 * (-0.3 * radii[index]).exp() + 0.0002 * (index + 1) as Real
        });
        let large_components =
            Array2::from_shape_fn((radial_count, orbital_count), |(row, col)| {
                let orbital = (col + 1) as Real;
                (0.03 * orbital + 0.0007 * (row + 1) as Real) * (-0.05 * orbital * radii[row]).exp()
            });
        let small_components =
            Array2::from_shape_fn((radial_count, orbital_count), |(row, col)| {
                let orbital = (col + 1) as Real;
                (-0.01 * orbital + 0.0003 * (row + 1) as Real)
                    * (-0.03 * orbital * radii[row]).exp()
            });
        let occupations = [2.0, 2.0, 1.5, 0.5, 0.0];
        let orbital_energies = [-0.85, -0.55, -0.21, -0.08, 0.04];
        let kappas = [-1, 1, -2, 2, -1];

        let form_factor = atomic_form_factor(AtomicFormFactorInput {
            atomic_number: 26,
            hole_orbital_1based: 2,
            radial_step,
            total_energy: -2.345,
            radii: radii.view(),
            density_4pi: density_4pi.view(),
            initial_large_component: initial_large_component.view(),
            initial_small_component: initial_small_component.view(),
            large_components: large_components.view(),
            small_components: small_components.view(),
            occupations: &occupations,
            orbital_energies: &orbital_energies,
            kappas: &kappas,
        })?;

        assert_eq!(form_factor.atomic_number, 26);
        assert_close_with(form_factor.total_energy_fprime, -2.081_24e-4, 5.0e-10);
        assert_close_with(form_factor.relativistic_correction, -6.478_75e-2, 5.0e-8);
        assert_eq!(form_factor.oscillators.len(), 3);
        let expected_oscillators = [(2.0, -0.55, 2), (0.104_07, -0.85, 1), (0.003_60, -0.08, 4)];
        for (actual, (strength, energy, index)) in
            form_factor.oscillators.iter().zip(expected_oscillators)
        {
            assert_close_with(actual.oscillator_strength, strength, 5.0e-6);
            assert_close_with(actual.excitation_energy, energy, 5.0e-13);
            assert_eq!(actual.orbital_index_1based, index);
        }
        assert_eq!(form_factor.form_factor.len(), 81);
        let expected_rows = [
            (0, 0.0, 760.5215),
            (1, 0.5, -4.0195),
            (2, 1.0, 16.7054),
            (3, 1.5, -1.1065),
            (4, 2.0, -0.5452),
            (10, 5.0, 1.4707),
            (20, 10.0, -0.1129),
            (40, 20.0, -0.6736),
            (80, 40.0, 0.1214),
        ];
        for (index, momentum, value) in expected_rows {
            assert_close_with(form_factor.form_factor_momentum[index], momentum, 1.0e-13);
            assert_close_with(form_factor.form_factor[index], value, 5.5e-5);
        }
        Ok(())
    }

    #[allow(clippy::excessive_precision)]
    #[test]
    fn atom_schmidt_orthogonalization_matches_feff_ortdat_reference() -> Result<(), AtomMathError> {
        let fixture = sample_schmidt_fixture();
        let all_orbitals =
            atomic_schmidt_orthogonalization(fixture.as_input(None), sample_schmidt_integral)?;
        assert_eq!(all_orbitals.active_lengths, vec![3, 4, 3, 5]);
        assert_columns_close(
            &all_orbitals.large_components,
            &[
                [0.18, 0.25, 0.32, 0.39, 0.46],
                [
                    0.333_475_933_348_347_96,
                    0.403_443_338_654_020_99,
                    0.473_410_743_959_694_18,
                    0.697_998_855_802_804_52,
                    0.57,
                ],
                [
                    0.487_117_140_335_587_17,
                    0.572_362_639_894_314_91,
                    0.657_608_139_453_042_64,
                    0.61,
                    0.68,
                ],
                [
                    0.086_758_208_000_696_446,
                    0.041_346_281_239_887_581,
                    -0.004_065_645_520_921_706_5,
                    -0.041_673_823_238_614_134,
                    0.979_213_171_940_273_24,
                ],
            ],
            1.0e-12,
        );
        assert_columns_close(
            &all_orbitals.small_components,
            &[
                [0.01, 0.04, 0.07, 0.1, 0.13],
                [
                    -0.017_924_610_617_016_022,
                    0.012_061_420_228_272_458,
                    0.042_047_451_073_560_942,
                    0.111_679_816_928_448_71,
                    0.11,
                ],
                [
                    -0.036_533_785_525_169_032,
                    0.0,
                    0.036_533_785_525_169_032,
                    0.06,
                    0.09,
                ],
                [
                    -0.043_493_187_919_062_107,
                    -0.062_955_442_245_123_172,
                    -0.082_417_696_571_184_237,
                    -0.099_878_989_604_138_421,
                    0.086_765_724_095_973_565,
                ],
            ],
            1.0e-12,
        );
        assert_columns_close(
            &all_orbitals.large_coefficients,
            &[
                [0.25, 0.45, 0.65, 0.85],
                [
                    0.319_683_475_957_684_54,
                    0.519_590_348_259_607_70,
                    0.719_497_220_561_530_87,
                    0.919_404_092_863_454_04,
                ],
                [
                    0.426_227_497_793_638_78,
                    0.669_786_067_961_432_36,
                    0.913_344_638_129_225_95,
                    1.156_903_208_297_019_4,
                ],
                [
                    -0.069_671_028_191_237_896,
                    -0.199_419_390_364_978_30,
                    -0.329_167_752_538_718_77,
                    -0.458_916_114_712_459_13,
                ],
            ],
            1.0e-12,
        );
        assert_columns_close(
            &all_orbitals.small_coefficients,
            &[
                [0.01, -0.02, -0.05, -0.08],
                [
                    0.065_835_252_079_320_519,
                    0.035_849_221_234_032_044,
                    0.005_863_190_388_743_565_9,
                    -0.024_122_840_456_544_916,
                ],
                [
                    0.109_601_356_575_507_10,
                    0.073_067_571_050_338_065,
                    0.036_533_785_525_169_032,
                    0.0,
                ],
                [
                    0.067_524_121_512_063_162,
                    0.086_986_375_838_124_214,
                    0.106_448_630_164_185_28,
                    0.125_910_884_490_246_34,
                ],
            ],
            1.0e-12,
        );

        let active_two =
            atomic_schmidt_orthogonalization(fixture.as_input(Some(2)), sample_schmidt_integral)?;
        assert_eq!(active_two.active_lengths, vec![3, 5, 3, 5]);
        assert_columns_close(
            &active_two.large_components,
            &[
                [0.18, 0.25, 0.32, 0.39, 0.46],
                [
                    -0.257_731_473_167_008_73,
                    -0.271_503_234_760_490_32,
                    -0.285_274_996_353_971_69,
                    -0.160_996_405_265_147_69,
                    -0.860_433_208_548_678_89,
                ],
                [0.4, 0.47, 0.54, 0.61, 0.68],
                [0.51, 0.58, 0.65, 0.72, 0.79],
            ],
            1.0e-12,
        );
        assert_columns_close(
            &active_two.small_components,
            &[
                [0.01, 0.04, 0.07, 0.1, 0.13],
                [
                    0.038_454_127_655_123_280,
                    0.032_551_944_115_059_794,
                    0.026_649_760_574_996_302,
                    0.056_145_103_363_729_076,
                    -0.076_240_917_213_174_053,
                ],
                [-0.03, 0.0, 0.03, 0.06, 0.09],
                [-0.05, -0.02, 0.01, 0.04, 0.07],
            ],
            1.0e-12,
        );
        assert_columns_close(
            &active_two.large_coefficients,
            &[
                [0.25, 0.45, 0.65, 0.85],
                [
                    -0.150_238_668_255_056_88,
                    -0.189_586_558_522_146_90,
                    -0.228_934_448_789_236_94,
                    -0.268_282_339_056_326_81,
                ],
                [0.35, 0.55, 0.75, 0.95],
                [0.4, 0.6, 0.8, 1.0],
            ],
            1.0e-12,
        );
        assert_columns_close(
            &active_two.small_coefficients,
            &[
                [0.01, -0.02, -0.05, -0.08],
                [
                    -0.082_810_438_850_310_059,
                    -0.076_908_255_310_246_559,
                    -0.071_006_071_770_183_060,
                    -0.065_103_888_230_119_589,
                ],
                [0.09, 0.06, 0.03, 0.0],
                [0.13, 0.10, 0.07, 0.04],
            ],
            1.0e-12,
        );

        Ok(())
    }

    #[test]
    fn atom_overlap_amplitude_reduction_matches_feff_s02at_reference() -> Result<(), AtomMathError>
    {
        let kappas = [-1, -1, 1, 1, -2, -3];
        let occupations = [2.0, 1.0, 1.5, 0.5, 3.0, 2.5];
        let overlaps = sample_s02at_overlaps();
        let cases = [
            (None, 9.680_452_235_999_996e-3),
            (Some(1), 9.680_452_235_999_996e-3),
            (Some(2), 0.327_600_000_000_000_1),
            (Some(3), 9.680_452_235_999_996e-3),
            (Some(4), 9.020_027_472_527_463e-2),
            (Some(5), 9.680_452_235_999_996e-3),
            (Some(6), 9.680_452_235_999_996e-3),
        ];

        for (hole_orbital_1based, expected) in cases {
            let actual =
                atomic_overlap_amplitude_reduction(AtomicOverlapAmplitudeReductionInput {
                    hole_orbital_1based,
                    kappas: &kappas,
                    occupations: &occupations,
                    overlap_integrals: overlaps.view(),
                })?;
            assert_close(actual, expected);
        }
        Ok(())
    }

    #[test]
    fn atom_helper_kernels_reject_invalid_inputs() {
        assert!(matches!(
            atomic_polynomial_product_coefficient(&[1.0], &[2.0], 2),
            Err(AtomMathError::InvalidPolynomialTerm { .. })
        ));
        assert!(matches!(
            atomic_convergence_mix(0.5, Real::INFINITY, 1.0),
            Err(AtomMathError::NonFiniteScalar {
                field: "current_error",
                ..
            })
        ));
        assert!(matches!(
            thomas_fermi_density_potential(0.0, 1.0, 0.0),
            Err(AtomMathError::NonPositiveRadius { .. })
        ));
        assert!(matches!(
            atomic_occupation_product(&[1.0], &[], 0, 0),
            Err(AtomMathError::OccupationKappaLengthMismatch { .. })
        ));
        assert!(matches!(
            atomic_occupation_product(&[1.0], &[0], 0, 0),
            Err(AtomMathError::ZeroKappa)
        ));

        let coefficients = Array3::zeros((2, 2, 1));
        assert!(matches!(
            atomic_direct_coulomb_coefficient(coefficients.view(), 0, 0, 4),
            Err(AtomMathError::CoefficientChannelOutOfRange { .. })
        ));

        let coefficients = Array3::zeros((2, 3, 1));
        assert!(matches!(
            atomic_direct_coulomb_coefficient(coefficients.view(), 1, 2, 0),
            Err(AtomMathError::CoefficientTableShape { .. })
        ));
        assert!(matches!(
            atomic_coulomb_coefficients(AtomicCoulombCoefficientInput {
                kappas: &[1],
                occupations: &[],
                valence_occupations: &[0.0],
            }),
            Err(AtomMathError::OrbitalTableLengthMismatch { .. })
        ));
        assert!(matches!(
            atomic_coulomb_coefficients(AtomicCoulombCoefficientInput {
                kappas: &[0],
                occupations: &[1.0],
                valence_occupations: &[0.0],
            }),
            Err(AtomMathError::InvalidKappa { .. })
        ));
        assert!(matches!(
            atomic_coulomb_coefficients(AtomicCoulombCoefficientInput {
                kappas: &[1],
                occupations: &[Real::NAN],
                valence_occupations: &[0.0],
            }),
            Err(AtomMathError::NonFiniteScalar {
                field: "occupation",
                ..
            })
        ));
        assert!(matches!(
            atomic_coulomb_coefficients(AtomicCoulombCoefficientInput {
                kappas: &[6],
                occupations: &[2.0],
                valence_occupations: &[0.0],
            }),
            Err(AtomMathError::CoefficientChannelOutOfRange { .. })
        ));
        assert!(matches!(
            atomic_nuclear_potential(AtomicNuclearPotentialInput {
                nuclear_charge: 0.0,
                step: 0.05,
                requested_nucleus_index: 1,
                radial_count: 251,
                coefficient_count: 10,
                first_radius_times_charge: 1.0,
            }),
            Err(AtomMathError::InvalidNuclearPotentialScalar { .. })
        ));
        assert!(matches!(
            atomic_nuclear_potential(AtomicNuclearPotentialInput {
                nuclear_charge: 92.0,
                step: 0.05,
                requested_nucleus_index: -11,
                radial_count: 5,
                coefficient_count: 10,
                first_radius_times_charge: 92.0 * (-8.8_f64).exp(),
            }),
            Err(AtomMathError::NuclearRadiusOutOfRange { .. })
        ));
        assert!(matches!(
            atomic_nuclear_potential(AtomicNuclearPotentialInput {
                nuclear_charge: 26.0,
                step: 0.05,
                requested_nucleus_index: 1,
                radial_count: 251,
                coefficient_count: 4,
                first_radius_times_charge: 26.0 * (-8.8_f64).exp(),
            }),
            Err(AtomMathError::InvalidNuclearPotentialCount { .. })
        ));
        let dsordf = sample_dsordf_fixture();
        assert!(matches!(
            atomic_differential_integral(dsordf.input(
                AtomicDifferentialIntegralKind::DerivativeNorm { active_len: 8 },
                0,
                0.45,
            )),
            Err(AtomMathError::InvalidDifferentialIntegralActiveLength { .. })
        ));
        assert!(matches!(
            atomic_differential_integral(dsordf.input(
                AtomicDifferentialIntegralKind::ComponentOverlap {
                    left_orbital_1based: 0,
                    right_orbital_1based: 1,
                    multiply_by_derivative: false,
                },
                0,
                0.0,
            )),
            Err(AtomMathError::ActiveOrbitalOutOfRange { .. })
        ));
        assert!(matches!(
            atomic_differential_integral(dsordf.input(
                AtomicDifferentialIntegralKind::DerivativeNorm { active_len: 9 },
                0,
                -1.0,
            )),
            Err(AtomMathError::ZeroDifferentialIntegralOriginExponent)
        ));
        let bad_radii = Array1::from_vec(vec![0.0; 11]);
        assert!(matches!(
            atomic_differential_integral(AtomicDifferentialIntegralInput {
                radii: bad_radii.view(),
                ..dsordf.input(
                    AtomicDifferentialIntegralKind::DerivativeNorm { active_len: 9 },
                    0,
                    0.45,
                )
            }),
            Err(AtomMathError::NonPositiveRadius { .. })
        ));
        let bad_derivative_coefficients = Array1::from_vec(vec![0.1; 5]);
        assert!(matches!(
            atomic_differential_integral(AtomicDifferentialIntegralInput {
                derivative_large_coefficients: bad_derivative_coefficients.view(),
                ..dsordf.input(
                    AtomicDifferentialIntegralKind::DerivativeNorm { active_len: 9 },
                    0,
                    0.45,
                )
            }),
            Err(AtomMathError::CoefficientTableLengthMismatch { .. })
        ));
        let yzkteg = sample_yzkteg_fixture();
        assert!(matches!(
            atomic_yk_zk_transform(AtomicYkZkTransformInput {
                active_len: 3,
                ..yzkteg.input()
            }),
            Err(AtomMathError::InvalidDifferentialIntegralActiveLength { .. })
        ));
        assert!(matches!(
            atomic_yk_zk_transform(AtomicYkZkTransformInput {
                step: 0.0,
                ..yzkteg.input()
            }),
            Err(AtomMathError::ZeroYkZkDenominator { field: "step" })
        ));
        assert!(matches!(
            atomic_yk_zk_transform(AtomicYkZkTransformInput {
                initial_power: 2.0,
                ..yzkteg.input()
            }),
            Err(AtomMathError::ZeroYkZkDenominator { field: "yk_origin" })
        ));
        let yzkrdf = sample_yzkrdf_fixture();
        assert!(matches!(
            atomic_yk_zk_exchange(AtomicYkZkExchangeInput {
                left_orbital_1based: 0,
                ..yzkrdf.yzkrdf_input(1, 2, 2, false)
            }),
            Err(AtomMathError::ActiveOrbitalOutOfRange { .. })
        ));
        assert!(matches!(
            AtomicLocalDensityExchangeMode::try_from(4),
            Err(AtomMathError::InvalidExchangeMode { idfock: 4 })
        ));
        let vlda = sample_vlda_fixture();
        assert!(matches!(
            atomic_local_density_potential(AtomicLocalDensityPotentialInput {
                speed_of_light: 0.0,
                ..vlda.input(AtomicLocalDensityExchangeMode::TotalDensity, false)
            }),
            Err(AtomMathError::NonPositiveScalar { .. })
        ));
        assert!(matches!(
            atomic_orbital_initialization(AtomicOrbitalInitializationInput {
                atomic_number: 4,
                ionicity: 0.0,
                principal_quantum_numbers: &[2],
                kappas: &[1],
                occupations: &[1.0],
            }),
            Err(AtomMathError::ElectronCountMismatch { .. })
        ));
        assert!(matches!(
            atomic_orbital_initialization(AtomicOrbitalInitializationInput {
                atomic_number: 1,
                ionicity: 0.0,
                principal_quantum_numbers: &[1],
                kappas: &[1],
                occupations: &[1.0],
            }),
            Err(AtomMathError::OrbitalAngularMomentumOutOfRange { .. })
        ));
        let soldir_norm = sample_soldir_norm_fixture();
        assert!(matches!(
            atomic_dirac_normalization(soldir_norm.input(1, 6, 0.177, 0.82, 10, 5)),
            Err(AtomMathError::InvalidDiracNormalizationActiveLength { .. })
        ));
        assert!(matches!(
            atomic_dirac_normalization(soldir_norm.input(1, 6, 0.177, 0.82, 11, 0)),
            Err(AtomMathError::DiracNormalizationMatchingIndexOutOfRange { .. })
        ));
        assert!(matches!(
            atomic_dirac_normalization(soldir_norm.input(2, 6, 0.177, -0.5, 11, 5)),
            Err(AtomMathError::ZeroDiracNormalizationOriginExponent)
        ));
        let soldir_solution_norm = sample_soldir_solution_normalization_fixture(false, false);
        assert!(matches!(
            atomic_dirac_solution_normalization(AtomicDiracSolutionNormalizationInput {
                active_len: 0,
                ..soldir_solution_norm.input(6.25, 0.8, -0.4)
            }),
            Err(AtomMathError::InvalidDiracSolutionNormalizationActiveLength { .. })
        ));
        assert!(matches!(
            atomic_dirac_solution_normalization(AtomicDiracSolutionNormalizationInput {
                norm: 0.0,
                ..soldir_solution_norm.input(6.25, 0.8, -0.4)
            }),
            Err(AtomMathError::NonPositiveScalar {
                field: "soldir_solution_norm",
                ..
            })
        ));
        let soldir_nodes = sample_soldir_node_count_component();
        assert!(matches!(
            atomic_dirac_node_count(AtomicDiracNodeCountInput {
                large_component: soldir_nodes.view(),
                matching_index_1based: 0,
                scan_index_1based: 3,
            }),
            Err(AtomMathError::InvalidDiracNodeCountIndex { .. })
        ));
        assert!(matches!(
            atomic_dirac_method_one_energy_correction(AtomicDiracMethodOneEnergyCorrectionInput {
                speed_of_light: 137.0373,
                norm: 1.0,
                large_component: soldir_nodes.view(),
                small_component: soldir_nodes.view(),
                matching_small_component: 0.0,
                matching_index_1based: 0,
            },),
            Err(AtomMathError::DiracEnergyCorrectionMatchingIndexOutOfRange { .. })
        ));
        assert!(matches!(
            atomic_dirac_energy_step(AtomicDiracEnergyStepInput {
                energy: 0.0,
                correction: 0.0,
                mismatch: 0.0,
                energy_sup: -1.0,
                energy_inf: -0.1,
                mismatch_precision: 0.1,
                zero_energy_precision: 1.0e-7,
            }),
            Err(AtomMathError::ZeroDiracEnergyCorrectionDenominator)
        ));
        assert!(matches!(
            atomic_dirac_rematch_attempt(AtomicDiracRematchAttemptInput {
                mismatch: Real::NAN,
                mismatch_precision: 0.1,
                match_attempt_count: 0,
                max_attempt_count: 50,
            }),
            Err(AtomMathError::NonFiniteScalar { .. })
        ));
        assert!(matches!(
            atomic_dirac_rematch_attempt(AtomicDiracRematchAttemptInput {
                mismatch: 1.0,
                mismatch_precision: 0.1,
                match_attempt_count: usize::MAX,
                max_attempt_count: usize::MAX,
            }),
            Err(AtomMathError::DiracRematchAttemptCountOutOfRange { .. })
        ));
        assert!(matches!(
            atomic_dirac_shooting_pass_setup(AtomicDiracShootingPassSetupInput {
                energy: 0.0,
                previous_energy: -0.1,
            }),
            Err(AtomMathError::ZeroDiracShootingPassEnergy)
        ));
        assert!(matches!(
            atomic_dirac_shooting_pass_setup(AtomicDiracShootingPassSetupInput {
                energy: Real::NAN,
                previous_energy: -0.1,
            }),
            Err(AtomMathError::NonFiniteScalar { .. })
        ));
        assert!(matches!(
            atomic_dirac_node_energy_search(AtomicDiracNodeEnergySearchInput {
                energy: Real::NAN,
                node_count: 2,
                target_node_count: 4,
                energy_sup: -5.0,
                energy_inf: 1.0,
                energy_floor: -5.0,
                energy_precision: 1.0e-7,
                search_attempt_count: 0,
                max_attempt_count: 50,
            }),
            Err(AtomMathError::NonFiniteScalar { .. })
        ));
        assert!(matches!(
            atomic_dirac_node_energy_search(AtomicDiracNodeEnergySearchInput {
                energy: -0.5,
                node_count: 2,
                target_node_count: 4,
                energy_sup: -5.0,
                energy_inf: 1.0,
                energy_floor: -5.0,
                energy_precision: 0.0,
                search_attempt_count: 0,
                max_attempt_count: 50,
            }),
            Err(AtomMathError::NonPositiveScalar { .. })
        ));
        assert!(matches!(
            atomic_dirac_node_energy_search(AtomicDiracNodeEnergySearchInput {
                energy: -0.5,
                node_count: 2,
                target_node_count: 4,
                energy_sup: -5.0,
                energy_inf: 1.0,
                energy_floor: -5.0,
                energy_precision: 1.0e-7,
                search_attempt_count: usize::MAX,
                max_attempt_count: usize::MAX,
            }),
            Err(AtomMathError::DiracNodeEnergyAttemptCountOutOfRange { .. })
        ));
        assert!(matches!(
            atomic_dirac_inhomogeneous_seed(AtomicDiracInhomogeneousSeedInput {
                large_source: soldir_nodes.view(),
                small_source: soldir_nodes
                    .view()
                    .slice_axis(Axis(0), Slice::from(..soldir_nodes.len() - 1)),
                large_source_coefficients: soldir_nodes.view(),
                small_source_coefficients: soldir_nodes.view(),
                coefficient_count: 1,
            }),
            Err(AtomMathError::RadialTableLengthMismatch { .. })
        ));
        assert!(matches!(
            atomic_dirac_inhomogeneous_seed(AtomicDiracInhomogeneousSeedInput {
                large_source: soldir_nodes.view(),
                small_source: soldir_nodes.view(),
                large_source_coefficients: soldir_nodes
                    .view()
                    .slice_axis(Axis(0), Slice::from(..1)),
                small_source_coefficients: soldir_nodes.view(),
                coefficient_count: 2,
            }),
            Err(AtomMathError::CoefficientTableLengthMismatch { .. })
        ));
        assert!(matches!(
            atomic_dirac_homogeneous_seed(AtomicDiracHomogeneousSeedInput {
                radial_len: 0,
                coefficient_len: 1,
            }),
            Err(AtomMathError::InvalidCount {
                field: "soldir_homogeneous_seed_radial_len",
                ..
            })
        ));
        assert!(matches!(
            atomic_dirac_homogeneous_seed(AtomicDiracHomogeneousSeedInput {
                radial_len: 1,
                coefficient_len: 0,
            }),
            Err(AtomMathError::InvalidCount {
                field: "soldir_homogeneous_seed_coefficient_len",
                ..
            })
        ));
        assert!(matches!(
            atomic_dirac_large_component_match(AtomicDiracLargeComponentMatchInput {
                large_component: soldir_nodes.view(),
                small_component: soldir_nodes.view(),
                homogeneous_large_component: soldir_nodes.view(),
                homogeneous_small_component: soldir_nodes.view(),
                matching_large_component: 0.0,
                active_len: 0,
                matching_index_1based: 1,
            }),
            Err(AtomMathError::InvalidDiracMatchActiveLength { .. })
        ));
        assert!(matches!(
            atomic_dirac_large_component_match(AtomicDiracLargeComponentMatchInput {
                large_component: soldir_nodes.view(),
                small_component: soldir_nodes.view(),
                homogeneous_large_component: soldir_nodes.view(),
                homogeneous_small_component: soldir_nodes.view(),
                matching_large_component: 0.0,
                active_len: soldir_nodes.len(),
                matching_index_1based: 0,
            }),
            Err(AtomMathError::DiracMatchMatchingIndexOutOfRange { .. })
        ));
        let zero_match_denominator = Array1::<Real>::zeros(soldir_nodes.len());
        assert!(matches!(
            atomic_dirac_large_component_match(AtomicDiracLargeComponentMatchInput {
                large_component: soldir_nodes.view(),
                small_component: soldir_nodes.view(),
                homogeneous_large_component: zero_match_denominator.view(),
                homogeneous_small_component: soldir_nodes.view(),
                matching_large_component: 0.0,
                active_len: soldir_nodes.len(),
                matching_index_1based: 1,
            }),
            Err(AtomMathError::ZeroDiracMatchDenominator { .. })
        ));
        assert!(matches!(
            atomic_dirac_homogeneous_match(AtomicDiracHomogeneousMatchInput {
                large_component: soldir_nodes.view(),
                small_component: soldir_nodes.view(),
                matching_large_component: 0.0,
                active_len: 0,
                matching_index_1based: 1,
            }),
            Err(AtomMathError::InvalidDiracMatchActiveLength { .. })
        ));
        assert!(matches!(
            atomic_dirac_homogeneous_match(AtomicDiracHomogeneousMatchInput {
                large_component: soldir_nodes.view(),
                small_component: soldir_nodes.view(),
                matching_large_component: 0.0,
                active_len: soldir_nodes.len(),
                matching_index_1based: 4,
            }),
            Err(AtomMathError::ZeroDiracMatchDenominator { .. })
        ));
        assert!(matches!(
            atomic_dirac_two_component_match(AtomicDiracTwoComponentMatchInput {
                large_component: soldir_nodes.view(),
                small_component: soldir_nodes.view(),
                large_coefficients: soldir_nodes.view(),
                small_coefficients: soldir_nodes.view(),
                homogeneous_large_component: soldir_nodes.view(),
                homogeneous_small_component: soldir_nodes.view(),
                homogeneous_large_coefficients: soldir_nodes.view(),
                homogeneous_small_coefficients: soldir_nodes.view(),
                matching_large_component: 0.0,
                matching_small_component: 0.0,
                homogeneous_matching_large_component: 1.0,
                homogeneous_matching_small_component: 1.0,
                coefficient_count: 1,
                active_len: soldir_nodes.len(),
                matching_index_1based: 1,
            }),
            Err(AtomMathError::ZeroDiracMatchDenominator { .. })
        ));
        assert!(matches!(
            atomic_dirac_energy_disagreement_match(AtomicDiracEnergyDisagreementMatchInput {
                large_derivative: soldir_nodes.view(),
                small_derivative: soldir_nodes.view(),
                large_derivative_coefficients: soldir_nodes.view(),
                small_derivative_coefficients: soldir_nodes.view(),
                homogeneous_large_component: soldir_nodes.view(),
                homogeneous_small_component: soldir_nodes.view(),
                homogeneous_large_coefficients: soldir_nodes.view(),
                homogeneous_small_coefficients: soldir_nodes.view(),
                matching_large_derivative: 0.0,
                matching_small_derivative: 0.0,
                homogeneous_matching_large_component: 1.0,
                homogeneous_matching_small_component: 1.0,
                coefficient_count: 1,
                active_len: 0,
                matching_index_1based: 1,
            }),
            Err(AtomMathError::InvalidDiracMatchActiveLength { .. })
        ));
        let positive_radii = Array1::from_elem(soldir_nodes.len(), 0.1);
        assert!(matches!(
            atomic_dirac_energy_disagreement_source(AtomicDiracEnergyDisagreementSourceInput {
                large_component: soldir_nodes.view(),
                small_component: soldir_nodes.view(),
                large_coefficients: soldir_nodes.view(),
                small_coefficients: soldir_nodes.view(),
                radii: positive_radii.view(),
                speed_of_light: 137.0373,
                coefficient_count: 1,
                active_len: 0,
            }),
            Err(AtomMathError::InvalidDiracEnergyDisagreementActiveLength { .. })
        ));
        assert!(matches!(
            atomic_dirac_energy_disagreement_source(AtomicDiracEnergyDisagreementSourceInput {
                large_component: soldir_nodes.view(),
                small_component: soldir_nodes.view(),
                large_coefficients: soldir_nodes.view(),
                small_coefficients: soldir_nodes.view(),
                radii: positive_radii.view(),
                speed_of_light: 0.0,
                coefficient_count: 1,
                active_len: soldir_nodes.len(),
            }),
            Err(AtomMathError::NonPositiveScalar { .. })
        ));
        assert!(matches!(
            atomic_dirac_energy_disagreement_correction(
                AtomicDiracEnergyDisagreementCorrectionInput {
                    radii: positive_radii.view(),
                    large_component: soldir_nodes.view(),
                    small_component: soldir_nodes.view(),
                    large_derivative: soldir_nodes.view(),
                    small_derivative: soldir_nodes.view(),
                    large_coefficients: soldir_nodes.view(),
                    small_coefficients: soldir_nodes.view(),
                    large_derivative_coefficients: soldir_nodes.view(),
                    small_derivative_coefficients: soldir_nodes.view(),
                    norm: 1.0,
                    step: 0.11,
                    origin_power: 1.0,
                    coefficient_count: 1,
                    active_len: 8,
                },
            ),
            Err(AtomMathError::InvalidDiracEnergyDisagreementCorrectionActiveLength { .. })
        ));
        assert!(matches!(
            atomic_dirac_energy_disagreement_correction(
                AtomicDiracEnergyDisagreementCorrectionInput {
                    radii: positive_radii.view(),
                    large_component: soldir_nodes.view(),
                    small_component: soldir_nodes.view(),
                    large_derivative: soldir_nodes.view(),
                    small_derivative: soldir_nodes.view(),
                    large_coefficients: soldir_nodes.view(),
                    small_coefficients: soldir_nodes.view(),
                    large_derivative_coefficients: soldir_nodes.view(),
                    small_derivative_coefficients: soldir_nodes.view(),
                    norm: 1.0,
                    step: 0.11,
                    origin_power: -0.5,
                    coefficient_count: 1,
                    active_len: soldir_nodes.len(),
                },
            ),
            Err(AtomMathError::ZeroDiracEnergyDisagreementCorrectionOriginExponent)
        ));
        let zero_derivative = Array1::<Real>::zeros(soldir_nodes.len());
        assert!(matches!(
            atomic_dirac_energy_disagreement_correction(
                AtomicDiracEnergyDisagreementCorrectionInput {
                    radii: positive_radii.view(),
                    large_component: soldir_nodes.view(),
                    small_component: soldir_nodes.view(),
                    large_derivative: zero_derivative.view(),
                    small_derivative: zero_derivative.view(),
                    large_coefficients: soldir_nodes.view(),
                    small_coefficients: soldir_nodes.view(),
                    large_derivative_coefficients: soldir_nodes.view(),
                    small_derivative_coefficients: soldir_nodes.view(),
                    norm: 0.9,
                    step: 0.11,
                    origin_power: 1.0,
                    coefficient_count: 1,
                    active_len: soldir_nodes.len(),
                },
            ),
            Err(AtomMathError::ZeroDiracEnergyDisagreementCorrectionIntegral)
        ));
        assert!(matches!(
            atomic_dirac_matching_point_update(AtomicDiracMatchingPointUpdateInput {
                large_component: soldir_nodes.view(),
                active_len: 9,
                matching_index_1based: 5,
                already_relocated: false,
            }),
            Err(AtomMathError::InvalidDiracMatchActiveLength { .. })
        ));
        let zero_matching_component = Array1::<Real>::zeros(13);
        assert!(matches!(
            atomic_dirac_matching_point_update(AtomicDiracMatchingPointUpdateInput {
                large_component: zero_matching_component.view(),
                active_len: 13,
                matching_index_1based: 5,
                already_relocated: false,
            }),
            Err(AtomMathError::DiracMatchingPointNotFound { .. })
        ));
        let intdir = sample_intdir_fixture();
        assert!(matches!(
            atomic_dirac_integration(AtomicDiracIntegrationInput {
                active_len: 12,
                ..intdir.input(AtomicDiracIntegrationMode::SearchMatchingPoint, 0, 0)
            }),
            Err(AtomMathError::InvalidDiracIntegrationActiveLength { .. })
        ));
        assert!(matches!(
            atomic_dirac_integration(intdir.input(
                AtomicDiracIntegrationMode::FixedMatchingPoint,
                5,
                139
            )),
            Err(AtomMathError::DiracIntegrationMatchingIndexOutOfRange { .. })
        ));
        assert!(matches!(
            atomic_dirac_integration(intdir.input(AtomicDiracIntegrationMode::InwardOnly, 65, 68)),
            Err(AtomMathError::DiracIntegrationMaxIndexOutOfRange { .. })
        ));
        assert!(matches!(
            atomic_dirac_integration(AtomicDiracIntegrationInput {
                energy: 0.01,
                ..intdir.input(AtomicDiracIntegrationMode::InwardOnly, 65, 139)
            }),
            Err(AtomMathError::InvalidDiracIntegrationEnergy { .. })
        ));
        let soldir_setup = sample_soldir_setup_fixture();
        assert!(matches!(
            atomic_dirac_solver_setup(AtomicDiracSolverSetupInput {
                active_len: 0,
                ..soldir_setup.input(-0.2, 2, 2, 4, true)
            }),
            Err(AtomMathError::InvalidDiracSolverSetupActiveLength { .. })
        ));
        assert!(matches!(
            atomic_dirac_solver_setup(AtomicDiracSolverSetupInput {
                kappa: 0,
                ..soldir_setup.input(-0.2, 2, 2, 4, true)
            }),
            Err(AtomMathError::InvalidKappa { .. })
        ));
        assert!(matches!(
            atomic_dirac_solver_setup(AtomicDiracSolverSetupInput {
                principal_quantum_number: 0,
                ..soldir_setup.input(-0.2, 2, 2, 4, true)
            }),
            Err(AtomMathError::InvalidDiracSolverPrincipalQuantumNumber { .. })
        ));
        let positive_potential = Array1::from_vec(vec![0.25; 7]);
        assert!(matches!(
            atomic_dirac_solver_setup(AtomicDiracSolverSetupInput {
                potential: positive_potential.view(),
                kappa: 2,
                ..soldir_setup.input(-0.2, 2, 2, 4, true)
            }),
            Err(AtomMathError::DiracSolverPotentialNotAttractive { .. })
        ));
        let potrdf = sample_potrdf_fixture();
        assert!(matches!(
            atomic_orbital_potential(AtomicOrbitalPotentialInput {
                active_orbital_1based: 0,
                ..potrdf.input(true, true)
            }),
            Err(AtomMathError::ActiveOrbitalOutOfRange { .. })
        ));
        let kappas = [-1, 1, -2];
        assert!(matches!(
            atomic_radial_integral(yzkrdf.fdrirk_input(
                AtomicRadialIntegralRequest {
                    first_left: 0,
                    first_right: 0,
                    second_left: 1,
                    second_right: 2,
                    rank: 1,
                },
                &kappas,
                false,
                None,
            )),
            Err(AtomMathError::MissingRadialFirstFactor)
        ));
        let coefficients = Array3::zeros((2, 2, 1));
        assert!(matches!(
            atomic_lagrange_parameters(
                AtomicLagrangeParametersInput {
                    active_orbital_1based: Some(0),
                    kappas: &[1, 1],
                    occupations: &[1.0, 2.0],
                    shell_markers: &[1, 1],
                    include_exchange: true,
                    coulomb_coefficients: coefficients.view(),
                },
                |_| Ok(0.0),
            ),
            Err(AtomMathError::ActiveOrbitalOutOfRange { .. })
        ));
        assert!(matches!(
            atomic_lagrange_parameters(
                AtomicLagrangeParametersInput {
                    active_orbital_1based: None,
                    kappas: &[1, 1],
                    occupations: &[0.0, 2.0],
                    shell_markers: &[1, 1],
                    include_exchange: true,
                    coulomb_coefficients: coefficients.view(),
                },
                |_| Ok(0.0),
            ),
            Err(AtomMathError::NonPositiveOccupation { .. })
        ));
        assert!(matches!(
            atomic_tabulation(
                AtomicTabulationInput {
                    principal_quantum_numbers: &[1],
                    kappas: &[1, 1],
                    occupations: &[1.0, 2.0],
                    orbital_energies: &[-0.1, -0.2],
                },
                sample_atomic_tabrat_integral,
            ),
            Err(AtomMathError::OrbitalTableLengthMismatch { .. })
        ));
        assert!(matches!(
            atomic_tabulation(
                AtomicTabulationInput {
                    principal_quantum_numbers: &[0],
                    kappas: &[1],
                    occupations: &[1.0],
                    orbital_energies: &[-0.1],
                },
                sample_atomic_tabrat_integral,
            ),
            Err(AtomMathError::InvalidPrincipalQuantumNumber { .. })
        ));
        assert!(matches!(
            atomic_tabulation(
                AtomicTabulationInput {
                    principal_quantum_numbers: &[1],
                    kappas: &[5],
                    occupations: &[1.0],
                    orbital_energies: &[-0.1],
                },
                sample_atomic_tabrat_integral,
            ),
            Err(AtomMathError::OrbitalLabelKappaOutOfRange { .. })
        ));
        assert!(matches!(
            atomic_tabulation(
                AtomicTabulationInput {
                    principal_quantum_numbers: &[1],
                    kappas: &[1],
                    occupations: &[1.0],
                    orbital_energies: &[-0.1],
                },
                |_| Ok(Real::NAN),
            ),
            Err(AtomMathError::NonFiniteScalar {
                field: "tabrat_integral",
                ..
            })
        ));
        let fpf0_radii = Array1::from_vec(vec![1.0, 1.2]);
        let fpf0_values = Array1::from_vec(vec![0.1, 0.2]);
        let fpf0_components = Array2::zeros((2, 1));
        let fpf0_input = AtomicFormFactorInput {
            atomic_number: 26,
            hole_orbital_1based: 1,
            radial_step: 0.05,
            total_energy: -1.0,
            radii: fpf0_radii.view(),
            density_4pi: fpf0_values.view(),
            initial_large_component: fpf0_values.view(),
            initial_small_component: fpf0_values.view(),
            large_components: fpf0_components.view(),
            small_components: fpf0_components.view(),
            occupations: &[1.0],
            orbital_energies: &[-0.2],
            kappas: &[1],
        };
        assert!(matches!(
            atomic_form_factor(AtomicFormFactorInput {
                atomic_number: 0,
                ..fpf0_input
            }),
            Err(AtomMathError::InvalidFormFactorAtomicNumber { .. })
        ));
        assert!(matches!(
            atomic_form_factor(AtomicFormFactorInput {
                hole_orbital_1based: 2,
                ..fpf0_input
            }),
            Err(AtomMathError::HoleOrbitalOutOfRange { .. })
        ));
        let bad_fpf0_density = Array1::from_vec(vec![0.1]);
        assert!(matches!(
            atomic_form_factor(AtomicFormFactorInput {
                density_4pi: bad_fpf0_density.view(),
                ..fpf0_input
            }),
            Err(AtomMathError::RadialTableLengthMismatch { .. })
        ));
        let schmidt = sample_schmidt_fixture();
        assert!(matches!(
            atomic_schmidt_orthogonalization(schmidt.as_input(Some(5)), sample_schmidt_integral),
            Err(AtomMathError::ActiveOrbitalOutOfRange { .. })
        ));

        let bad_small_components = Array2::<Real>::zeros((4, 4));
        assert!(matches!(
            atomic_schmidt_orthogonalization(
                AtomicSchmidtOrthogonalizationInput {
                    small_components: bad_small_components.view(),
                    ..schmidt.as_input(None)
                },
                sample_schmidt_integral,
            ),
            Err(AtomMathError::MatrixShape { .. })
        ));

        let bad_active_lengths = [6, 4, 3, 5];
        assert!(matches!(
            atomic_schmidt_orthogonalization(
                AtomicSchmidtOrthogonalizationInput {
                    active_lengths: &bad_active_lengths,
                    ..schmidt.as_input(None)
                },
                sample_schmidt_integral,
            ),
            Err(AtomMathError::ActiveLengthOutOfRange { .. })
        ));

        assert!(matches!(
            atomic_schmidt_orthogonalization(schmidt.as_input(Some(1)), |request| match request {
                AtomicSchmidtIntegralRequest::Projection(_) => Ok(0.0),
                AtomicSchmidtIntegralRequest::Norm(_) => Ok(0.0),
            }),
            Err(AtomMathError::NonPositiveNorm { .. })
        ));
        assert!(matches!(
            atomic_breit_angular_coefficients(0, -1, 1),
            Err(AtomMathError::InvalidKappa { .. })
        ));
        assert!(matches!(
            atomic_breit_angular_coefficients(i32::MIN, -1, 1),
            Err(AtomMathError::InvalidKappa { .. })
        ));
        assert!(matches!(
            atomic_breit_angular_coefficients(1, -1, usize::MAX),
            Err(AtomMathError::BreitRankOutOfRange { .. })
        ));

        let coefficients = Array3::zeros((2, 2, 1));
        let input = AtomicTotalEnergyInput {
            kappas: &[1],
            occupations: &[],
            valence_occupations: &[0.0],
            orbital_energies: &[0.0],
            coulomb_coefficients: coefficients.view(),
        };
        assert!(matches!(
            atomic_total_energy(input, |_| Ok(0.0)),
            Err(AtomMathError::OrbitalTableLengthMismatch { .. })
        ));

        let input = AtomicTotalEnergyInput {
            kappas: &[1],
            occupations: &[1.0],
            valence_occupations: &[0.0],
            orbital_energies: &[0.0],
            coulomb_coefficients: coefficients.view(),
        };
        assert!(matches!(
            atomic_total_energy(input, |_| Ok(Real::NAN)),
            Err(AtomMathError::NonFiniteScalar {
                field: "radial_integral",
                ..
            })
        ));

        let single_overlap = Array2::from_elem((1, 1), 1.0);
        let input = AtomicOverlapAmplitudeReductionInput {
            hole_orbital_1based: Some(0),
            kappas: &[1],
            occupations: &[1.0],
            overlap_integrals: single_overlap.view(),
        };
        assert!(matches!(
            atomic_overlap_amplitude_reduction(input),
            Err(AtomMathError::HoleOrbitalOutOfRange { .. })
        ));

        let input = AtomicOverlapAmplitudeReductionInput {
            hole_orbital_1based: None,
            kappas: &[1, 1],
            occupations: &[1.0, 1.0],
            overlap_integrals: single_overlap.view(),
        };
        assert!(matches!(
            atomic_overlap_amplitude_reduction(input),
            Err(AtomMathError::OverlapMatrixShape { .. })
        ));

        let too_many_kappas = [1; 9];
        let too_many_occupations = [1.0; 9];
        let too_many_overlaps = Array2::from_diag(&Array1::from_elem(9, 1.0));
        let input = AtomicOverlapAmplitudeReductionInput {
            hole_orbital_1based: None,
            kappas: &too_many_kappas,
            occupations: &too_many_occupations,
            overlap_integrals: too_many_overlaps.view(),
        };
        assert!(matches!(
            atomic_overlap_amplitude_reduction(input),
            Err(AtomMathError::KappaGroupTooLarge { .. })
        ));
    }

    struct SchmidtFixture {
        kappas: Vec<i32>,
        active_lengths: Vec<usize>,
        orbital_powers: Vec<Real>,
        large_components: Array2<Real>,
        small_components: Array2<Real>,
        large_coefficients: Array2<Real>,
        small_coefficients: Array2<Real>,
    }

    struct DsordfFixture {
        radii: Array1<Real>,
        active_lengths: Vec<usize>,
        orbital_powers: Vec<Real>,
        large_components: Array2<Real>,
        small_components: Array2<Real>,
        large_coefficients: Array2<Real>,
        small_coefficients: Array2<Real>,
        derivative_large: Array1<Real>,
        derivative_small: Array1<Real>,
        derivative_large_coefficients: Array1<Real>,
        derivative_small_coefficients: Array1<Real>,
    }

    struct YzktegFixture {
        source: Array1<Real>,
        source_coefficients: Array1<Real>,
        radii: Array1<Real>,
    }

    struct VldaFixture {
        radii: Array1<Real>,
        active_lengths: Vec<usize>,
        occupations: Vec<Real>,
        valence_occupations: Vec<Real>,
        large_components: Array2<Real>,
        small_components: Array2<Real>,
        initial_potential: Array1<Real>,
        initial_development_coefficients: Array1<Real>,
        initial_energy_density: Array1<Real>,
    }

    struct PotrdfFixture {
        radii: Array1<Real>,
        active_lengths: Vec<usize>,
        kappas: Vec<i32>,
        orbital_powers: Vec<Real>,
        occupations: Vec<Real>,
        shell_markers: Vec<i32>,
        origin_scales: Vec<Real>,
        coulomb_coefficients: Array3<Real>,
        lagrange_parameters: Array1<Real>,
        nuclear_potential: Array1<Real>,
        nuclear_development_coefficients: Array1<Real>,
        large_components: Array2<Real>,
        small_components: Array2<Real>,
        large_coefficients: Array2<Real>,
        small_coefficients: Array2<Real>,
    }

    struct SoldirNormFixture {
        radii: Array1<Real>,
        large_component: Array1<Real>,
        small_component: Array1<Real>,
        large_coefficients: Array1<Real>,
        small_coefficients: Array1<Real>,
    }

    struct SoldirSolutionNormalizationFixture {
        large_component: Array1<Real>,
        small_component: Array1<Real>,
        large_coefficients: Array1<Real>,
        small_coefficients: Array1<Real>,
    }

    struct SoldirSetupFixture {
        radii: Array1<Real>,
        potential: Array1<Real>,
        potential_coefficients: Array1<Real>,
        positive_origin_coefficients: Array1<Real>,
    }

    struct IntdirFixture {
        radii: Array1<Real>,
        potential: Array1<Real>,
        potential_coefficients: Array1<Real>,
        large_source: Array1<Real>,
        small_source: Array1<Real>,
        large_coefficients: Array1<Real>,
        small_coefficients: Array1<Real>,
    }

    impl SoldirNormFixture {
        fn input(
            &self,
            method: i32,
            coefficient_count: usize,
            matching_small_component: Real,
            origin_power: Real,
            active_len: usize,
            matching_index_1based: usize,
        ) -> AtomicDiracNormalizationInput<'_> {
            AtomicDiracNormalizationInput {
                radii: self.radii.view(),
                large_component: self.large_component.view(),
                small_component: self.small_component.view(),
                large_coefficients: self.large_coefficients.view(),
                small_coefficients: self.small_coefficients.view(),
                method,
                step: 0.05,
                coefficient_count,
                matching_small_component,
                origin_power,
                active_len,
                matching_index_1based,
            }
        }
    }

    impl SoldirSolutionNormalizationFixture {
        fn input(
            &self,
            norm: Real,
            initial_large_coefficient: Real,
            initial_small_coefficient: Real,
        ) -> AtomicDiracSolutionNormalizationInput<'_> {
            AtomicDiracSolutionNormalizationInput {
                norm,
                initial_large_coefficient,
                initial_small_coefficient,
                large_component: self.large_component.view(),
                small_component: self.small_component.view(),
                large_coefficients: self.large_coefficients.view(),
                small_coefficients: self.small_coefficients.view(),
                coefficient_count: 4,
                active_len: 7,
            }
        }
    }

    impl SoldirSetupFixture {
        fn input(
            &self,
            energy: Real,
            method: i32,
            kappa: i32,
            principal_quantum_number: usize,
            negative_origin: bool,
        ) -> AtomicDiracSolverSetupInput<'_> {
            AtomicDiracSolverSetupInput {
                energy,
                origin_power: 1.25,
                initial_large_coefficient: 0.82,
                initial_small_coefficient: -0.006,
                principal_quantum_number,
                kappa,
                speed_of_light: 137.0373,
                method,
                radii: self.radii.view(),
                potential: self.potential.view(),
                potential_coefficients: if negative_origin {
                    self.potential_coefficients.view()
                } else {
                    self.positive_origin_coefficients.view()
                },
                active_len: 7,
            }
        }
    }

    impl IntdirFixture {
        fn input(
            &self,
            mode: AtomicDiracIntegrationMode,
            matching_index_1based: usize,
            max_index_1based: usize,
        ) -> AtomicDiracIntegrationInput<'_> {
            AtomicDiracIntegrationInput {
                large_source: self.large_source.view(),
                small_source: self.small_source.view(),
                large_coefficients: self.large_coefficients.view(),
                small_coefficients: self.small_coefficients.view(),
                radii: self.radii.view(),
                potential: self.potential.view(),
                potential_coefficients: self.potential_coefficients.view(),
                energy: -0.08,
                origin_power: 0.999,
                initial_large_coefficient: 0.85,
                initial_small_coefficient: -0.004,
                asymptotic_large_component: 0.02,
                kappa: -1,
                speed_of_light: 137.0373,
                step: 0.05,
                matching_precision: 1.0e-7,
                coefficient_count: 6,
                active_len: 151,
                mode,
                matching_index_1based,
                max_index_1based,
            }
        }
    }

    impl DsordfFixture {
        fn input(
            &self,
            kind: AtomicDifferentialIntegralKind,
            power: i32,
            origin_power: Real,
        ) -> AtomicDifferentialIntegralInput<'_> {
            AtomicDifferentialIntegralInput {
                kind,
                power,
                origin_power,
                step: 0.05,
                radii: self.radii.view(),
                active_lengths: &self.active_lengths,
                orbital_powers: &self.orbital_powers,
                large_components: self.large_components.view(),
                small_components: self.small_components.view(),
                large_coefficients: self.large_coefficients.view(),
                small_coefficients: self.small_coefficients.view(),
                derivative_large: self.derivative_large.view(),
                derivative_small: self.derivative_small.view(),
                derivative_large_coefficients: self.derivative_large_coefficients.view(),
                derivative_small_coefficients: self.derivative_small_coefficients.view(),
            }
        }

        fn yzkrdf_input(
            &self,
            left_orbital_1based: usize,
            right_orbital_1based: usize,
            angular_momentum: usize,
            large_small: bool,
        ) -> AtomicYkZkExchangeInput<'_> {
            AtomicYkZkExchangeInput {
                left_orbital_1based,
                right_orbital_1based,
                large_small,
                angular_momentum,
                step: 0.05,
                radii: self.radii.view(),
                active_lengths: &self.active_lengths,
                orbital_powers: &self.orbital_powers,
                large_components: self.large_components.view(),
                small_components: self.small_components.view(),
                large_coefficients: self.large_coefficients.view(),
                small_coefficients: self.small_coefficients.view(),
            }
        }

        fn fdrirk_input<'a>(
            &'a self,
            request: AtomicRadialIntegralRequest,
            kappas: &'a [i32],
            large_small: bool,
            previous_first_factor: Option<AtomicRadialFirstFactorView<'a>>,
        ) -> AtomicRadialIntegralInput<'a> {
            AtomicRadialIntegralInput {
                request,
                large_small,
                previous_first_factor,
                kappas,
                step: 0.05,
                radii: self.radii.view(),
                active_lengths: &self.active_lengths,
                orbital_powers: &self.orbital_powers,
                large_components: self.large_components.view(),
                small_components: self.small_components.view(),
                large_coefficients: self.large_coefficients.view(),
                small_coefficients: self.small_coefficients.view(),
            }
        }
    }

    impl YzktegFixture {
        fn input(&self) -> AtomicYkZkTransformInput<'_> {
            AtomicYkZkTransformInput {
                source: self.source.view(),
                source_coefficients: self.source_coefficients.view(),
                radii: self.radii.view(),
                initial_power: 0.65,
                step: 0.05,
                angular_momentum: 2,
                coefficient_count: 6,
                source_len: 9,
                active_len: 13,
            }
        }

        fn prepared_input(
            &self,
            source_len: usize,
            angular_momentum: usize,
        ) -> AtomicYkZkPreparedSourceInput<'_> {
            AtomicYkZkPreparedSourceInput {
                source: self.source.view(),
                source_coefficients: self.source_coefficients.view(),
                radii: self.radii.view(),
                step: 0.05,
                angular_momentum,
                coefficient_count: 6,
                source_len,
                active_len: 13,
            }
        }
    }

    impl VldaFixture {
        fn input(
            &self,
            mode: AtomicLocalDensityExchangeMode,
            accumulate_energy_density: bool,
        ) -> AtomicLocalDensityPotentialInput<'_> {
            AtomicLocalDensityPotentialInput {
                mode,
                accumulate_energy_density,
                speed_of_light: 137.035_999,
                radii: self.radii.view(),
                active_lengths: &self.active_lengths,
                occupations: &self.occupations,
                valence_occupations: &self.valence_occupations,
                large_components: self.large_components.view(),
                small_components: self.small_components.view(),
                initial_potential: self.initial_potential.view(),
                initial_development_coefficients: self.initial_development_coefficients.view(),
                initial_energy_density: self.initial_energy_density.view(),
            }
        }
    }

    impl PotrdfFixture {
        fn input(
            &self,
            include_exchange: bool,
            include_lagrange: bool,
        ) -> AtomicOrbitalPotentialInput<'_> {
            AtomicOrbitalPotentialInput {
                active_orbital_1based: 2,
                include_exchange,
                include_lagrange,
                self_consistent_count: 3,
                speed_of_light: 137.035_999,
                step: 0.05,
                radii: self.radii.view(),
                active_lengths: &self.active_lengths,
                kappas: &self.kappas,
                orbital_powers: &self.orbital_powers,
                occupations: &self.occupations,
                shell_markers: &self.shell_markers,
                origin_scales: &self.origin_scales,
                coulomb_coefficients: self.coulomb_coefficients.view(),
                lagrange_parameters: self.lagrange_parameters.view(),
                nuclear_potential: self.nuclear_potential.view(),
                nuclear_development_coefficients: self.nuclear_development_coefficients.view(),
                large_components: self.large_components.view(),
                small_components: self.small_components.view(),
                large_coefficients: self.large_coefficients.view(),
                small_coefficients: self.small_coefficients.view(),
            }
        }
    }

    fn sample_dsordf_fixture() -> DsordfFixture {
        sample_atomic_radial_fixture(11)
    }

    fn sample_yzkrdf_fixture() -> DsordfFixture {
        sample_atomic_radial_fixture(13)
    }

    fn sample_soldir_norm_fixture() -> SoldirNormFixture {
        SoldirNormFixture {
            radii: Array1::from_shape_fn(251, |row| (-8.8 + 0.05 * row as Real).exp()),
            large_component: Array1::from_shape_fn(251, |row| {
                let index = (row + 1) as Real;
                0.03 * index + 0.002 * (0.17 * index).sin()
            }),
            small_component: Array1::from_shape_fn(251, |row| {
                let index = (row + 1) as Real;
                -0.014 * index + 0.003 * (0.11 * index).cos()
            }),
            large_coefficients: Array1::from_shape_fn(10, |row| {
                let index = (row + 1) as Real;
                0.021 * index - 0.0007 * index * index
            }),
            small_coefficients: Array1::from_shape_fn(10, |row| {
                let index = (row + 1) as Real;
                -0.013 * index + 0.0004 * index * index
            }),
        }
    }

    fn sample_soldir_solution_normalization_fixture(
        flip_coefficients: bool,
        flip_components: bool,
    ) -> SoldirSolutionNormalizationFixture {
        let mut large_coefficients = Array1::from_shape_fn(10, |row| {
            let index = (row + 1) as Real;
            0.2 * index + 0.01 * index * index
        });
        let small_coefficients = Array1::from_shape_fn(10, |row| {
            let index = (row + 1) as Real;
            -0.11 * index + 0.003 * index * index
        });
        let mut large_component = Array1::from_shape_fn(9, |row| {
            let index = (row + 1) as Real;
            0.04 * index + 0.001 * index * index
        });
        let small_component = Array1::from_shape_fn(9, |row| {
            let index = (row + 1) as Real;
            -0.03 * index + 0.0005 * index * index
        });

        if flip_coefficients {
            large_coefficients[0] = -large_coefficients[0];
        }
        if flip_components {
            large_component[0] = -large_component[0];
        }

        SoldirSolutionNormalizationFixture {
            large_component,
            small_component,
            large_coefficients,
            small_coefficients,
        }
    }

    fn sample_soldir_node_count_component() -> Array1<Real> {
        Array1::from_vec(vec![0.2, 0.1, -0.05, 0.0, 0.0, 0.03, -0.02, -0.01, 0.01])
    }

    fn sample_soldir_setup_fixture() -> SoldirSetupFixture {
        SoldirSetupFixture {
            radii: Array1::from_shape_fn(7, |row| 0.08 * (0.11 * row as Real).exp()),
            potential: Array1::from_shape_fn(7, |row| {
                let radius = 0.08 * (0.11 * row as Real).exp();
                -0.42 * (-0.30 * radius).exp() + 0.008 * row as Real
            }),
            potential_coefficients: Array1::from_vec(vec![-0.058_378_260_164_777, 0.0006, -0.0003]),
            positive_origin_coefficients: Array1::from_vec(vec![0.021, 0.0006, -0.0003]),
        }
    }

    fn sample_intdir_fixture() -> IntdirFixture {
        let speed_of_light = 137.0373;
        let step = 0.05;
        let nuclear_charge = 8.0;
        IntdirFixture {
            radii: Array1::from_shape_fn(251, |row| 0.03 * (step * row as Real).exp()),
            potential: Array1::from_shape_fn(251, |row| {
                let radius = 0.03 * (step * row as Real).exp();
                -0.25 * (-0.40 * radius).exp()
            }),
            potential_coefficients: Array1::from_shape_fn(10, |row| {
                if row == 0 {
                    -nuclear_charge / speed_of_light
                } else {
                    0.0003 * row as Real * (-1.0_f64).powi((row + 1) as i32)
                }
            }),
            large_source: Array1::from_shape_fn(251, |row| {
                let index = (row + 1) as Real;
                0.001 * (0.17 * index).sin() + 0.0002 * (0.03 * index).cos()
            }),
            small_source: Array1::from_shape_fn(251, |row| {
                let index = (row + 1) as Real;
                0.0007 * (0.11 * index).cos() - 0.0001 * (0.05 * index).sin()
            }),
            large_coefficients: Array1::from_shape_fn(10, |row| {
                let index = (row + 1) as Real;
                0.0002 * index * (-1.0_f64).powi((row + 1) as i32)
            }),
            small_coefficients: Array1::from_shape_fn(10, |row| {
                let index = (row + 1) as Real;
                -0.00015 * index * (-1.0_f64).powi((row + 1) as i32)
            }),
        }
    }

    fn sample_vlda_fixture() -> VldaFixture {
        let radial_count = 13;
        let orbital_count = 3;
        VldaFixture {
            radii: Array1::from_shape_fn(radial_count, |row| (-4.2 + 0.05 * row as Real).exp()),
            active_lengths: vec![9, 11, 7],
            occupations: vec![2.0, 1.6, 0.7],
            valence_occupations: vec![1.0, 0.4, 0.2],
            large_components: Array2::from_shape_fn((radial_count, orbital_count), |(row, col)| {
                let radial = (row + 1) as Real;
                let orbital = (col + 1) as Real;
                0.02 * orbital + 0.0015 * radial + 0.00003 * radial * orbital
            }),
            small_components: Array2::from_shape_fn((radial_count, orbital_count), |(row, col)| {
                let radial = (row + 1) as Real;
                let orbital = (col + 1) as Real;
                -0.006 * orbital + 0.0008 * radial - 0.00001 * radial * orbital
            }),
            initial_potential: Array1::from_shape_fn(radial_count, |row| {
                0.0001 * (row + 1) as Real
            }),
            initial_development_coefficients: Array1::from_shape_fn(6, |row| {
                0.01 * (row + 1) as Real
            }),
            initial_energy_density: Array1::from_shape_fn(radial_count, |row| {
                0.002 * (row + 1) as Real
            }),
        }
    }

    fn sample_potrdf_fixture() -> PotrdfFixture {
        let radial_count = 13;
        let orbital_count = 3;
        let coefficient_count = 6;
        PotrdfFixture {
            radii: Array1::from_shape_fn(radial_count, |row| (-4.2 + 0.05 * row as Real).exp()),
            active_lengths: vec![9, 11, 7],
            kappas: vec![-1, 1, 1],
            orbital_powers: (1..=orbital_count)
                .map(|orbital| 0.12 + 0.09 * orbital as Real)
                .collect(),
            occupations: vec![2.0, 1.6, 0.7],
            shell_markers: vec![-1, 1, 1],
            origin_scales: vec![1.05, 0.95, 1.10],
            coulomb_coefficients: Array3::from_shape_fn(
                (orbital_count, orbital_count, 5),
                |(left, right, rank)| {
                    0.015 * (left + 1) as Real + 0.011 * (right + 1) as Real + 0.003 * rank as Real
                },
            ),
            lagrange_parameters: Array1::from_shape_fn(3, |row| 0.012 * (row + 1) as Real),
            nuclear_potential: Array1::from_shape_fn(radial_count, |row| {
                -0.2 + 0.001 * (row + 1) as Real
            }),
            nuclear_development_coefficients: Array1::from_shape_fn(coefficient_count, |row| {
                -0.03 * (row + 1) as Real
            }),
            large_components: Array2::from_shape_fn((radial_count, orbital_count), |(row, col)| {
                let radial = (row + 1) as Real;
                let orbital = (col + 1) as Real;
                0.02 * orbital + 0.0015 * radial + 0.00003 * radial * orbital
            }),
            small_components: Array2::from_shape_fn((radial_count, orbital_count), |(row, col)| {
                let radial = (row + 1) as Real;
                let orbital = (col + 1) as Real;
                -0.006 * orbital + 0.0008 * radial - 0.00001 * radial * orbital
            }),
            large_coefficients: Array2::from_shape_fn(
                (coefficient_count, orbital_count),
                |(row, col)| 0.08 * (row + 1) as Real + 0.015 * (col + 1) as Real,
            ),
            small_coefficients: Array2::from_shape_fn(
                (coefficient_count, orbital_count),
                |(row, col)| -0.02 * (row + 1) as Real + 0.01 * (col + 1) as Real,
            ),
        }
    }

    fn sample_atomic_radial_fixture(radial_count: usize) -> DsordfFixture {
        let orbital_count = 3;
        let coefficient_count = 6;
        DsordfFixture {
            radii: Array1::from_shape_fn(radial_count, |row| (-4.2 + 0.05 * row as Real).exp()),
            active_lengths: vec![9, 11, 7],
            orbital_powers: (1..=orbital_count)
                .map(|orbital| 0.12 + 0.09 * orbital as Real)
                .collect(),
            large_components: Array2::from_shape_fn((radial_count, orbital_count), |(row, col)| {
                let radial = (row + 1) as Real;
                let orbital = (col + 1) as Real;
                0.02 * orbital + 0.0015 * radial + 0.00003 * radial * orbital
            }),
            small_components: Array2::from_shape_fn((radial_count, orbital_count), |(row, col)| {
                let radial = (row + 1) as Real;
                let orbital = (col + 1) as Real;
                -0.006 * orbital + 0.0008 * radial - 0.00001 * radial * orbital
            }),
            large_coefficients: Array2::from_shape_fn(
                (coefficient_count, orbital_count),
                |(row, col)| {
                    let coefficient = (row + 1) as Real;
                    let orbital = (col + 1) as Real;
                    0.08 * coefficient + 0.015 * orbital
                },
            ),
            small_coefficients: Array2::from_shape_fn(
                (coefficient_count, orbital_count),
                |(row, col)| {
                    let coefficient = (row + 1) as Real;
                    let orbital = (col + 1) as Real;
                    -0.02 * coefficient + 0.01 * orbital
                },
            ),
            derivative_large: Array1::from_shape_fn(radial_count, |row| {
                let radial = (row + 1) as Real;
                0.015 * radial - 0.00007 * radial * radial
            }),
            derivative_small: Array1::from_shape_fn(radial_count, |row| {
                let radial = (row + 1) as Real;
                -0.004 * radial + 0.00013 * radial * radial
            }),
            derivative_large_coefficients: Array1::from_shape_fn(coefficient_count, |row| {
                let coefficient = (row + 1) as Real;
                0.05 * coefficient - 0.003
            }),
            derivative_small_coefficients: Array1::from_shape_fn(coefficient_count, |row| {
                let coefficient = (row + 1) as Real;
                -0.015 * coefficient + 0.004
            }),
        }
    }

    fn sample_yzkteg_fixture() -> YzktegFixture {
        let active_len = 13;
        let coefficient_count = 6;
        YzktegFixture {
            source: Array1::from_shape_fn(active_len, |row| {
                let row = (row + 1) as Real;
                0.017 * row + 0.0008 * row * row - 0.00001 * row * row * row
            }),
            source_coefficients: Array1::from_shape_fn(coefficient_count, |row| {
                let row = (row + 1) as Real;
                0.04 * row - 0.0015 * row * row
            }),
            radii: Array1::from_shape_fn(active_len, |row| (-4.2 + 0.05 * row as Real).exp()),
        }
    }

    impl SchmidtFixture {
        fn as_input(
            &self,
            active_orbital_1based: Option<usize>,
        ) -> AtomicSchmidtOrthogonalizationInput<'_> {
            AtomicSchmidtOrthogonalizationInput {
                active_orbital_1based,
                kappas: &self.kappas,
                active_lengths: &self.active_lengths,
                orbital_powers: &self.orbital_powers,
                large_components: self.large_components.view(),
                small_components: self.small_components.view(),
                large_coefficients: self.large_coefficients.view(),
                small_coefficients: self.small_coefficients.view(),
            }
        }
    }

    fn sample_schmidt_fixture() -> SchmidtFixture {
        SchmidtFixture {
            kappas: vec![-1, -1, 1, -1],
            active_lengths: vec![3, 4, 3, 5],
            orbital_powers: (1..=4).map(|orbital| 0.1 * orbital as Real).collect(),
            large_components: Array2::from_shape_fn((5, 4), |(row, orbital)| {
                0.07 * (row + 1) as Real + 0.11 * (orbital + 1) as Real
            }),
            small_components: Array2::from_shape_fn((5, 4), |(row, orbital)| {
                0.03 * (row + 1) as Real - 0.02 * (orbital + 1) as Real
            }),
            large_coefficients: Array2::from_shape_fn((4, 4), |(row, orbital)| {
                0.2 * (row + 1) as Real + 0.05 * (orbital + 1) as Real
            }),
            small_coefficients: Array2::from_shape_fn((4, 4), |(row, orbital)| {
                -0.03 * (row + 1) as Real + 0.04 * (orbital + 1) as Real
            }),
        }
    }

    fn sample_schmidt_integral(
        request: AtomicSchmidtIntegralRequest<'_>,
    ) -> Result<Real, AtomMathError> {
        match request {
            AtomicSchmidtIntegralRequest::Projection(request) => Ok(request
                .target_large
                .iter()
                .zip(request.reference_large.iter())
                .map(|(&target, &reference)| target * reference)
                .sum::<Real>()
                + request
                    .target_small
                    .iter()
                    .zip(request.reference_small.iter())
                    .map(|(&target, &reference)| target * reference)
                    .sum::<Real>()),
            AtomicSchmidtIntegralRequest::Norm(request) => Ok(request
                .target_large
                .iter()
                .map(|&value| value * value)
                .sum::<Real>()
                + request
                    .target_small
                    .iter()
                    .map(|&value| value * value)
                    .sum::<Real>()),
        }
    }

    fn assert_columns_close<const ROWS: usize, const COLUMNS: usize>(
        actual: &Array2<Real>,
        expected_columns: &[[Real; ROWS]; COLUMNS],
        tolerance: Real,
    ) {
        assert_eq!(actual.nrows(), ROWS);
        assert_eq!(actual.ncols(), COLUMNS);
        for (column, expected_column) in expected_columns.iter().enumerate() {
            for (row, &expected) in expected_column.iter().enumerate() {
                assert_close_with(actual[(row, column)], expected, tolerance);
            }
        }
    }

    fn sample_s02at_overlaps() -> Array2<Real> {
        let mut overlaps =
            Array2::from_shape_fn((6, 6), |(row, column)| 0.02 * (row + column + 2) as Real);
        for index in 0..6 {
            overlaps[(index, index)] = 1.0;
        }
        overlaps[(0, 1)] = 0.91;
        overlaps[(1, 0)] = 0.91;
        overlaps[(2, 3)] = 0.82;
        overlaps[(3, 2)] = 0.82;
        overlaps
    }

    fn sample_atomic_radial_integral(
        request: AtomicRadialIntegralRequest,
    ) -> Result<Real, AtomMathError> {
        Ok(0.0001 * (request.rank + 1) as Real
            + 0.001 * request.first_left as Real
            + 0.0002 * request.first_right as Real
            + 0.00003 * request.second_left as Real
            + 0.000004 * request.second_right as Real)
    }

    fn sample_atomic_tabrat_integral(
        request: AtomicTabulationIntegralRequest,
    ) -> Result<Real, AtomMathError> {
        Ok(0.01 * (request.left + 1) as Real
            + 0.02 * (request.right + 1) as Real
            + 0.001 * request.power as Real
            + 0.1)
    }
}
