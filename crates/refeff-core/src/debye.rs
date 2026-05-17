//! Debye and Einstein-model cumulant helpers ported from FEFF.
//!
//! This module starts with `DEBYE/sigm3.f90`, the correlated Einstein model
//! with a Morse potential used for first and third cumulant estimates.

use ndarray::{Array1, Array2, ArrayView1, ArrayView2, ArrayView3, ArrayView4};
use refeff_linalg::{SymmetricTriangle, real64_symmetric_eigen};

use crate::{Real, atomic::atomic_weight as feff_atomic_weight};

const BOHR_ANGSTROM: Real = 0.529_177_249;
/// FEFF DMDW conversion factor from Angstrom to Bohr.
pub const DMDW_ANGSTROM_TO_BOHR: Real = 1.889_726_663_510_319_2;
const HBAR: Real = 1.054_572_7e-34_f32 as Real;
const ATOMIC_MASS_UNIT: Real = 1.660_54e-27_f32 as Real;
const BOLTZMANN: Real = 1.380_658e-23_f32 as Real;
const DEBYE_CORRELATION_FACTOR: Real = 48.508_46_f32 as Real;
const DEBYE_ROMBERG_TOLERANCE: Real = 1.0e-5;
const DEBYE_ROMBERG_MAX_ITERATIONS: usize = 10;
const AU_FORCE_TO_NEWTON_PER_METER: Real = 1_556.892_791_61;
const NEWTON_PER_METER_TO_AMU_PER_PS2: Real = 602.214_198_280;
const DMDW_DYNAMICAL_MATRIX_SCALE: Real =
    AU_FORCE_TO_NEWTON_PER_METER * NEWTON_PER_METER_TO_AMU_PER_PS2;
const DMDW_AMU_EV: Real = 9.314_78e8;
const DMDW_LIGHT_SPEED_ANGSTROM_PER_PS: Real = 2.997_924_58e6;
const DMDW_BOLTZMANN_EV_PER_K: Real = 8.617_385e-5;
const DMDW_HBAR_EV_PS: Real = 6.582_122e-4;
const DMDW_HBARC_EV_ANGSTROM: Real = 1_973.27;
const DMDW_GAS_CONSTANT_J_PER_MOL_K: Real = 8.314_713_470;
const DMDW_THZ_TO_KELVIN: Real = 47.990_874_194_2;
const DMDW_AMU_THZ2_TO_NEWTON_PER_METER: Real = 0.001_660_538_730_00;
const DMDW_LANCZOS_POLE_SEARCH_LIMIT: Real = 810_000.0;
const DMDW_LANCZOS_DEFAULT_SAMPLES_PER_POLE: usize = 100_000;
const DMDW_IMAGINARY_POLE_SMALL_WEIGHT: Real = 0.01;
const DMDW_IMAGINARY_POLE_LARGE_WEIGHT: Real = 0.05;

/// First and third cumulants from FEFF `sigm3`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MorseCumulants {
    /// FEFF `sig1`: first cumulant.
    pub first: Real,
    /// FEFF `sig3`: third cumulant.
    pub third: Real,
    /// FEFF mutates `alphat` from inverse angstrom to inverse bohr; Rust returns
    /// the scaled value explicitly.
    pub scaled_thermal_expansion: Real,
}

/// First and third cumulants from FEFF `sigte3`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ThermalExpansionCumulants {
    /// FEFF `sig1`: first cumulant.
    pub first: Real,
    /// FEFF `sig3`: third cumulant.
    pub third: Real,
}

/// Correlated Debye-model displacement correlation from FEFF `corrfn`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DebyeCorrelation {
    /// Correlation value returned as FEFF `cij`.
    pub value: Real,
    /// Relative Romberg estimate used by FEFF `bingrt`.
    pub estimated_error: Real,
    /// Number of binary-refinement iterations used by the integration.
    pub iterations: usize,
}

/// Reduced path mass and normalized initial vector from FEFF DMDW `Calc_DW`.
#[derive(Debug, Clone, PartialEq)]
pub struct DmdwPathMotion {
    /// FEFF `mu`, the path reduced mass in the same units as the input masses.
    pub reduced_mass: Real,
    /// FEFF `mu_inv`, the inverse reduced mass accumulated from director cosines.
    pub inverse_reduced_mass: Real,
    /// FEFF `qj0`, arranged as component-major blocks:
    /// `component * atom_count + atom_index`.
    pub initial_vector: Array1<Real>,
}

/// FEFF DMDW path descriptor from `Paths_Info%Desc`.
///
/// Selectors use FEFF's convention: `0` expands over every atom, while a
/// positive value selects that 1-based atom index. `max_effective_length` must
/// use the same distance unit as the coordinate table supplied to
/// [`dmdw_expand_path_descriptor`].
#[derive(Debug, Clone, PartialEq)]
pub struct DmdwPathDescriptor {
    /// FEFF path selectors, one per path leg.
    pub selectors: Vec<i32>,
    /// Maximum effective path length after FEFF's closure-distance adjustment.
    pub max_effective_length: Real,
}

/// One concrete path generated from a DMDW descriptor.
#[derive(Debug, Clone, PartialEq)]
pub struct DmdwExpandedPath {
    /// Zero-based atom indices in FEFF path order.
    pub atoms: Vec<usize>,
    /// FEFF `Desc_Paths%Len`, the effective half-closed path length.
    pub effective_length: Real,
}

/// Full mass-weighted DMDW matrix and FEFF symmetry diagnostics.
#[derive(Debug, Clone, PartialEq)]
pub struct DmdwDynamicalMatrix {
    /// Component-major full matrix with indices `component * atom_count + atom`.
    pub matrix: Array2<Real>,
    /// FEFF `Avg_Val`: average absolute matrix value.
    pub average_value: Real,
    /// FEFF `Max_Val`: maximum absolute matrix value.
    pub max_value: Real,
    /// FEFF `Avg_Asym`: average absolute antisymmetric component.
    pub average_asymmetry: Real,
    /// FEFF `Asym_T1`: asymmetry as percent of `average_value`.
    pub asymmetry_percent_average: Real,
    /// FEFF `Asym_T2`: asymmetry as percent of `max_value`.
    pub asymmetry_percent_max: Real,
}

/// FEFF DMDW rigid-body projection basis from `Make_TrfD`.
#[derive(Debug, Clone, PartialEq)]
pub struct DmdwRigidBodyModes {
    /// FEFF `R_CM`, the mass-weighted center of mass.
    pub center_of_mass: [Real; 3],
    /// Coordinates shifted to the center of mass while building rotations.
    pub centered_positions: Array2<Real>,
    /// FEFF `ToI`, the tensor of inertia about the center of mass.
    pub inertia_tensor: Array2<Real>,
    /// FEFF `MoI`, principal moments of inertia.
    pub moments_of_inertia: Array1<Real>,
    /// FEFF `PAoR`, principal axes of rotation stored column-wise.
    pub principal_axes: Array2<Real>,
    /// First six FEFF `TrfD` columns, normalized and stored column-wise.
    pub projection_modes: Array2<Real>,
}

impl DmdwDynamicalMatrix {
    /// Whether FEFF would skip the "not symmetric" warning for this matrix.
    pub fn passes_feff_symmetry_check(&self) -> bool {
        !(self.asymmetry_percent_average > 50.0 || self.asymmetry_percent_max > 5.0)
    }
}

/// Tridiagonal coefficients from FEFF DMDW `Lanczos`.
#[derive(Debug, Clone, PartialEq)]
pub struct DmdwLanczosCoefficients {
    /// FEFF `anj(0:nPoles)` diagonal coefficients.
    pub alpha: Array1<Real>,
    /// FEFF `bnj(0:nPoles+1)` off-diagonal coefficients, with `beta[0] = 0`.
    pub beta: Array1<Real>,
    /// FEFF `SPole_EinsteinFreq`, `sqrt(alpha[0]) / (2*pi)`.
    pub single_pole_frequency: Real,
}

/// Pole and weight tables from FEFF DMDW `Lanczos`.
#[derive(Debug, Clone, PartialEq)]
pub struct DmdwLanczosPoleSpectrum {
    /// Requested FEFF `nPoles`.
    pub expected_poles: usize,
    /// FEFF `xnull`, the roots of `Poly_Y('S', ...)`.
    pub squared_angular_frequencies: Array1<Real>,
    /// FEFF `w_pole`, with negative values representing imaginary modes.
    pub angular_frequencies: Array1<Real>,
    /// FEFF `DW_Out%Poles_Frq`, `w_pole / (2*pi)`.
    pub frequencies: Array1<Real>,
    /// FEFF `wil`, the Lanczos pole weights.
    pub weights: Array1<Real>,
    /// FEFF-style diagnostics for imaginary-frequency poles with notable
    /// positive weight.
    pub imaginary_warnings: Vec<DmdwImaginaryPoleWarning>,
}

/// FEFF DMDW Einstein-frequency summary from pole data.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DmdwEinsteinSummary {
    /// Frequency in THz.
    pub frequency_thz: Real,
    /// Associated Einstein temperature in Kelvin.
    pub temperature_kelvin: Real,
    /// Effective force constant in N/m.
    pub effective_force_constant_n_per_m: Real,
}

/// FEFF DMDW moment-derived Einstein summary row.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DmdwMomentSummary {
    /// Moment order `n`.
    pub order: i32,
    /// Normalized projected-DOS moment in `THz^n`.
    pub moment_thz_power_n: Real,
    /// Derived Einstein frequency in THz. FEFF leaves this blank for `n = 0`.
    pub frequency_thz: Option<Real>,
    /// Derived Einstein temperature in Kelvin. FEFF leaves this blank for `n = 0`.
    pub temperature_kelvin: Option<Real>,
    /// Derived effective force constant in N/m. FEFF leaves this blank for `n = 0`.
    pub effective_force_constant_n_per_m: Option<Real>,
}

impl DmdwLanczosPoleSpectrum {
    /// Whether FEFF's root scan found the requested number of poles.
    pub fn has_expected_pole_count(&self) -> bool {
        self.squared_angular_frequencies.len() == self.expected_poles
    }
}

/// Severity of a FEFF DMDW imaginary-pole diagnostic.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DmdwImaginaryPoleSeverity {
    /// FEFF warning branch: `0.01 <= weight <= 0.05`.
    SmallWeight,
    /// FEFF error branch: `weight >= 0.05`.
    LargeWeight,
}

/// FEFF DMDW imaginary-pole diagnostic data.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DmdwImaginaryPoleWarning {
    /// Zero-based pole index in the found pole table.
    pub pole_index: usize,
    /// FEFF `xnull` root for this pole.
    pub squared_angular_frequency: Real,
    /// FEFF `w_pole`, negative for imaginary modes.
    pub angular_frequency: Real,
    /// FEFF printed frequency, `w_pole / (2*pi)`.
    pub frequency: Real,
    /// FEFF `wil` weight.
    pub weight: Real,
    /// FEFF warning or error branch.
    pub severity: DmdwImaginaryPoleSeverity,
}

/// Error returned by Debye/Einstein cumulant helpers.
#[derive(Debug, Clone, Copy, PartialEq, thiserror::Error)]
pub enum DebyeError {
    /// Inputs must be finite real values.
    #[error("Debye input {name} must be finite, got {value}")]
    NonFinite { name: &'static str, value: Real },
    /// Inputs used as scales must be strictly positive.
    #[error("Debye input {name} must be positive, got {value}")]
    NonPositive { name: &'static str, value: Real },
    /// Inputs used as nonnegative values must be zero or positive.
    #[error("Debye input {name} must be nonnegative, got {value}")]
    Negative { name: &'static str, value: Real },
    /// FEFF's periodic table covers atomic numbers 1 through 139.
    #[error("Debye atomic number must be in 1..=139, got {z}")]
    InvalidAtomicNumber { z: usize },
    /// A computed output became non-finite.
    #[error("Debye output {name} must be finite, got {value}")]
    NonFiniteOutput { name: &'static str, value: Real },
    /// Path coordinates must be an `(nleg + 1) x 3` array.
    #[error(
        "Debye path coordinates must have at least 2 rows and exactly 3 columns, got {rows}x{columns}"
    )]
    InvalidPathShape { rows: usize, columns: usize },
    /// The atomic-number list must align with the coordinate rows.
    #[error("Debye path has {positions} coordinate rows but {atomic_numbers} atomic numbers")]
    InvalidAtomicNumberCount {
        positions: usize,
        atomic_numbers: usize,
    },
    /// Consecutive path coordinates must not be identical.
    #[error("Debye path leg {leg} has zero length")]
    ZeroLengthPathLeg { leg: usize },
    /// DMDW atom positions must be an `natom x 3` array.
    #[error("DMDW atom positions must have exactly 3 columns, got {rows}x{columns}")]
    InvalidDmdwAtomShape { rows: usize, columns: usize },
    /// DMDW atom masses must align with coordinate rows.
    #[error("DMDW has {positions} atom position rows but {masses} atom masses")]
    InvalidDmdwMassCount { positions: usize, masses: usize },
    /// DMDW atom tables must contain at least one atom.
    #[error("DMDW atom table must contain at least one atom")]
    EmptyDmdwAtomTable,
    /// DMDW seed projection modes must align with the seed vector.
    #[error("DMDW seed projection has seed length {seed_len} but mode matrix is {rows}x{columns}")]
    InvalidDmdwProjectionShape {
        seed_len: usize,
        rows: usize,
        columns: usize,
    },
    /// DMDW block dynamical matrix must be `(atom, atom, 3, 3)`.
    #[error(
        "DMDW block matrix has shape {atoms_i}x{atoms_j}x{components_i}x{components_j} for {masses} masses"
    )]
    InvalidDmdwBlockShape {
        atoms_i: usize,
        atoms_j: usize,
        components_i: usize,
        components_j: usize,
        masses: usize,
    },
    /// DMDW IR dipole-derivative table must be `(atom, displacement, dipole)`.
    #[error(
        "DMDW IR dipole derivatives have shape {atoms}x{displacements}x{dipoles} for {masses} masses"
    )]
    InvalidDmdwDipoleDerivativeShape {
        atoms: usize,
        displacements: usize,
        dipoles: usize,
        masses: usize,
    },
    /// DMDW Lanczos requires a square matrix aligned with the seed vector.
    #[error("DMDW Lanczos matrix is {rows}x{columns} for seed length {seed_len}")]
    InvalidDmdwLanczosShape {
        rows: usize,
        columns: usize,
        seed_len: usize,
    },
    /// DMDW Lanczos polynomial inputs must cover the requested order.
    #[error(
        "DMDW Lanczos polynomial order {order} needs alpha/beta lengths >= order, got {alpha_len}/{beta_len}"
    )]
    InvalidDmdwLanczosPolynomialShape {
        order: usize,
        alpha_len: usize,
        beta_len: usize,
    },
    /// DMDW pole frequencies and weights must have matching lengths.
    #[error("DMDW pole table has {frequencies} frequencies but {weights} weights")]
    InvalidDmdwPoleTableShape { frequencies: usize, weights: usize },
    /// DMDW pole summaries require at least one pole.
    #[error("DMDW pole table must contain at least one pole")]
    EmptyDmdwPoleTable,
    /// DMDW temperature tables must contain at least one temperature.
    #[error("DMDW temperature table must contain at least one value")]
    EmptyDmdwTemperatureTable,
    /// DMDW paths must contain at least one atom index.
    #[error("DMDW path must contain at least one atom index")]
    EmptyDmdwPath,
    /// DMDW seed vectors must not be empty.
    #[error("DMDW seed vector must not be empty")]
    EmptyDmdwSeed,
    /// DMDW path atom index was outside the available atom table.
    #[error("DMDW path atom index {index} is outside 0..{atom_count}")]
    InvalidDmdwPathAtomIndex { index: usize, atom_count: usize },
    /// DMDW path descriptors use FEFF selectors: zero or 1-based atom indices.
    #[error("DMDW path selector {selector} is outside 0..={atom_count}")]
    InvalidDmdwPathSelector { selector: i32, atom_count: usize },
    /// DMDW wildcard path expansion overflowed `usize`.
    #[error("DMDW path descriptor with {selectors} selectors is too large for {atom_count} atoms")]
    DmdwPathExpansionTooLarge { selectors: usize, atom_count: usize },
    /// DMDW director cosines require distinct atom positions.
    #[error("DMDW atom pair {first}-{second} has zero distance")]
    ZeroLengthDmdwAtomPair { first: usize, second: usize },
    /// DMDW seed normalization requires a nonzero vector.
    #[error("DMDW seed vector has zero norm")]
    ZeroDmdwSeedNorm,
    /// DMDW Lanczos recursion cannot continue after a zero residual norm.
    #[error("DMDW Lanczos recursion broke down at iteration {iteration}")]
    DmdwLanczosBreakdown { iteration: usize },
    /// DMDW rigid-body projection requires at least two atoms.
    #[error("DMDW rigid-body projection requires at least two atoms, got {atoms}")]
    TooFewDmdwRigidBodyAtoms { atoms: usize },
    /// DMDW Lanczos pole search step count overflowed `usize`.
    #[error(
        "DMDW Lanczos pole search is too large for order {order} and {samples_per_pole} samples per pole"
    )]
    DmdwLanczosPoleSearchTooLarge {
        order: usize,
        samples_per_pole: usize,
    },
    /// DMDW Lanczos pole weights require a nonzero derivative.
    #[error("DMDW Lanczos pole {pole_index} has zero polynomial derivative at {x}")]
    ZeroDmdwLanczosPoleDerivative { pole_index: usize, x: Real },
    /// FEFF `Make_TrfD` cannot normalize a zero rigid-body mode.
    #[error("DMDW rigid-body projection mode {mode} has zero norm")]
    ZeroDmdwProjectionModeNorm { mode: usize },
    /// FEFF `Make_TrfD` principal-axis decomposition did not converge.
    #[error("DMDW rigid-body principal-axis eigensolver did not converge")]
    DmdwRigidBodyEigenDidNotConverge,
    /// FEFF Romberg integration did not converge within the configured limit.
    #[error(
        "{routine} did not converge after {iterations} iterations; estimated error {estimated_error}"
    )]
    IntegrationDidNotConverge {
        routine: &'static str,
        iterations: usize,
        estimated_error: Real,
    },
}

/// Port of FEFF `sigm3`: correlated Einstein-model Morse cumulants.
///
/// `mean_square_relative_displacement` is FEFF `sig2`, `temperature` is `tk`,
/// `thermal_expansion` is `alphat` in inverse angstrom, and
/// `einstein_temperature` is `thetae`. FEFF stores several intermediates as
/// single precision `real`; this port keeps those roundings to match the
/// reference values.
pub fn morse_einstein_cumulants(
    mean_square_relative_displacement: Real,
    temperature: Real,
    thermal_expansion: Real,
    einstein_temperature: Real,
) -> Result<MorseCumulants, DebyeError> {
    ensure_positive("sig2", mean_square_relative_displacement)?;
    ensure_positive("tk", temperature)?;
    ensure_finite("alphat", thermal_expansion)?;
    ensure_positive("thetae", einstein_temperature)?;

    let scaled_thermal_expansion = thermal_expansion * BOHR_ANGSTROM;
    let z = to_feff_real((-einstein_temperature / temperature).exp());
    let occupation_ratio = to_feff_real(((1.0_f32 - z as f32) / (1.0_f32 + z as f32)) as Real);
    let sig02 = to_feff_real(occupation_ratio * mean_square_relative_displacement);
    let sig01 = to_feff_real(scaled_thermal_expansion * sig02 * 0.75);
    let first = sig01 * mean_square_relative_displacement / sig02;
    let third = (2.0
        - ((4.0_f32 / 3.0_f32) as Real) * (sig02 / mean_square_relative_displacement).powi(2))
        * first
        * mean_square_relative_displacement;

    ensure_finite_output("sig1", first)?;
    ensure_finite_output("sig3", third)?;
    ensure_finite_output("alphat", scaled_thermal_expansion)?;

    Ok(MorseCumulants {
        first,
        third,
        scaled_thermal_expansion,
    })
}

/// Port of FEFF `sigte3`: thermal-expansion first and third cumulants.
///
/// `central_atomic_number` and `neighbor_atomic_number` are FEFF `iz1` and
/// `iz2`; `mean_square_relative_displacement` is `sig2`; `thermal_expansion`
/// is `alphat`; `debye_temperature` is `thetad`; and
/// `effective_distance_angstrom` is FEFF's single-precision `reff`.
pub fn thermal_expansion_cumulants(
    central_atomic_number: usize,
    neighbor_atomic_number: usize,
    mean_square_relative_displacement: Real,
    thermal_expansion: Real,
    debye_temperature: Real,
    effective_distance_angstrom: Real,
) -> Result<ThermalExpansionCumulants, DebyeError> {
    let central_mass = atomic_weight(central_atomic_number)? * ATOMIC_MASS_UNIT;
    let neighbor_mass = atomic_weight(neighbor_atomic_number)? * ATOMIC_MASS_UNIT;
    ensure_positive("sig2", mean_square_relative_displacement)?;
    ensure_finite("alphat", thermal_expansion)?;
    ensure_positive("thetad", debye_temperature)?;
    ensure_positive("reff", effective_distance_angstrom)?;

    let reff = to_feff_real(effective_distance_angstrom);
    let reduced_mass = 1.0 / (1.0 / central_mass + 1.0 / neighbor_mass);
    let omega = (2.0 * BOLTZMANN * debye_temperature) / (3.0 * HBAR);
    let spring_constant = reduced_mass * omega.powi(2);
    let cubic_force_constant =
        spring_constant.powi(2) * reff * thermal_expansion / (3.0 * BOLTZMANN);
    let sig02 = HBAR * omega / spring_constant;
    let first = -3.0 * (cubic_force_constant / spring_constant) * mean_square_relative_displacement;
    let third = (2.0
        - ((4.0_f32 / 3.0_f32) as Real) * (sig02 / mean_square_relative_displacement).powi(2))
        * first
        * mean_square_relative_displacement;

    ensure_finite_output("sig1", first)?;
    ensure_finite_output("sig3", third)?;

    Ok(ThermalExpansionCumulants { first, third })
}

/// Port of FEFF `corrfn`: quantum Debye displacement correlation.
///
/// `distance_angstrom` is FEFF `rij`, `debye_temperature` is `thetad`,
/// `temperature` is `tk`, the atomic numbers are `iz1`/`iz2`, and
/// `average_wigner_seitz_radius_bohr` is `rsavg`.
pub fn quantum_debye_correlation(
    distance_angstrom: Real,
    debye_temperature: Real,
    temperature: Real,
    first_atomic_number: usize,
    second_atomic_number: usize,
    average_wigner_seitz_radius_bohr: Real,
) -> Result<DebyeCorrelation, DebyeError> {
    debye_correlation(DebyeCorrelationInput {
        routine: "corrfn",
        distance_angstrom,
        debye_temperature,
        temperature,
        first_atomic_number,
        second_atomic_number,
        average_wigner_seitz_radius_bohr,
        integrand: quantum_debye_integrand,
    })
}

/// Port of FEFF `corrfn2`: classical Debye displacement correlation.
pub fn classical_debye_correlation(
    distance_angstrom: Real,
    debye_temperature: Real,
    temperature: Real,
    first_atomic_number: usize,
    second_atomic_number: usize,
    average_wigner_seitz_radius_bohr: Real,
) -> Result<DebyeCorrelation, DebyeError> {
    debye_correlation(DebyeCorrelationInput {
        routine: "corrfn2",
        distance_angstrom,
        debye_temperature,
        temperature,
        first_atomic_number,
        second_atomic_number,
        average_wigner_seitz_radius_bohr,
        integrand: classical_debye_integrand,
    })
}

/// Port of FEFF `sigms`: quantum Debye-Waller factor for a scattering path.
///
/// `positions_angstrom` is an `(nleg + 1) x 3` ndarray view. Row `0` and the
/// final row correspond to FEFF's central-atom endpoints for a closed path.
/// `atomic_numbers` must contain one atomic number for each row.
pub fn quantum_debye_waller_factor(
    temperature: Real,
    debye_temperature: Real,
    average_wigner_seitz_radius_bohr: Real,
    positions_angstrom: ArrayView2<'_, Real>,
    atomic_numbers: &[usize],
) -> Result<Real, DebyeError> {
    debye_waller_factor(DebyeWallerInput {
        temperature,
        debye_temperature,
        average_wigner_seitz_radius_bohr,
        positions_angstrom,
        atomic_numbers,
        correlation: quantum_debye_correlation,
    })
}

/// Port of FEFF `sigcl`: classical Debye-Waller factor for a scattering path.
pub fn classical_debye_waller_factor(
    temperature: Real,
    debye_temperature: Real,
    average_wigner_seitz_radius_bohr: Real,
    positions_angstrom: ArrayView2<'_, Real>,
    atomic_numbers: &[usize],
) -> Result<Real, DebyeError> {
    debye_waller_factor(DebyeWallerInput {
        temperature,
        debye_temperature,
        average_wigner_seitz_radius_bohr,
        positions_angstrom,
        atomic_numbers,
        correlation: classical_debye_correlation,
    })
}

/// Port the DMDW `Calc_DW` path mass and initial-vector setup.
///
/// `atom_positions_angstrom` is the DMDW atom table as `(atom, xyz)`,
/// `atom_masses` is FEFF `dym_In%am`, and `path_atoms` is FEFF `lpath` after
/// conversion to zero-based local atom indices. The returned `initial_vector`
/// matches FEFF's component-major `qj0` layout: all x components, then all y,
/// then all z components.
pub fn dmdw_path_motion(
    atom_positions_angstrom: ArrayView2<'_, Real>,
    atom_masses: ArrayView1<'_, Real>,
    path_atoms: &[usize],
) -> Result<DmdwPathMotion, DebyeError> {
    validate_dmdw_path_input(atom_positions_angstrom, atom_masses, path_atoms)?;

    let inverse_reduced_mass = if path_atoms.len() == 1 {
        1.0 / atom_masses[path_atoms[0]]
    } else {
        path_atoms
            .iter()
            .enumerate()
            .map(|(path_index, &atom)| {
                let previous = path_atoms[(path_index + path_atoms.len() - 1) % path_atoms.len()];
                let next = path_atoms[(path_index + 1) % path_atoms.len()];
                let director_sum =
                    dmdw_director_sum(atom_positions_angstrom, atom, previous, next)?;
                Ok(dot(director_sum, director_sum) / (4.0 * atom_masses[atom]))
            })
            .sum::<Result<Real, DebyeError>>()?
    };

    ensure_positive("mu_inv", inverse_reduced_mass)?;
    let reduced_mass = 1.0 / inverse_reduced_mass;
    let mut initial_vector = Array1::<Real>::zeros(atom_positions_angstrom.nrows() * 3);

    if path_atoms.len() > 1 {
        for (path_index, &atom) in path_atoms.iter().enumerate() {
            let previous = path_atoms[(path_index + path_atoms.len() - 1) % path_atoms.len()];
            let next = path_atoms[(path_index + 1) % path_atoms.len()];
            let director_sum = dmdw_director_sum(atom_positions_angstrom, atom, previous, next)?;
            let scale = 0.5 * (reduced_mass / atom_masses[atom]).sqrt();
            for component in 0..3 {
                initial_vector[component * atom_positions_angstrom.nrows() + atom] =
                    scale * director_sum[component];
            }
        }
    }

    ensure_finite_output("mu", reduced_mass)?;
    for value in initial_vector.iter().copied() {
        ensure_finite_output("qj0", value)?;
    }

    Ok(DmdwPathMotion {
        reduced_mass,
        inverse_reduced_mass,
        initial_vector,
    })
}

/// Port FEFF DMDW run-type 4's IR Lanczos seed construction.
///
/// `atom_masses` is FEFF `dym_In%am`, and `dipole_derivatives` is the type 3
/// `.dym` payload arranged as `(atom, displacement_component, dipole_component)`.
/// FEFF's active branch uses the second dipole component (`jq = 2`) squared,
/// scaled by `sqrt(mass)`, in component-major seed order before normalizing.
pub fn dmdw_ir_dipole_seed_vector(
    atom_masses: ArrayView1<'_, Real>,
    dipole_derivatives: ArrayView3<'_, Real>,
) -> Result<Array1<Real>, DebyeError> {
    validate_dmdw_ir_dipoles(atom_masses, dipole_derivatives)?;

    let atom_count = atom_masses.len();
    let mut seed = Array1::<Real>::zeros(atom_count * 3);
    for atom in 0..atom_count {
        let mass_scale = atom_masses[atom].sqrt();
        for displacement_component in 0..3 {
            let derivative = dipole_derivatives[(atom, displacement_component, 1)];
            seed[displacement_component * atom_count + atom] = mass_scale * derivative.powi(2);
        }
    }
    dmdw_normalize_seed_vector(seed.view())
}

/// Port FEFF DMDW `Paths_Init` descriptor expansion and path pruning.
///
/// `atom_positions` is the DMDW atom table as `(atom, xyz)`. It must use the
/// same distance unit as `descriptor.max_effective_length`. Descriptor
/// selectors preserve FEFF's convention: `0` expands over all atoms, while a
/// positive selector chooses that 1-based atom index. Returned atom indices are
/// zero-based for Rust callers.
pub fn dmdw_expand_path_descriptor(
    atom_positions: ArrayView2<'_, Real>,
    descriptor: &DmdwPathDescriptor,
) -> Result<Vec<DmdwExpandedPath>, DebyeError> {
    validate_dmdw_path_descriptor(atom_positions, descriptor)?;

    if descriptor.selectors.len() == 1 {
        return Ok(dmdw_expand_single_atom_descriptor(
            atom_positions.nrows(),
            descriptor.selectors[0],
        ));
    }

    let atom_count = atom_positions.nrows();
    let selector_ranges = descriptor
        .selectors
        .iter()
        .copied()
        .map(|selector| dmdw_selector_range(selector, atom_count))
        .collect::<Result<Vec<_>, DebyeError>>()?;
    let strides = dmdw_descriptor_loop_strides(
        selector_ranges.iter().map(Vec::len),
        descriptor.selectors.len(),
        atom_count,
    )?;
    let total_paths = strides.total_count;
    let mut paths = Vec::new();

    for path_index in 0..total_paths {
        let atoms = dmdw_descriptor_atoms_for_index(path_index, &selector_ranges, &strides.strides);
        if dmdw_path_has_pruned_repetition(&atoms) {
            continue;
        }

        let effective_length = dmdw_effective_path_length(atom_positions, &atoms)?;
        if effective_length <= descriptor.max_effective_length {
            paths.push(DmdwExpandedPath {
                atoms,
                effective_length,
            });
        }
    }

    Ok(paths)
}

/// Expand a sequence of FEFF DMDW path descriptors in input order.
pub fn dmdw_expand_path_descriptors(
    atom_positions: ArrayView2<'_, Real>,
    descriptors: &[DmdwPathDescriptor],
) -> Result<Vec<DmdwExpandedPath>, DebyeError> {
    let mut paths = Vec::new();
    for descriptor in descriptors {
        paths.extend(dmdw_expand_path_descriptor(atom_positions, descriptor)?);
    }
    Ok(paths)
}

/// Port FEFF DMDW `Make_DM`: mass-weight a block force-constant matrix.
///
/// `force_blocks` is FEFF `dym_In%dm_block(iAt,jAt,ip,jq)`, shaped as
/// `(atom_i, atom_j, component_i, component_j)`. The returned matrix uses
/// FEFF's component-major coordinate order: all x atom coordinates, then all
/// y, then all z. Values are scaled by FEFF's `auf2npm * npm2amups2` and
/// divided by `sqrt(m_i * m_j)`.
pub fn dmdw_mass_weighted_dynamical_matrix(
    force_blocks: ArrayView4<'_, Real>,
    atom_masses: ArrayView1<'_, Real>,
) -> Result<DmdwDynamicalMatrix, DebyeError> {
    validate_dmdw_force_blocks(force_blocks, atom_masses)?;
    let atom_count = atom_masses.len();
    let coordinate_count = atom_count * 3;
    let mut matrix = Array2::<Real>::zeros((coordinate_count, coordinate_count));

    for atom_i in 0..atom_count {
        for atom_j in 0..atom_count {
            let mass_scale =
                DMDW_DYNAMICAL_MATRIX_SCALE / (atom_masses[atom_i] * atom_masses[atom_j]).sqrt();
            for component_i in 0..3 {
                for component_j in 0..3 {
                    matrix[(
                        component_i * atom_count + atom_i,
                        component_j * atom_count + atom_j,
                    )] = mass_scale * force_blocks[(atom_i, atom_j, component_i, component_j)];
                }
            }
        }
    }

    let element_count = (coordinate_count * coordinate_count) as Real;
    let average_value = matrix.iter().map(|value| value.abs()).sum::<Real>() / element_count;
    let max_value = matrix.iter().map(|value| value.abs()).fold(0.0, Real::max);
    let average_asymmetry = matrix
        .indexed_iter()
        .map(|((row, column), value)| (value - matrix[(column, row)]).abs())
        .sum::<Real>()
        / element_count;
    let asymmetry_percent_average = percent_or_zero(average_asymmetry, average_value);
    let asymmetry_percent_max = percent_or_zero(average_asymmetry, max_value);

    for value in matrix.iter().copied() {
        ensure_finite_output("DMDW dynamical matrix", value)?;
    }
    ensure_finite_output("DMDW matrix average value", average_value)?;
    ensure_finite_output("DMDW matrix max value", max_value)?;
    ensure_finite_output("DMDW matrix average asymmetry", average_asymmetry)?;
    ensure_finite_output(
        "DMDW matrix average asymmetry percent",
        asymmetry_percent_average,
    )?;
    ensure_finite_output("DMDW matrix max asymmetry percent", asymmetry_percent_max)?;

    Ok(DmdwDynamicalMatrix {
        matrix,
        average_value,
        max_value,
        average_asymmetry,
        asymmetry_percent_average,
        asymmetry_percent_max,
    })
}

/// Port FEFF DMDW `Lanczos` tridiagonal-recursion coefficients.
///
/// FEFF applies the dynamical matrix by taking dot products with matrix
/// columns. For symmetric DMDW matrices this is equivalent to the usual
/// matrix-vector product, but this helper preserves the exact column
/// convention. The seed is normalized with the same Euclidean norm used by FEFF
/// before recursion.
pub fn dmdw_lanczos_coefficients(
    dynamical_matrix: ArrayView2<'_, Real>,
    seed: ArrayView1<'_, Real>,
    pole_count: usize,
) -> Result<DmdwLanczosCoefficients, DebyeError> {
    validate_dmdw_lanczos_inputs(dynamical_matrix, seed, pole_count)?;

    let mut alpha = Array1::<Real>::zeros(pole_count + 1);
    let mut beta = Array1::<Real>::zeros(pole_count + 2);
    let mut qj = dmdw_normalize_seed_vector(seed)?;

    let applied = dmdw_apply_dynamical_matrix(dynamical_matrix, qj.view());
    let alpha0 = dot_array_views(qj.view(), applied.view());
    alpha[0] = alpha0;
    ensure_finite_output("DMDW Lanczos alpha", alpha0)?;
    let single_pole_frequency = alpha0.sqrt() / (2.0 * std::f64::consts::PI);
    ensure_finite_output("DMDW single-pole frequency", single_pole_frequency)?;

    let mut qp = lanczos_residual(applied, qj.view(), alpha0, None);
    beta[1] = array_vector_norm(qp.view());
    qp = normalize_lanczos_vector(qp, beta[1], 1)?;

    for iteration in 1..=pole_count {
        let qm = qj;
        qj = qp;
        let applied = dmdw_apply_dynamical_matrix(dynamical_matrix, qj.view());
        let alpha_i = dot_array_views(qj.view(), applied.view());
        alpha[iteration] = alpha_i;
        ensure_finite_output("DMDW Lanczos alpha", alpha_i)?;

        qp = lanczos_residual(
            applied,
            qj.view(),
            alpha_i,
            Some((beta[iteration], qm.view())),
        );
        beta[iteration + 1] = array_vector_norm(qp.view());
        qp = normalize_lanczos_vector(qp, beta[iteration + 1], iteration + 1)?;
    }

    Ok(DmdwLanczosCoefficients {
        alpha,
        beta,
        single_pole_frequency,
    })
}

/// Port FEFF DMDW `Poly_Y('S', ...)` for Lanczos pole locations.
pub fn dmdw_lanczos_s_polynomial(
    order: usize,
    x: Real,
    alpha: ArrayView1<'_, Real>,
    beta: ArrayView1<'_, Real>,
) -> Result<Real, DebyeError> {
    validate_dmdw_lanczos_polynomial_inputs(order, x, alpha, beta)?;
    let mut previous = 1.0;
    let mut value = x - alpha[0];
    for n in 2..=order {
        let older = previous;
        previous = value;
        value = (x - alpha[n - 1]) * previous - beta[n - 1].powi(2) * older;
    }
    ensure_finite_output("DMDW Lanczos S polynomial", value)?;
    Ok(value)
}

/// Port FEFF DMDW `Poly_Y('R', ...)`.
///
/// FEFF's `'P'` branch is identical to `'R'`; callers can use this function
/// for both recurrence variants.
pub fn dmdw_lanczos_r_polynomial(
    order: usize,
    x: Real,
    alpha: ArrayView1<'_, Real>,
    beta: ArrayView1<'_, Real>,
) -> Result<Real, DebyeError> {
    validate_dmdw_lanczos_polynomial_inputs(order, x, alpha, beta)?;
    let mut previous = 0.0;
    let mut value = 1.0;
    for n in 2..=order {
        let older = previous;
        previous = value;
        value = (x - alpha[n - 1]) * previous - beta[n - 1].powi(2) * older;
    }
    ensure_finite_output("DMDW Lanczos R polynomial", value)?;
    Ok(value)
}

/// Port FEFF DMDW `PolyD_Y('S', ...)`.
pub fn dmdw_lanczos_s_polynomial_derivative(
    order: usize,
    x: Real,
    alpha: ArrayView1<'_, Real>,
    beta: ArrayView1<'_, Real>,
) -> Result<Real, DebyeError> {
    validate_dmdw_lanczos_polynomial_inputs(order, x, alpha, beta)?;
    let mut y_previous_2 = 0.0;
    let mut y_previous_1 = 1.0;
    let mut derivative_previous_1 = 0.0;
    let mut derivative = 1.0;
    for n in 2..=order {
        let y_previous_3 = y_previous_2;
        y_previous_2 = y_previous_1;
        y_previous_1 = (x - alpha[n - 2]) * y_previous_2 - beta[n - 2].powi(2) * y_previous_3;
        let derivative_previous_2 = derivative_previous_1;
        derivative_previous_1 = derivative;
        derivative = y_previous_1 + (x - alpha[n - 1]) * derivative_previous_1
            - beta[n - 1].powi(2) * derivative_previous_2;
    }
    ensure_finite_output("DMDW Lanczos S polynomial derivative", derivative)?;
    Ok(derivative)
}

/// Port FEFF DMDW `Lanczos` pole search and `wil` weight calculation.
///
/// This uses FEFF's default scan range, `[-810000, 810000]`, and 100000 scan
/// samples per requested pole. The returned angular frequencies match FEFF
/// `w_pole`; the `frequencies` field is the `DW_Out%Poles_Frq` value after
/// division by `2*pi`.
pub fn dmdw_lanczos_pole_spectrum(
    order: usize,
    alpha: ArrayView1<'_, Real>,
    beta: ArrayView1<'_, Real>,
) -> Result<DmdwLanczosPoleSpectrum, DebyeError> {
    dmdw_lanczos_pole_spectrum_with_search(
        order,
        alpha,
        beta,
        DMDW_LANCZOS_POLE_SEARCH_LIMIT,
        DMDW_LANCZOS_DEFAULT_SAMPLES_PER_POLE,
    )
}

/// Port FEFF DMDW `Lanczos` pole search with a configurable scan grid.
///
/// FEFF uses linear interpolation inside sign-changing grid intervals. This
/// helper keeps that behavior while exposing the grid for focused tests and
/// benchmarks.
pub fn dmdw_lanczos_pole_spectrum_with_search(
    order: usize,
    alpha: ArrayView1<'_, Real>,
    beta: ArrayView1<'_, Real>,
    search_limit: Real,
    samples_per_pole: usize,
) -> Result<DmdwLanczosPoleSpectrum, DebyeError> {
    validate_dmdw_lanczos_pole_search_inputs(order, alpha, beta, search_limit, samples_per_pole)?;
    let total_steps =
        order
            .checked_mul(samples_per_pole)
            .ok_or(DebyeError::DmdwLanczosPoleSearchTooLarge {
                order,
                samples_per_pole,
            })?;
    let step = 2.0 * search_limit / total_steps as Real;
    let mut roots = Vec::new();
    let mut previous_sample: Option<(Real, Real)> = None;

    for step_index in 1..=total_steps {
        let x = -search_limit + step * step_index as Real;
        let value = dmdw_lanczos_s_polynomial(order, x, alpha, beta)?;
        if let Some((previous_x, previous_value)) = previous_sample {
            if value == 0.0 {
                roots.push(x);
            } else if value * previous_value < 0.0 {
                let ratio = previous_value.abs() / (previous_value.abs() + value.abs());
                roots.push(ratio * (x - previous_x) + previous_x);
            }
        }
        previous_sample = Some((x, value));
    }

    let mut angular_frequencies = Vec::with_capacity(roots.len());
    let mut frequencies = Vec::with_capacity(roots.len());
    let mut weights = Vec::with_capacity(roots.len());
    let mut imaginary_warnings = Vec::new();

    for (pole_index, &root) in roots.iter().enumerate() {
        ensure_finite_output("DMDW Lanczos pole root", root)?;
        let angular_frequency = if root < 0.0 {
            -(-root).sqrt()
        } else {
            root.sqrt()
        };
        let frequency = angular_frequency / (2.0 * std::f64::consts::PI);
        let derivative = dmdw_lanczos_s_polynomial_derivative(order, root, alpha, beta)?;
        if derivative == 0.0 {
            return Err(DebyeError::ZeroDmdwLanczosPoleDerivative {
                pole_index,
                x: root,
            });
        }
        let weight = dmdw_lanczos_r_polynomial(order, root, alpha, beta)? / derivative;
        ensure_finite_output("DMDW Lanczos pole angular frequency", angular_frequency)?;
        ensure_finite_output("DMDW Lanczos pole frequency", frequency)?;
        ensure_finite_output("DMDW Lanczos pole weight", weight)?;

        if root < 0.0 {
            let severity = if weight >= DMDW_IMAGINARY_POLE_LARGE_WEIGHT {
                Some(DmdwImaginaryPoleSeverity::LargeWeight)
            } else if (DMDW_IMAGINARY_POLE_SMALL_WEIGHT..=DMDW_IMAGINARY_POLE_LARGE_WEIGHT)
                .contains(&weight)
            {
                Some(DmdwImaginaryPoleSeverity::SmallWeight)
            } else {
                None
            };
            if let Some(severity) = severity {
                imaginary_warnings.push(DmdwImaginaryPoleWarning {
                    pole_index,
                    squared_angular_frequency: root,
                    angular_frequency,
                    frequency,
                    weight,
                    severity,
                });
            }
        }

        angular_frequencies.push(angular_frequency);
        frequencies.push(frequency);
        weights.push(weight);
    }

    Ok(DmdwLanczosPoleSpectrum {
        expected_poles: order,
        squared_angular_frequencies: Array1::from_vec(roots),
        angular_frequencies: Array1::from_vec(angular_frequencies),
        frequencies: Array1::from_vec(frequencies),
        weights: Array1::from_vec(weights),
        imaginary_warnings,
    })
}

/// Port FEFF DMDW `Calc_DW` `sig2` accumulation for `RunTyp` 0 and 3.
///
/// `temperatures` are FEFF `Lanc_In%T`, `reduced_mass` is FEFF `mu`, and
/// `angular_frequencies`/`weights` are FEFF `w_pole`/`wil`. Imaginary and
/// zero-frequency poles are ignored exactly as in FEFF's guarded pole loop.
/// The returned values are FEFF `DW_Out%s2` or `DW_Out%u2` in square angstrom.
pub fn dmdw_debye_waller_factors_from_poles(
    temperatures: ArrayView1<'_, Real>,
    reduced_mass: Real,
    angular_frequencies: ArrayView1<'_, Real>,
    weights: ArrayView1<'_, Real>,
) -> Result<Array1<Real>, DebyeError> {
    validate_dmdw_pole_thermal_inputs(temperatures, angular_frequencies, weights)?;
    ensure_positive("DMDW reduced mass", reduced_mass)?;
    let scale = DMDW_HBARC_EV_ANGSTROM * DMDW_LIGHT_SPEED_ANGSTROM_PER_PS
        / (2.0 * reduced_mass * DMDW_AMU_EV);
    ensure_finite_output("DMDW Debye-Waller scale", scale)?;

    temperatures
        .iter()
        .copied()
        .map(|temperature| {
            let cotarg = dmdw_coth_argument_scale(temperature)?;
            let sigma = angular_frequencies
                .iter()
                .zip(weights.iter())
                .filter(|(frequency, _)| **frequency > 0.0)
                .map(|(&frequency, &weight)| {
                    let coth = 1.0 / (cotarg * frequency).tanh();
                    weight / frequency * coth
                })
                .sum::<Real>();
            let value = scale * sigma;
            ensure_finite_output("DMDW Debye-Waller factor", value)?;
            Ok(value)
        })
        .collect()
}

/// Port FEFF DMDW `Calc_DW` vibrational free-energy accumulation for `RunTyp` 1.
///
/// The returned values are FEFF `DW_Out%vfe` in J/mol. FEFF prints these values
/// as eV by dividing by `Jpmol2eV` at output time.
pub fn dmdw_vibrational_free_energy_from_poles(
    temperatures: ArrayView1<'_, Real>,
    angular_frequencies: ArrayView1<'_, Real>,
    weights: ArrayView1<'_, Real>,
) -> Result<Array1<Real>, DebyeError> {
    validate_dmdw_pole_thermal_inputs(temperatures, angular_frequencies, weights)?;

    temperatures
        .iter()
        .copied()
        .map(|temperature| {
            let cotarg = dmdw_coth_argument_scale(temperature)?;
            let entropy_sum = angular_frequencies
                .iter()
                .zip(weights.iter())
                .filter(|(frequency, _)| **frequency > 0.0)
                .map(|(&frequency, &weight)| {
                    let argument = cotarg * frequency;
                    let logarithm = if argument <= 50.0 {
                        (2.0 * argument.sinh()).ln()
                    } else {
                        argument
                    };
                    weight * logarithm
                })
                .sum::<Real>();
            let value = DMDW_GAS_CONSTANT_J_PER_MOL_K * temperature * entropy_sum;
            ensure_finite_output("DMDW vibrational free energy", value)?;
            Ok(value)
        })
        .collect()
}

/// Build FEFF DMDW's single-pole Einstein diagnostic row.
///
/// `frequency_thz` is FEFF `SPole_Frq`, and `reduced_mass` is FEFF `RedMass`
/// in AMU. The returned temperature and force constant match `Print_DW_Out`.
pub fn dmdw_single_pole_einstein_summary(
    frequency_thz: Real,
    reduced_mass: Real,
) -> Result<DmdwEinsteinSummary, DebyeError> {
    ensure_positive("DMDW Einstein frequency", frequency_thz)?;
    dmdw_einstein_summary(frequency_thz, reduced_mass)
}

/// Build FEFF DMDW `n = -2..=2` projected-DOS moment summaries.
///
/// `frequencies_thz` and `weights` are FEFF `DW_Out%Poles_Frq` and
/// `DW_Out%Poles_Wgt`. Imaginary and zero-frequency poles are excluded from
/// the moment sum, and the remaining weights are renormalized by FEFF's
/// `1 - Corr` correction.
pub fn dmdw_moment_summaries_from_poles(
    reduced_mass: Real,
    frequencies_thz: ArrayView1<'_, Real>,
    weights: ArrayView1<'_, Real>,
) -> Result<Vec<DmdwMomentSummary>, DebyeError> {
    validate_dmdw_frequency_weight_poles(frequencies_thz, weights)?;
    ensure_positive("DMDW reduced mass", reduced_mass)?;

    let removed_weight = frequencies_thz
        .iter()
        .zip(weights.iter())
        .filter(|(frequency, _)| **frequency <= 0.0)
        .map(|(_, &weight)| weight)
        .sum::<Real>();
    let normalization = 1.0 - removed_weight;
    ensure_positive("DMDW positive pole weight normalization", normalization)?;

    (-2..=2)
        .map(|order| {
            let moment = frequencies_thz
                .iter()
                .zip(weights.iter())
                .filter(|(frequency, _)| **frequency > 0.0)
                .map(|(&frequency, &weight)| frequency.powi(order) * weight)
                .sum::<Real>()
                / normalization;
            ensure_finite_output("DMDW pole moment", moment)?;

            if order == 0 {
                Ok(DmdwMomentSummary {
                    order,
                    moment_thz_power_n: moment,
                    frequency_thz: None,
                    temperature_kelvin: None,
                    effective_force_constant_n_per_m: None,
                })
            } else {
                ensure_positive("DMDW nonzero-order pole moment", moment)?;
                let frequency = moment.powf(1.0 / f64::from(order));
                let summary = dmdw_einstein_summary(frequency, reduced_mass)?;
                Ok(DmdwMomentSummary {
                    order,
                    moment_thz_power_n: moment,
                    frequency_thz: Some(summary.frequency_thz),
                    temperature_kelvin: Some(summary.temperature_kelvin),
                    effective_force_constant_n_per_m: Some(
                        summary.effective_force_constant_n_per_m,
                    ),
                })
            }
        })
        .collect()
}

/// Port FEFF DMDW `Calc_R_CM`: mass-weighted center of mass.
///
/// `atom_positions` is an `(atom, xyz)` table in any consistent distance unit,
/// and `atom_masses` is FEFF `dym_In%am`.
pub fn dmdw_center_of_mass(
    atom_positions: ArrayView2<'_, Real>,
    atom_masses: ArrayView1<'_, Real>,
) -> Result<[Real; 3], DebyeError> {
    validate_dmdw_atoms(atom_positions, atom_masses)?;
    let total_mass = atom_masses.iter().copied().sum::<Real>();
    let mut center = [0.0; 3];
    for (component, value) in center.iter_mut().enumerate() {
        *value = atom_positions
            .column(component)
            .iter()
            .zip(atom_masses.iter())
            .map(|(&coordinate, &mass)| coordinate * mass)
            .sum::<Real>()
            / total_mass;
    }
    Ok(center)
}

/// Port FEFF DMDW `Calc_ToI`: tensor of inertia about the supplied origin.
///
/// FEFF calls this after shifting coordinates to the center of mass. This
/// function preserves that explicit calling convention: pass centered
/// coordinates when a center-of-mass tensor is required.
pub fn dmdw_inertia_tensor(
    atom_positions: ArrayView2<'_, Real>,
    atom_masses: ArrayView1<'_, Real>,
) -> Result<Array2<Real>, DebyeError> {
    validate_dmdw_atoms(atom_positions, atom_masses)?;
    let mut tensor = Array2::<Real>::zeros((3, 3));
    for (atom, row) in atom_positions.rows().into_iter().enumerate() {
        let mass = atom_masses[atom];
        let x = row[0];
        let y = row[1];
        let z = row[2];
        tensor[(0, 0)] += mass * (y * y + z * z);
        tensor[(1, 1)] += mass * (x * x + z * z);
        tensor[(2, 2)] += mass * (x * x + y * y);
        tensor[(1, 0)] -= mass * y * x;
        tensor[(2, 0)] -= mass * z * x;
        tensor[(2, 1)] -= mass * z * y;
    }
    tensor[(0, 1)] = tensor[(1, 0)];
    tensor[(0, 2)] = tensor[(2, 0)];
    tensor[(1, 2)] = tensor[(2, 1)];
    Ok(tensor)
}

/// Port FEFF DMDW `Make_TrfD` rigid translation/rotation basis.
///
/// The returned `projection_modes` matrix contains the first six normalized
/// `TrfD` columns used by FEFF to project translations and rotations out of a
/// DMDW Lanczos seed. Rows use FEFF's component-major coordinate order: all x
/// atom components, then all y, then all z.
pub fn dmdw_rigid_body_projection_modes(
    atom_positions: ArrayView2<'_, Real>,
    atom_masses: ArrayView1<'_, Real>,
) -> Result<DmdwRigidBodyModes, DebyeError> {
    validate_dmdw_atoms(atom_positions, atom_masses)?;
    let atom_count = atom_masses.len();
    if atom_count < 2 {
        return Err(DebyeError::TooFewDmdwRigidBodyAtoms { atoms: atom_count });
    }

    let center_of_mass = dmdw_center_of_mass(atom_positions, atom_masses)?;
    let centered_positions = Array2::from_shape_fn((atom_count, 3), |(atom, component)| {
        atom_positions[(atom, component)] - center_of_mass[component]
    });
    let inertia_tensor = dmdw_inertia_tensor(centered_positions.view(), atom_masses)?;
    let eigensystem = real64_symmetric_eigen(inertia_tensor.view(), SymmetricTriangle::Lower)
        .map_err(|_| DebyeError::DmdwRigidBodyEigenDidNotConverge)?;
    let moments_of_inertia = eigensystem.eigenvalues().to_owned();
    let principal_axes = eigensystem.eigenvectors().to_owned();
    let mut projection_modes = Array2::<Real>::zeros((atom_count * 3, 6));

    for atom in 0..atom_count {
        let mass_root = atom_masses[atom].sqrt();
        projection_modes[(atom, 0)] = mass_root;
        projection_modes[(atom_count + atom, 1)] = mass_root;
        projection_modes[(2 * atom_count + atom, 2)] = mass_root;
    }

    for axis_index in 0..3 {
        let axis = [
            principal_axes[(0, axis_index)],
            principal_axes[(1, axis_index)],
            principal_axes[(2, axis_index)],
        ];
        for atom in 0..atom_count {
            let position = [
                centered_positions[(atom, 0)],
                centered_positions[(atom, 1)],
                centered_positions[(atom, 2)],
            ];
            let rotation = cross(axis, position);
            let mass_root = atom_masses[atom].sqrt();
            for component in 0..3 {
                projection_modes[(component * atom_count + atom, 3 + axis_index)] =
                    rotation[component] * mass_root;
            }
        }
    }

    normalize_dmdw_projection_modes(&mut projection_modes)?;

    Ok(DmdwRigidBodyModes {
        center_of_mass,
        centered_positions,
        inertia_tensor,
        moments_of_inertia,
        principal_axes,
        projection_modes,
    })
}

/// Project a DMDW seed vector out of rigid-body modes and normalize it.
///
/// This ports the FEFF `qj0 = qj0 - sum(qj0*TrfD(:,i))*TrfD(:,i)` loop used
/// before Lanczos recursion. `projection_modes` uses FEFF's `TrfD` orientation:
/// rows are seed-vector components and columns are modes to remove. The modes
/// are expected to be pre-normalized, matching `Make_TrfD`.
pub fn dmdw_project_seed_vector(
    seed: ArrayView1<'_, Real>,
    projection_modes: ArrayView2<'_, Real>,
) -> Result<Array1<Real>, DebyeError> {
    validate_dmdw_seed_projection(seed, projection_modes)?;
    let mut projected = seed.to_owned();
    for mode in projection_modes.columns() {
        let projection = projected
            .iter()
            .zip(mode.iter())
            .map(|(&seed_value, &mode_value)| seed_value * mode_value)
            .sum::<Real>();
        for (value, &mode_value) in projected.iter_mut().zip(mode.iter()) {
            *value -= projection * mode_value;
        }
    }
    dmdw_normalize_seed_vector(projected.view())
}

/// Normalize a DMDW Lanczos seed vector with FEFF's Euclidean norm.
pub fn dmdw_normalize_seed_vector(seed: ArrayView1<'_, Real>) -> Result<Array1<Real>, DebyeError> {
    validate_dmdw_seed(seed)?;
    let norm = seed.iter().map(|value| value * value).sum::<Real>().sqrt();
    ensure_finite_output("DMDW seed norm", norm)?;
    if norm == 0.0 {
        return Err(DebyeError::ZeroDmdwSeedNorm);
    }
    Ok(Array1::from_iter(seed.iter().map(|value| value / norm)))
}

type CorrelationFn =
    fn(Real, Real, Real, usize, usize, Real) -> Result<DebyeCorrelation, DebyeError>;

struct DmdwDescriptorLoopStrides {
    strides: Vec<usize>,
    total_count: usize,
}

fn validate_dmdw_atom_positions(atom_positions: ArrayView2<'_, Real>) -> Result<(), DebyeError> {
    if atom_positions.ncols() != 3 {
        return Err(DebyeError::InvalidDmdwAtomShape {
            rows: atom_positions.nrows(),
            columns: atom_positions.ncols(),
        });
    }
    if atom_positions.nrows() == 0 {
        return Err(DebyeError::EmptyDmdwAtomTable);
    }
    for value in atom_positions.iter().copied() {
        ensure_finite("DMDW atom coordinate", value)?;
    }
    Ok(())
}

fn validate_dmdw_atoms(
    atom_positions: ArrayView2<'_, Real>,
    atom_masses: ArrayView1<'_, Real>,
) -> Result<(), DebyeError> {
    validate_dmdw_atom_positions(atom_positions)?;
    if atom_positions.nrows() != atom_masses.len() {
        return Err(DebyeError::InvalidDmdwMassCount {
            positions: atom_positions.nrows(),
            masses: atom_masses.len(),
        });
    }
    for mass in atom_masses.iter().copied() {
        ensure_positive("DMDW atom mass", mass)?;
    }
    Ok(())
}

fn validate_dmdw_path_descriptor(
    atom_positions: ArrayView2<'_, Real>,
    descriptor: &DmdwPathDescriptor,
) -> Result<(), DebyeError> {
    validate_dmdw_atom_positions(atom_positions)?;
    if descriptor.selectors.is_empty() {
        return Err(DebyeError::EmptyDmdwPath);
    }
    ensure_nonnegative(
        "DMDW path descriptor maximum effective length",
        descriptor.max_effective_length,
    )?;
    for &selector in &descriptor.selectors {
        validate_dmdw_path_selector(selector, atom_positions.nrows())?;
    }
    Ok(())
}

fn validate_dmdw_path_selector(selector: i32, atom_count: usize) -> Result<(), DebyeError> {
    if selector < 0 || selector as usize > atom_count {
        Err(DebyeError::InvalidDmdwPathSelector {
            selector,
            atom_count,
        })
    } else {
        Ok(())
    }
}

fn dmdw_expand_single_atom_descriptor(atom_count: usize, selector: i32) -> Vec<DmdwExpandedPath> {
    let atoms = if selector == 0 {
        (0..atom_count).collect::<Vec<_>>()
    } else {
        vec![selector as usize - 1]
    };
    atoms
        .into_iter()
        .map(|atom| DmdwExpandedPath {
            atoms: vec![atom],
            effective_length: 0.0,
        })
        .collect()
}

fn dmdw_selector_range(selector: i32, atom_count: usize) -> Result<Vec<usize>, DebyeError> {
    validate_dmdw_path_selector(selector, atom_count)?;
    if selector == 0 {
        Ok((0..atom_count).collect())
    } else {
        Ok(vec![selector as usize - 1])
    }
}

fn dmdw_descriptor_loop_strides(
    lengths: impl Iterator<Item = usize>,
    selector_count: usize,
    atom_count: usize,
) -> Result<DmdwDescriptorLoopStrides, DebyeError> {
    let lengths = lengths.collect::<Vec<_>>();
    let mut strides = vec![1; lengths.len()];
    let mut total_count = 1usize;
    for index in (0..lengths.len()).rev() {
        strides[index] = total_count;
        total_count = total_count.checked_mul(lengths[index]).ok_or(
            DebyeError::DmdwPathExpansionTooLarge {
                selectors: selector_count,
                atom_count,
            },
        )?;
    }
    Ok(DmdwDescriptorLoopStrides {
        strides,
        total_count,
    })
}

fn dmdw_descriptor_atoms_for_index(
    path_index: usize,
    selector_ranges: &[Vec<usize>],
    strides: &[usize],
) -> Vec<usize> {
    let mut remainder = path_index;
    selector_ranges
        .iter()
        .zip(strides.iter())
        .map(|(range, &stride)| {
            let range_index = remainder / stride;
            remainder %= stride;
            range[range_index]
        })
        .collect()
}

fn dmdw_path_has_pruned_repetition(atoms: &[usize]) -> bool {
    let has_consecutive_repeat = atoms.windows(2).any(|pair| pair[0] == pair[1]);
    let closes_on_same_atom = match (atoms.first(), atoms.last()) {
        (Some(first), Some(last)) => first == last,
        _ => false,
    };
    has_consecutive_repeat || closes_on_same_atom
}

fn dmdw_effective_path_length(
    atom_positions: ArrayView2<'_, Real>,
    atoms: &[usize],
) -> Result<Real, DebyeError> {
    if atoms.len() <= 1 {
        return Ok(0.0);
    }

    let segment_length = atoms
        .windows(2)
        .map(|pair| dmdw_atom_distance(atom_positions, pair[0], pair[1]))
        .sum::<Result<Real, DebyeError>>()?;
    let closing_length = dmdw_atom_distance(atom_positions, atoms[0], atoms[atoms.len() - 1])?;
    let effective_length = 0.5 * (segment_length + closing_length);
    ensure_finite_output("DMDW effective path length", effective_length)?;
    Ok(effective_length)
}

fn dmdw_atom_distance(
    atom_positions: ArrayView2<'_, Real>,
    left: usize,
    right: usize,
) -> Result<Real, DebyeError> {
    let squared = (0..3)
        .map(|component| {
            let difference = atom_positions[(left, component)] - atom_positions[(right, component)];
            difference * difference
        })
        .sum::<Real>();
    let distance = squared.sqrt();
    ensure_finite_output("DMDW atom distance", distance)?;
    Ok(distance)
}

fn validate_dmdw_path_input(
    atom_positions: ArrayView2<'_, Real>,
    atom_masses: ArrayView1<'_, Real>,
    path_atoms: &[usize],
) -> Result<(), DebyeError> {
    validate_dmdw_atoms(atom_positions, atom_masses)?;
    if path_atoms.is_empty() {
        return Err(DebyeError::EmptyDmdwPath);
    }
    for &index in path_atoms {
        if index >= atom_positions.nrows() {
            return Err(DebyeError::InvalidDmdwPathAtomIndex {
                index,
                atom_count: atom_positions.nrows(),
            });
        }
    }
    Ok(())
}

fn validate_dmdw_seed_projection(
    seed: ArrayView1<'_, Real>,
    projection_modes: ArrayView2<'_, Real>,
) -> Result<(), DebyeError> {
    validate_dmdw_seed(seed)?;
    if projection_modes.nrows() != seed.len() {
        return Err(DebyeError::InvalidDmdwProjectionShape {
            seed_len: seed.len(),
            rows: projection_modes.nrows(),
            columns: projection_modes.ncols(),
        });
    }
    for value in projection_modes.iter().copied() {
        ensure_finite("DMDW projection mode", value)?;
    }
    Ok(())
}

fn normalize_dmdw_projection_modes(modes: &mut Array2<Real>) -> Result<(), DebyeError> {
    for mode_index in 0..modes.ncols() {
        let norm = modes
            .column(mode_index)
            .iter()
            .map(|value| value * value)
            .sum::<Real>()
            .sqrt();
        ensure_finite_output("DMDW projection mode norm", norm)?;
        if norm == 0.0 {
            return Err(DebyeError::ZeroDmdwProjectionModeNorm { mode: mode_index });
        }
        for value in modes.column_mut(mode_index) {
            *value /= norm;
        }
    }
    Ok(())
}

fn validate_dmdw_force_blocks(
    force_blocks: ArrayView4<'_, Real>,
    atom_masses: ArrayView1<'_, Real>,
) -> Result<(), DebyeError> {
    let shape = force_blocks.shape();
    if atom_masses.is_empty() {
        return Err(DebyeError::EmptyDmdwAtomTable);
    }
    if shape[0] != atom_masses.len()
        || shape[1] != atom_masses.len()
        || shape[2] != 3
        || shape[3] != 3
    {
        return Err(DebyeError::InvalidDmdwBlockShape {
            atoms_i: shape[0],
            atoms_j: shape[1],
            components_i: shape[2],
            components_j: shape[3],
            masses: atom_masses.len(),
        });
    }
    for mass in atom_masses.iter().copied() {
        ensure_positive("DMDW atom mass", mass)?;
    }
    for value in force_blocks.iter().copied() {
        ensure_finite("DMDW force block", value)?;
    }
    Ok(())
}

fn validate_dmdw_ir_dipoles(
    atom_masses: ArrayView1<'_, Real>,
    dipole_derivatives: ArrayView3<'_, Real>,
) -> Result<(), DebyeError> {
    if atom_masses.is_empty() {
        return Err(DebyeError::EmptyDmdwAtomTable);
    }
    if dipole_derivatives.shape() != [atom_masses.len(), 3, 3] {
        let shape = dipole_derivatives.shape();
        return Err(DebyeError::InvalidDmdwDipoleDerivativeShape {
            atoms: shape[0],
            displacements: shape[1],
            dipoles: shape[2],
            masses: atom_masses.len(),
        });
    }
    for mass in atom_masses.iter().copied() {
        ensure_positive("DMDW atom mass", mass)?;
    }
    for value in dipole_derivatives.iter().copied() {
        ensure_finite("DMDW IR dipole derivative", value)?;
    }
    Ok(())
}

fn validate_dmdw_lanczos_inputs(
    dynamical_matrix: ArrayView2<'_, Real>,
    seed: ArrayView1<'_, Real>,
    pole_count: usize,
) -> Result<(), DebyeError> {
    if pole_count == 0 {
        return Err(DebyeError::NonPositive {
            name: "DMDW Lanczos pole count",
            value: pole_count as Real,
        });
    }
    if dynamical_matrix.nrows() == 0
        || dynamical_matrix.nrows() != dynamical_matrix.ncols()
        || dynamical_matrix.nrows() != seed.len()
    {
        return Err(DebyeError::InvalidDmdwLanczosShape {
            rows: dynamical_matrix.nrows(),
            columns: dynamical_matrix.ncols(),
            seed_len: seed.len(),
        });
    }
    for value in dynamical_matrix.iter().copied() {
        ensure_finite("DMDW Lanczos matrix", value)?;
    }
    validate_dmdw_seed(seed)?;
    Ok(())
}

fn validate_dmdw_lanczos_polynomial_inputs(
    order: usize,
    x: Real,
    alpha: ArrayView1<'_, Real>,
    beta: ArrayView1<'_, Real>,
) -> Result<(), DebyeError> {
    if order == 0 {
        return Err(DebyeError::NonPositive {
            name: "DMDW Lanczos polynomial order",
            value: order as Real,
        });
    }
    if alpha.len() < order || beta.len() < order {
        return Err(DebyeError::InvalidDmdwLanczosPolynomialShape {
            order,
            alpha_len: alpha.len(),
            beta_len: beta.len(),
        });
    }
    ensure_finite("DMDW Lanczos polynomial x", x)?;
    for value in alpha.iter().take(order).copied() {
        ensure_finite("DMDW Lanczos alpha", value)?;
    }
    for value in beta.iter().take(order).copied() {
        ensure_finite("DMDW Lanczos beta", value)?;
    }
    Ok(())
}

fn validate_dmdw_lanczos_pole_search_inputs(
    order: usize,
    alpha: ArrayView1<'_, Real>,
    beta: ArrayView1<'_, Real>,
    search_limit: Real,
    samples_per_pole: usize,
) -> Result<(), DebyeError> {
    validate_dmdw_lanczos_polynomial_inputs(order, 0.0, alpha, beta)?;
    ensure_positive("DMDW Lanczos pole search limit", search_limit)?;
    if samples_per_pole == 0 {
        return Err(DebyeError::NonPositive {
            name: "DMDW Lanczos pole samples per pole",
            value: samples_per_pole as Real,
        });
    }
    Ok(())
}

fn validate_dmdw_pole_thermal_inputs(
    temperatures: ArrayView1<'_, Real>,
    angular_frequencies: ArrayView1<'_, Real>,
    weights: ArrayView1<'_, Real>,
) -> Result<(), DebyeError> {
    if temperatures.is_empty() {
        return Err(DebyeError::EmptyDmdwTemperatureTable);
    }
    validate_dmdw_frequency_weight_poles(angular_frequencies, weights)?;
    for temperature in temperatures.iter().copied() {
        ensure_positive("DMDW temperature", temperature)?;
    }
    Ok(())
}

fn validate_dmdw_frequency_weight_poles(
    frequencies: ArrayView1<'_, Real>,
    weights: ArrayView1<'_, Real>,
) -> Result<(), DebyeError> {
    if frequencies.len() != weights.len() {
        return Err(DebyeError::InvalidDmdwPoleTableShape {
            frequencies: frequencies.len(),
            weights: weights.len(),
        });
    }
    if frequencies.is_empty() {
        return Err(DebyeError::EmptyDmdwPoleTable);
    }
    for frequency in frequencies.iter().copied() {
        ensure_finite("DMDW pole frequency", frequency)?;
    }
    for weight in weights.iter().copied() {
        ensure_finite("DMDW pole weight", weight)?;
    }
    Ok(())
}

fn dmdw_einstein_summary(
    frequency_thz: Real,
    reduced_mass: Real,
) -> Result<DmdwEinsteinSummary, DebyeError> {
    ensure_positive("DMDW reduced mass", reduced_mass)?;
    ensure_positive("DMDW Einstein frequency", frequency_thz)?;
    let temperature_kelvin = frequency_thz * DMDW_THZ_TO_KELVIN;
    let effective_force_constant_n_per_m = reduced_mass
        * (2.0 * std::f64::consts::PI * frequency_thz).powi(2)
        * DMDW_AMU_THZ2_TO_NEWTON_PER_METER;
    ensure_finite_output("DMDW Einstein temperature", temperature_kelvin)?;
    ensure_finite_output(
        "DMDW Einstein effective force constant",
        effective_force_constant_n_per_m,
    )?;
    Ok(DmdwEinsteinSummary {
        frequency_thz,
        temperature_kelvin,
        effective_force_constant_n_per_m,
    })
}

fn dmdw_coth_argument_scale(temperature: Real) -> Result<Real, DebyeError> {
    ensure_positive("DMDW temperature", temperature)?;
    let beta = 1.0 / (DMDW_BOLTZMANN_EV_PER_K * temperature);
    let scale = 0.5 * DMDW_HBAR_EV_PS * beta;
    ensure_finite_output("DMDW coth argument scale", scale)?;
    Ok(scale)
}

fn validate_dmdw_seed(seed: ArrayView1<'_, Real>) -> Result<(), DebyeError> {
    if seed.is_empty() {
        return Err(DebyeError::EmptyDmdwSeed);
    }
    for value in seed.iter().copied() {
        ensure_finite("DMDW seed", value)?;
    }
    Ok(())
}

fn dmdw_apply_dynamical_matrix(
    matrix: ArrayView2<'_, Real>,
    vector: ArrayView1<'_, Real>,
) -> Array1<Real> {
    Array1::from_iter(matrix.columns().into_iter().map(|column| {
        column
            .iter()
            .zip(vector.iter())
            .map(|(&matrix_value, &vector_value)| matrix_value * vector_value)
            .sum()
    }))
}

fn lanczos_residual(
    mut applied: Array1<Real>,
    current: ArrayView1<'_, Real>,
    alpha: Real,
    previous: Option<(Real, ArrayView1<'_, Real>)>,
) -> Array1<Real> {
    for (value, &current_value) in applied.iter_mut().zip(current.iter()) {
        *value -= alpha * current_value;
    }
    if let Some((beta, previous_vector)) = previous {
        for (value, &previous_value) in applied.iter_mut().zip(previous_vector.iter()) {
            *value -= beta * previous_value;
        }
    }
    applied
}

fn normalize_lanczos_vector(
    mut vector: Array1<Real>,
    norm: Real,
    iteration: usize,
) -> Result<Array1<Real>, DebyeError> {
    ensure_finite_output("DMDW Lanczos beta", norm)?;
    if norm == 0.0 {
        return Err(DebyeError::DmdwLanczosBreakdown { iteration });
    }
    vector.mapv_inplace(|value| value / norm);
    Ok(vector)
}

fn dot_array_views(lhs: ArrayView1<'_, Real>, rhs: ArrayView1<'_, Real>) -> Real {
    lhs.iter()
        .zip(rhs.iter())
        .map(|(&lhs_value, &rhs_value)| lhs_value * rhs_value)
        .sum()
}

fn array_vector_norm(vector: ArrayView1<'_, Real>) -> Real {
    vector
        .iter()
        .map(|value| value * value)
        .sum::<Real>()
        .sqrt()
}

fn percent_or_zero(numerator: Real, denominator: Real) -> Real {
    if denominator == 0.0 {
        0.0
    } else {
        numerator / denominator * 100.0
    }
}

fn dmdw_director_sum(
    atom_positions: ArrayView2<'_, Real>,
    atom: usize,
    previous: usize,
    next: usize,
) -> Result<[Real; 3], DebyeError> {
    let previous_vector = dmdw_director_cosine(atom_positions, atom, previous)?;
    let next_vector = dmdw_director_cosine(atom_positions, atom, next)?;
    Ok([
        previous_vector[0] + next_vector[0],
        previous_vector[1] + next_vector[1],
        previous_vector[2] + next_vector[2],
    ])
}

fn dmdw_director_cosine(
    atom_positions: ArrayView2<'_, Real>,
    atom: usize,
    neighbor: usize,
) -> Result<[Real; 3], DebyeError> {
    if atom == neighbor {
        return Ok([0.0; 3]);
    }
    let vector = [
        atom_positions[(atom, 0)] - atom_positions[(neighbor, 0)],
        atom_positions[(atom, 1)] - atom_positions[(neighbor, 1)],
        atom_positions[(atom, 2)] - atom_positions[(neighbor, 2)],
    ];
    let norm = vector_norm(vector);
    if norm == 0.0 {
        return Err(DebyeError::ZeroLengthDmdwAtomPair {
            first: atom,
            second: neighbor,
        });
    }
    Ok([vector[0] / norm, vector[1] / norm, vector[2] / norm])
}

struct DebyeWallerInput<'a> {
    temperature: Real,
    debye_temperature: Real,
    average_wigner_seitz_radius_bohr: Real,
    positions_angstrom: ArrayView2<'a, Real>,
    atomic_numbers: &'a [usize],
    correlation: CorrelationFn,
}

fn debye_waller_factor(input: DebyeWallerInput<'_>) -> Result<Real, DebyeError> {
    let DebyeWallerInput {
        temperature,
        debye_temperature,
        average_wigner_seitz_radius_bohr,
        positions_angstrom,
        atomic_numbers,
        correlation,
    } = input;
    validate_path(positions_angstrom, atomic_numbers)?;
    ensure_nonnegative("tk", temperature)?;
    ensure_positive("thetad", debye_temperature)?;
    ensure_positive("rsavg", average_wigner_seitz_radius_bohr)?;

    let nleg = positions_angstrom.nrows() - 1;
    let mut total = 0.0;
    for il in 1..=nleg {
        for jl in il..=nleg {
            let cij = correlation_between_path_atoms(
                positions_angstrom,
                atomic_numbers,
                il,
                jl,
                DebyePathCorrelation {
                    debye_temperature,
                    temperature,
                    average_wigner_seitz_radius_bohr,
                    correlation,
                },
            )?;
            let cimjm = correlation_between_path_atoms(
                positions_angstrom,
                atomic_numbers,
                il - 1,
                jl - 1,
                DebyePathCorrelation {
                    debye_temperature,
                    temperature,
                    average_wigner_seitz_radius_bohr,
                    correlation,
                },
            )?;
            let cijm = correlation_between_path_atoms(
                positions_angstrom,
                atomic_numbers,
                il,
                jl - 1,
                DebyePathCorrelation {
                    debye_temperature,
                    temperature,
                    average_wigner_seitz_radius_bohr,
                    correlation,
                },
            )?;
            let cimj = correlation_between_path_atoms(
                positions_angstrom,
                atomic_numbers,
                il - 1,
                jl,
                DebyePathCorrelation {
                    debye_temperature,
                    temperature,
                    average_wigner_seitz_radius_bohr,
                    correlation,
                },
            )?;
            let first_leg = path_segment(positions_angstrom, il)?;
            let second_leg = path_segment(positions_angstrom, jl)?;
            let first_norm = vector_norm(first_leg);
            let second_norm = vector_norm(second_leg);
            let leg_projection = dot(first_leg, second_leg) / (first_norm * second_norm);
            let mut contribution = cij + cimjm - cijm - cimj;
            if jl != il {
                contribution *= 2.0;
            }
            total += contribution * leg_projection;
        }
    }

    let value = total / 4.0;
    ensure_finite_output("sig2", value)?;
    Ok(value)
}

#[derive(Clone, Copy)]
struct DebyePathCorrelation {
    debye_temperature: Real,
    temperature: Real,
    average_wigner_seitz_radius_bohr: Real,
    correlation: CorrelationFn,
}

fn validate_path(
    positions: ArrayView2<'_, Real>,
    atomic_numbers: &[usize],
) -> Result<(), DebyeError> {
    if positions.nrows() < 2 || positions.ncols() != 3 {
        return Err(DebyeError::InvalidPathShape {
            rows: positions.nrows(),
            columns: positions.ncols(),
        });
    }
    if positions.nrows() != atomic_numbers.len() {
        return Err(DebyeError::InvalidAtomicNumberCount {
            positions: positions.nrows(),
            atomic_numbers: atomic_numbers.len(),
        });
    }
    for value in positions.iter().copied() {
        ensure_finite("path coordinate", value)?;
    }
    for &atomic_number in atomic_numbers {
        atomic_weight(atomic_number)?;
    }
    Ok(())
}

fn correlation_between_path_atoms(
    positions: ArrayView2<'_, Real>,
    atomic_numbers: &[usize],
    first: usize,
    second: usize,
    path_correlation: DebyePathCorrelation,
) -> Result<Real, DebyeError> {
    let distance = path_distance(positions, first, second);
    Ok((path_correlation.correlation)(
        distance,
        path_correlation.debye_temperature,
        path_correlation.temperature,
        atomic_numbers[first],
        atomic_numbers[second],
        path_correlation.average_wigner_seitz_radius_bohr,
    )?
    .value)
}

fn path_distance(positions: ArrayView2<'_, Real>, first: usize, second: usize) -> Real {
    vector_norm([
        positions[(first, 0)] - positions[(second, 0)],
        positions[(first, 1)] - positions[(second, 1)],
        positions[(first, 2)] - positions[(second, 2)],
    ])
}

fn path_segment(positions: ArrayView2<'_, Real>, leg: usize) -> Result<[Real; 3], DebyeError> {
    let segment = [
        positions[(leg, 0)] - positions[(leg - 1, 0)],
        positions[(leg, 1)] - positions[(leg - 1, 1)],
        positions[(leg, 2)] - positions[(leg - 1, 2)],
    ];
    if vector_norm(segment) == 0.0 {
        Err(DebyeError::ZeroLengthPathLeg { leg })
    } else {
        Ok(segment)
    }
}

fn vector_norm(vector: [Real; 3]) -> Real {
    dot(vector, vector).sqrt()
}

fn dot(left: [Real; 3], right: [Real; 3]) -> Real {
    left.iter()
        .zip(right.iter())
        .map(|(&left, &right)| left * right)
        .sum()
}

fn cross(left: [Real; 3], right: [Real; 3]) -> [Real; 3] {
    [
        left[1] * right[2] - left[2] * right[1],
        left[2] * right[0] - left[0] * right[2],
        left[0] * right[1] - left[1] * right[0],
    ]
}

struct DebyeCorrelationInput {
    routine: &'static str,
    distance_angstrom: Real,
    debye_temperature: Real,
    temperature: Real,
    first_atomic_number: usize,
    second_atomic_number: usize,
    average_wigner_seitz_radius_bohr: Real,
    integrand: fn(Real, Real, Real) -> Real,
}

fn debye_correlation(input: DebyeCorrelationInput) -> Result<DebyeCorrelation, DebyeError> {
    let DebyeCorrelationInput {
        routine,
        distance_angstrom,
        debye_temperature,
        temperature,
        first_atomic_number,
        second_atomic_number,
        average_wigner_seitz_radius_bohr,
        integrand,
    } = input;

    ensure_nonnegative("rij", distance_angstrom)?;
    ensure_positive("thetad", debye_temperature)?;
    ensure_nonnegative("tk", temperature)?;
    ensure_positive("rsavg", average_wigner_seitz_radius_bohr)?;
    let first_mass = atomic_weight(first_atomic_number)?;
    let second_mass = atomic_weight(second_atomic_number)?;

    let y_inverse = temperature / debye_temperature;
    let debye_wave_number = (9.0 * std::f64::consts::PI / 2.0).powf(1.0 / 3.0)
        / (average_wigner_seitz_radius_bohr * BOHR_ANGSTROM);
    let x = debye_wave_number * distance_angstrom;
    let factor = ((3.0_f32 / 2.0_f32) as Real) * DEBYE_CORRELATION_FACTOR
        / (debye_temperature * (first_mass * second_mass).sqrt());
    let (integral, estimated_error, iterations) =
        integrate_debye_romberg(routine, |w| integrand(w, x, y_inverse))?;
    let value = factor * integral;
    ensure_finite_output(routine, value)?;
    Ok(DebyeCorrelation {
        value,
        estimated_error,
        iterations,
    })
}

fn quantum_debye_integrand(w: Real, x: Real, y_inverse: Real) -> Real {
    if w < 1.0e-20 {
        return 2.0 * y_inverse;
    }
    let factor = if x > 0.0 { (w * x).sin() / x } else { w };
    let exp_term = (-w / y_inverse).exp();
    factor * (1.0 + exp_term) / (1.0 - exp_term)
}

fn classical_debye_integrand(w: Real, x: Real, y_inverse: Real) -> Real {
    if w < 1.0e-20 {
        return 2.0 * y_inverse;
    }
    let factor = if x > 0.0 { (w * x).sin() / x } else { w };
    factor * 2.0 * y_inverse / w
}

fn integrate_debye_romberg(
    routine: &'static str,
    integrand: impl Fn(Real) -> Real,
) -> Result<(Real, Real, usize), DebyeError> {
    let mut intervals = 1;
    let mut delta = 1.0;
    let mut previous_trapezoid = (integrand(0.0) + integrand(1.0)) / 2.0;
    let mut previous_extrapolated = previous_trapezoid;
    let mut estimated_error = Real::INFINITY;

    for iteration in 1..=DEBYE_ROMBERG_MAX_ITERATIONS {
        delta /= 2.0;
        let midpoint_sum = (1..=intervals)
            .map(|index| {
                let z = (2 * index - 1) as Real * delta;
                integrand(z)
            })
            .sum::<Real>();
        let trapezoid = previous_trapezoid / 2.0 + delta * midpoint_sum;
        let extrapolated = (4.0 * trapezoid - previous_trapezoid) / 3.0;
        estimated_error = relative_error(extrapolated, previous_extrapolated);
        if estimated_error < DEBYE_ROMBERG_TOLERANCE {
            return Ok((extrapolated, estimated_error, iteration));
        }
        previous_trapezoid = trapezoid;
        previous_extrapolated = extrapolated;
        intervals *= 2;
    }

    Err(DebyeError::IntegrationDidNotConverge {
        routine,
        iterations: DEBYE_ROMBERG_MAX_ITERATIONS,
        estimated_error,
    })
}

fn relative_error(current: Real, previous: Real) -> Real {
    if current == 0.0 {
        if previous == 0.0 { 0.0 } else { Real::INFINITY }
    } else {
        ((current - previous) / current).abs()
    }
}

fn atomic_weight(atomic_number: usize) -> Result<Real, DebyeError> {
    feff_atomic_weight(atomic_number)
        .map_err(|_| DebyeError::InvalidAtomicNumber { z: atomic_number })
}

fn ensure_nonnegative(name: &'static str, value: Real) -> Result<(), DebyeError> {
    ensure_finite(name, value)?;
    if value >= 0.0 {
        Ok(())
    } else {
        Err(DebyeError::Negative { name, value })
    }
}

fn to_feff_real(value: Real) -> Real {
    (value as f32) as Real
}

fn ensure_finite(name: &'static str, value: Real) -> Result<(), DebyeError> {
    if value.is_finite() {
        Ok(())
    } else {
        Err(DebyeError::NonFinite { name, value })
    }
}

fn ensure_positive(name: &'static str, value: Real) -> Result<(), DebyeError> {
    ensure_finite(name, value)?;
    if value > 0.0 {
        Ok(())
    } else {
        Err(DebyeError::NonPositive { name, value })
    }
}

fn ensure_finite_output(name: &'static str, value: Real) -> Result<(), DebyeError> {
    if value.is_finite() {
        Ok(())
    } else {
        Err(DebyeError::NonFiniteOutput { name, value })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn morse_einstein_cumulants_match_feff_reference() -> Result<(), DebyeError> {
        let first = morse_einstein_cumulants(0.003, 300.0, 1.0e-5, 400.0)?;
        assert_close(first.first, 1.190_648_842_682_321_3e-8);
        assert_close(first.third, 5.526_344_214_607_83e-11);
        assert_close(first.scaled_thermal_expansion, 5.291_772_49e-6);

        let second = morse_einstein_cumulants(0.0075, 800.0, 2.5e-5, 250.0)?;
        assert_close(second.first, 7.441_554_786_684_262e-8);
        assert_close(second.third, 1.098_357_016_560_439_2e-9);
        assert_close(second.scaled_thermal_expansion, 1.322_943_122_5e-5);

        let negative_alpha = morse_einstein_cumulants(0.0012, 120.0, -7.0e-6, 350.0)?;
        assert_close(negative_alpha.first, -3.333_816_545_419_16e-9);
        assert_close(negative_alpha.third, -3.706_146_208_663_239e-12);
        assert_close(negative_alpha.scaled_thermal_expansion, -3.704_240_743e-6);
        Ok(())
    }

    #[test]
    fn morse_einstein_cumulants_reject_invalid_inputs() {
        assert!(matches!(
            morse_einstein_cumulants(0.0, 300.0, 1.0e-5, 400.0),
            Err(DebyeError::NonPositive { name: "sig2", .. })
        ));
        assert!(matches!(
            morse_einstein_cumulants(0.003, Real::NAN, 1.0e-5, 400.0),
            Err(DebyeError::NonFinite { name: "tk", .. })
        ));
        assert!(matches!(
            morse_einstein_cumulants(0.003, 300.0, Real::INFINITY, 400.0),
            Err(DebyeError::NonFinite { name: "alphat", .. })
        ));
        assert!(matches!(
            morse_einstein_cumulants(0.003, 300.0, 1.0e-5, -1.0),
            Err(DebyeError::NonPositive { name: "thetae", .. })
        ));
    }

    #[test]
    fn thermal_expansion_cumulants_match_feff_reference() -> Result<(), DebyeError> {
        let copper = thermal_expansion_cumulants(29, 29, 0.003, 1.0e-5, 400.0, 2.55)?;
        assert_relative_close(copper.first, -3.563_418_839_026_406e17);
        assert_relative_close(copper.third, -2.138_051_303_415_843_5e15);

        let copper_oxygen = thermal_expansion_cumulants(29, 8, 0.0042, 1.8e-5, 650.0, 1.91)?;
        assert_relative_close(copper_oxygen.first, -7.144_230_125_822_932e17);
        assert_relative_close(copper_oxygen.third, -6.001_153_305_691_263e15);

        let carbon_hydrogen = thermal_expansion_cumulants(6, 1, 0.0015, -6.0e-6, 300.0, 1.09)?;
        assert_relative_close(carbon_hydrogen.first, 7.521_958_969_413_031e14);
        assert_relative_close(carbon_hydrogen.third, 2.256_587_690_823_909e12);
        Ok(())
    }

    #[test]
    fn thermal_expansion_cumulants_reject_invalid_inputs() {
        assert!(matches!(
            thermal_expansion_cumulants(0, 29, 0.003, 1.0e-5, 400.0, 2.55),
            Err(DebyeError::InvalidAtomicNumber { z: 0 })
        ));
        assert!(matches!(
            thermal_expansion_cumulants(29, 140, 0.003, 1.0e-5, 400.0, 2.55),
            Err(DebyeError::InvalidAtomicNumber { z: 140 })
        ));
        assert!(matches!(
            thermal_expansion_cumulants(29, 29, -0.003, 1.0e-5, 400.0, 2.55),
            Err(DebyeError::NonPositive { name: "sig2", .. })
        ));
        assert!(matches!(
            thermal_expansion_cumulants(29, 29, 0.003, 1.0e-5, 0.0, 2.55),
            Err(DebyeError::NonPositive { name: "thetad", .. })
        ));
        assert!(matches!(
            thermal_expansion_cumulants(29, 29, 0.003, 1.0e-5, 400.0, Real::NAN),
            Err(DebyeError::NonFinite { name: "reff", .. })
        ));
    }

    #[test]
    fn debye_correlations_match_feff_reference() -> Result<(), DebyeError> {
        let zero = quantum_debye_correlation(0.0, 400.0, 300.0, 29, 29, 2.7)?;
        assert_close(zero.value, 4.501_999_849_393_054e-3);
        let copper = quantum_debye_correlation(2.55, 400.0, 300.0, 29, 29, 2.7)?;
        assert_close(copper.value, 1.691_640_883_386_128e-3);
        let copper_oxygen = quantum_debye_correlation(1.91, 650.0, 120.0, 29, 8, 2.3)?;
        assert_close(copper_oxygen.value, 7.447_746_368_694_431e-4);

        let classical_zero = classical_debye_correlation(0.0, 400.0, 300.0, 29, 29, 2.7)?;
        assert_close(classical_zero.value, 4.293_628_582_101_32e-3);
        let classical_copper = classical_debye_correlation(2.55, 400.0, 300.0, 29, 29, 2.7)?;
        assert_close(classical_copper.value, 1.685_437_153_407_153e-3);
        let classical_copper_oxygen = classical_debye_correlation(1.91, 650.0, 120.0, 29, 8, 2.3)?;
        assert_close(classical_copper_oxygen.value, 6.129_399_740_209_465e-4);
        Ok(())
    }

    #[test]
    fn debye_correlations_reject_invalid_inputs() {
        assert!(matches!(
            quantum_debye_correlation(-1.0, 400.0, 300.0, 29, 29, 2.7),
            Err(DebyeError::Negative { name: "rij", .. })
        ));
        assert!(matches!(
            quantum_debye_correlation(1.0, 400.0, -1.0, 29, 29, 2.7),
            Err(DebyeError::Negative { name: "tk", .. })
        ));
        assert!(matches!(
            classical_debye_correlation(1.0, 400.0, 300.0, 29, 0, 2.7),
            Err(DebyeError::InvalidAtomicNumber { z: 0 })
        ));
    }

    #[test]
    fn debye_waller_factors_match_feff_reference() -> Result<(), DebyeError> {
        let copper_path = ndarray::arr2(&[[0.0, 0.0, 0.0], [2.55, 0.0, 0.0], [0.0, 0.0, 0.0]]);
        let copper_atomic_numbers = [29, 29, 29];
        assert_close(
            quantum_debye_waller_factor(
                300.0,
                400.0,
                2.7,
                copper_path.view(),
                &copper_atomic_numbers,
            )?,
            5.620_717_932_013_852e-3,
        );
        assert_close(
            classical_debye_waller_factor(
                300.0,
                400.0,
                2.7,
                copper_path.view(),
                &copper_atomic_numbers,
            )?,
            5.216_382_857_388_334e-3,
        );

        let triangle_path = ndarray::arr2(&[
            [0.0, 0.0, 0.0],
            [1.91, 0.25, 0.10],
            [2.60, 1.40, -0.20],
            [0.0, 0.0, 0.0],
        ]);
        let triangle_atomic_numbers = [29, 8, 29, 29];
        assert_close(
            quantum_debye_waller_factor(
                180.0,
                650.0,
                2.3,
                triangle_path.view(),
                &triangle_atomic_numbers,
            )?,
            2.623_124_881_997_499_5e-3,
        );
        assert_close(
            classical_debye_waller_factor(
                180.0,
                650.0,
                2.3,
                triangle_path.view(),
                &triangle_atomic_numbers,
            )?,
            1.796_449_763_322_294e-3,
        );
        Ok(())
    }

    #[test]
    fn debye_waller_factors_reject_invalid_inputs() {
        let positions = ndarray::arr2(&[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]);
        assert!(matches!(
            quantum_debye_waller_factor(300.0, 400.0, 2.7, positions.view(), &[29, 29]),
            Err(DebyeError::ZeroLengthPathLeg { leg: 1 })
        ));
        assert!(matches!(
            quantum_debye_waller_factor(300.0, 400.0, 2.7, positions.view(), &[29]),
            Err(DebyeError::InvalidAtomicNumberCount { .. })
        ));
        let bad_shape = ndarray::Array2::<Real>::zeros((1, 3));
        assert!(matches!(
            quantum_debye_waller_factor(300.0, 400.0, 2.7, bad_shape.view(), &[29]),
            Err(DebyeError::InvalidPathShape { .. })
        ));
    }

    #[test]
    fn dmdw_path_descriptor_expands_single_atom_feff_branches() -> Result<(), DebyeError> {
        let positions = ndarray::arr2(&[[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 2.0, 0.0]]);
        let all_atoms = DmdwPathDescriptor {
            selectors: vec![0],
            max_effective_length: 0.0,
        };
        let selected_atom = DmdwPathDescriptor {
            selectors: vec![2],
            max_effective_length: 0.0,
        };

        assert_eq!(
            dmdw_expand_path_descriptor(positions.view(), &all_atoms)?,
            vec![
                DmdwExpandedPath {
                    atoms: vec![0],
                    effective_length: 0.0,
                },
                DmdwExpandedPath {
                    atoms: vec![1],
                    effective_length: 0.0,
                },
                DmdwExpandedPath {
                    atoms: vec![2],
                    effective_length: 0.0,
                },
            ]
        );
        assert_eq!(
            dmdw_expand_path_descriptor(positions.view(), &selected_atom)?,
            vec![DmdwExpandedPath {
                atoms: vec![1],
                effective_length: 0.0,
            }]
        );
        Ok(())
    }

    #[test]
    fn dmdw_path_descriptor_expands_multi_atom_feff_order_and_pruning() -> Result<(), DebyeError> {
        let positions = ndarray::arr2(&[[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 2.0, 0.0]]);
        let pairs = DmdwPathDescriptor {
            selectors: vec![0, 0],
            max_effective_length: 2.1,
        };

        let expanded = dmdw_expand_path_descriptor(positions.view(), &pairs)?;
        let expanded_atoms = expanded
            .iter()
            .map(|path| path.atoms.clone())
            .collect::<Vec<_>>();
        assert_eq!(
            expanded_atoms,
            vec![vec![0, 1], vec![0, 2], vec![1, 0], vec![2, 0]]
        );
        for path in &expanded {
            assert_dmdw_close(path.effective_length, 2.0);
        }

        let triple = DmdwPathDescriptor {
            selectors: vec![1, 0, 3],
            max_effective_length: 3.5,
        };
        let expanded = dmdw_expand_path_descriptor(positions.view(), &triple)?;
        assert_eq!(expanded.len(), 1);
        assert_eq!(expanded[0].atoms, vec![0, 1, 2]);
        assert_dmdw_close(
            expanded[0].effective_length,
            0.5 * (2.0 + 8.0_f64.sqrt() + 2.0),
        );
        Ok(())
    }

    #[test]
    fn dmdw_path_descriptor_rejects_invalid_inputs() {
        let positions = ndarray::arr2(&[[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 2.0, 0.0]]);
        let bad_shape = ndarray::Array2::<Real>::zeros((3, 2));

        assert!(matches!(
            dmdw_expand_path_descriptor(
                bad_shape.view(),
                &DmdwPathDescriptor {
                    selectors: vec![0, 0],
                    max_effective_length: 1.0,
                }
            ),
            Err(DebyeError::InvalidDmdwAtomShape { .. })
        ));
        assert!(matches!(
            dmdw_expand_path_descriptor(
                positions.view(),
                &DmdwPathDescriptor {
                    selectors: Vec::new(),
                    max_effective_length: 1.0,
                }
            ),
            Err(DebyeError::EmptyDmdwPath)
        ));
        assert!(matches!(
            dmdw_expand_path_descriptor(
                positions.view(),
                &DmdwPathDescriptor {
                    selectors: vec![-1],
                    max_effective_length: 1.0,
                }
            ),
            Err(DebyeError::InvalidDmdwPathSelector { selector: -1, .. })
        ));
        assert!(matches!(
            dmdw_expand_path_descriptor(
                positions.view(),
                &DmdwPathDescriptor {
                    selectors: vec![4],
                    max_effective_length: 1.0,
                }
            ),
            Err(DebyeError::InvalidDmdwPathSelector { selector: 4, .. })
        ));
        assert!(matches!(
            dmdw_expand_path_descriptor(
                positions.view(),
                &DmdwPathDescriptor {
                    selectors: vec![1],
                    max_effective_length: -1.0,
                }
            ),
            Err(DebyeError::Negative {
                name: "DMDW path descriptor maximum effective length",
                ..
            })
        ));
    }

    #[test]
    fn dmdw_path_motion_matches_feff_two_atom_path() -> Result<(), DebyeError> {
        let positions = ndarray::arr2(&[[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]]);
        let masses = ndarray::arr1(&[10.0, 20.0]);
        let motion = dmdw_path_motion(positions.view(), masses.view(), &[0, 1])?;

        assert_dmdw_close(motion.inverse_reduced_mass, 0.15);
        assert_dmdw_close(motion.reduced_mass, 6.666_666_666_666_667);
        assert_vector_close(
            &motion.initial_vector,
            &[
                -0.816_496_580_927_726,
                0.577_350_269_189_625_8,
                0.0,
                0.0,
                0.0,
                0.0,
            ],
        );
        assert_dmdw_close(
            motion
                .initial_vector
                .iter()
                .map(|value| value * value)
                .sum(),
            1.0,
        );
        Ok(())
    }

    #[test]
    fn dmdw_path_motion_matches_feff_bent_three_atom_path() -> Result<(), DebyeError> {
        let positions = ndarray::arr2(&[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]);
        let masses = ndarray::arr1(&[10.0, 20.0, 30.0]);
        let motion = dmdw_path_motion(positions.view(), masses.view(), &[0, 1, 2])?;

        assert_dmdw_close(motion.inverse_reduced_mass, 0.121_129_449_216_106_15);
        assert_dmdw_close(motion.reduced_mass, 8.255_630_703_115_866);
        assert_vector_close(
            &motion.initial_vector,
            &[
                -0.454_302_506_682_383,
                0.548_391_636_526_351_4,
                -0.185_468_221_706_530_54,
                -0.454_302_506_682_383,
                -0.227_151_253_341_191_5,
                0.447_759_896_233_126_1,
                0.0,
                0.0,
                0.0,
            ],
        );
        assert_dmdw_close(
            motion
                .initial_vector
                .iter()
                .map(|value| value * value)
                .sum(),
            1.0,
        );
        Ok(())
    }

    #[test]
    fn dmdw_path_motion_matches_feff_single_atom_mass_branch() -> Result<(), DebyeError> {
        let positions = ndarray::arr2(&[[0.0, 0.0, 0.0]]);
        let masses = ndarray::arr1(&[63.546]);
        let motion = dmdw_path_motion(positions.view(), masses.view(), &[0])?;

        assert_dmdw_close(motion.inverse_reduced_mass, 1.0 / 63.546);
        assert_dmdw_close(motion.reduced_mass, 63.546);
        assert_vector_close(&motion.initial_vector, &[0.0, 0.0, 0.0]);
        Ok(())
    }

    #[test]
    fn dmdw_path_motion_rejects_invalid_inputs() {
        let positions = ndarray::arr2(&[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]);
        let masses = ndarray::arr1(&[10.0, 20.0]);
        assert!(matches!(
            dmdw_path_motion(positions.view(), masses.view(), &[]),
            Err(DebyeError::EmptyDmdwPath)
        ));
        assert!(matches!(
            dmdw_path_motion(positions.view(), masses.view(), &[0, 2]),
            Err(DebyeError::InvalidDmdwPathAtomIndex { index: 2, .. })
        ));
        assert!(matches!(
            dmdw_path_motion(positions.view(), masses.view(), &[0, 1]),
            Err(DebyeError::ZeroLengthDmdwAtomPair {
                first: 0,
                second: 1
            })
        ));

        let bad_masses = ndarray::arr1(&[10.0]);
        assert!(matches!(
            dmdw_path_motion(positions.view(), bad_masses.view(), &[0]),
            Err(DebyeError::InvalidDmdwMassCount { .. })
        ));
        let bad_shape = ndarray::Array2::<Real>::zeros((2, 2));
        assert!(matches!(
            dmdw_path_motion(bad_shape.view(), masses.view(), &[0]),
            Err(DebyeError::InvalidDmdwAtomShape { .. })
        ));
    }

    #[test]
    fn dmdw_ir_dipole_seed_matches_feff_type4_branch() -> Result<(), DebyeError> {
        let masses = ndarray::arr1(&[4.0, 9.0]);
        let dipoles = ndarray::arr3(&[
            [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]],
            [[1.0, 1.1, 1.2], [1.3, 1.4, 1.5], [1.6, 1.7, 1.8]],
        ]);

        let seed = dmdw_ir_dipole_seed_vector(masses.view(), dipoles.view())?;

        assert_vector_close(
            &seed,
            &[
                0.007_160_718_421_688_271,
                0.324_917_598_384_105_3,
                0.044_754_490_135_551_696,
                0.526_312_803_994_088,
                0.114_571_494_747_012_34,
                0.776_042_858_950_466_4,
            ],
        );
        Ok(())
    }

    #[test]
    fn dmdw_ir_dipole_seed_rejects_invalid_inputs() {
        let masses = ndarray::arr1(&[4.0, 9.0]);
        let bad_shape = ndarray::Array3::<Real>::zeros((2, 3, 2));
        assert!(matches!(
            dmdw_ir_dipole_seed_vector(masses.view(), bad_shape.view()),
            Err(DebyeError::InvalidDmdwDipoleDerivativeShape { .. })
        ));

        let zero_dipoles = ndarray::Array3::<Real>::zeros((2, 3, 3));
        assert!(matches!(
            dmdw_ir_dipole_seed_vector(masses.view(), zero_dipoles.view()),
            Err(DebyeError::ZeroDmdwSeedNorm)
        ));
    }

    #[test]
    fn dmdw_mass_weighted_dynamical_matrix_matches_feff_make_dm() -> Result<(), DebyeError> {
        let mut blocks = ndarray::Array4::<Real>::zeros((2, 2, 3, 3));
        blocks[(0, 0, 0, 0)] = 2.0;
        blocks[(0, 1, 0, 1)] = 3.0;
        blocks[(1, 0, 1, 0)] = 6.0;
        blocks[(1, 1, 2, 2)] = 18.0;
        let masses = ndarray::arr1(&[4.0, 9.0]);
        let scale = 1_556.892_791_61 * 602.214_198_280;

        let result = dmdw_mass_weighted_dynamical_matrix(blocks.view(), masses.view())?;

        assert_eq!(result.matrix.shape(), &[6, 6]);
        assert_dmdw_close(result.matrix[(0, 0)], 0.5 * scale);
        assert_dmdw_close(result.matrix[(0, 3)], 0.5 * scale);
        assert_dmdw_close(result.matrix[(3, 0)], scale);
        assert_dmdw_close(result.matrix[(5, 5)], 2.0 * scale);
        assert_dmdw_close(result.average_value, scale / 9.0);
        assert_dmdw_close(result.average_asymmetry, scale / 36.0);
        assert_dmdw_close(result.asymmetry_percent_average, 25.0);
        assert_dmdw_close(result.asymmetry_percent_max, 100.0 / 72.0);
        assert!(result.passes_feff_symmetry_check());
        Ok(())
    }

    #[test]
    fn dmdw_mass_weighted_dynamical_matrix_reports_feff_asymmetry_warning() -> Result<(), DebyeError>
    {
        let mut blocks = ndarray::Array4::<Real>::zeros((2, 2, 3, 3));
        blocks[(0, 1, 0, 1)] = 6.0;
        let masses = ndarray::arr1(&[4.0, 9.0]);

        let result = dmdw_mass_weighted_dynamical_matrix(blocks.view(), masses.view())?;

        assert_dmdw_close(result.asymmetry_percent_average, 200.0);
        assert_dmdw_close(result.asymmetry_percent_max, 100.0 / 18.0);
        assert!(!result.passes_feff_symmetry_check());
        Ok(())
    }

    #[test]
    fn dmdw_mass_weighted_dynamical_matrix_rejects_invalid_inputs() {
        let masses = ndarray::arr1(&[4.0, 9.0]);
        let bad_shape = ndarray::Array4::<Real>::zeros((1, 2, 3, 3));
        assert!(matches!(
            dmdw_mass_weighted_dynamical_matrix(bad_shape.view(), masses.view()),
            Err(DebyeError::InvalidDmdwBlockShape { .. })
        ));

        let empty_blocks = ndarray::Array4::<Real>::zeros((0, 0, 3, 3));
        let empty_masses = ndarray::Array1::<Real>::zeros(0);
        assert!(matches!(
            dmdw_mass_weighted_dynamical_matrix(empty_blocks.view(), empty_masses.view()),
            Err(DebyeError::EmptyDmdwAtomTable)
        ));

        let bad_masses = ndarray::arr1(&[4.0, 0.0]);
        let blocks = ndarray::Array4::<Real>::zeros((2, 2, 3, 3));
        assert!(matches!(
            dmdw_mass_weighted_dynamical_matrix(blocks.view(), bad_masses.view()),
            Err(DebyeError::NonPositive {
                name: "DMDW atom mass",
                ..
            })
        ));
    }

    #[test]
    fn dmdw_lanczos_coefficients_match_feff_recurrence() -> Result<(), DebyeError> {
        let matrix = ndarray::arr2(&[[1.0, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 9.0]]);
        let seed = ndarray::arr1(&[1.0, 1.0, 1.0]);

        let coefficients = dmdw_lanczos_coefficients(matrix.view(), seed.view(), 1)?;

        assert_vector_close(
            &coefficients.alpha,
            &[4.666_666_666_666_667, 5.639_455_782_312_925],
        );
        assert_vector_close(
            &coefficients.beta,
            &[0.0, 3.299_831_645_537_221_6, 2.120_878_539_880_258],
        );
        assert_dmdw_close(coefficients.single_pole_frequency, 0.343_813_972_349_477_75);
        Ok(())
    }

    #[test]
    fn dmdw_lanczos_coefficients_preserve_feff_column_product() -> Result<(), DebyeError> {
        let matrix = ndarray::arr2(&[[1.0, 10.0], [0.0, 2.0]]);
        let seed = ndarray::arr1(&[1.0, 0.0]);

        let coefficients = dmdw_lanczos_coefficients(matrix.view(), seed.view(), 1)?;

        assert_vector_close(&coefficients.alpha, &[1.0, 2.0]);
        assert_vector_close(&coefficients.beta, &[0.0, 10.0, 10.0]);
        Ok(())
    }

    #[test]
    fn dmdw_lanczos_coefficients_reject_invalid_inputs() {
        let matrix = ndarray::arr2(&[[1.0, 0.0], [0.0, 2.0]]);
        let seed = ndarray::arr1(&[1.0, 0.0]);
        assert!(matches!(
            dmdw_lanczos_coefficients(matrix.view(), seed.view(), 0),
            Err(DebyeError::NonPositive {
                name: "DMDW Lanczos pole count",
                ..
            })
        ));

        let bad_matrix = ndarray::Array2::<Real>::zeros((2, 3));
        assert!(matches!(
            dmdw_lanczos_coefficients(bad_matrix.view(), seed.view(), 1),
            Err(DebyeError::InvalidDmdwLanczosShape { .. })
        ));

        let eigen_seed = ndarray::arr1(&[1.0, 0.0]);
        assert!(matches!(
            dmdw_lanczos_coefficients(matrix.view(), eigen_seed.view(), 1),
            Err(DebyeError::DmdwLanczosBreakdown { iteration: 1 })
        ));
    }

    #[test]
    fn dmdw_lanczos_polynomials_match_feff_recurrences() -> Result<(), DebyeError> {
        let alpha = ndarray::arr1(&[4.666_666_666_666_667, 5.639_455_782_312_925]);
        let beta = ndarray::arr1(&[0.0, 3.299_831_645_537_221_6]);

        assert_dmdw_close(
            dmdw_lanczos_s_polynomial(2, 7.0, alpha.view(), beta.view())?,
            -7.714_285_714_285_713_5,
        );
        assert_dmdw_close(
            dmdw_lanczos_r_polynomial(2, 7.0, alpha.view(), beta.view())?,
            1.360_544_217_687_074_6,
        );
        assert_dmdw_close(
            dmdw_lanczos_s_polynomial_derivative(2, 7.0, alpha.view(), beta.view())?,
            3.693_877_551_020_406_7,
        );
        assert_dmdw_close(
            dmdw_lanczos_s_polynomial(1, 7.0, alpha.view(), beta.view())?,
            2.333_333_333_333_333,
        );
        assert_dmdw_close(
            dmdw_lanczos_r_polynomial(1, 7.0, alpha.view(), beta.view())?,
            1.0,
        );
        assert_dmdw_close(
            dmdw_lanczos_s_polynomial_derivative(1, 7.0, alpha.view(), beta.view())?,
            1.0,
        );
        Ok(())
    }

    #[test]
    fn dmdw_lanczos_polynomials_reject_invalid_inputs() {
        let alpha = ndarray::arr1(&[1.0]);
        let beta = ndarray::arr1(&[0.0]);
        assert!(matches!(
            dmdw_lanczos_s_polynomial(0, 1.0, alpha.view(), beta.view()),
            Err(DebyeError::NonPositive {
                name: "DMDW Lanczos polynomial order",
                ..
            })
        ));
        assert!(matches!(
            dmdw_lanczos_s_polynomial(2, 1.0, alpha.view(), beta.view()),
            Err(DebyeError::InvalidDmdwLanczosPolynomialShape { .. })
        ));
        assert!(matches!(
            dmdw_lanczos_s_polynomial(1, Real::NAN, alpha.view(), beta.view()),
            Err(DebyeError::NonFinite {
                name: "DMDW Lanczos polynomial x",
                ..
            })
        ));
    }

    #[test]
    fn dmdw_lanczos_pole_spectrum_matches_feff_scan() -> Result<(), DebyeError> {
        let alpha = ndarray::arr1(&[16.0, 16.0]);
        let beta = ndarray::arr1(&[0.0, 8.0]);
        let spectrum =
            dmdw_lanczos_pole_spectrum_with_search(2, alpha.view(), beta.view(), 32.0, 8192)?;

        assert!(spectrum.has_expected_pole_count());
        assert_vector_close(&spectrum.squared_angular_frequencies, &[8.0, 24.0]);
        assert_vector_close(
            &spectrum.angular_frequencies,
            &[8.0_f64.sqrt(), 24.0_f64.sqrt()],
        );
        assert_vector_close(
            &spectrum.frequencies,
            &[
                8.0_f64.sqrt() / (2.0 * std::f64::consts::PI),
                24.0_f64.sqrt() / (2.0 * std::f64::consts::PI),
            ],
        );
        assert_vector_close(&spectrum.weights, &[0.5, 0.5]);
        assert!(spectrum.imaginary_warnings.is_empty());
        Ok(())
    }

    #[test]
    fn dmdw_lanczos_pole_spectrum_reports_imaginary_weight_warnings() -> Result<(), DebyeError> {
        let alpha = ndarray::arr1(&[-16.0, -16.0]);
        let beta = ndarray::arr1(&[0.0, 8.0]);
        let spectrum =
            dmdw_lanczos_pole_spectrum_with_search(2, alpha.view(), beta.view(), 32.0, 8192)?;

        assert!(spectrum.has_expected_pole_count());
        assert_vector_close(&spectrum.squared_angular_frequencies, &[-24.0, -8.0]);
        assert_vector_close(
            &spectrum.angular_frequencies,
            &[-24.0_f64.sqrt(), -8.0_f64.sqrt()],
        );
        assert_vector_close(&spectrum.weights, &[0.5, 0.5]);
        assert_eq!(spectrum.imaginary_warnings.len(), 2);
        assert_eq!(
            spectrum.imaginary_warnings[0].severity,
            DmdwImaginaryPoleSeverity::LargeWeight
        );
        assert_eq!(spectrum.imaginary_warnings[0].pole_index, 0);
        assert_dmdw_close(spectrum.imaginary_warnings[0].weight, 0.5);
        Ok(())
    }

    #[test]
    fn dmdw_lanczos_pole_spectrum_rejects_invalid_inputs() {
        let alpha = ndarray::arr1(&[1.0, 1.0]);
        let beta = ndarray::arr1(&[0.0, 0.0]);
        assert!(matches!(
            dmdw_lanczos_pole_spectrum_with_search(0, alpha.view(), beta.view(), 2.0, 1),
            Err(DebyeError::NonPositive {
                name: "DMDW Lanczos polynomial order",
                ..
            })
        ));
        assert!(matches!(
            dmdw_lanczos_pole_spectrum_with_search(1, alpha.view(), beta.view(), 0.0, 1),
            Err(DebyeError::NonPositive {
                name: "DMDW Lanczos pole search limit",
                ..
            })
        ));
        assert!(matches!(
            dmdw_lanczos_pole_spectrum_with_search(1, alpha.view(), beta.view(), 2.0, 0),
            Err(DebyeError::NonPositive {
                name: "DMDW Lanczos pole samples per pole",
                ..
            })
        ));
        assert!(matches!(
            dmdw_lanczos_pole_spectrum_with_search(2, alpha.view(), beta.view(), 2.0, 2),
            Err(DebyeError::ZeroDmdwLanczosPoleDerivative { .. })
        ));
    }

    #[test]
    fn dmdw_debye_waller_factors_from_poles_match_feff_accumulation() -> Result<(), DebyeError> {
        let temperatures = ndarray::arr1(&[300.0, 600.0]);
        let angular_frequencies = ndarray::arr1(&[2.0, 4.0, -3.0, 0.0]);
        let weights = ndarray::arr1(&[0.25, 0.75, 0.9, 10.0]);
        let factors = dmdw_debye_waller_factors_from_poles(
            temperatures.view(),
            5.0,
            angular_frequencies.view(),
            weights.view(),
        )?;

        assert_vector_close(&factors, &[5.459_186_287_610_058, 10.914_330_842_743_967]);
        Ok(())
    }

    #[test]
    fn dmdw_debye_waller_factors_use_zero_temperature_coth_limit() -> Result<(), DebyeError> {
        let temperatures = ndarray::arr1(&[0.001]);
        let angular_frequencies = ndarray::arr1(&[2.0]);
        let weights = ndarray::arr1(&[1.0]);
        let factors = dmdw_debye_waller_factors_from_poles(
            temperatures.view(),
            5.0,
            angular_frequencies.view(),
            weights.view(),
        )?;

        assert_vector_close(&factors, &[0.317_544_517_206_879_8]);
        Ok(())
    }

    #[test]
    fn dmdw_vibrational_free_energy_from_poles_matches_feff_accumulation() -> Result<(), DebyeError>
    {
        let temperatures = ndarray::arr1(&[300.0, 600.0]);
        let angular_frequencies = ndarray::arr1(&[2.0, 4.0, -3.0, 0.0]);
        let weights = ndarray::arr1(&[0.25, 0.75, 0.9, 10.0]);
        let free_energy = dmdw_vibrational_free_energy_from_poles(
            temperatures.view(),
            angular_frequencies.view(),
            weights.view(),
        )?;

        assert_vector_close(
            &free_energy,
            &[-6_129.431_830_672_452, -15_718.169_449_997_833],
        );
        Ok(())
    }

    #[test]
    fn dmdw_einstein_and_moment_summaries_match_feff_print_formulas() -> Result<(), DebyeError> {
        let reduced_mass = 10.0;
        let summary = dmdw_single_pole_einstein_summary(3.5, reduced_mass)?;
        assert_dmdw_close(summary.frequency_thz, 3.5);
        assert_dmdw_close(summary.temperature_kelvin, 3.5 * DMDW_THZ_TO_KELVIN);
        assert_dmdw_close(
            summary.effective_force_constant_n_per_m,
            reduced_mass
                * (2.0 * std::f64::consts::PI * 3.5).powi(2)
                * DMDW_AMU_THZ2_TO_NEWTON_PER_METER,
        );

        let frequencies = ndarray::arr1(&[-1.0, 2.0, 4.0]);
        let weights = ndarray::arr1(&[0.2, 0.2, 0.6]);
        let moments =
            dmdw_moment_summaries_from_poles(reduced_mass, frequencies.view(), weights.view())?;

        assert_eq!(
            moments
                .iter()
                .map(|moment| moment.order)
                .collect::<Vec<_>>(),
            vec![-2, -1, 0, 1, 2]
        );
        assert_moment_summary(
            &moments[0],
            0.109_375,
            0.109_375_f64.powf(-0.5),
            reduced_mass,
        )?;
        assert_moment_summary(&moments[1], 0.312_5, 3.2, reduced_mass)?;
        assert_dmdw_close(moments[2].moment_thz_power_n, 1.0);
        assert_eq!(moments[2].frequency_thz, None);
        assert_eq!(moments[2].temperature_kelvin, None);
        assert_eq!(moments[2].effective_force_constant_n_per_m, None);
        assert_moment_summary(&moments[3], 3.5, 3.5, reduced_mass)?;
        assert_moment_summary(&moments[4], 13.0, 13.0_f64.sqrt(), reduced_mass)?;
        Ok(())
    }

    #[test]
    fn dmdw_pole_thermal_helpers_reject_invalid_inputs() {
        let temperatures = ndarray::arr1(&[300.0]);
        let frequencies = ndarray::arr1(&[1.0, 2.0]);
        let weights = ndarray::arr1(&[1.0]);
        assert!(matches!(
            dmdw_debye_waller_factors_from_poles(
                temperatures.view(),
                1.0,
                frequencies.view(),
                weights.view()
            ),
            Err(DebyeError::InvalidDmdwPoleTableShape { .. })
        ));

        let empty_temperatures = ndarray::arr1(&[]);
        assert!(matches!(
            dmdw_vibrational_free_energy_from_poles(
                empty_temperatures.view(),
                weights.view(),
                weights.view()
            ),
            Err(DebyeError::EmptyDmdwTemperatureTable)
        ));

        let bad_temperatures = ndarray::arr1(&[0.0]);
        assert!(matches!(
            dmdw_vibrational_free_energy_from_poles(
                bad_temperatures.view(),
                weights.view(),
                weights.view()
            ),
            Err(DebyeError::NonPositive {
                name: "DMDW temperature",
                ..
            })
        ));

        assert!(matches!(
            dmdw_debye_waller_factors_from_poles(
                temperatures.view(),
                0.0,
                weights.view(),
                weights.view()
            ),
            Err(DebyeError::NonPositive {
                name: "DMDW reduced mass",
                ..
            })
        ));
    }

    #[test]
    fn dmdw_pole_summary_helpers_reject_invalid_inputs() {
        assert!(matches!(
            dmdw_single_pole_einstein_summary(0.0, 1.0),
            Err(DebyeError::NonPositive {
                name: "DMDW Einstein frequency",
                ..
            })
        ));

        let empty = ndarray::arr1(&[]);
        assert!(matches!(
            dmdw_moment_summaries_from_poles(1.0, empty.view(), empty.view()),
            Err(DebyeError::EmptyDmdwPoleTable)
        ));

        let imaginary_frequencies = ndarray::arr1(&[-1.0]);
        let weights = ndarray::arr1(&[1.0]);
        assert!(matches!(
            dmdw_moment_summaries_from_poles(1.0, imaginary_frequencies.view(), weights.view()),
            Err(DebyeError::NonPositive {
                name: "DMDW positive pole weight normalization",
                ..
            })
        ));
    }

    #[test]
    fn dmdw_center_of_mass_and_inertia_match_feff_reference_formulas() -> Result<(), DebyeError> {
        let positions = ndarray::arr2(&[[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 3.0, 0.0]]);
        let masses = ndarray::arr1(&[2.0, 3.0, 5.0]);

        let center = dmdw_center_of_mass(positions.view(), masses.view())?;
        assert_slice_close(&center, &[0.6, 1.5, 0.0]);

        let centered = ndarray::arr2(&[[-0.6, -1.5, 0.0], [1.4, -1.5, 0.0], [-0.6, 1.5, 0.0]]);
        let tensor = dmdw_inertia_tensor(centered.view(), masses.view())?;
        assert_matrix_close(
            tensor.view(),
            &[[22.5, 9.0, 0.0], [9.0, 8.4, 0.0], [0.0, 0.0, 30.9]],
        );
        Ok(())
    }

    #[test]
    fn dmdw_rigid_body_projection_modes_match_feff_make_trfd_formulas() -> Result<(), DebyeError> {
        let positions = ndarray::arr2(&[
            [1.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0, 2.0, 0.0],
            [0.0, -2.0, 0.0],
            [0.0, 0.0, 3.0],
            [0.0, 0.0, -3.0],
        ]);
        let masses = ndarray::arr1(&[1.0; 6]);
        let modes = dmdw_rigid_body_projection_modes(positions.view(), masses.view())?;

        assert_slice_close(&modes.center_of_mass, &[0.0, 0.0, 0.0]);
        assert_vector_close(&modes.moments_of_inertia, &[10.0, 20.0, 26.0]);
        assert_matrix_abs_close(
            modes.principal_axes.view(),
            &[[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]],
        );

        let projection = modes.projection_modes;
        assert_eq!(projection.shape(), &[18, 6]);
        for left in 0..6 {
            assert_dmdw_close(column_dot(projection.view(), left, left), 1.0);
            for right in (left + 1)..6 {
                assert_dmdw_close(column_dot(projection.view(), left, right), 0.0);
            }
        }

        let translation_scale = 1.0 / 6.0_f64.sqrt();
        for atom in 0..6 {
            assert_dmdw_close(projection[(atom, 0)], translation_scale);
            assert_dmdw_close(projection[(6 + atom, 1)], translation_scale);
            assert_dmdw_close(projection[(12 + atom, 2)], translation_scale);
        }

        let rotation_z = ndarray::arr1(&[
            0.0,
            0.0,
            -2.0 / 10.0_f64.sqrt(),
            2.0 / 10.0_f64.sqrt(),
            0.0,
            0.0,
            1.0 / 10.0_f64.sqrt(),
            -1.0 / 10.0_f64.sqrt(),
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ]);
        let rotation_y = ndarray::arr1(&[
            0.0,
            0.0,
            0.0,
            0.0,
            3.0 / 20.0_f64.sqrt(),
            -3.0 / 20.0_f64.sqrt(),
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            -1.0 / 20.0_f64.sqrt(),
            1.0 / 20.0_f64.sqrt(),
            0.0,
            0.0,
            0.0,
            0.0,
        ]);
        let rotation_x = ndarray::arr1(&[
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            -3.0 / 26.0_f64.sqrt(),
            3.0 / 26.0_f64.sqrt(),
            0.0,
            0.0,
            2.0 / 26.0_f64.sqrt(),
            -2.0 / 26.0_f64.sqrt(),
            0.0,
            0.0,
        ]);
        assert_dmdw_close(projection.column(3).dot(&rotation_z).abs(), 1.0);
        assert_dmdw_close(projection.column(4).dot(&rotation_y).abs(), 1.0);
        assert_dmdw_close(projection.column(5).dot(&rotation_x).abs(), 1.0);
        Ok(())
    }

    #[test]
    fn dmdw_seed_projection_matches_feff_qj0_loop() -> Result<(), DebyeError> {
        let seed = ndarray::arr1(&[1.0, 2.0, 3.0, 4.0]);
        let inv_sqrt_two = 0.5_f64.sqrt();
        let modes = ndarray::arr2(&[
            [1.0, 0.0],
            [0.0, inv_sqrt_two],
            [0.0, inv_sqrt_two],
            [0.0, 0.0],
        ]);

        let projected = dmdw_project_seed_vector(seed.view(), modes.view())?;
        assert_vector_close(
            &projected,
            &[
                0.0,
                -0.123_091_490_979_332_72,
                0.123_091_490_979_332_72,
                0.984_731_927_834_661_8,
            ],
        );
        assert_dmdw_close(projected.iter().map(|value| value * value).sum(), 1.0);

        let normalized = dmdw_normalize_seed_vector(seed.view())?;
        assert_vector_close(
            &normalized,
            &[
                0.182_574_185_835_055_36,
                0.365_148_371_670_110_7,
                0.547_722_557_505_166_1,
                0.730_296_743_340_221_4,
            ],
        );
        Ok(())
    }

    #[test]
    fn dmdw_rigid_body_helpers_reject_invalid_inputs() {
        let empty_positions = ndarray::Array2::<Real>::zeros((0, 3));
        let empty_masses = ndarray::Array1::<Real>::zeros(0);
        assert!(matches!(
            dmdw_center_of_mass(empty_positions.view(), empty_masses.view()),
            Err(DebyeError::EmptyDmdwAtomTable)
        ));

        let positions = ndarray::arr2(&[[0.0, 0.0, 0.0]]);
        let bad_masses = ndarray::arr1(&[-1.0]);
        assert!(matches!(
            dmdw_inertia_tensor(positions.view(), bad_masses.view()),
            Err(DebyeError::NonPositive {
                name: "DMDW atom mass",
                ..
            })
        ));

        let positions = ndarray::arr2(&[[0.0, 0.0, 0.0]]);
        let masses = ndarray::arr1(&[1.0]);
        assert!(matches!(
            dmdw_rigid_body_projection_modes(positions.view(), masses.view()),
            Err(DebyeError::TooFewDmdwRigidBodyAtoms { atoms: 1 })
        ));

        let collinear_positions = ndarray::arr2(&[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]);
        let collinear_masses = ndarray::arr1(&[1.0, 1.0]);
        assert!(matches!(
            dmdw_rigid_body_projection_modes(collinear_positions.view(), collinear_masses.view()),
            Err(DebyeError::ZeroDmdwProjectionModeNorm { .. })
        ));
    }

    #[test]
    fn dmdw_seed_projection_rejects_invalid_inputs() {
        let seed = ndarray::arr1(&[1.0, 2.0]);
        let bad_modes = ndarray::Array2::<Real>::zeros((3, 1));
        assert!(matches!(
            dmdw_project_seed_vector(seed.view(), bad_modes.view()),
            Err(DebyeError::InvalidDmdwProjectionShape { .. })
        ));

        let empty_seed = ndarray::Array1::<Real>::zeros(0);
        assert!(matches!(
            dmdw_normalize_seed_vector(empty_seed.view()),
            Err(DebyeError::EmptyDmdwSeed)
        ));

        let zero_seed = ndarray::arr1(&[0.0, 0.0]);
        assert!(matches!(
            dmdw_normalize_seed_vector(zero_seed.view()),
            Err(DebyeError::ZeroDmdwSeedNorm)
        ));
    }

    fn assert_close(actual: Real, expected: Real) {
        assert!(
            (actual - expected).abs() < 1.0e-18,
            "actual={actual} expected={expected} diff={}",
            (actual - expected).abs()
        );
    }

    fn assert_slice_close(actual: &[Real], expected: &[Real]) {
        assert_eq!(actual.len(), expected.len());
        for (actual, expected) in actual.iter().zip(expected.iter()) {
            assert_dmdw_close(*actual, *expected);
        }
    }

    fn assert_matrix_close(actual: ArrayView2<'_, Real>, expected: &[[Real; 3]; 3]) {
        assert_eq!(actual.shape(), &[3, 3]);
        for row in 0..3 {
            for column in 0..3 {
                assert_dmdw_close(actual[(row, column)], expected[row][column]);
            }
        }
    }

    fn assert_matrix_abs_close(actual: ArrayView2<'_, Real>, expected: &[[Real; 3]; 3]) {
        assert_eq!(actual.shape(), &[3, 3]);
        for row in 0..3 {
            for column in 0..3 {
                assert_dmdw_close(actual[(row, column)].abs(), expected[row][column].abs());
            }
        }
    }

    fn assert_vector_close(actual: &Array1<Real>, expected: &[Real]) {
        assert_eq!(actual.len(), expected.len());
        for (actual, expected) in actual.iter().zip(expected.iter()) {
            assert_dmdw_close(*actual, *expected);
        }
    }

    fn column_dot(matrix: ArrayView2<'_, Real>, left: usize, right: usize) -> Real {
        let left_column = matrix.column(left);
        let right_column = matrix.column(right);
        left_column
            .iter()
            .zip(right_column.iter())
            .map(|(&left, &right)| left * right)
            .sum()
    }

    fn assert_moment_summary(
        actual: &DmdwMomentSummary,
        expected_moment: Real,
        expected_frequency: Real,
        reduced_mass: Real,
    ) -> Result<(), DebyeError> {
        assert_dmdw_close(actual.moment_thz_power_n, expected_moment);
        let expected = dmdw_single_pole_einstein_summary(expected_frequency, reduced_mass)?;
        assert_dmdw_close(
            actual.frequency_thz.ok_or(DebyeError::NonFiniteOutput {
                name: "test moment frequency",
                value: Real::NAN,
            })?,
            expected.frequency_thz,
        );
        assert_dmdw_close(
            actual
                .temperature_kelvin
                .ok_or(DebyeError::NonFiniteOutput {
                    name: "test moment temperature",
                    value: Real::NAN,
                })?,
            expected.temperature_kelvin,
        );
        assert_dmdw_close(
            actual
                .effective_force_constant_n_per_m
                .ok_or(DebyeError::NonFiniteOutput {
                    name: "test moment force constant",
                    value: Real::NAN,
                })?,
            expected.effective_force_constant_n_per_m,
        );
        Ok(())
    }

    fn assert_dmdw_close(actual: Real, expected: Real) {
        let tolerance = expected.abs().max(1.0) * 1.0e-14;
        assert!(
            (actual - expected).abs() <= tolerance,
            "actual={actual} expected={expected} diff={}",
            (actual - expected).abs()
        );
    }

    fn assert_relative_close(actual: Real, expected: Real) {
        let tolerance = expected.abs().max(1.0) * 1.0e-14;
        assert!(
            (actual - expected).abs() <= tolerance,
            "actual={actual} expected={expected} diff={}",
            (actual - expected).abs()
        );
    }
}
