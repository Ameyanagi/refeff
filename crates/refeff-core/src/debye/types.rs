//! Public Debye and DMDW data types.

use ndarray::{Array1, Array2};

use crate::special::SpecialFunctionError;
use crate::{Complex, Real};

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
/// [`crate::debye::dmdw_expand_path_descriptor`].
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

/// FEFF DMDW run-type 2 phonon-coupling table derived from PDS and `a2f`.
#[derive(Debug, Clone, PartialEq)]
pub struct DmdwPhononCoupling {
    /// Energy grid in Hartree, from FEFF's final `a2f` row read.
    pub energy_hartree: Array1<Real>,
    /// Energy grid in eV, using FEFF's run-type 2 conversion constant.
    pub energy_ev: Array1<Real>,
    /// Eliashberg coupling column from the `a2f` input table.
    pub eliashberg: Array1<Real>,
    /// FEFF `a2(2,j)`, the coupling divided by projected phonon DOS.
    pub matrix_element: Array1<Real>,
    /// FEFF `norm`, accumulated from projected phonon DOS and energy steps.
    pub normalization: Real,
}

/// FEFF DMDW run-type 2 pole-weight `a2f` diagnostic.
#[derive(Debug, Clone, PartialEq)]
pub struct DmdwPoleWeightedA2f {
    /// FEFF diagnostic pole frequencies, `w_pole / 6.28`, in THz.
    pub lanczos_frequency_thz: Array1<Real>,
    /// FEFF `wil` projected-DOS weights.
    pub lanczos_weight: Array1<Real>,
    /// FEFF projected-DOS normalization from the PDS table.
    pub normalization: Real,
    /// Pole energies written in eV.
    pub pole_energy_ev: Array1<Real>,
    /// Pole-weight `a2f` values.
    pub pole_weight: Array1<Real>,
    /// FEFF `lambda` mass-enhancement diagnostic.
    pub mass_enhancement: Real,
    /// FEFF `w0` characteristic phonon energy in eV.
    pub characteristic_energy_ev: Real,
}

/// FEFF DMDW run-type 2 unique-atom group used for `a2f` pole generation.
///
/// FEFF's type-2 branch only uses the central atom indices when constructing
/// the Lanczos displacement seeds. Other `.dym` metadata remains part of the IO
/// layer and can be preserved independently.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DmdwType2AtomGroup {
    /// Zero-based central atom indices for degenerate representatives.
    pub center_atom_indices: Vec<usize>,
}

/// FEFF DMDW run-type 2 phonon self-energy table.
#[derive(Debug, Clone, PartialEq)]
pub struct DmdwSelfEnergyGrid {
    /// Electron energy grid in eV.
    pub energy_ev: Array1<Real>,
    /// FEFF `SE_a2f` values on the energy grid.
    pub self_energy: Array1<Complex>,
}

/// FEFF DMDW run-type 2 cumulant spectral-function table.
#[derive(Debug, Clone, PartialEq)]
pub struct DmdwSpectralFunctionGrid {
    /// Electron-energy samples in units of the characteristic phonon energy.
    pub energy_w0: Array1<Real>,
    /// FEFF `Akw` values after normalization, before meV scaling for output.
    pub spectral_function: Array1<Complex>,
    /// FEFF normalization factor applied to the spectral function.
    pub normalization: Real,
    /// Lorentzian damping width in units of the characteristic phonon energy.
    pub gamma_w0: Real,
}

impl DmdwPhononCoupling {
    /// Number of phonon-coupling grid points.
    #[must_use]
    pub fn point_count(&self) -> usize {
        self.energy_hartree.len()
    }
}

impl DmdwPoleWeightedA2f {
    /// Number of pole-weight rows.
    #[must_use]
    pub fn pole_count(&self) -> usize {
        self.pole_energy_ev.len()
    }
}

impl DmdwSelfEnergyGrid {
    /// Number of self-energy samples.
    #[must_use]
    pub fn point_count(&self) -> usize {
        self.energy_ev.len()
    }
}

impl DmdwSpectralFunctionGrid {
    /// Number of spectral-function samples.
    #[must_use]
    pub fn point_count(&self) -> usize {
        self.energy_w0.len()
    }
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
#[non_exhaustive]
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
    /// FEFF `spring.inp` parser or spring-matrix setup rejected the input.
    #[error("Debye spring input is invalid: {reason}")]
    InvalidSpringInput { reason: &'static str },
    /// FEFF `spring.inp` atom index is outside the current atom table.
    #[error("Debye spring atom index {index} is outside 0..{atom_count}")]
    InvalidSpringAtomIndex { index: usize, atom_count: usize },
    /// FEFF spring/RM path coordinates must map onto the atom table.
    #[error("Debye spring path leg {leg} did not match any atom coordinate")]
    UnmatchedSpringPathAtom { leg: usize },
    /// FEFF spring/RM path setup produced a zero reduced-mass denominator.
    #[error("Debye spring path has zero reduced-mass denominator")]
    ZeroSpringReducedMassDenominator,
    /// FEFF spring/RM characteristic frequency setup failed.
    #[error("Debye spring characteristic frequency must be positive, got {value}")]
    NonPositiveSpringFrequency { value: Real },
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
    /// DMDW PDS and a2f coupling tables must have matching lengths.
    #[error(
        "DMDW coupling tables have lengths pds_energy={pds_energy}, phonon_dos={phonon_dos}, a2f_energy={a2f_energy}, eliashberg={eliashberg}"
    )]
    InvalidDmdwCouplingTableShape {
        pds_energy: usize,
        phonon_dos: usize,
        a2f_energy: usize,
        eliashberg: usize,
    },
    /// DMDW phonon-coupling tables must contain at least one point.
    #[error("DMDW coupling table must contain at least one point")]
    EmptyDmdwCouplingTable,
    /// DMDW PDS and a2f energy grids must match row by row.
    #[error(
        "DMDW coupling energy row {row} differs between PDS ({pds_energy}) and a2f ({a2f_energy})"
    )]
    MismatchedDmdwCouplingEnergyGrid {
        row: usize,
        pds_energy: Real,
        a2f_energy: Real,
    },
    /// DMDW PDS values are divisors in the type-2 coupling transform.
    #[error("DMDW phonon density row {row} must be positive, got {value}")]
    NonPositiveDmdwPhononDensity { row: usize, value: Real },
    /// DMDW type-2 pole-weight `a2f` matching did not cover a Lanczos pole.
    #[error(
        "DMDW a2f diagnostic pole {pole_index} at {frequency_thz} THz was not covered by {coupling_points} coupling grid point(s)"
    )]
    UnmatchedDmdwA2fPole {
        pole_index: usize,
        frequency_thz: Real,
        coupling_points: usize,
    },
    /// DMDW type-2 `.dym` metadata must identify at least one unique atom.
    #[error("DMDW type 2 unique-atom metadata must contain at least one group")]
    EmptyDmdwType2UniqueAtomTable,
    /// DMDW type-2 unique atom groups need at least one central atom.
    #[error("DMDW type 2 unique atom group {group} contains no center atoms")]
    EmptyDmdwType2CenterAtomGroup { group: usize },
    /// DMDW type-2 central atom indices must refer to the `.dym` atom table.
    #[error("DMDW type 2 center atom index {index} in group {group} is outside 0..{atom_count}")]
    InvalidDmdwType2CenterAtomIndex {
        group: usize,
        index: usize,
        atom_count: usize,
    },
    /// DMDW type-2 displacement option must be all directions or one axis.
    #[error("DMDW type 2 displacement option {option} is outside 0..=3")]
    InvalidDmdwType2DisplacementOption { option: i32 },
    /// DMDW self-energy pole energies and weights must have matching lengths.
    #[error("DMDW self-energy pole table has {energies} energies but {weights} weights")]
    InvalidDmdwSelfEnergyPoleTableShape { energies: usize, weights: usize },
    /// DMDW self-energy energy grids must contain at least one point.
    #[error("DMDW self-energy energy grid must contain at least one point")]
    EmptyDmdwSelfEnergyGrid,
    /// DMDW spectral function needs at least two uniformly spaced energies.
    #[error("DMDW spectral energy grid needs at least two points, got {points}")]
    InvalidDmdwSpectralEnergyGrid { points: usize },
    /// DMDW spectral function energy grid must be uniformly increasing.
    #[error("DMDW spectral energy grid step at row {row} is {step}, expected {expected_step}")]
    NonUniformDmdwSpectralEnergyGrid {
        row: usize,
        step: Real,
        expected_step: Real,
    },
    /// DMDW spectral function uses FEFF's odd time grid around zero.
    #[error("DMDW spectral time grid needs an odd point count >= 3, got {points}")]
    InvalidDmdwSpectralTimeGrid { points: usize },
    /// DMDW self-energy complex values must be finite.
    #[error("DMDW complex value {name} must be finite, got {value:?}")]
    NonFiniteComplex { name: &'static str, value: Complex },
    /// DMDW self-energy special-function evaluation failed.
    #[error("DMDW self-energy special function failed: {source}")]
    DmdwSpecialFunction {
        #[from]
        source: SpecialFunctionError,
    },
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
