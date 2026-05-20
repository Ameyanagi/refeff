//! FEFF EELS numerical helpers.
//!
//! This module ports the small kernels from `EELS/wavelength.f90`,
//! `EELS/euler.f90`, `EELS/productmatvect.f90`, `EELS/qmesh.f90`, and the
//! `EELS/readsp.f90` spectrum assembly, and spectrum/angular/GOS accumulation
//! loops in `EELS/eels.f90`,
//! `EELS/writeangulardependence1.f90`, `EELS/writeangulardependence2.f90`, and
//! `EELS/writeangulardependence3.f90`. The functions keep FEFF's constants and
//! matrix convention while validating inputs instead of producing NaN/Inf
//! outputs.

use ndarray::{Array1, Array2, Array3, ArrayView1, ArrayView2, ArrayView3, ShapeBuilder};
use thiserror::Error;

use crate::{Real, RealMat, RealVec};

/// FEFF electron rest energy `m_e c^2` in eV, from `COMMON/m_constants.f90`.
pub const FEFF_ELECTRON_REST_ENERGY_EV: Real = 511_004.0;
/// FEFF `HOnSqrtTwoMe` constant for electron wavelengths in atomic units.
pub const FEFF_H_ON_SQRT_TWO_ME: Real = 23.1761;
/// FEFF `hbarc_eV`, `hbar*c` in eV atomic-radius units.
pub const FEFF_HBARC_EV: Real = 1973.2708 / 0.529177;
/// FEFF `hbarc_atomic`, `hbar*c` in Hartree atomic units.
pub const FEFF_HBARC_ATOMIC: Real = 137.04188;
/// FEFF hardcoded GOS q-grid count from `EELS/writeangulardependence3.f90`.
pub const FEFF_EELS_GOS_Q_COUNT: usize = 20;
/// FEFF tensor component count from `EELS/readsp.f90` columns `s(:,2:10)`.
pub const FEFF_EELS_TRANSITION_TENSOR_COMPONENT_COUNT: usize = 9;
/// FEFF angular-dependence table column count from `EELS/writeangulardependence1.f90`.
pub const FEFF_EELS_ANGULAR_DEPENDENCE_COLUMN_COUNT: usize = 9;
/// FEFF collection-angle dependence table column count from `EELS/writeangulardependence2.f90`.
pub const FEFF_EELS_COLLECTION_DEPENDENCE_COLUMN_COUNT: usize = 5;

const FEFF_EELS_GOS_Q_BASE: Real = 0.44;
const FEFF_EELS_GOS_Q_STEP_SEED: Real = 0.1950;
const FEFF_EELS_GOS_EDGE_PARAMETER: Real = 100.0;
const FEFF_EELS_GOS_ENERGY_START_EV: Real = 100.0;
const FEFF_EELS_GOS_ENERGY_STEP_EV: Real = 10.0;
const FEFF_EELS_GOS_A0: Real = 0.529177;
const FEFF_EELS_GOS_RYDBERG_EV: Real = 13.6;
const FEFF_EELS_ANGULAR_DEPENDENCE_PARTIAL_COUNT: usize = 10;

/// Error returned by FEFF EELS helper routines.
#[derive(Debug, Clone, Copy, PartialEq, Error)]
pub enum EelsError {
    /// Scalar EELS inputs must be finite real values.
    #[error("EELS input {name} must be finite, got {value}")]
    NonFiniteInput { name: &'static str, value: Real },
    /// FEFF EELS wavelength calculation requires a positive beam energy.
    #[error("EELS beam energy must be positive, got {value}")]
    InvalidBeamEnergy { value: Real },
    /// A result became non-finite after evaluating the FEFF formula.
    #[error("EELS result {name} must be finite, got {value}")]
    NonFiniteResult { name: &'static str, value: Real },
    /// FEFF `ProductMatVect` only accepts a `3 x 3` matrix.
    #[error("EELS matrix must have shape (3, 3), got ({rows}, {columns})")]
    InvalidMatrixShape { rows: usize, columns: usize },
    /// FEFF `ProductMatVect` only accepts a 3-vector.
    #[error("EELS vector must have length 3, got {length}")]
    InvalidVectorLength { length: usize },
    /// FEFF `QMesh` needs aligned angular coordinate arrays.
    #[error("EELS q-mesh theta_x length {theta_x_len} does not match theta_y length {theta_y_len}")]
    QMeshLengthMismatch {
        theta_x_len: usize,
        theta_y_len: usize,
    },
    /// FEFF `eels.f90` requires spectrum arrays aligned with the energy grid.
    #[error("EELS spectrum {name} length {actual} does not match energy length {expected}")]
    SpectrumLengthMismatch {
        name: &'static str,
        expected: usize,
        actual: usize,
    },
    /// FEFF EELS transition tensors are 3 by 3 for each energy point.
    #[error(
        "EELS transition tensor has shape ({energies}, {rows}, {columns}), expected ({expected_energies}, 3, 3)"
    )]
    InvalidSpectrumTensorShape {
        expected_energies: usize,
        energies: usize,
        rows: usize,
        columns: usize,
    },
    /// FEFF EELS energy losses must leave a positive scattered beam energy.
    #[error(
        "EELS energy loss[{index}] must be positive and below incident energy {incident_energy_ev}, got {value}"
    )]
    InvalidEnergyLoss {
        index: usize,
        value: Real,
        incident_energy_ev: Real,
    },
    /// FEFF EELS wave numbers must be positive finite values.
    #[error("EELS wave number must be positive and finite, got {value}")]
    InvalidWaveNumber { value: Real },
    /// FEFF EELS input tables must have the expected dimensions.
    #[error(
        "EELS table {name} has shape ({rows}, {columns}), expected ({expected_rows}, {expected_columns})"
    )]
    InvalidTableShape {
        name: &'static str,
        rows: usize,
        columns: usize,
        expected_rows: usize,
        expected_columns: usize,
    },
    /// FEFF EELS angular-dependence weights must be positive finite values.
    #[error("EELS integration weight[{index}] must be positive and finite, got {value}")]
    InvalidWeight { index: usize, value: Real },
    /// FEFF EELS scattering-angle denominator became singular.
    #[error("EELS scattering-angle denominator is singular at position {position}")]
    SingularScatteringAngle { position: usize },
    /// FEFF EELS q-dependent denominator became singular.
    #[error("EELS q-factor is singular at energy index {energy_index}, position {position}")]
    SingularQFactor {
        energy_index: usize,
        position: usize,
    },
    /// FEFF EELS mesh counts must be positive.
    #[error("EELS mesh count {name} must be positive, got {value}")]
    InvalidMeshCount { name: &'static str, value: usize },
    /// FEFF EELS angular widths must be nonnegative finite values.
    #[error("EELS mesh angle {name} must be nonnegative and finite, got {value}")]
    InvalidMeshAngle { name: &'static str, value: Real },
    /// FEFF EELS logarithmic meshes need positive finite scale parameters.
    #[error("EELS logarithmic mesh parameter {name} must be positive and finite, got {value}")]
    InvalidLogMeshParameter { name: &'static str, value: Real },
    /// FEFF EELS mesh dimensions overflowed `usize`.
    #[error("EELS mesh point count overflows usize")]
    MeshSizeOverflow,
    /// The generated mesh did not match FEFF's expected point count.
    #[error("EELS mesh generated {actual} points but expected {expected}")]
    MeshSizeMismatch { expected: usize, actual: usize },
    /// FEFF EELS polarization controls are one-based and bounded by files 1..10.
    #[error(
        "EELS polarization range must satisfy 1 <= min <= max <= 10 and step > 0, got min={min}, step={step}, max={max}"
    )]
    InvalidPolarizationRange { min: usize, step: usize, max: usize },
    /// FEFF `readsp` accepts polarization file indices 1..10.
    #[error("EELS polarization source index must be in 1..=10, got {value}")]
    InvalidPolarizationIndex { value: usize },
    /// FEFF `readsp` expects at most one source spectrum per polarization index.
    #[error("EELS duplicate polarization source {index}")]
    DuplicatePolarizationSource { index: usize },
    /// FEFF `readsp` could not find a requested source spectrum.
    #[error("EELS missing polarization source {index}")]
    MissingPolarizationSource { index: usize },
    /// FEFF `readsp` source columns must align.
    #[error(
        "EELS polarization source {polarization_index} field {name} length {actual} does not match energy length {expected}"
    )]
    ReadSpectrumLengthMismatch {
        polarization_index: usize,
        name: &'static str,
        expected: usize,
        actual: usize,
    },
    /// FEFF `readsp` assumes every source uses the same energy grid.
    #[error(
        "EELS polarization source {polarization_index} energy row {row} got {actual}, expected {expected}"
    )]
    ReadSpectrumEnergyMismatch {
        polarization_index: usize,
        row: usize,
        expected: Real,
        actual: Real,
    },
}

/// FEFF EELS q-mesh sampling mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EelsMeshMode {
    /// FEFF `qmodus = 'U'`: uniform radial rings.
    Uniform,
    /// FEFF `qmodus = 'L'`: logarithmic radial rings.
    Logarithmic,
    /// FEFF `qmodus = '1'`: one-dimensional logarithmic radial mesh.
    OneDimensional,
}

/// Inputs for FEFF `EELS/angularmesh.f90` and `EELS/calculateweights.f90`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EelsMeshInput {
    /// Collection semiangle `acoll`, in radians.
    pub collection_angle: Real,
    /// Convergence semiangle `aconv`, in radians.
    pub convergence_angle: Real,
    /// FEFF logarithmic mesh inner angle `th0`, in radians.
    pub theta0: Real,
    /// Detector x-center, matching FEFF `ThetaXCenter`.
    pub theta_x_center: Real,
    /// Detector y-center, matching FEFF `ThetaYCenter`.
    pub theta_y_center: Real,
    /// FEFF radial mesh count `nqr`.
    pub radial_count: usize,
    /// FEFF angular mesh factor `nqf`.
    pub angular_count: usize,
    /// FEFF q-mesh mode.
    pub mode: EelsMeshMode,
}

/// FEFF EELS mesh metadata after `init_work` adjustments.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EelsMeshSetup {
    /// FEFF `nqr` after any zero-angle reset.
    pub radial_count: usize,
    /// FEFF `nqf` after mode-specific adjustment.
    pub angular_count: usize,
    /// FEFF `npos`.
    pub point_count: usize,
    /// FEFF `ThPart`.
    pub theta_part: Real,
    /// FEFF q-mesh mode.
    pub mode: EelsMeshMode,
}

/// FEFF EELS angular sample coordinates.
#[derive(Debug, Clone, PartialEq)]
pub struct EelsAngularMesh {
    /// FEFF `ThXV`.
    pub theta_x: RealVec,
    /// FEFF `ThYV`.
    pub theta_y: RealVec,
    /// Mesh setup values used to generate the coordinates.
    pub setup: EelsMeshSetup,
}

/// FEFF EELS angular coordinates and integration weights.
#[derive(Debug, Clone, PartialEq)]
pub struct EelsIntegrationMesh {
    /// FEFF `ThXV` after recentering on the requested detector position.
    pub theta_x: RealVec,
    /// FEFF `ThYV` after recentering on the requested detector position.
    pub theta_y: RealVec,
    /// FEFF `WeightV`.
    pub weights: RealVec,
    /// Mesh setup values used to generate the coordinates and weights.
    pub setup: EelsMeshSetup,
}

/// Inputs for FEFF `EELS/qmesh.f90`.
#[derive(Debug, Clone, Copy)]
pub struct EelsQMeshInput<'a> {
    /// Incident beam-electron energy `Energy`, in eV.
    pub incident_energy_ev: Real,
    /// Scattered beam-electron energy `Energy2`, in eV.
    pub scattered_energy_ev: Real,
    /// FEFF incident beam direction `xivec`.
    pub beam_direction: [Real; 3],
    /// Detector-plane x angular samples `ThXV`.
    pub theta_x: ArrayView1<'a, Real>,
    /// Detector-plane y angular samples `ThYV`.
    pub theta_y: ArrayView1<'a, Real>,
    /// Whether to apply FEFF's relativistic q-vector shortening, `RelatQ`.
    pub relativistic: bool,
}

/// FEFF EELS q-vector mesh for one scattered-electron energy.
#[derive(Debug, Clone, PartialEq)]
pub struct EelsQMesh {
    /// Rotated q vectors as `(component, position)`, matching FEFF `QV(1:3,1,ipos)`.
    pub q_vectors: RealMat,
    /// Relativistically corrected q-vector lengths, FEFF `QLenV`.
    pub q_lengths: RealVec,
    /// Classical q-vector lengths before the relativistic z correction, FEFF `QLenVClas`.
    pub classical_q_lengths: RealVec,
    /// Euler angles used to rotate from the observer frame to the FEFF crystal frame.
    pub euler_angles: [Real; 3],
    /// Rotation matrix from FEFF `euler.f90`.
    pub rotation_matrix: RealMat,
}

/// Inputs for the FEFF `EELS/eels.f90` spectrum accumulation loop.
#[derive(Debug, Clone, Copy)]
pub struct EelsSpectrumInput<'a> {
    /// Incident beam-electron energy `ebeam`, in eV.
    pub incident_energy_ev: Real,
    /// FEFF incident beam direction `xivec`.
    pub beam_direction: [Real; 3],
    /// Angular integration mesh controls from `eels.inp`.
    pub mesh: EelsMeshInput,
    /// Energy losses `s(:,1)`, in eV.
    pub energy_loss_ev: ArrayView1<'a, Real>,
    /// FEFF tensor spectra `s(:,2:10)` as `(energy, row, column)`.
    pub transition_tensor: ArrayView3<'a, Real>,
    /// Atomic-background spectrum `s(:,11)`.
    pub atomic_background: ArrayView1<'a, Real>,
    /// Whether to apply FEFF's relativistic q-dependent denominator.
    pub relativistic: bool,
}

/// EELS spectrum rows produced by the FEFF `EELS/eels.f90` accumulation loop.
#[derive(Debug, Clone, PartialEq)]
pub struct EelsSpectrum {
    /// FEFF output column `total`.
    pub total: RealVec,
    /// FEFF output column `atomic-bg`.
    pub background: RealVec,
    /// FEFF output column `fine-struct`, equal to `total - background`.
    pub fine_structure: RealVec,
    /// FEFF partial tensor contributions `xx, xy, ..., zz` as `(energy, partial)`.
    pub partials: RealMat,
    /// Angular coordinates and weights used for q-vector integration.
    pub integration_mesh: EelsIntegrationMesh,
}

/// One already-read source spectrum for FEFF `EELS/readsp.f90`.
#[derive(Debug, Clone, Copy)]
pub struct EelsReadSpectrumSource<'a> {
    /// FEFF polarization file index `ip`, in the range `1..=10`.
    pub polarization_index: usize,
    /// Source energy grid, matching `xmufile(:,1,ip)`.
    pub energy_loss_ev: ArrayView1<'a, Real>,
    /// Source spectrum column selected by FEFF `spcol`.
    pub selected_spectrum: ArrayView1<'a, Real>,
    /// Atomic-background column `xmufile(:,5,ip)`.
    pub atomic_background: ArrayView1<'a, Real>,
}

/// Controls for FEFF `EELS/readsp.f90` spectrum assembly.
#[derive(Debug, Clone, Copy)]
pub struct EelsReadSpectrumInput<'a> {
    /// Source spectra that correspond to already-read `xmuNN.dat` or `opconsKKNN.dat` files.
    pub sources: &'a [EelsReadSpectrumSource<'a>],
    /// FEFF `aver`: orientation-average spectra onto the diagonal tensor.
    pub orientation_averaged: bool,
    /// FEFF `cross`: retain cross-term polarization files when available.
    pub cross_terms: bool,
    /// FEFF `ipmin`.
    pub polarization_min: usize,
    /// FEFF `ipstep`.
    pub polarization_step: usize,
    /// FEFF `ipmax`.
    pub polarization_max: usize,
}

/// Output of FEFF `EELS/readsp.f90` spectrum assembly.
#[derive(Debug, Clone, PartialEq)]
pub struct EelsReadSpectrum {
    /// FEFF `s(:,1)` energy grid.
    pub energy_loss_ev: RealVec,
    /// FEFF `s(:,2:10)` tensor spectra as `(energy, row, column)`.
    pub transition_tensor: Array3<Real>,
    /// FEFF `s(:,11)` atomic-background spectrum.
    pub atomic_background: RealVec,
    /// FEFF `ipsteplocal` after the cross-term compatibility adjustment.
    pub effective_polarization_step: usize,
}

/// Inputs for FEFF `EELS/writeangulardependence1.f90`.
#[derive(Debug, Clone, Copy)]
pub struct EelsAngularDependenceInput<'a> {
    /// FEFF `QVs(1:3,1:npos)` in spherical coordinates `(q, theta_q, phi_q)`.
    pub q_vectors_spherical: ArrayView2<'a, Real>,
    /// FEFF angular integration weights `WeightV(1:npos)`.
    pub weights: ArrayView1<'a, Real>,
    /// FEFF `sdlm(1:10,1:npos,1)` partial spectra for the first edge.
    pub partial_spectra: ArrayView2<'a, Real>,
    /// FEFF incident wave-number length `k0len`.
    pub incident_wave_number: Real,
}

/// FEFF angular-dependence output rows from `writeangulardependence1`.
#[derive(Debug, Clone, PartialEq)]
pub struct EelsAngularDependenceTable {
    /// `(position, column)` rows matching FEFF's file-60 output:
    /// `theta`, `pi`, `sigma`, `total`, `sigmadipole`, `totaldipole`,
    /// `monopole`, `quadrupole`, `octupole`.
    pub rows: RealMat,
}

/// Inputs for FEFF `EELS/writeangulardependence2.f90`.
#[derive(Debug, Clone, Copy)]
pub struct EelsCollectionDependenceInput<'a> {
    /// Incident beam-electron energy `ebeam`, in eV.
    pub incident_energy_ev: Real,
    /// FEFF incident beam direction `xivec`.
    pub beam_direction: [Real; 3],
    /// Original angular integration mesh controls from `eels.inp`.
    pub mesh: EelsMeshInput,
    /// FEFF `emagic` selector for the energy row used in the plot.
    pub magic_energy_ev: Real,
    /// Energy losses `s(:,1)`, in eV.
    pub energy_loss_ev: ArrayView1<'a, Real>,
    /// FEFF `s(:,2)` x-dipole spectrum.
    pub sigma_x_spectrum: ArrayView1<'a, Real>,
    /// FEFF `s(:,6)` y-dipole spectrum.
    pub sigma_y_spectrum: ArrayView1<'a, Real>,
    /// FEFF `s(:,10)` z/pi spectrum.
    pub pi_spectrum: ArrayView1<'a, Real>,
    /// Whether to use FEFF's relativistic q-dependent denominator.
    pub relativistic: bool,
}

/// FEFF collection-angle dependence output from `writeangulardependence2`.
#[derive(Debug, Clone, PartialEq)]
pub struct EelsCollectionDependenceTable {
    /// `(collection, column)` rows: `beta`, `sp2`, `pi`, `sigmadip`, `total`.
    pub rows: RealMat,
    /// FEFF `npos` used for each collection semiangle.
    pub point_counts: Array1<usize>,
    /// Zero-based energy row selected by FEFF's `emagic` logic.
    pub magic_index: usize,
    /// Energy loss at `magic_index`, in eV.
    pub magic_energy_loss_ev: Real,
}

/// Inputs for FEFF `EELS/writeangulardependence3.f90` GOS table construction.
#[derive(Debug, Clone, Copy)]
pub struct EelsGosInput<'a> {
    /// Incident beam-electron energy `ebeam`, in eV.
    pub incident_energy_ev: Real,
    /// Energy losses `s(:,1)`, in eV.
    pub energy_loss_ev: ArrayView1<'a, Real>,
    /// Orientation-averaged EELS spectrum `s(:,2)`.
    pub averaged_spectrum: ArrayView1<'a, Real>,
    /// Whether to use FEFF's relativistic q-dependent denominator.
    pub relativistic: bool,
}

/// FEFF generalized oscillator strength table from `writeangulardependence3`.
#[derive(Debug, Clone, PartialEq)]
pub struct EelsGosTable {
    /// FEFF `qq(1:nqq)` q grid.
    pub q_values: RealVec,
    /// FEFF `xq(1:nqq,1:ne)` as `(q, energy)` in Fortran-order storage.
    pub strengths: RealMat,
    /// Header value `info1_1` after FEFF's q-grid normalization.
    pub q_scale: Real,
    /// Header value `info1_2` after FEFF's q-grid normalization.
    pub q_log_step: Real,
    /// Header value `info1_3`.
    pub edge_parameter: Real,
    /// Header value `info2_1`.
    pub energy_start_ev: Real,
    /// Header value `info2_2`.
    pub energy_step_ev: Real,
}

/// Return FEFF's relativistic electron wavelength in atomic units.
///
/// This ports `EELS/wavelength.f90`:
/// `HOnSqrtTwoMe / sqrt(E + E**2 / (2 * MeC2))`, with `E` in eV.
pub fn electron_wavelength_atomic_units(energy_ev: Real) -> Result<Real, EelsError> {
    validate_finite("energy_ev", energy_ev)?;
    if energy_ev <= 0.0 {
        return Err(EelsError::InvalidBeamEnergy { value: energy_ev });
    }

    let denominator =
        (energy_ev + energy_ev * energy_ev / (2.0 * FEFF_ELECTRON_REST_ENERGY_EV)).sqrt();
    let wavelength = FEFF_H_ON_SQRT_TWO_ME / denominator;
    if !wavelength.is_finite() {
        return Err(EelsError::NonFiniteResult {
            name: "wavelength",
            value: wavelength,
        });
    }
    Ok(wavelength)
}

/// Build FEFF's EELS Euler rotation matrix.
///
/// The three angles correspond to FEFF `a`, `b`, and `g`. The returned matrix
/// is shaped `(3, 3)` in Fortran-order `ndarray` storage and preserves the
/// `E(row,column)` assignments in `EELS/euler.f90`.
pub fn eels_euler_rotation_matrix(
    alpha: Real,
    beta: Real,
    gamma: Real,
) -> Result<RealMat, EelsError> {
    validate_finite("alpha", alpha)?;
    validate_finite("beta", beta)?;
    validate_finite("gamma", gamma)?;

    let (sin_alpha, cos_alpha) = alpha.sin_cos();
    let (sin_beta, cos_beta) = beta.sin_cos();
    let (sin_gamma, cos_gamma) = gamma.sin_cos();

    let mut matrix = Array2::zeros((3, 3).f());
    matrix[(0, 0)] = cos_alpha * cos_beta * cos_gamma - sin_alpha * sin_gamma;
    matrix[(1, 0)] = sin_alpha * cos_beta * cos_gamma + cos_alpha * sin_gamma;
    matrix[(0, 1)] = -cos_alpha * cos_beta * sin_gamma - sin_alpha * cos_gamma;
    matrix[(1, 1)] = -sin_alpha * cos_beta * sin_gamma + cos_alpha * cos_gamma;
    matrix[(0, 2)] = cos_alpha * sin_beta;
    matrix[(1, 2)] = sin_alpha * sin_beta;
    matrix[(2, 2)] = cos_beta;
    matrix[(2, 0)] = -sin_beta * cos_gamma;
    matrix[(2, 1)] = sin_beta * sin_gamma;

    for &value in &matrix {
        if !value.is_finite() {
            return Err(EelsError::NonFiniteResult {
                name: "euler_matrix",
                value,
            });
        }
    }
    Ok(matrix)
}

/// Port of FEFF `EELS/productmatvect.f90`.
///
/// Multiplies a `3 x 3` matrix by a 3-vector with FEFF's row/column
/// convention: `Vout(i) = sum_k M(i,k) * Vin(k)`.
pub fn eels_product_matrix_vector(
    matrix: ArrayView2<'_, Real>,
    vector: ArrayView1<'_, Real>,
) -> Result<RealVec, EelsError> {
    let (rows, columns) = matrix.dim();
    if (rows, columns) != (3, 3) {
        return Err(EelsError::InvalidMatrixShape { rows, columns });
    }
    if vector.len() != 3 {
        return Err(EelsError::InvalidVectorLength {
            length: vector.len(),
        });
    }
    for &value in matrix.iter() {
        if !value.is_finite() {
            return Err(EelsError::NonFiniteInput {
                name: "matrix",
                value,
            });
        }
    }
    for &value in &vector {
        validate_finite("vector", value)?;
    }

    let product = Array1::from_shape_fn(3, |row| {
        (0..3)
            .map(|column| matrix[(row, column)] * vector[column])
            .sum::<Real>()
    });
    for &value in &product {
        if !value.is_finite() {
            return Err(EelsError::NonFiniteResult {
                name: "matrix_vector_product",
                value,
            });
        }
    }
    Ok(product)
}

/// Port of FEFF `EELS/qmesh.f90`.
///
/// Builds momentum-transfer vectors for one scattered-electron energy from the
/// detector-plane angular mesh. FEFF currently rotates the observer-frame
/// q-vector into a single local basis using the Euler angles implied by
/// `xivec`; this function returns the same `(3, npos)` q-vector table together
/// with the relativistic and classical q lengths.
pub fn eels_qmesh(input: EelsQMeshInput<'_>) -> Result<EelsQMesh, EelsError> {
    validate_qmesh_input(input)?;

    let euler_angles = eels_qmesh_euler_angles(input.beam_direction);
    let rotation_matrix =
        eels_euler_rotation_matrix(euler_angles[0], euler_angles[1], euler_angles[2])?;
    let incident_wave_number =
        std::f64::consts::TAU / electron_wavelength_atomic_units(input.incident_energy_ev)?;
    let scattered_wave_number =
        std::f64::consts::TAU / electron_wavelength_atomic_units(input.scattered_energy_ev)?;
    let beta = ((2.0 + input.incident_energy_ev / FEFF_ELECTRON_REST_ENERGY_EV)
        / (2.0
            + input.incident_energy_ev / FEFF_ELECTRON_REST_ENERGY_EV
            + FEFF_ELECTRON_REST_ENERGY_EV / input.incident_energy_ev))
        .sqrt();
    let relativistic_factor = if input.relativistic {
        1.0 - beta * beta
    } else {
        1.0
    };

    let position_count = input.theta_x.len();
    let mut q_vectors = Array2::<Real>::zeros((3, position_count).f());
    let mut q_lengths = Array1::<Real>::zeros(position_count);
    let mut classical_q_lengths = Array1::<Real>::zeros(position_count);

    for position in 0..position_count {
        let theta_x = input.theta_x[position];
        let theta_y = input.theta_y[position];
        let theta = theta_x.hypot(theta_y);
        let phi = eels_qmesh_phi(theta_x, theta_y);
        let mut q = [
            -scattered_wave_number * theta.sin() * phi.cos(),
            -scattered_wave_number * theta.sin() * phi.sin(),
            scattered_wave_number * theta.cos() - incident_wave_number,
        ];
        classical_q_lengths[position] = q[0].hypot(q[1]).hypot(q[2]);
        q[2] *= relativistic_factor;
        q_lengths[position] = q[0].hypot(q[1]).hypot(q[2]);

        for row in 0..3 {
            q_vectors[(row, position)] = (0..3)
                .map(|column| rotation_matrix[(row, column)] * q[column])
                .sum::<Real>();
        }
    }

    validate_finite_matrix("q_vectors", q_vectors.view())?;
    validate_finite_array("q_lengths", q_lengths.view())?;
    validate_finite_array("classical_q_lengths", classical_q_lengths.view())?;

    Ok(EelsQMesh {
        q_vectors,
        q_lengths,
        classical_q_lengths,
        euler_angles,
        rotation_matrix,
    })
}

/// Port of the FEFF `EELS/eels.f90` spectrum accumulation loop.
///
/// The input tensor corresponds to FEFF `s(:,2:10)` after `readsp`: row/column
/// order is Cartesian `xx, xy, ..., zz`. FEFF first applies the beam-energy
/// prefactor to both the tensor spectra and atomic background, then integrates
/// `q_i q_j / qfac` over the angular mesh. This function returns the same
/// total, atomic background, fine-structure, and partial tensor columns without
/// doing any file I/O.
pub fn eels_spectrum(input: EelsSpectrumInput<'_>) -> Result<EelsSpectrum, EelsError> {
    validate_spectrum_input(input)?;
    let integration_mesh = eels_integration_mesh(input.mesh)?;
    let energy_count = input.energy_loss_ev.len();
    let mut total = Array1::<Real>::zeros(energy_count);
    let mut background = Array1::<Real>::zeros(energy_count);
    let mut partials = Array2::<Real>::zeros((energy_count, 9).f());
    let incident_wavelength = electron_wavelength_atomic_units(input.incident_energy_ev)?;
    let beam_factor = (1.0 + input.incident_energy_ev / FEFF_ELECTRON_REST_ENERGY_EV).powi(2)
        / std::f64::consts::PI
        * FEFF_HBARC_ATOMIC;

    for energy_index in 0..energy_count {
        let loss = input.energy_loss_ev[energy_index];
        let scattered_energy = input.incident_energy_ev - loss;
        let prefactor = incident_wavelength / electron_wavelength_atomic_units(scattered_energy)?
            * beam_factor
            / loss;
        let qmesh = eels_qmesh(EelsQMeshInput {
            incident_energy_ev: input.incident_energy_ev,
            scattered_energy_ev: scattered_energy,
            beam_direction: input.beam_direction,
            theta_x: integration_mesh.theta_x.view(),
            theta_y: integration_mesh.theta_y.view(),
            relativistic: input.relativistic,
        })?;
        let scaled_background = input.atomic_background[energy_index] * prefactor;

        for position in 0..integration_mesh.setup.point_count {
            let classical_len = qmesh.classical_q_lengths[position];
            let qfac = if input.relativistic {
                (classical_len.powi(2) - (loss / FEFF_HBARC_EV).powi(2)).powi(2)
            } else {
                classical_len.powi(4)
            };
            if !qfac.is_finite() || qfac.abs() <= Real::MIN_POSITIVE {
                return Err(EelsError::SingularQFactor {
                    energy_index,
                    position,
                });
            }
            let weight = integration_mesh.weights[position] / qfac;
            for row in 0..3 {
                let q_row = qmesh.q_vectors[(row, position)];
                for column in 0..3 {
                    let partial_index = 3 * row + column;
                    let contribution = weight
                        * q_row
                        * qmesh.q_vectors[(column, position)]
                        * input.transition_tensor[(energy_index, row, column)]
                        * prefactor;
                    total[energy_index] += contribution;
                    partials[(energy_index, partial_index)] += contribution;
                    if row == column {
                        background[energy_index] += weight * q_row * q_row * scaled_background;
                    }
                }
            }
        }
    }

    let fine_structure = &total - &background;
    validate_finite_array("total", total.view())?;
    validate_finite_array("background", background.view())?;
    validate_finite_array("fine_structure", fine_structure.view())?;
    validate_finite_matrix("partials", partials.view())?;

    Ok(EelsSpectrum {
        total,
        background,
        fine_structure,
        partials,
        integration_mesh,
    })
}

/// Port of FEFF `EELS/readsp.f90` after file parsing.
///
/// FEFF reads `xmuNN.dat` or `opconsKKNN.dat` files into `xmufile`, then maps
/// the selected spectrum column into `s(:,2:10)`. This helper performs that
/// polarization-index reduction on already-read source columns: orientation
/// sensitive runs keep the requested tensor components, no-cross runs suppress
/// off-diagonal files when FEFF would set `ipsteplocal = 4`, and averaged runs
/// copy either file 10 or the average of files 1, 5, and 9 onto the diagonal.
pub fn eels_read_spectrum(input: EelsReadSpectrumInput<'_>) -> Result<EelsReadSpectrum, EelsError> {
    let sources = validate_read_spectrum_input(input)?;
    let reference = read_spectrum_source(&sources, input.polarization_min)?;
    validate_read_spectrum_energy_grids(&sources, reference.energy_loss_ev)?;

    let energy_count = reference.energy_loss_ev.len();
    let mut transition_tensor = Array3::<Real>::zeros((energy_count, 3, 3).f());
    let effective_step = if input.orientation_averaged {
        assemble_averaged_read_spectrum(&sources, input, &mut transition_tensor)?
    } else {
        assemble_sensitive_read_spectrum(&sources, input, &mut transition_tensor)?
    };

    validate_finite_tensor("read_spectrum_tensor", transition_tensor.view())?;
    Ok(EelsReadSpectrum {
        energy_loss_ev: reference.energy_loss_ev.to_owned(),
        transition_tensor,
        atomic_background: reference.atomic_background.to_owned(),
        effective_polarization_step: effective_step,
    })
}

/// Port of FEFF `EELS/writeangulardependence1.f90`.
///
/// FEFF removes the angular integration weights from `sdlm` partial spectra and
/// maps each spherical q-vector to a small-angle scattering angle in mrad. This
/// function returns the same nine output columns without doing file I/O.
pub fn eels_angular_dependence(
    input: EelsAngularDependenceInput<'_>,
) -> Result<EelsAngularDependenceTable, EelsError> {
    validate_angular_dependence_input(input)?;

    let position_count = input.weights.len();
    let mut rows =
        Array2::<Real>::zeros((position_count, FEFF_EELS_ANGULAR_DEPENDENCE_COLUMN_COUNT).f());
    for position in 0..position_count {
        let weight = input.weights[position];
        let q = input.q_vectors_spherical[(0, position)];
        let theta_q = input.q_vectors_spherical[(1, position)];
        let denominator = input.incident_wave_number.powi(2) + q.powi(2)
            - 2.0 * input.incident_wave_number * q * theta_q.cos();
        if !denominator.is_finite() || denominator <= 0.0 {
            return Err(EelsError::SingularScatteringAngle { position });
        }

        let pi_component = input.partial_spectra[(2, position)] / weight;
        let sigma_dipole =
            (input.partial_spectra[(1, position)] + input.partial_spectra[(3, position)]) / weight;
        let sigma = (input.partial_spectra[(1, position)]
            + input.partial_spectra[(3, position)]
            + input.partial_spectra[(0, position)])
            / weight;
        let quadrupole = (4..=8)
            .map(|partial| input.partial_spectra[(partial, position)])
            .sum::<Real>()
            / weight;
        let octupole = input.partial_spectra[(9, position)] / weight;
        let monopole = sigma - sigma_dipole;
        let total_dipole = pi_component + sigma_dipole;
        let total = pi_component + sigma + quadrupole;

        rows[(position, 0)] = q * theta_q.sin() * -1000.0 / denominator.sqrt();
        rows[(position, 1)] = pi_component;
        rows[(position, 2)] = sigma;
        rows[(position, 3)] = total;
        rows[(position, 4)] = sigma_dipole;
        rows[(position, 5)] = total_dipole;
        rows[(position, 6)] = monopole;
        rows[(position, 7)] = quadrupole;
        rows[(position, 8)] = octupole;
    }

    validate_finite_matrix("angular_dependence", rows.view())?;
    Ok(EelsAngularDependenceTable { rows })
}

/// Port of FEFF `EELS/writeangulardependence2.f90`.
///
/// FEFF builds q-vectors once on the original full mesh, then recomputes only
/// the integration weights while sweeping the collection semiangle. The output
/// is the five floating-point columns that FEFF writes to file 59, plus the
/// integer `npos` column as metadata.
pub fn eels_collection_angle_dependence(
    input: EelsCollectionDependenceInput<'_>,
) -> Result<EelsCollectionDependenceTable, EelsError> {
    validate_collection_dependence_input(input)?;

    let magic_index = eels_magic_energy_index(
        input.energy_loss_ev,
        input.sigma_x_spectrum,
        input.magic_energy_ev,
    );
    let magic_energy_loss = input.energy_loss_ev[magic_index];
    if magic_energy_loss <= 0.0 || magic_energy_loss >= input.incident_energy_ev {
        return Err(EelsError::InvalidEnergyLoss {
            index: magic_index,
            value: magic_energy_loss,
            incident_energy_ev: input.incident_energy_ev,
        });
    }

    let collections = eels_collection_sweep(input.mesh)?;
    let original_mesh = eels_angular_mesh(input.mesh)?;
    let qmesh = eels_qmesh(EelsQMeshInput {
        incident_energy_ev: input.incident_energy_ev,
        scattered_energy_ev: input.incident_energy_ev - magic_energy_loss,
        beam_direction: input.beam_direction,
        theta_x: original_mesh.theta_x.view(),
        theta_y: original_mesh.theta_y.view(),
        relativistic: input.relativistic,
    })?;

    let mut rows = Array2::<Real>::zeros(
        (
            collections.len(),
            FEFF_EELS_COLLECTION_DEPENDENCE_COLUMN_COUNT,
        )
            .f(),
    );
    let mut point_counts = Vec::with_capacity(collections.len());
    for (collection_index, &collection_angle) in collections.iter().enumerate() {
        let radial_count = collection_index + 1;
        let (weights, setup) =
            eels_collection_sweep_weights(input.mesh, collection_angle, radial_count)?;
        if setup.point_count > qmesh.classical_q_lengths.len() {
            return Err(EelsError::MeshSizeMismatch {
                expected: setup.point_count,
                actual: qmesh.classical_q_lengths.len(),
            });
        }

        let mut pi_component = 0.0;
        let mut sigma_dipole = 0.0;
        for position in 0..setup.point_count {
            let classical_len = qmesh.classical_q_lengths[position];
            let qfac = if input.relativistic {
                (classical_len.powi(2) - (magic_energy_loss / FEFF_HBARC_EV).powi(2)).powi(2)
            } else {
                classical_len.powi(4)
            };
            if !qfac.is_finite() || qfac.abs() <= Real::MIN_POSITIVE {
                return Err(EelsError::SingularQFactor {
                    energy_index: magic_index,
                    position,
                });
            }
            let weight = weights[position] / qfac;
            let qx = qmesh.q_vectors[(0, position)];
            let qy = qmesh.q_vectors[(1, position)];
            let qz = qmesh.q_vectors[(2, position)];
            pi_component += weight * qz * qz * input.pi_spectrum[magic_index];
            sigma_dipole += weight
                * (qx * qx * input.sigma_x_spectrum[magic_index]
                    + qy * qy * input.sigma_y_spectrum[magic_index]);
        }
        let total = pi_component + sigma_dipole;
        rows[(collection_index, 0)] = collection_angle;
        rows[(collection_index, 1)] = if total.abs() > 0.0 {
            pi_component / total
        } else {
            0.0
        };
        rows[(collection_index, 2)] = pi_component;
        rows[(collection_index, 3)] = sigma_dipole;
        rows[(collection_index, 4)] = total;
        point_counts.push(setup.point_count);
    }

    validate_finite_matrix("collection_dependence", rows.view())?;
    Ok(EelsCollectionDependenceTable {
        rows,
        point_counts: Array1::from_vec(point_counts),
        magic_index,
        magic_energy_loss_ev: magic_energy_loss,
    })
}

/// Port of FEFF `EELS/writeangulardependence3.f90`.
///
/// FEFF uses this path to write `gos1.txt` and `gos2.txt` for an
/// orientation-averaged EELS calculation. The q-grid defaults and prefactor are
/// intentionally the hardcoded values from the reference routine. File-format
/// rendering is left to the caller; this function returns the q grid and
/// generalized oscillator strength matrix.
pub fn eels_generalized_oscillator_strength(
    input: EelsGosInput<'_>,
) -> Result<EelsGosTable, EelsError> {
    validate_gos_input(input)?;

    let (q_values, q_scale, q_log_step) = eels_gos_q_grid()?;
    let energy_count = input.energy_loss_ev.len();
    let mut strengths = Array2::<Real>::zeros((FEFF_EELS_GOS_Q_COUNT, energy_count).f());
    let gamma = 1.0 + input.incident_energy_ev / FEFF_ELECTRON_REST_ENERGY_EV;
    let beam_factor = input.incident_energy_ev * (1.0 + gamma)
        / (2.0 * gamma.powi(2))
        / (4.0 * std::f64::consts::PI * FEFF_EELS_GOS_RYDBERG_EV.powi(2))
        * 1000.0;

    for energy_index in 0..energy_count {
        let loss = input.energy_loss_ev[energy_index];
        let prefactor = loss * beam_factor;
        for q_index in 0..FEFF_EELS_GOS_Q_COUNT {
            let q = q_values[q_index];
            let qfac = if input.relativistic {
                (q.powi(2) - (loss / FEFF_HBARC_EV).powi(2)).powi(2)
            } else {
                q.powi(4)
            };
            if !qfac.is_finite() || qfac.abs() <= Real::MIN_POSITIVE {
                return Err(EelsError::SingularQFactor {
                    energy_index,
                    position: q_index,
                });
            }
            strengths[(q_index, energy_index)] =
                q.powi(2) / qfac * input.averaged_spectrum[energy_index] * prefactor;
        }
    }

    validate_finite_matrix("gos_strengths", strengths.view())?;
    Ok(EelsGosTable {
        q_values,
        strengths,
        q_scale,
        q_log_step,
        edge_parameter: FEFF_EELS_GOS_EDGE_PARAMETER,
        energy_start_ev: FEFF_EELS_GOS_ENERGY_START_EV,
        energy_step_ev: FEFF_EELS_GOS_ENERGY_STEP_EV,
    })
}

/// Return FEFF EELS mesh metadata after `init_work` rules are applied.
pub fn eels_mesh_setup(input: EelsMeshInput) -> Result<EelsMeshSetup, EelsError> {
    validate_mesh_inputs(input)?;

    let mut radial_count = input.radial_count;
    let mut angular_count = input.angular_count;
    let angle_sum = input.collection_angle + input.convergence_angle;
    let theta_part = if input.collection_angle > 1.0e-6 || input.convergence_angle > 1.0e-6 {
        angle_sum / (2.0 * radial_count as Real)
    } else if radial_count
        .checked_add(angular_count)
        .ok_or(EelsError::MeshSizeOverflow)?
        > 2
    {
        radial_count = 1;
        angular_count = 1;
        0.0
    } else {
        0.0
    };

    let point_count = match input.mode {
        EelsMeshMode::Uniform | EelsMeshMode::Logarithmic => radial_count
            .checked_mul(radial_count)
            .and_then(|value| value.checked_mul(angular_count))
            .ok_or(EelsError::MeshSizeOverflow)?,
        EelsMeshMode::OneDimensional => {
            angular_count = 1;
            radial_count
        }
    };

    if matches!(
        input.mode,
        EelsMeshMode::Logarithmic | EelsMeshMode::OneDimensional
    ) && point_count > 1
    {
        if radial_count <= 1 {
            return Err(EelsError::InvalidMeshCount {
                name: "radial_count",
                value: radial_count,
            });
        }
        if input.theta0 <= 0.0 || !input.theta0.is_finite() {
            return Err(EelsError::InvalidLogMeshParameter {
                name: "theta0",
                value: input.theta0,
            });
        }
        if angle_sum <= 0.0 || !angle_sum.is_finite() {
            return Err(EelsError::InvalidLogMeshParameter {
                name: "angle_sum",
                value: angle_sum,
            });
        }
    }

    Ok(EelsMeshSetup {
        radial_count,
        angular_count,
        point_count,
        theta_part,
        mode: input.mode,
    })
}

/// Port of FEFF `EELS/angularmesh.f90`.
///
/// The returned coordinates are FEFF `ThXV` and `ThYV` after applying the
/// requested detector center and q-mesh mode.
pub fn eels_angular_mesh(input: EelsMeshInput) -> Result<EelsAngularMesh, EelsError> {
    let setup = eels_mesh_setup(input)?;
    angular_mesh_with_setup(input, setup)
}

/// Port of FEFF `EELS/calculateweights.f90`.
///
/// FEFF computes `WeightV` from a zero-centered angular mesh, then regenerates
/// `ThXV` and `ThYV` around the detector center for the rest of the EELS
/// calculation. This function returns that final centered mesh and the weights.
pub fn eels_integration_mesh(input: EelsMeshInput) -> Result<EelsIntegrationMesh, EelsError> {
    let setup = eels_mesh_setup(input)?;
    let zero_center = EelsMeshInput {
        theta_x_center: 0.0,
        theta_y_center: 0.0,
        ..input
    };
    let zero_mesh = angular_mesh_with_setup(zero_center, setup)?;
    let weights = calculate_weights(input, setup, &zero_mesh)?;
    let centered_mesh = angular_mesh_with_setup(input, setup)?;

    Ok(EelsIntegrationMesh {
        theta_x: centered_mesh.theta_x,
        theta_y: centered_mesh.theta_y,
        weights,
        setup,
    })
}

fn validate_finite(name: &'static str, value: Real) -> Result<(), EelsError> {
    if !value.is_finite() {
        return Err(EelsError::NonFiniteInput { name, value });
    }
    Ok(())
}

fn validate_finite_array(
    name: &'static str,
    values: ArrayView1<'_, Real>,
) -> Result<(), EelsError> {
    for &value in &values {
        validate_finite(name, value)?;
    }
    Ok(())
}

fn validate_finite_matrix(
    name: &'static str,
    values: ArrayView2<'_, Real>,
) -> Result<(), EelsError> {
    for &value in &values {
        validate_finite(name, value)?;
    }
    Ok(())
}

fn validate_finite_tensor(
    name: &'static str,
    values: ArrayView3<'_, Real>,
) -> Result<(), EelsError> {
    for &value in &values {
        validate_finite(name, value)?;
    }
    Ok(())
}

fn validate_qmesh_input(input: EelsQMeshInput<'_>) -> Result<(), EelsError> {
    if input.theta_x.len() != input.theta_y.len() {
        return Err(EelsError::QMeshLengthMismatch {
            theta_x_len: input.theta_x.len(),
            theta_y_len: input.theta_y.len(),
        });
    }
    for &value in &input.beam_direction {
        validate_finite("beam_direction", value)?;
    }
    validate_finite_array("theta_x", input.theta_x)?;
    validate_finite_array("theta_y", input.theta_y)?;
    Ok(())
}

fn validate_spectrum_input(input: EelsSpectrumInput<'_>) -> Result<(), EelsError> {
    validate_finite("incident_energy_ev", input.incident_energy_ev)?;
    if input.incident_energy_ev <= 0.0 {
        return Err(EelsError::InvalidBeamEnergy {
            value: input.incident_energy_ev,
        });
    }
    for &value in &input.beam_direction {
        validate_finite("beam_direction", value)?;
    }
    let energy_count = input.energy_loss_ev.len();
    if energy_count == 0 {
        return Err(EelsError::InvalidMeshCount {
            name: "energy_count",
            value: 0,
        });
    }
    let (tensor_energies, tensor_rows, tensor_columns) = input.transition_tensor.dim();
    if (tensor_energies, tensor_rows, tensor_columns) != (energy_count, 3, 3) {
        return Err(EelsError::InvalidSpectrumTensorShape {
            expected_energies: energy_count,
            energies: tensor_energies,
            rows: tensor_rows,
            columns: tensor_columns,
        });
    }
    if input.atomic_background.len() != energy_count {
        return Err(EelsError::SpectrumLengthMismatch {
            name: "atomic_background",
            expected: energy_count,
            actual: input.atomic_background.len(),
        });
    }
    for (index, &loss) in input.energy_loss_ev.iter().enumerate() {
        validate_finite("energy_loss_ev", loss)?;
        if loss <= 0.0 || loss >= input.incident_energy_ev {
            return Err(EelsError::InvalidEnergyLoss {
                index,
                value: loss,
                incident_energy_ev: input.incident_energy_ev,
            });
        }
    }
    validate_finite_tensor("transition_tensor", input.transition_tensor)?;
    validate_finite_array("atomic_background", input.atomic_background)?;
    Ok(())
}

fn validate_read_spectrum_input<'a>(
    input: EelsReadSpectrumInput<'a>,
) -> Result<[Option<EelsReadSpectrumSource<'a>>; 11], EelsError> {
    if input.polarization_step == 0
        || input.polarization_min == 0
        || input.polarization_min > input.polarization_max
        || input.polarization_max > 10
    {
        return Err(EelsError::InvalidPolarizationRange {
            min: input.polarization_min,
            step: input.polarization_step,
            max: input.polarization_max,
        });
    }

    let mut sources = [None; 11];
    for &source in input.sources {
        let index = source.polarization_index;
        if !(1..=10).contains(&index) {
            return Err(EelsError::InvalidPolarizationIndex { value: index });
        }
        if sources[index].is_some() {
            return Err(EelsError::DuplicatePolarizationSource { index });
        }

        let energy_count = source.energy_loss_ev.len();
        if energy_count == 0 {
            return Err(EelsError::InvalidMeshCount {
                name: "energy_count",
                value: 0,
            });
        }
        validate_read_spectrum_len(
            index,
            "selected_spectrum",
            source.selected_spectrum.len(),
            energy_count,
        )?;
        validate_read_spectrum_len(
            index,
            "atomic_background",
            source.atomic_background.len(),
            energy_count,
        )?;
        validate_finite_array("readsp_energy_loss_ev", source.energy_loss_ev)?;
        validate_finite_array("readsp_selected_spectrum", source.selected_spectrum)?;
        validate_finite_array("readsp_atomic_background", source.atomic_background)?;
        sources[index] = Some(source);
    }

    Ok(sources)
}

fn validate_read_spectrum_len(
    polarization_index: usize,
    name: &'static str,
    actual: usize,
    expected: usize,
) -> Result<(), EelsError> {
    if actual == expected {
        Ok(())
    } else {
        Err(EelsError::ReadSpectrumLengthMismatch {
            polarization_index,
            name,
            expected,
            actual,
        })
    }
}

fn validate_read_spectrum_energy_grids(
    sources: &[Option<EelsReadSpectrumSource<'_>>; 11],
    reference_energy: ArrayView1<'_, Real>,
) -> Result<(), EelsError> {
    for source in sources.iter().flatten() {
        if source.energy_loss_ev.len() != reference_energy.len() {
            return Err(EelsError::ReadSpectrumLengthMismatch {
                polarization_index: source.polarization_index,
                name: "energy_loss_ev",
                expected: reference_energy.len(),
                actual: source.energy_loss_ev.len(),
            });
        }
        for (row, (&expected, &actual)) in reference_energy
            .iter()
            .zip(source.energy_loss_ev.iter())
            .enumerate()
        {
            if actual != expected {
                return Err(EelsError::ReadSpectrumEnergyMismatch {
                    polarization_index: source.polarization_index,
                    row,
                    expected,
                    actual,
                });
            }
        }
    }
    Ok(())
}

fn read_spectrum_source<'a>(
    sources: &[Option<EelsReadSpectrumSource<'a>>; 11],
    index: usize,
) -> Result<EelsReadSpectrumSource<'a>, EelsError> {
    sources
        .get(index)
        .and_then(|source| *source)
        .ok_or(EelsError::MissingPolarizationSource { index })
}

fn assemble_sensitive_read_spectrum(
    sources: &[Option<EelsReadSpectrumSource<'_>>; 11],
    input: EelsReadSpectrumInput<'_>,
    transition_tensor: &mut Array3<Real>,
) -> Result<usize, EelsError> {
    if input.polarization_min != 1 || input.polarization_max != 9 {
        return Err(EelsError::InvalidPolarizationRange {
            min: input.polarization_min,
            step: input.polarization_step,
            max: input.polarization_max,
        });
    }
    if input.cross_terms && input.polarization_step != 1 {
        return Err(EelsError::InvalidPolarizationRange {
            min: input.polarization_min,
            step: input.polarization_step,
            max: input.polarization_max,
        });
    }

    let effective_step = if input.cross_terms || input.polarization_step != 1 {
        input.polarization_step
    } else {
        4
    };

    for polarization_index in
        (input.polarization_min..=input.polarization_max).step_by(effective_step)
    {
        let source = read_spectrum_source(sources, polarization_index)?;
        copy_read_spectrum_component(transition_tensor, polarization_index, source);
    }
    Ok(effective_step)
}

fn assemble_averaged_read_spectrum(
    sources: &[Option<EelsReadSpectrumSource<'_>>; 11],
    input: EelsReadSpectrumInput<'_>,
    transition_tensor: &mut Array3<Real>,
) -> Result<usize, EelsError> {
    match (input.polarization_min, input.polarization_max) {
        (10, 10) => {
            let source = read_spectrum_source(sources, 10)?;
            copy_read_spectrum_diagonal(transition_tensor, source.selected_spectrum);
        }
        (1, 9) => {
            let x = read_spectrum_source(sources, 1)?;
            let y = read_spectrum_source(sources, 5)?;
            let z = read_spectrum_source(sources, 9)?;
            for energy_index in 0..transition_tensor.dim().0 {
                let averaged = (x.selected_spectrum[energy_index]
                    + y.selected_spectrum[energy_index]
                    + z.selected_spectrum[energy_index])
                    / 3.0;
                for diagonal in 0..3 {
                    transition_tensor[(energy_index, diagonal, diagonal)] = averaged;
                }
            }
        }
        _ => {
            return Err(EelsError::InvalidPolarizationRange {
                min: input.polarization_min,
                step: input.polarization_step,
                max: input.polarization_max,
            });
        }
    }
    Ok(input.polarization_step)
}

fn copy_read_spectrum_component(
    transition_tensor: &mut Array3<Real>,
    polarization_index: usize,
    source: EelsReadSpectrumSource<'_>,
) {
    let component = polarization_index - 1;
    let row = component / 3;
    let column = component % 3;
    for (energy_index, &value) in source.selected_spectrum.iter().enumerate() {
        transition_tensor[(energy_index, row, column)] = value;
    }
}

fn copy_read_spectrum_diagonal(transition_tensor: &mut Array3<Real>, values: ArrayView1<'_, Real>) {
    for (energy_index, &value) in values.iter().enumerate() {
        for diagonal in 0..3 {
            transition_tensor[(energy_index, diagonal, diagonal)] = value;
        }
    }
}

fn validate_angular_dependence_input(
    input: EelsAngularDependenceInput<'_>,
) -> Result<(), EelsError> {
    validate_finite("incident_wave_number", input.incident_wave_number)?;
    if input.incident_wave_number <= 0.0 {
        return Err(EelsError::InvalidWaveNumber {
            value: input.incident_wave_number,
        });
    }

    let position_count = input.weights.len();
    if position_count == 0 {
        return Err(EelsError::InvalidMeshCount {
            name: "position_count",
            value: 0,
        });
    }
    let (q_rows, q_columns) = input.q_vectors_spherical.dim();
    if (q_rows, q_columns) != (3, position_count) {
        return Err(EelsError::InvalidTableShape {
            name: "q_vectors_spherical",
            rows: q_rows,
            columns: q_columns,
            expected_rows: 3,
            expected_columns: position_count,
        });
    }
    let (partial_rows, partial_columns) = input.partial_spectra.dim();
    if (partial_rows, partial_columns)
        != (FEFF_EELS_ANGULAR_DEPENDENCE_PARTIAL_COUNT, position_count)
    {
        return Err(EelsError::InvalidTableShape {
            name: "partial_spectra",
            rows: partial_rows,
            columns: partial_columns,
            expected_rows: FEFF_EELS_ANGULAR_DEPENDENCE_PARTIAL_COUNT,
            expected_columns: position_count,
        });
    }

    for (index, &weight) in input.weights.iter().enumerate() {
        if !weight.is_finite() || weight <= 0.0 {
            return Err(EelsError::InvalidWeight {
                index,
                value: weight,
            });
        }
    }
    validate_finite_matrix("q_vectors_spherical", input.q_vectors_spherical)?;
    validate_finite_matrix("partial_spectra", input.partial_spectra)?;
    Ok(())
}

fn validate_collection_dependence_input(
    input: EelsCollectionDependenceInput<'_>,
) -> Result<(), EelsError> {
    validate_finite("incident_energy_ev", input.incident_energy_ev)?;
    if input.incident_energy_ev <= 0.0 {
        return Err(EelsError::InvalidBeamEnergy {
            value: input.incident_energy_ev,
        });
    }
    validate_finite("magic_energy_ev", input.magic_energy_ev)?;
    for &value in &input.beam_direction {
        validate_finite("beam_direction", value)?;
    }
    validate_mesh_inputs(input.mesh)?;
    let energy_count = input.energy_loss_ev.len();
    if energy_count == 0 {
        return Err(EelsError::InvalidMeshCount {
            name: "energy_count",
            value: 0,
        });
    }
    for (name, spectrum) in [
        ("sigma_x_spectrum", input.sigma_x_spectrum),
        ("sigma_y_spectrum", input.sigma_y_spectrum),
        ("pi_spectrum", input.pi_spectrum),
    ] {
        if spectrum.len() != energy_count {
            return Err(EelsError::SpectrumLengthMismatch {
                name,
                expected: energy_count,
                actual: spectrum.len(),
            });
        }
        validate_finite_array(name, spectrum)?;
    }
    for (index, &loss) in input.energy_loss_ev.iter().enumerate() {
        validate_finite("energy_loss_ev", loss)?;
        if loss <= 0.0 || loss >= input.incident_energy_ev {
            return Err(EelsError::InvalidEnergyLoss {
                index,
                value: loss,
                incident_energy_ev: input.incident_energy_ev,
            });
        }
    }
    Ok(())
}

fn eels_magic_energy_index(
    energy_loss_ev: ArrayView1<'_, Real>,
    reference_spectrum: ArrayView1<'_, Real>,
    magic_energy_ev: Real,
) -> usize {
    let mut origin = -5.0;
    for (index, (&loss, &spectrum)) in energy_loss_ev
        .iter()
        .zip(reference_spectrum.iter())
        .enumerate()
    {
        if spectrum > 1.0e-6 && origin < 0.0 {
            origin = loss;
        }
        if magic_energy_ev > loss - origin && origin >= 0.0 {
            return index;
        }
    }
    energy_loss_ev.len() - 1
}

fn eels_collection_sweep(input: EelsMeshInput) -> Result<RealVec, EelsError> {
    let count = match input.mode {
        EelsMeshMode::Logarithmic | EelsMeshMode::OneDimensional => {
            if input.radial_count <= 1 {
                return Err(EelsError::InvalidMeshCount {
                    name: "radial_count",
                    value: input.radial_count,
                });
            }
            if input.collection_angle <= 0.0 || !input.collection_angle.is_finite() {
                return Err(EelsError::InvalidLogMeshParameter {
                    name: "collection_angle",
                    value: input.collection_angle,
                });
            }
            if input.theta0 <= 0.0 || !input.theta0.is_finite() {
                return Err(EelsError::InvalidLogMeshParameter {
                    name: "theta0",
                    value: input.theta0,
                });
            }
            let dx = ((input.collection_angle + input.convergence_angle) / input.theta0).ln()
                / (input.radial_count as Real - 1.0);
            if !dx.is_finite() || dx <= 0.0 {
                return Err(EelsError::InvalidLogMeshParameter {
                    name: "dx",
                    value: dx,
                });
            }
            1 + (input.collection_angle / input.theta0).ln().div_euclid(dx) as usize
        }
        EelsMeshMode::Uniform => {
            if input.collection_angle + input.convergence_angle <= 0.0 {
                return Err(EelsError::InvalidMeshAngle {
                    name: "angle_sum",
                    value: input.collection_angle + input.convergence_angle,
                });
            }
            (input.collection_angle / (input.collection_angle + input.convergence_angle)
                * input.radial_count as Real) as usize
        }
    };
    if count == 0 {
        return Err(EelsError::InvalidMeshCount {
            name: "collection_count",
            value: 0,
        });
    }

    let values = match input.mode {
        EelsMeshMode::Logarithmic | EelsMeshMode::OneDimensional => {
            let dx = ((input.collection_angle + input.convergence_angle) / input.theta0).ln()
                / (input.radial_count as Real - 1.0);
            Array1::from_shape_fn(count, |index| {
                if index == 0 {
                    input.theta0
                } else {
                    input.theta0 * (index as Real * dx).exp()
                }
            })
        }
        EelsMeshMode::Uniform => {
            let beta_step =
                (input.convergence_angle + input.collection_angle) / input.radial_count as Real;
            Array1::from_shape_fn(count, |index| beta_step * (index + 1) as Real)
        }
    };
    validate_finite_array("collection_angles", values.view())?;
    Ok(values)
}

fn eels_collection_sweep_weights(
    original: EelsMeshInput,
    collection_angle: Real,
    radial_count: usize,
) -> Result<(RealVec, EelsMeshSetup), EelsError> {
    let angular_count = if original.mode == EelsMeshMode::OneDimensional {
        1
    } else {
        original.angular_count
    };
    let point_count = match original.mode {
        EelsMeshMode::Uniform | EelsMeshMode::Logarithmic => radial_count
            .checked_mul(radial_count)
            .and_then(|value| value.checked_mul(angular_count))
            .ok_or(EelsError::MeshSizeOverflow)?,
        EelsMeshMode::OneDimensional => radial_count,
    };
    let setup = EelsMeshSetup {
        radial_count,
        angular_count,
        point_count,
        theta_part: (collection_angle + original.convergence_angle) / (2.0 * radial_count as Real),
        mode: original.mode,
    };
    let mesh = EelsMeshInput {
        collection_angle,
        radial_count,
        angular_count,
        ..original
    };
    validate_angle("collection_angle", collection_angle)?;
    let zero_mesh = angular_mesh_with_setup(
        EelsMeshInput {
            theta_x_center: 0.0,
            theta_y_center: 0.0,
            ..mesh
        },
        setup,
    )?;
    let mut weights = calculate_weights(mesh, setup, &zero_mesh)?;
    if setup.point_count == 1 {
        weights[0] = if original.convergence_angle > 1.0e-5 {
            std::f64::consts::PI
                * ((original.convergence_angle + collection_angle)
                    * original.convergence_angle.min(collection_angle)
                    / original.convergence_angle)
                    .powi(2)
        } else {
            std::f64::consts::PI * (original.convergence_angle + collection_angle).powi(2)
        };
    }
    validate_finite_array("collection_weights", weights.view())?;
    Ok((weights, setup))
}

fn validate_gos_input(input: EelsGosInput<'_>) -> Result<(), EelsError> {
    validate_finite("incident_energy_ev", input.incident_energy_ev)?;
    if input.incident_energy_ev <= 0.0 {
        return Err(EelsError::InvalidBeamEnergy {
            value: input.incident_energy_ev,
        });
    }
    let energy_count = input.energy_loss_ev.len();
    if energy_count == 0 {
        return Err(EelsError::InvalidMeshCount {
            name: "energy_count",
            value: 0,
        });
    }
    if input.averaged_spectrum.len() != energy_count {
        return Err(EelsError::SpectrumLengthMismatch {
            name: "averaged_spectrum",
            expected: energy_count,
            actual: input.averaged_spectrum.len(),
        });
    }
    for (index, &loss) in input.energy_loss_ev.iter().enumerate() {
        validate_finite("energy_loss_ev", loss)?;
        if loss <= 0.0 || loss >= input.incident_energy_ev {
            return Err(EelsError::InvalidEnergyLoss {
                index,
                value: loss,
                incident_energy_ev: input.incident_energy_ev,
            });
        }
    }
    validate_finite_array("averaged_spectrum", input.averaged_spectrum)?;
    Ok(())
}

fn eels_gos_q_grid() -> Result<(RealVec, Real, Real), EelsError> {
    let q_min = FEFF_EELS_GOS_Q_BASE * (FEFF_EELS_GOS_Q_STEP_SEED.exp() - 1.0) * FEFF_EELS_GOS_A0;
    let q_max = FEFF_EELS_GOS_Q_BASE
        * ((FEFF_EELS_GOS_Q_COUNT as Real * FEFF_EELS_GOS_Q_STEP_SEED).exp() - 1.0)
        * FEFF_EELS_GOS_A0;
    let q_log_step = ((1.0 + q_max) / (1.0 + q_min)).ln() / (FEFF_EELS_GOS_Q_COUNT as Real - 1.0);
    let q_scale = q_min / (FEFF_EELS_GOS_A0 * (q_log_step.exp() - 1.0));
    validate_finite("q_scale", q_scale)?;
    validate_finite("q_log_step", q_log_step)?;

    let q_values = Array1::from_shape_fn(FEFF_EELS_GOS_Q_COUNT, |index| {
        q_scale * (((index + 1) as Real * q_log_step).exp() - 1.0) * FEFF_EELS_GOS_A0
    });
    validate_finite_array("q_values", q_values.view())?;
    Ok((q_values, q_scale, q_log_step))
}

fn eels_qmesh_euler_angles(beam_direction: [Real; 3]) -> [Real; 3] {
    let alpha1 = if beam_direction[0].abs() < 0.0001 {
        if beam_direction[1] > 0.0001 {
            std::f64::consts::FRAC_PI_2
        } else {
            0.0
        }
    } else {
        (beam_direction[1] / beam_direction[0]).atan()
    };
    let alpha2 = if beam_direction[2].abs() < 0.0001 {
        std::f64::consts::FRAC_PI_2
    } else {
        (beam_direction[0].hypot(beam_direction[1]) / beam_direction[2]).atan()
    };
    [alpha1, alpha2, 0.0]
}

fn eels_qmesh_phi(theta_x: Real, theta_y: Real) -> Real {
    if theta_x.abs() < 0.000001 {
        if theta_y > 0.0 {
            std::f64::consts::FRAC_PI_2
        } else {
            -std::f64::consts::FRAC_PI_2
        }
    } else {
        let mut phi = (theta_y / theta_x).atan().abs();
        if theta_y < 0.0 && theta_x < 0.0 {
            phi += std::f64::consts::PI;
        } else if theta_x < 0.0 {
            phi = std::f64::consts::PI - phi;
        } else if theta_y < 0.0 {
            phi = -phi;
        }
        phi
    }
}

fn validate_mesh_inputs(input: EelsMeshInput) -> Result<(), EelsError> {
    validate_angle("collection_angle", input.collection_angle)?;
    validate_angle("convergence_angle", input.convergence_angle)?;
    validate_finite("theta0", input.theta0)?;
    validate_finite("theta_x_center", input.theta_x_center)?;
    validate_finite("theta_y_center", input.theta_y_center)?;
    if input.radial_count == 0 {
        return Err(EelsError::InvalidMeshCount {
            name: "radial_count",
            value: input.radial_count,
        });
    }
    if input.angular_count == 0 {
        return Err(EelsError::InvalidMeshCount {
            name: "angular_count",
            value: input.angular_count,
        });
    }
    Ok(())
}

fn validate_angle(name: &'static str, value: Real) -> Result<(), EelsError> {
    if value < 0.0 || !value.is_finite() {
        return Err(EelsError::InvalidMeshAngle { name, value });
    }
    Ok(())
}

fn angular_mesh_with_setup(
    input: EelsMeshInput,
    setup: EelsMeshSetup,
) -> Result<EelsAngularMesh, EelsError> {
    let mut theta_x = Vec::with_capacity(setup.point_count);
    let mut theta_y = Vec::with_capacity(setup.point_count);
    if setup.point_count == 1 {
        theta_x.push(input.theta_x_center);
        theta_y.push(input.theta_y_center);
        return Ok(EelsAngularMesh {
            theta_x: Array1::from_vec(theta_x),
            theta_y: Array1::from_vec(theta_y),
            setup,
        });
    }

    let dxx = if matches!(
        setup.mode,
        EelsMeshMode::Logarithmic | EelsMeshMode::OneDimensional
    ) {
        ((input.collection_angle + input.convergence_angle) / input.theta0).ln()
            / (setup.radial_count as Real - 1.0)
    } else {
        0.0
    };
    let exp_dxx = dxx.exp();

    for iray in 1..=setup.radial_count {
        let present_tour = if setup.mode == EelsMeshMode::OneDimensional {
            1
        } else {
            setup.angular_count * (2 * iray - 1)
        };
        let inter_angle = std::f64::consts::TAU / present_tour as Real;
        for itour in 1..=present_tour {
            let (sin_angle, cos_angle) = (inter_angle * itour as Real).sin_cos();
            let radius = if matches!(
                setup.mode,
                EelsMeshMode::Logarithmic | EelsMeshMode::OneDimensional
            ) {
                if iray == 1 {
                    input.theta0 / 2.0
                } else {
                    input.theta0 * (dxx * (iray as Real - 2.0)).exp() * (1.0 + exp_dxx) / 2.0
                }
            } else {
                setup.theta_part * (2 * iray - 1) as Real
            };
            theta_x.push(input.theta_x_center + radius * cos_angle);
            theta_y.push(input.theta_y_center + radius * sin_angle);
        }
    }

    ensure_point_count(setup.point_count, theta_x.len())?;
    Ok(EelsAngularMesh {
        theta_x: Array1::from_vec(theta_x),
        theta_y: Array1::from_vec(theta_y),
        setup,
    })
}

fn calculate_weights(
    input: EelsMeshInput,
    setup: EelsMeshSetup,
    zero_mesh: &EelsAngularMesh,
) -> Result<RealVec, EelsError> {
    let mut weights = Vec::with_capacity(setup.point_count);
    if setup.point_count == 1 {
        weights.push(1.0);
        return Ok(Array1::from_vec(weights));
    }

    let dxx = if matches!(
        setup.mode,
        EelsMeshMode::Logarithmic | EelsMeshMode::OneDimensional
    ) {
        ((input.collection_angle + input.convergence_angle) / input.theta0).ln()
            / (setup.radial_count as Real - 1.0)
    } else {
        0.0
    };
    let exp_2dxx = (2.0 * dxx).exp();
    let sa = input.collection_angle;
    let ca = input.convergence_angle;
    let mut index_pos = 0usize;

    for iray in 1..=setup.radial_count {
        let present_tour = if setup.mode == EelsMeshMode::OneDimensional {
            1
        } else {
            setup.angular_count * (2 * iray - 1)
        };
        let theta = *zero_mesh.theta_x.get(index_pos + present_tour - 1).ok_or(
            EelsError::MeshSizeMismatch {
                expected: setup.point_count,
                actual: zero_mesh.theta_x.len(),
            },
        )?;
        let convol_value = convolution_overlap_value(theta, sa, ca);
        for _ in 0..present_tour {
            let mut weight = setup.theta_part.powi(2) / present_tour as Real
                * std::f64::consts::PI
                * 4.0
                * (2 * iray - 1) as Real
                * convol_value;
            if matches!(
                setup.mode,
                EelsMeshMode::Logarithmic | EelsMeshMode::OneDimensional
            ) {
                let lfactor = if iray == 1 {
                    (setup.radial_count as Real * input.theta0 / (sa + ca)).powi(2)
                        * setup.angular_count as Real
                        / present_tour as Real
                } else {
                    (setup.radial_count as Real * input.theta0 * (dxx * (iray as Real - 2.0)).exp()
                        / (sa + ca))
                        .powi(2)
                        * (exp_2dxx - 1.0)
                        * setup.angular_count as Real
                        / present_tour as Real
                };
                weight *= lfactor;
            }
            if !weight.is_finite() {
                return Err(EelsError::NonFiniteResult {
                    name: "weight",
                    value: weight,
                });
            }
            weights.push(weight);
        }
        index_pos += present_tour;
    }

    ensure_point_count(setup.point_count, weights.len())?;
    Ok(Array1::from_vec(weights))
}

fn convolution_overlap_value(theta: Real, collection_angle: Real, convergence_angle: Real) -> Real {
    let sa = collection_angle;
    let ca = convergence_angle;
    if theta <= (sa - ca).abs() {
        if ca > 1.0e-6 && sa > 1.0e-6 {
            sa.min(ca).powi(2) / ca.powi(2)
        } else {
            1.0
        }
    } else if theta >= sa + ca {
        0.0
    } else {
        let p = (theta * theta + ca * ca - sa * sa) / (2.0 * theta);
        let value = std::f64::consts::PI / 2.0 * (ca * ca + sa * sa)
            - p * (ca * ca - p * p).sqrt()
            - (theta - p) * (sa * sa - (theta - p) * (theta - p)).sqrt()
            - sa * sa * ((theta - p) / sa).asin()
            - ca * ca * (p / ca).asin();
        value / (std::f64::consts::PI * ca * ca)
    }
}

fn ensure_point_count(expected: usize, actual: usize) -> Result<(), EelsError> {
    if expected != actual {
        return Err(EelsError::MeshSizeMismatch { expected, actual });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{Array2, Array3, ArrayView1, ArrayView2, ArrayView3, arr1, arr2};

    #[test]
    fn electron_wavelength_matches_feff_reference() -> Result<(), EelsError> {
        assert_close(
            electron_wavelength_atomic_units(1_000.0)?,
            0.732_534_340_476_640,
        );
        assert_close(
            electron_wavelength_atomic_units(100_000.0)?,
            0.069_947_069_983_283,
        );
        assert_close(
            electron_wavelength_atomic_units(300_000.0)?,
            0.037_204_017_054_112,
        );
        Ok(())
    }

    #[test]
    fn eels_euler_rotation_matrix_matches_feff_reference() -> Result<(), EelsError> {
        assert_matrix_close(
            eels_euler_rotation_matrix(0.3, 0.4, -0.2)?.view(),
            arr2(&[
                [
                    0.921_094_097_834_994,
                    -0.114_815_729_042_654,
                    0.372_025_551_942_260,
                ],
                [
                    0.076_970_353_575_606,
                    0.990_369_592_951_021,
                    0.115_080_988_996_769,
                ],
                [
                    -0.381_655_902_095_048,
                    -0.077_365_481_465_782,
                    0.921_060_994_002_885,
                ],
            ])
            .view(),
        );
        assert_matrix_close(
            eels_euler_rotation_matrix(-1.1, 0.75, 1.4)?.view(),
            arr2(&[
                [
                    0.934_650_656_964_861,
                    -0.175_586_157_235_345,
                    0.309_188_697_759_924,
                ],
                [
                    0.336_162_895_167_387,
                    0.719_694_907_282_947,
                    -0.607_481_479_835_946,
                ],
                [
                    -0.115_856_192_531_229,
                    0.671_720_732_014_663,
                    0.731_688_868_873_821,
                ],
            ])
            .view(),
        );
        Ok(())
    }

    #[test]
    fn eels_euler_rotation_matrix_uses_fortran_order_storage() -> Result<(), EelsError> {
        let matrix = eels_euler_rotation_matrix(0.3, 0.4, -0.2)?;
        let mut expected = Vec::new();
        for column in 0..3 {
            for row in 0..3 {
                expected.push(matrix[(row, column)]);
            }
        }
        assert_eq!(matrix.as_slice_memory_order(), Some(expected.as_slice()));
        Ok(())
    }

    #[test]
    fn eels_product_matrix_vector_matches_feff_reference() -> Result<(), EelsError> {
        let first_matrix = arr2(&[[1.25, 2.0, -0.25], [-0.5, 0.125, 3.0], [0.75, -1.5, 0.5]]);
        let first_vector = arr1(&[0.2, -1.5, 4.0]);
        assert_vector_close(
            eels_product_matrix_vector(first_matrix.view(), first_vector.view())?.view(),
            arr1(&[-3.75, 11.7125, 4.4]).view(),
        );

        let second_matrix = arr2(&[[0.0, -3.5, 2.25], [1.0, 0.25, -0.75], [-2.0, 4.0, 0.5]]);
        let second_vector = arr1(&[-2.0, 0.5, 1.25]);
        assert_vector_close(
            eels_product_matrix_vector(second_matrix.view(), second_vector.view())?.view(),
            arr1(&[1.0625, -2.8125, 6.625]).view(),
        );
        Ok(())
    }

    #[test]
    fn eels_qmesh_matches_feff_reference() -> Result<(), EelsError> {
        let theta_x = arr1(&[0.0, 0.0015, -0.002, -0.001]);
        let theta_y = arr1(&[0.0, -0.0025, 0.001, -0.003]);
        let relativistic = eels_qmesh(EelsQMeshInput {
            incident_energy_ev: 300_000.0,
            scattered_energy_ev: 299_880.0,
            beam_direction: [0.2, 0.3, 0.9],
            theta_x: theta_x.view(),
            theta_y: theta_y.view(),
            relativistic: true,
        })?;
        assert_vector_close(
            arr1(&relativistic.euler_angles).view(),
            arr1(&[0.982793723247329, 0.38103799535731686, 0.0]).view(),
        );
        assert_rect_matrix_close(
            relativistic.q_vectors.view(),
            arr2(&[
                [
                    -0.003394132349274313,
                    -0.4850774133237736,
                    0.31093731394495,
                    -0.337980548857258,
                ],
                [
                    -0.00509119852391147,
                    0.03334859520894076,
                    0.16201990728101914,
                    0.40618660665810785,
                ],
                [
                    -0.015273595571734404,
                    0.07864696951859113,
                    -0.14100925915007487,
                    -0.07837471841393498,
                ],
            ])
            .view(),
        );
        assert_vector_close(
            relativistic.q_lengths.view(),
            arr1(&[
                0.016453667022982246,
                0.49254194900916926,
                0.3779101410715296,
                0.534191919932102,
            ])
            .view(),
        );
        assert_vector_close(
            relativistic.classical_q_lengths.view(),
            arr1(&[
                0.04144385038566156,
                0.4940596914878048,
                0.37985861484381056,
                0.5356000588077447,
            ])
            .view(),
        );

        let classical = eels_qmesh(EelsQMeshInput {
            incident_energy_ev: 100_000.0,
            scattered_energy_ev: 99_800.0,
            beam_direction: [0.0, 1.0, 0.0],
            theta_x: theta_x.view(),
            theta_y: theta_y.view(),
            relativistic: false,
        })?;
        assert_vector_close(
            arr1(&classical.euler_angles).view(),
            arr1(&[
                std::f64::consts::FRAC_PI_2,
                std::f64::consts::FRAC_PI_2,
                0.0,
            ])
            .view(),
        );
        assert_rect_matrix_close(
            classical.q_vectors.view(),
            arr2(&[
                [
                    -5.992870093576868e-18,
                    -0.22432428648555633,
                    0.08972976693659487,
                    -0.26918907648534857,
                ],
                [
                    -0.09787099591081017,
                    -0.09825234746796241,
                    -0.09809532042162061,
                    -0.09831964474550145,
                ],
                [
                    -5.992870093576868e-18,
                    0.13459457189133378,
                    -0.1794595338731897,
                    -0.08972969216178289,
                ],
            ])
            .view(),
        );
        assert_vector_close(
            classical.q_lengths.view(),
            arr1(&[
                0.09787099591081017,
                0.2794469682655916,
                0.22333796645688922,
                0.30030139709525955,
            ])
            .view(),
        );
        assert_vector_close(
            classical.q_lengths.view(),
            classical.classical_q_lengths.view(),
        );
        Ok(())
    }

    #[test]
    fn eels_spectrum_matches_feff_reference() -> Result<(), EelsError> {
        let energy_loss = arr1(&[12.5, 28.0, 64.0]);
        let transition_tensor = Array3::from_shape_fn((3, 3, 3), |(energy, row, column)| {
            let i = (energy + 1) as Real;
            let j1 = (row + 1) as Real;
            let j2 = (column + 1) as Real;
            0.015 * i + 0.11 * j1 - 0.045 * j2 + 0.002 * i * j1 * j2
        });
        let atomic_background = arr1(&[0.092, 0.104, 0.116]);

        let spectrum = eels_spectrum(EelsSpectrumInput {
            incident_energy_ev: 200_000.0,
            beam_direction: [0.25, -0.15, 0.95],
            mesh: EelsMeshInput {
                collection_angle: 0.014,
                convergence_angle: 0.006,
                theta0: 0.0007,
                theta_x_center: 0.0012,
                theta_y_center: -0.0008,
                radial_count: 2,
                angular_count: 2,
                mode: EelsMeshMode::Uniform,
            },
            energy_loss_ev: energy_loss.view(),
            transition_tensor: transition_tensor.view(),
            atomic_background: atomic_background.view(),
            relativistic: true,
        })?;

        assert_vector_close(
            spectrum.total.view(),
            arr1(&[
                5.330409013028863e-5,
                3.468472190648792e-5,
                1.95390880411704e-5,
            ])
            .view(),
        );
        assert_vector_close(
            spectrum.background.view(),
            arr1(&[
                5.631994485295036e-4,
                2.8415578845250556e-4,
                1.385024135506364e-4,
            ])
            .view(),
        );
        assert_vector_close(
            spectrum.fine_structure.view(),
            arr1(&[
                -5.098953583992149e-4,
                -2.4947106654601764e-4,
                -1.18963325509466e-4,
            ])
            .view(),
        );
        assert_rect_matrix_close(
            spectrum.partials.view(),
            arr2(&[
                [
                    3.628_362_866_235_717e-4,
                    -6.954606789113099e-5,
                    5.850424513429709e-6,
                    -3.45947106945626e-4,
                    1.839675708822848e-4,
                    7.416082245471633e-5,
                    -4.4755747527737183e-4,
                    1.7679410353043987e-4,
                    1.1274553223997504e-4,
                ],
                [
                    1.9510588328310328e-4,
                    -4.606423836113137e-5,
                    -1.121262380465124e-5,
                    -1.6916694432622382e-4,
                    9.436020466361983e-5,
                    4.1257244610055344e-5,
                    -2.1567811671299733e-4,
                    8.726352457090846e-5,
                    5.881978798380475e-5,
                ],
                [
                    9.938837828312541e-5,
                    -2.658737781501426e-5,
                    -1.1208163048553978e-5,
                    -8.010742406601703e-5,
                    4.6522662357093414e-5,
                    2.1737585896153796e-5,
                    -1.0264317739202065e-4,
                    4.203472935340584e-5,
                    3.040187447299786e-5,
                ],
            ])
            .view(),
        );
        assert_close(spectrum.integration_mesh.weights[0], 1.5707963267948965e-4);
        assert_close(spectrum.integration_mesh.weights[3], 5.542237284087798e-5);
        assert_close(spectrum.integration_mesh.weights[7], 5.542237284087798e-5);
        Ok(())
    }

    #[test]
    fn eels_read_spectrum_matches_feff_readsp_reference() -> Result<(), EelsError> {
        let owned_sources = sample_readsp_sources();
        let sources = readsp_source_views(&owned_sources);

        let sensitive_cross = eels_read_spectrum(EelsReadSpectrumInput {
            sources: &sources,
            orientation_averaged: false,
            cross_terms: true,
            polarization_min: 1,
            polarization_step: 1,
            polarization_max: 9,
        })?;
        assert_eq!(sensitive_cross.effective_polarization_step, 1);
        assert_vector_close(
            sensitive_cross.energy_loss_ev.view(),
            arr1(&[10.25, 20.25, 30.25]).view(),
        );
        assert_vector_close(
            sensitive_cross.atomic_background.view(),
            arr1(&[1.23, 1.26, 1.29]).view(),
        );
        assert_readsp_tensor_rows(
            sensitive_cross.transition_tensor.view(),
            &[
                [
                    0.111, 0.212, 0.313, 0.414, 0.515, 0.616, 0.717, 0.818, 0.919,
                ],
                [
                    0.122, 0.224, 0.326, 0.428, 0.530, 0.632, 0.734, 0.836, 0.938,
                ],
                [
                    0.133, 0.236, 0.339, 0.442, 0.545, 0.648, 0.751, 0.854, 0.957,
                ],
            ],
        );

        let sensitive_no_cross = eels_read_spectrum(EelsReadSpectrumInput {
            sources: &sources,
            orientation_averaged: false,
            cross_terms: false,
            polarization_min: 1,
            polarization_step: 1,
            polarization_max: 9,
        })?;
        assert_eq!(sensitive_no_cross.effective_polarization_step, 4);
        assert_readsp_tensor_rows(
            sensitive_no_cross.transition_tensor.view(),
            &[
                [0.111, 0.0, 0.0, 0.0, 0.515, 0.0, 0.0, 0.0, 0.919],
                [0.122, 0.0, 0.0, 0.0, 0.530, 0.0, 0.0, 0.0, 0.938],
                [0.133, 0.0, 0.0, 0.0, 0.545, 0.0, 0.0, 0.0, 0.957],
            ],
        );

        let average_from_diagonal = eels_read_spectrum(EelsReadSpectrumInput {
            sources: &sources,
            orientation_averaged: true,
            cross_terms: true,
            polarization_min: 1,
            polarization_step: 1,
            polarization_max: 9,
        })?;
        assert_eq!(average_from_diagonal.effective_polarization_step, 1);
        assert_vector_close(
            average_from_diagonal.atomic_background.view(),
            arr1(&[1.23, 1.26, 1.29]).view(),
        );
        assert_readsp_tensor_rows(
            average_from_diagonal.transition_tensor.view(),
            &[
                [0.515, 0.0, 0.0, 0.0, 0.515, 0.0, 0.0, 0.0, 0.515],
                [0.530, 0.0, 0.0, 0.0, 0.530, 0.0, 0.0, 0.0, 0.530],
                [0.545, 0.0, 0.0, 0.0, 0.545, 0.0, 0.0, 0.0, 0.545],
            ],
        );

        let average_from_ten = eels_read_spectrum(EelsReadSpectrumInput {
            sources: &sources,
            orientation_averaged: true,
            cross_terms: false,
            polarization_min: 10,
            polarization_step: 1,
            polarization_max: 10,
        })?;
        assert_vector_close(
            average_from_ten.atomic_background.view(),
            arr1(&[3.03, 3.06, 3.09]).view(),
        );
        assert_readsp_tensor_rows(
            average_from_ten.transition_tensor.view(),
            &[
                [1.02, 0.0, 0.0, 0.0, 1.02, 0.0, 0.0, 0.0, 1.02],
                [1.04, 0.0, 0.0, 0.0, 1.04, 0.0, 0.0, 0.0, 1.04],
                [1.06, 0.0, 0.0, 0.0, 1.06, 0.0, 0.0, 0.0, 1.06],
            ],
        );

        Ok(())
    }

    #[test]
    fn eels_gos_matches_feff_reference() -> Result<(), EelsError> {
        let energy_loss = arr1(&[12.5, 28.0, 64.0, 92.0]);
        let averaged = arr1(&[0.0045, 0.0062, 0.0087, 0.011]);
        let relativistic = eels_generalized_oscillator_strength(EelsGosInput {
            incident_energy_ev: 200_000.0,
            energy_loss_ev: energy_loss.view(),
            averaged_spectrum: averaged.view(),
            relativistic: true,
        })?;
        assert_close(relativistic.q_scale, 0.6858854501070719);
        assert_close(relativistic.q_log_step, 0.12938077038704135);
        assert_close(relativistic.edge_parameter, 100.0);
        assert_close(relativistic.energy_start_ev, 100.0);
        assert_close(relativistic.energy_step_ev, 10.0);
        assert_eq!(relativistic.q_values.len(), FEFF_EELS_GOS_Q_COUNT);
        assert_selected_q_values(&relativistic.q_values);
        assert_selected_gos_rows(
            relativistic.strengths.view(),
            &[
                [
                    1.200166939655954e6,
                    2.606956723601938e5,
                    2.743180061971628e4,
                    3.2396868649661033e3,
                    1.500_423_530_532_838e2,
                ],
                [
                    3.841354508768833e6,
                    8.109311801422351e5,
                    8.473081327354828e4,
                    9.999371994697678e3,
                    4.630661427836753e2,
                ],
                [
                    1.510800565240755e7,
                    2.7128132396096834e6,
                    2.729555927210918e5,
                    3.2088282622040024e4,
                    1.485261481444802e3,
                ],
                [
                    3.7264225704868965e7,
                    5.219418883596917e6,
                    4.98984358650779e5,
                    5.8361116195723495e4,
                    2.699590549258333e3,
                ],
            ],
        );

        let classical = eels_generalized_oscillator_strength(EelsGosInput {
            incident_energy_ev: 200_000.0,
            energy_loss_ev: energy_loss.view(),
            averaged_spectrum: averaged.view(),
            relativistic: false,
        })?;
        assert_selected_q_values(&classical.q_values);
        assert_selected_gos_rows(
            classical.strengths.view(),
            &[
                [
                    1.1894589336479066e6,
                    2.601859956710956e5,
                    2.7426144922494695e4,
                    3.239607964002951e3,
                    1.5004218380799958e2,
                ],
                [
                    3.6709345934449174e6,
                    8.029_918_017_511_5e5,
                    8.46431779296903e4,
                    9.998150089793999e3,
                    4.630635219389996e2,
                ],
                [
                    1.1774057497869411e7,
                    2.575494442482951e6,
                    2.714822665394675e5,
                    3.206779936634388e4,
                    1.4852175634541184e3,
                ],
                [
                    2.139968783736322e7,
                    4.681035157673755e6,
                    4.9342682065003784e5,
                    5.8284146836817534e4,
                    2.699425600243477e3,
                ],
            ],
        );
        Ok(())
    }

    #[test]
    fn eels_angular_dependence_matches_feff_reference() -> Result<(), EelsError> {
        let q_vectors = arr2(&[
            [0.145, 0.310, 0.720, 1.350],
            [0.010, 0.045, 0.115, 0.210],
            [0.25, -0.40, 0.90, -1.20],
        ]);
        let weights = arr1(&[0.185, 0.295, 0.470, 0.815]);
        let partials = Array2::from_shape_fn((10, 4), |(partial, position)| {
            let l = (partial + 1) as Real;
            let k = (position + 1) as Real;
            0.003 * l.powi(2) + 0.017 * k + 0.0009 * l * k
        });

        let angular = eels_angular_dependence(EelsAngularDependenceInput {
            q_vectors_spherical: q_vectors.view(),
            weights: weights.view(),
            partial_spectra: partials.view(),
            incident_wave_number: 82.75,
        })?;

        assert_rect_matrix_close(
            angular.rows.view(),
            arr2(&[
                [
                    -1.7553122764623317e-2,
                    2.524324324324324e-1,
                    6.502702702702704e-1,
                    5.667567567567568,
                    5.372972972972974e-1,
                    7.897297297297298e-1,
                    1.1297297297297304e-1,
                    4.764864864864865,
                    1.7621621621621622,
                ],
                [
                    -1.6915622352268206e-1,
                    2.2508474576271187e-1,
                    6.020338983050848e-1,
                    4.210169491525424,
                    4.705084745762712e-1,
                    6.955932203389831e-1,
                    1.315254237288136e-1,
                    3.3830508474576275,
                    1.193220338983051,
                ],
                [
                    -1.0071045252090503,
                    1.8319148936170213e-1,
                    4.997872340425533e-1,
                    3.0542553191489366,
                    3.7914893617021284e-1,
                    5.62340425531915e-1,
                    1.2063829787234043e-1,
                    2.371276595744681,
                    8.042553191489362e-1,
                ],
                [
                    -3.4559789414201028,
                    1.2981595092024542e-1,
                    3.585276073619632e-1,
                    1.9987730061349693,
                    2.669938650306749e-1,
                    3.9680981595092035e-1,
                    9.15337423312883e-2,
                    1.5104294478527607,
                    4.957055214723926e-1,
                ],
            ])
            .view(),
        );
        Ok(())
    }

    #[test]
    fn eels_collection_angle_dependence_matches_feff_reference() -> Result<(), EelsError> {
        let energy_loss = arr1(&[12.5, 28.0, 64.0]);
        let sigma_x = arr1(&[0.0045, 0.0062, 0.0087]);
        let sigma_y = arr1(&[0.0051, 0.0068, 0.0094]);
        let pi_spectrum = arr1(&[0.0060, 0.0077, 0.0102]);
        let base = EelsCollectionDependenceInput {
            incident_energy_ev: 200_000.0,
            beam_direction: [0.25, -0.15, 0.95],
            mesh: EelsMeshInput {
                collection_angle: 0.020,
                convergence_angle: 0.006,
                theta0: 0.001,
                theta_x_center: 0.0012,
                theta_y_center: -0.0008,
                radial_count: 5,
                angular_count: 2,
                mode: EelsMeshMode::Uniform,
            },
            magic_energy_ev: 10.0,
            energy_loss_ev: energy_loss.view(),
            sigma_x_spectrum: sigma_x.view(),
            sigma_y_spectrum: sigma_y.view(),
            pi_spectrum: pi_spectrum.view(),
            relativistic: true,
        };

        let uniform = eels_collection_angle_dependence(base)?;
        assert_eq!(uniform.magic_index, 0);
        assert_close(uniform.magic_energy_loss_ev, 12.5);
        assert_eq!(uniform.point_counts.to_vec(), vec![2, 8, 18]);
        assert_collection_rows(
            uniform.rows.view(),
            &[
                [
                    5.200_000_000_000_001e-3,
                    9.31245474312255e-2,
                    7.40535193031613e-7,
                    7.211559216646076e-6,
                    7.952094409677688e-6,
                ],
                [
                    1.0400000000000001e-2,
                    9.08490159327226e-2,
                    1.2126539050039605e-6,
                    1.213535974769185e-5,
                    1.334801365269581e-5,
                ],
                [
                    1.5600000000000003e-2,
                    8.507558779221076e-2,
                    1.0342933558856195e-6,
                    1.1123052631682343e-5,
                    1.2157345987567962e-5,
                ],
            ],
        );

        let logarithmic = eels_collection_angle_dependence(EelsCollectionDependenceInput {
            mesh: EelsMeshInput {
                mode: EelsMeshMode::Logarithmic,
                ..base.mesh
            },
            ..base
        })?;
        assert_eq!(logarithmic.magic_index, 0);
        assert_eq!(logarithmic.point_counts.to_vec(), vec![2, 8, 18, 32]);
        assert_collection_rows(
            logarithmic.rows.view(),
            &[
                [
                    1e-3,
                    4.63719811734318e-2,
                    6.832195667109031e-10,
                    1.4050236917618574e-8,
                    1.4733456484329477e-8,
                ],
                [
                    2.2581008643532256e-3,
                    3.177958871763603e-2,
                    1.3650442476143092e-7,
                    4.158844579729555e-6,
                    4.295349004490987e-6,
                ],
                [
                    5.099019513592785e-3,
                    3.973498993079076e-2,
                    2.1691625690008808e-7,
                    5.242157906146296e-6,
                    5.459074163046384e-6,
                ],
                [
                    1.1514100370997834e-2,
                    4.482992649760372e-2,
                    3.500417701478247e-7,
                    7.458174693614065e-6,
                    7.808216463761889e-6,
                ],
            ],
        );
        Ok(())
    }

    #[test]
    fn eels_helpers_reject_invalid_inputs() {
        assert_eq!(
            electron_wavelength_atomic_units(0.0),
            Err(EelsError::InvalidBeamEnergy { value: 0.0 })
        );
        assert!(matches!(
            electron_wavelength_atomic_units(f64::NAN),
            Err(EelsError::NonFiniteInput {
                name: "energy_ev",
                ..
            })
        ));
        assert!(matches!(
            eels_euler_rotation_matrix(0.0, f64::INFINITY, 0.0),
            Err(EelsError::NonFiniteInput { name: "beta", .. })
        ));
        assert_eq!(
            eels_product_matrix_vector(
                arr2(&[[1.0, 2.0, 3.0]]).view(),
                arr1(&[1.0, 2.0, 3.0]).view()
            ),
            Err(EelsError::InvalidMatrixShape {
                rows: 1,
                columns: 3,
            })
        );
        assert_eq!(
            eels_product_matrix_vector(
                arr2(&[[1.0, 2.0], [3.0, 4.0]]).view(),
                arr1(&[1.0, 2.0]).view()
            ),
            Err(EelsError::InvalidMatrixShape {
                rows: 2,
                columns: 2,
            })
        );
        assert_eq!(
            eels_product_matrix_vector(
                arr2(&[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]).view(),
                arr1(&[1.0, 2.0]).view(),
            ),
            Err(EelsError::InvalidVectorLength { length: 2 })
        );
        assert_eq!(
            eels_qmesh(EelsQMeshInput {
                incident_energy_ev: 100_000.0,
                scattered_energy_ev: 99_000.0,
                beam_direction: [0.0, 0.0, 1.0],
                theta_x: arr1(&[0.0, 0.1]).view(),
                theta_y: arr1(&[0.0]).view(),
                relativistic: true,
            }),
            Err(EelsError::QMeshLengthMismatch {
                theta_x_len: 2,
                theta_y_len: 1,
            })
        );
        assert!(matches!(
            eels_qmesh(EelsQMeshInput {
                incident_energy_ev: 100_000.0,
                scattered_energy_ev: 99_000.0,
                beam_direction: [0.0, f64::NAN, 1.0],
                theta_x: arr1(&[0.0]).view(),
                theta_y: arr1(&[0.0]).view(),
                relativistic: true,
            }),
            Err(EelsError::NonFiniteInput {
                name: "beam_direction",
                ..
            })
        ));
        let losses = arr1(&[10.0]);
        let tensor = Array3::<Real>::zeros((1, 3, 3));
        assert_eq!(
            eels_spectrum(EelsSpectrumInput {
                incident_energy_ev: 100_000.0,
                beam_direction: [0.0, 0.0, 1.0],
                mesh: EelsMeshInput {
                    collection_angle: 0.01,
                    convergence_angle: 0.0,
                    theta0: 0.001,
                    theta_x_center: 0.0,
                    theta_y_center: 0.0,
                    radial_count: 1,
                    angular_count: 1,
                    mode: EelsMeshMode::Uniform,
                },
                energy_loss_ev: losses.view(),
                transition_tensor: tensor.view(),
                atomic_background: arr1(&[0.1, 0.2]).view(),
                relativistic: true,
            }),
            Err(EelsError::SpectrumLengthMismatch {
                name: "atomic_background",
                expected: 1,
                actual: 2,
            })
        );
        assert_eq!(
            eels_spectrum(EelsSpectrumInput {
                incident_energy_ev: 100_000.0,
                beam_direction: [0.0, 0.0, 1.0],
                mesh: EelsMeshInput {
                    collection_angle: 0.01,
                    convergence_angle: 0.0,
                    theta0: 0.001,
                    theta_x_center: 0.0,
                    theta_y_center: 0.0,
                    radial_count: 1,
                    angular_count: 1,
                    mode: EelsMeshMode::Uniform,
                },
                energy_loss_ev: arr1(&[100_000.0]).view(),
                transition_tensor: tensor.view(),
                atomic_background: arr1(&[0.1]).view(),
                relativistic: true,
            }),
            Err(EelsError::InvalidEnergyLoss {
                index: 0,
                value: 100_000.0,
                incident_energy_ev: 100_000.0,
            })
        );
        assert_eq!(
            eels_generalized_oscillator_strength(EelsGosInput {
                incident_energy_ev: 100_000.0,
                energy_loss_ev: arr1(&[10.0, 20.0]).view(),
                averaged_spectrum: arr1(&[0.1]).view(),
                relativistic: true,
            }),
            Err(EelsError::SpectrumLengthMismatch {
                name: "averaged_spectrum",
                expected: 2,
                actual: 1,
            })
        );
        assert_eq!(
            eels_generalized_oscillator_strength(EelsGosInput {
                incident_energy_ev: 100_000.0,
                energy_loss_ev: arr1(&[-10.0]).view(),
                averaged_spectrum: arr1(&[0.1]).view(),
                relativistic: true,
            }),
            Err(EelsError::InvalidEnergyLoss {
                index: 0,
                value: -10.0,
                incident_energy_ev: 100_000.0,
            })
        );
        assert_eq!(
            eels_angular_dependence(EelsAngularDependenceInput {
                q_vectors_spherical: arr2(&[[0.1], [0.01], [0.0]]).view(),
                weights: arr1(&[0.0]).view(),
                partial_spectra: Array2::<Real>::zeros((10, 1)).view(),
                incident_wave_number: 10.0,
            }),
            Err(EelsError::InvalidWeight {
                index: 0,
                value: 0.0,
            })
        );
        assert_eq!(
            eels_angular_dependence(EelsAngularDependenceInput {
                q_vectors_spherical: arr2(&[[0.1], [0.01]]).view(),
                weights: arr1(&[1.0]).view(),
                partial_spectra: Array2::<Real>::zeros((10, 1)).view(),
                incident_wave_number: 10.0,
            }),
            Err(EelsError::InvalidTableShape {
                name: "q_vectors_spherical",
                rows: 2,
                columns: 1,
                expected_rows: 3,
                expected_columns: 1,
            })
        );
        assert_eq!(
            eels_collection_angle_dependence(EelsCollectionDependenceInput {
                incident_energy_ev: 100_000.0,
                beam_direction: [0.0, 0.0, 1.0],
                mesh: EelsMeshInput {
                    collection_angle: 0.01,
                    convergence_angle: 0.002,
                    theta0: 0.001,
                    theta_x_center: 0.0,
                    theta_y_center: 0.0,
                    radial_count: 3,
                    angular_count: 2,
                    mode: EelsMeshMode::Uniform,
                },
                magic_energy_ev: 5.0,
                energy_loss_ev: arr1(&[10.0, 20.0]).view(),
                sigma_x_spectrum: arr1(&[0.1]).view(),
                sigma_y_spectrum: arr1(&[0.1, 0.2]).view(),
                pi_spectrum: arr1(&[0.1, 0.2]).view(),
                relativistic: true,
            }),
            Err(EelsError::SpectrumLengthMismatch {
                name: "sigma_x_spectrum",
                expected: 2,
                actual: 1,
            })
        );
        let owned_sources = sample_readsp_sources();
        let readsp_sources = readsp_source_views(&owned_sources);
        assert_eq!(
            eels_read_spectrum(EelsReadSpectrumInput {
                sources: &readsp_sources,
                orientation_averaged: false,
                cross_terms: true,
                polarization_min: 1,
                polarization_step: 4,
                polarization_max: 9,
            }),
            Err(EelsError::InvalidPolarizationRange {
                min: 1,
                step: 4,
                max: 9,
            })
        );
        assert_eq!(
            eels_read_spectrum(EelsReadSpectrumInput {
                sources: &readsp_sources[..1],
                orientation_averaged: true,
                cross_terms: false,
                polarization_min: 10,
                polarization_step: 1,
                polarization_max: 10,
            }),
            Err(EelsError::MissingPolarizationSource { index: 10 })
        );
    }

    #[test]
    fn eels_integration_mesh_matches_feff_uniform_reference() -> Result<(), EelsError> {
        assert_mesh_summary(
            eels_integration_mesh(EelsMeshInput {
                collection_angle: 0.015,
                convergence_angle: 0.008,
                theta0: 0.001,
                theta_x_center: 0.001,
                theta_y_center: -0.002,
                radial_count: 2,
                angular_count: 2,
                mode: EelsMeshMode::Uniform,
            })?,
            MeshSummary {
                radial_count: 2,
                angular_count: 2,
                point_count: 8,
                theta_part: 0.005_750_000_000_000,
                sum_x: 0.008_000_000_000_000,
                sum_y: -0.016_000_000_000_000,
                sum_weight: 0.000_762_080_895_545,
                weighted_x: 0.000_000_762_080_896,
                weighted_y: -0.000_001_524_161_791,
            },
            &[
                (
                    1,
                    -0.004_750_000_000_000,
                    -0.002_000_000_000_000,
                    0.000_207_737_814_219,
                ),
                (
                    4,
                    -0.007_625_000_000_000,
                    0.012_938_938_215_282,
                    0.000_057_767_544_518,
                ),
                (
                    8,
                    0.018_250_000_000_000,
                    -0.002_000_000_000_000,
                    0.000_057_767_544_518,
                ),
            ],
        );
        Ok(())
    }

    #[test]
    fn eels_integration_mesh_matches_feff_logarithmic_reference() -> Result<(), EelsError> {
        assert_mesh_summary(
            eels_integration_mesh(EelsMeshInput {
                collection_angle: 0.015,
                convergence_angle: 0.008,
                theta0: 0.001,
                theta_x_center: -0.0015,
                theta_y_center: 0.0005,
                radial_count: 3,
                angular_count: 2,
                mode: EelsMeshMode::Logarithmic,
            })?,
            MeshSummary {
                radial_count: 3,
                angular_count: 2,
                point_count: 18,
                theta_part: 0.003_833_333_333_333,
                sum_x: -0.027_000_000_000_000,
                sum_y: 0.009_000_000_000_000,
                sum_weight: 0.000_912_791_351_009,
                weighted_x: -0.000_001_369_187_027,
                weighted_y: 0.000_000_456_395_676,
            },
            &[
                (
                    1,
                    -0.002_000_000_000_000,
                    0.000_500_000_000_000,
                    0.000_001_570_796_327,
                ),
                (
                    9,
                    0.009_743_650_037_571,
                    0.008_668_989_922_305,
                    0.000_084_053_471_998,
                ),
                (
                    18,
                    0.012_397_915_761_656,
                    0.000_500_000_000_000,
                    0.000_084_053_471_998,
                ),
            ],
        );
        Ok(())
    }

    #[test]
    fn eels_integration_mesh_matches_feff_one_dimensional_reference() -> Result<(), EelsError> {
        assert_mesh_summary(
            eels_integration_mesh(EelsMeshInput {
                collection_angle: 0.015,
                convergence_angle: 0.008,
                theta0: 0.001,
                theta_x_center: 0.002,
                theta_y_center: 0.001,
                radial_count: 3,
                angular_count: 2,
                mode: EelsMeshMode::OneDimensional,
            })?,
            MeshSummary {
                radial_count: 3,
                angular_count: 1,
                point_count: 3,
                theta_part: 0.003_833_333_333_333,
                sum_x: 0.023_295_831_523_313,
                sum_y: 0.003_000_000_000_000,
                sum_weight: 0.004_413_160_307_671,
                weighted_x: 0.000_067_837_163_754,
                weighted_y: 0.000_004_413_160_308,
            },
            &[
                (
                    1,
                    0.002_500_000_000_000,
                    0.001_000_000_000_000,
                    0.000_003_141_592_654,
                ),
                (
                    1,
                    0.002_500_000_000_000,
                    0.001_000_000_000_000,
                    0.000_003_141_592_654,
                ),
                (
                    3,
                    0.015_897_915_761_656,
                    0.001_000_000_000_000,
                    0.004_202_673_599_880,
                ),
            ],
        );
        Ok(())
    }

    #[test]
    fn eels_mesh_rejects_invalid_inputs() {
        let input = EelsMeshInput {
            collection_angle: 0.015,
            convergence_angle: 0.008,
            theta0: 0.001,
            theta_x_center: 0.0,
            theta_y_center: 0.0,
            radial_count: 2,
            angular_count: 2,
            mode: EelsMeshMode::Uniform,
        };
        assert_eq!(
            eels_integration_mesh(EelsMeshInput {
                radial_count: 0,
                ..input
            }),
            Err(EelsError::InvalidMeshCount {
                name: "radial_count",
                value: 0,
            })
        );
        assert!(matches!(
            eels_integration_mesh(EelsMeshInput {
                collection_angle: -0.1,
                ..input
            }),
            Err(EelsError::InvalidMeshAngle {
                name: "collection_angle",
                ..
            })
        ));
        assert!(matches!(
            eels_integration_mesh(EelsMeshInput {
                theta0: 0.0,
                mode: EelsMeshMode::Logarithmic,
                ..input
            }),
            Err(EelsError::InvalidLogMeshParameter { name: "theta0", .. })
        ));
    }

    fn assert_close(actual: Real, expected: Real) {
        assert!(
            (actual - expected).abs() < 1.0e-14,
            "actual={actual}, expected={expected}, diff={}",
            (actual - expected).abs()
        );
    }

    fn assert_matrix_close(actual: ArrayView2<'_, Real>, expected: ArrayView2<'_, Real>) {
        assert_eq!(actual.dim(), expected.dim());
        for ((row, column), &actual) in actual.indexed_iter() {
            assert_close(actual, expected[(row, column)]);
        }
        assert_close(determinant_3x3(actual), 1.0);
    }

    fn assert_vector_close(actual: ArrayView1<'_, Real>, expected: ArrayView1<'_, Real>) {
        assert_eq!(actual.len(), expected.len());
        for (&actual, &expected) in actual.iter().zip(expected.iter()) {
            assert_close(actual, expected);
        }
    }

    fn assert_rect_matrix_close(actual: ArrayView2<'_, Real>, expected: ArrayView2<'_, Real>) {
        assert_eq!(actual.dim(), expected.dim());
        for ((row, column), &actual) in actual.indexed_iter() {
            assert_close(actual, expected[(row, column)]);
        }
    }

    fn assert_relative_close(actual: Real, expected: Real) {
        let tolerance = 1.0e-12 * expected.abs().max(1.0);
        assert!(
            (actual - expected).abs() <= tolerance,
            "actual={actual}, expected={expected}, diff={}, tolerance={tolerance}",
            (actual - expected).abs()
        );
    }

    fn assert_selected_q_values(q_values: &RealVec) {
        let indices = [0, 1, 4, 9, 19];
        let expected = [
            5.0132553634977525e-2,
            1.0718958629739758e-1,
            3.3015066023371864e-1,
            9.60612700842798e-1,
            4.463626683435738,
        ];
        for (&index, &expected) in indices.iter().zip(expected.iter()) {
            assert_close(q_values[index], expected);
        }
    }

    fn assert_selected_gos_rows(actual: ArrayView2<'_, Real>, expected: &[[Real; 5]; 4]) {
        assert_eq!(actual.dim(), (FEFF_EELS_GOS_Q_COUNT, 4));
        let q_indices = [0, 1, 4, 9, 19];
        for (energy_index, expected_row) in expected.iter().enumerate() {
            for (&q_index, &expected_value) in q_indices.iter().zip(expected_row.iter()) {
                assert_relative_close(actual[(q_index, energy_index)], expected_value);
            }
        }
    }

    fn assert_collection_rows(actual: ArrayView2<'_, Real>, expected: &[[Real; 5]]) {
        assert_eq!(
            actual.dim(),
            (expected.len(), FEFF_EELS_COLLECTION_DEPENDENCE_COLUMN_COUNT)
        );
        for (row, expected_row) in expected.iter().enumerate() {
            for (column, &expected_value) in expected_row.iter().enumerate() {
                assert_relative_close(actual[(row, column)], expected_value);
            }
        }
    }

    fn assert_readsp_tensor_rows(actual: ArrayView3<'_, Real>, expected: &[[Real; 9]; 3]) {
        assert_eq!(actual.dim(), (3, 3, 3));
        for (energy_index, expected_row) in expected.iter().enumerate() {
            for (component, &expected_value) in expected_row.iter().enumerate() {
                let row = component / 3;
                let column = component % 3;
                assert_close(actual[(energy_index, row, column)], expected_value);
            }
        }
    }

    fn determinant_3x3(matrix: ArrayView2<'_, Real>) -> Real {
        matrix[(0, 0)] * matrix[(1, 1)] * matrix[(2, 2)]
            + matrix[(0, 1)] * matrix[(1, 2)] * matrix[(2, 0)]
            + matrix[(1, 0)] * matrix[(2, 1)] * matrix[(0, 2)]
            - matrix[(2, 0)] * matrix[(1, 1)] * matrix[(0, 2)]
            - matrix[(1, 0)] * matrix[(0, 1)] * matrix[(2, 2)]
            - matrix[(0, 0)] * matrix[(2, 1)] * matrix[(1, 2)]
    }

    #[derive(Debug, Clone, Copy)]
    struct MeshSummary {
        radial_count: usize,
        angular_count: usize,
        point_count: usize,
        theta_part: Real,
        sum_x: Real,
        sum_y: Real,
        sum_weight: Real,
        weighted_x: Real,
        weighted_y: Real,
    }

    fn assert_mesh_summary(
        mesh: EelsIntegrationMesh,
        expected: MeshSummary,
        points: &[(usize, Real, Real, Real)],
    ) {
        assert_eq!(mesh.setup.radial_count, expected.radial_count);
        assert_eq!(mesh.setup.angular_count, expected.angular_count);
        assert_eq!(mesh.setup.point_count, expected.point_count);
        assert_eq!(mesh.theta_x.len(), expected.point_count);
        assert_eq!(mesh.theta_y.len(), expected.point_count);
        assert_eq!(mesh.weights.len(), expected.point_count);
        assert_close(mesh.setup.theta_part, expected.theta_part);
        assert_close(mesh.theta_x.sum(), expected.sum_x);
        assert_close(mesh.theta_y.sum(), expected.sum_y);
        assert_close(mesh.weights.sum(), expected.sum_weight);
        assert_close(
            mesh.theta_x
                .iter()
                .zip(mesh.weights.iter())
                .map(|(&theta, &weight)| theta * weight)
                .sum(),
            expected.weighted_x,
        );
        assert_close(
            mesh.theta_y
                .iter()
                .zip(mesh.weights.iter())
                .map(|(&theta, &weight)| theta * weight)
                .sum(),
            expected.weighted_y,
        );
        for &(index, theta_x, theta_y, weight) in points {
            let offset = index - 1;
            assert_close(mesh.theta_x[offset], theta_x);
            assert_close(mesh.theta_y[offset], theta_y);
            assert_close(mesh.weights[offset], weight);
        }
    }

    #[derive(Debug)]
    struct OwnedReadspSource {
        energy_loss_ev: RealVec,
        selected_spectrum: RealVec,
        atomic_background: RealVec,
    }

    fn sample_readsp_sources() -> Vec<OwnedReadspSource> {
        (1..=10)
            .map(|polarization_index| {
                let energy_loss_ev =
                    Array1::from_shape_fn(3, |energy| 10.0 * (energy + 1) as Real + 0.25);
                let selected_spectrum = Array1::from_shape_fn(3, |energy| {
                    let ip = polarization_index as Real;
                    let row = (energy + 1) as Real;
                    0.1 * ip + 0.01 * row + 0.001 * ip * row
                });
                let atomic_background = Array1::from_shape_fn(3, |energy| {
                    let ip = polarization_index as Real;
                    let row = (energy + 1) as Real;
                    1.0 + 0.2 * ip + 0.03 * row
                });
                OwnedReadspSource {
                    energy_loss_ev,
                    selected_spectrum,
                    atomic_background,
                }
            })
            .collect()
    }

    fn readsp_source_views(sources: &[OwnedReadspSource]) -> Vec<EelsReadSpectrumSource<'_>> {
        sources
            .iter()
            .enumerate()
            .map(|(index, source)| EelsReadSpectrumSource {
                polarization_index: index + 1,
                energy_loss_ev: source.energy_loss_ev.view(),
                selected_spectrum: source.selected_spectrum.view(),
                atomic_background: source.atomic_background.view(),
            })
            .collect()
    }
}
