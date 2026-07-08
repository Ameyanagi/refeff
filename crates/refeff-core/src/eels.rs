//! FEFF EELS numerical helpers.
//!
//! This module ports the small kernels from `EELS/wavelength.f90`,
//! `EELS/euler.f90`, `EELS/productmatvect.f90`, `EELS/qmesh.f90`, and the
//! `EELS/readsp.f90` spectrum assembly, the `EELSMDFF/mdff_qmesh.f90`
//! automatic q-vector grid, the `EELSMDFF/mdff_eels.f90` complex spectrum
//! reducer, and spectrum/angular/GOS accumulation loops in
//! `EELS/eels.f90`,
//! `EELS/writeangulardependence1.f90`, `EELS/writeangulardependence2.f90`, and
//! `EELS/writeangulardependence3.f90`. The functions keep FEFF's constants and
//! matrix convention while validating inputs instead of producing NaN/Inf
//! outputs.

use ndarray::{Array1, Array2, Array3, ArrayView1, ArrayView2, ArrayView3, ShapeBuilder};
use num_complex::Complex64;
use thiserror::Error;

use crate::{Real, RealMat, RealVec, constants::BOHR_ANGSTROM_EELS_LEGACY};

/// FEFF electron rest energy `m_e c^2` in eV, from `COMMON/m_constants.f90`.
pub const FEFF_ELECTRON_REST_ENERGY_EV: Real = 511_004.0;
/// FEFF `HOnSqrtTwoMe` constant for electron wavelengths in atomic units.
pub const FEFF_H_ON_SQRT_TWO_ME: Real = 23.1761;
/// FEFF `hbarc_eV`, `hbar*c` in eV atomic-radius units.
pub const FEFF_HBARC_EV: Real = 1973.2708 / BOHR_ANGSTROM_EELS_LEGACY;
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
/// FEFF's hardcoded `EELSMDFF/mdff_angularmesh.f90` automatic-q x positions.
pub const FEFF_MDFF_AUTOMATIC_THETA_X: [Real; 2] = [0.0, 0.0];
/// FEFF's hardcoded `EELSMDFF/mdff_angularmesh.f90` automatic-q y positions.
pub const FEFF_MDFF_AUTOMATIC_THETA_Y: [Real; 2] = [0.0, 0.002];

const FEFF_EELS_GOS_Q_BASE: Real = 0.44;
const FEFF_EELS_GOS_Q_STEP_SEED: Real = 0.1950;
const FEFF_EELS_GOS_EDGE_PARAMETER: Real = 100.0;
const FEFF_EELS_GOS_ENERGY_START_EV: Real = 100.0;
const FEFF_EELS_GOS_ENERGY_STEP_EV: Real = 10.0;
const FEFF_EELS_GOS_A0: Real = BOHR_ANGSTROM_EELS_LEGACY;
const FEFF_EELS_GOS_RYDBERG_EV: Real = 13.6;
const FEFF_EELS_ANGULAR_DEPENDENCE_PARTIAL_COUNT: usize = 10;

/// Error returned by FEFF EELS helper routines.
#[derive(Debug, Clone, Copy, PartialEq, Error)]
#[non_exhaustive]
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
    /// FEFF EELS-MDFF q-vector grids are 3-vectors for every energy point.
    #[error(
        "EELS-MDFF q-vector grid has shape ({energies}, {components}, {q_count}), expected ({expected_energies}, 3, {expected_q_count})"
    )]
    InvalidMdffQVectorShape {
        expected_energies: usize,
        expected_q_count: usize,
        energies: usize,
        components: usize,
        q_count: usize,
    },
    /// FEFF EELS-MDFF classical q lengths align with every energy point and q-vector.
    #[error(
        "EELS-MDFF q-length grid has shape ({energies}, {q_count}), expected ({expected_energies}, {expected_q_count})"
    )]
    InvalidMdffQLengthShape {
        expected_energies: usize,
        expected_q_count: usize,
        energies: usize,
        q_count: usize,
    },
    /// FEFF EELS-MDFF excitation amplitudes align with the q-vector count.
    #[error("EELS-MDFF amplitude length {actual} does not match q-vector count {expected}")]
    InvalidMdffAmplitudeLength { expected: usize, actual: usize },
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

/// Inputs for FEFF's manual-q EELS-MDFF branch in `mdff_eels.f90`.
#[derive(Debug, Clone, Copy)]
pub struct MdffManualQGridInput<'a> {
    /// Incident beam-electron energy `ebeam`, in eV.
    pub incident_energy_ev: Real,
    /// User-supplied q-vectors before FEFF's optional relativistic z shortening.
    pub q_vectors: ArrayView2<'a, Real>,
    /// Number of energy rows that will reuse the manual q-vectors.
    pub energy_count: usize,
    /// Whether to apply FEFF's relativistic manual-q z shortening.
    pub relativistic: bool,
}

/// Inputs for FEFF's automatic-q EELS-MDFF branch in `mdff_qmesh.f90`.
#[derive(Debug, Clone, Copy)]
pub struct MdffAutomaticQGridInput<'a> {
    /// Incident beam-electron energy `ebeam`, in eV.
    pub incident_energy_ev: Real,
    /// Energy losses `s(:,1)`, in eV.
    pub energy_loss_ev: ArrayView1<'a, Real>,
    /// FEFF incident beam direction `xivec`.
    pub beam_direction: [Real; 3],
    /// Detector-plane x angular samples `ThXV`.
    pub theta_x: ArrayView1<'a, Real>,
    /// Detector-plane y angular samples `ThYV`.
    pub theta_y: ArrayView1<'a, Real>,
    /// Whether to apply FEFF's relativistic q-vector shortening, `RelatQ`.
    pub relativistic: bool,
}

/// FEFF EELS-MDFF q-vectors and classical q lengths for the spectrum reducer.
#[derive(Debug, Clone, PartialEq)]
pub struct MdffQGrid {
    /// FEFF `qve(1:3,1:nq)` repeated as `(energy, component, q)`.
    pub q_vectors: Array3<Real>,
    /// FEFF `QLenVClas(1,1:nq)` repeated as `(energy, q)`.
    pub classical_q_lengths: RealMat,
}

/// Inputs for the FEFF `EELSMDFF/mdff_eels.f90` complex spectrum reducer.
#[derive(Debug, Clone, Copy)]
pub struct MdffSpectrumInput<'a> {
    /// Incident beam-electron energy `ebeam`, in eV.
    pub incident_energy_ev: Real,
    /// Energy losses `s(:,1)`, in eV.
    pub energy_loss_ev: ArrayView1<'a, Real>,
    /// FEFF tensor spectra `s(:,2:10)` as `(energy, row, column)`.
    pub transition_tensor: ArrayView3<'a, Real>,
    /// FEFF `qve(1:3,1:nq)` for each energy row, shaped `(energy, component, q)`.
    pub q_vectors: ArrayView3<'a, Real>,
    /// Classical q lengths before relativistic shortening, shaped `(energy, q)`.
    pub classical_q_lengths: ArrayView2<'a, Real>,
    /// Complex q-vector excitation amplitudes `aq(1:nq)`.
    pub amplitudes: ArrayView1<'a, Complex64>,
    /// Whether to use FEFF's relativistic q-dependent denominator.
    pub relativistic: bool,
}

/// Complex EELS-MDFF spectrum rows produced by FEFF `mdff_eels.f90`.
#[derive(Debug, Clone, PartialEq)]
pub struct MdffSpectrum {
    /// FEFF `s(:,1)` energy grid.
    pub energy_loss_ev: RealVec,
    /// FEFF `x(:,1:1+nq*nq)` as `(energy, channel)`.
    pub spectrum: Array2<Complex64>,
    /// FEFF `xpart(:,1:9,1:1+nq*nq)` as `(energy, tensor_component, channel)`.
    pub partials: Array3<Complex64>,
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

mod dependence;
mod gos;
mod kernels;
mod mdff;
mod mesh;
mod qmesh;
mod spectrum;
mod validation;

pub use dependence::{eels_angular_dependence, eels_collection_angle_dependence};
pub use gos::eels_generalized_oscillator_strength;
pub use kernels::{
    eels_euler_rotation_matrix, eels_product_matrix_vector, electron_wavelength_atomic_units,
};
pub use mdff::{mdff_automatic_q_grid, mdff_manual_q_grid, mdff_spectrum};
pub use mesh::{eels_angular_mesh, eels_integration_mesh, eels_mesh_setup};
pub use qmesh::eels_qmesh;
pub use spectrum::{eels_read_spectrum, eels_spectrum};

use mesh::{angular_mesh_with_setup, calculate_weights, validate_angle, validate_mesh_inputs};
use validation::*;

#[cfg(test)]
mod tests;
