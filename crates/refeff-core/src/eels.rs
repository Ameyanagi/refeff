//! FEFF EELS numerical helpers.
//!
//! This module ports the small kernels from `EELS/wavelength.f90`,
//! `EELS/euler.f90`, `EELS/productmatvect.f90`, `EELS/qmesh.f90`, and the
//! spectrum accumulation loop in `EELS/eels.f90`. The functions keep FEFF's
//! constants and matrix convention while validating inputs instead of producing
//! NaN/Inf outputs.

use ndarray::{Array1, Array2, ArrayView1, ArrayView2, ArrayView3, ShapeBuilder};
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
    use ndarray::{Array3, ArrayView1, ArrayView2, arr1, arr2};

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
}
