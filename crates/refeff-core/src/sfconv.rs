//! FEFF SFCONV numerical helpers.
//!
//! These kernels support spectral-function convolution. The full SFCONV driver
//! also depends on spectrum file orchestration, so this module keeps the
//! reusable numerical transforms independent and directly testable.

use ndarray::{Array1, ArrayView1, ArrayView2};
use thiserror::Error;

use crate::{Real, RealVec, RootError, real_polynomial_roots};

const SFCONV_GRATER_MAX_REGIONS: usize = 1_500;
const SFCONV_GRATER_MAX_SINGULARITIES: usize = 20;
const SFCONV_GRATER_DX: [Real; 3] = [
    0.112_701_66_f32 as Real,
    0.5_f32 as Real,
    0.887_298_35_f32 as Real,
];
const SFCONV_GRATER_WT: [Real; 3] = [
    0.277_777_8_f32 as Real,
    0.444_444_45_f32 as Real,
    0.277_777_8_f32 as Real,
];
const SFCONV_GRATER_WT9: [Real; 9] = [
    0.061_693_88_f32 as Real,
    0.108_384_23_f32 as Real,
    0.039_846_36_f32 as Real,
    0.175_209_03_f32 as Real,
    0.229_732_99_f32 as Real,
    0.175_209_03_f32 as Real,
    0.039_846_36_f32 as Real,
    0.108_384_23_f32 as Real,
    0.061_693_88_f32 as Real,
];

/// Inputs for FEFF `SFCONV/mkrmu.f90`.
#[derive(Debug, Clone, Copy)]
pub struct SfconvKramersKronigInput<'a> {
    /// Imaginary part of the spectrum-dependent function, FEFF `xmu`.
    pub imaginary: ArrayView1<'a, Real>,
    /// Reference imaginary part to subtract before the transform, FEFF `xmu0`.
    pub reference_imaginary: ArrayView1<'a, Real>,
    /// Energy grid, FEFF `wpts`.
    pub energy: ArrayView1<'a, Real>,
    /// Number of active rows, FEFF `npts`.
    pub active_len: usize,
}

/// Inputs for FEFF `SFCONV/sfconvsub.f90`.
#[derive(Debug, Clone, Copy)]
pub struct SfconvConvolutionInput<'a> {
    /// Photoelectron energy neglecting collective excitations, FEFF `ekp`.
    pub photoelectron_energy: Real,
    /// Chemical potential / edge position, FEFF `mu`.
    pub chemical_potential: Real,
    /// Core-hole lifetime width, FEFF `gammach`.
    pub core_hole_lifetime: Real,
    /// Signal energy grid, FEFF `wpts2`.
    pub signal_energy: ArrayView1<'a, Real>,
    /// Signal values on `signal_energy`, FEFF `xchi`.
    pub signal: ArrayView1<'a, Real>,
    /// Spectral-function energy grid, FEFF `wpts1`.
    pub spectral_energy: ArrayView1<'a, Real>,
    /// Spectral function values, FEFF `spectf`.
    pub spectral_function: ArrayView1<'a, Real>,
    /// FEFF eight-slot spectral weights array.
    pub weights: ArrayView1<'a, Real>,
    /// Include quasiparticle phase as an asymmetric `1 / omega` term.
    pub asymmetric_phase: bool,
    /// Apply FEFF's available-energy cutoff.
    pub cutoff: bool,
    /// Plasma frequency scale used by the asymmetric phase branch, FEFF `omp`.
    pub plasma_frequency: Real,
}

/// Inputs for FEFF `SFCONV/interpsf.f90`.
#[derive(Debug, Clone, Copy)]
pub struct SfconvSpectralInterpolationInput<'a> {
    /// Minimal spectral-function energy grid, FEFF `epts`.
    pub energy: ArrayView1<'a, Real>,
    /// Eight-row spectral-function table, FEFF `spectf(row, point)`.
    pub spectral_function: ArrayView2<'a, Real>,
    /// Number of rows in the uniform output grid, FEFF `npts`.
    pub output_len: usize,
}

/// Selected FEFF SFCONV pole parameters.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SfconvPole {
    /// Pole energy, FEFF `ompl`.
    pub energy: Real,
    /// Pole weight, FEFF `wt`.
    pub weight: Real,
    /// Pole broadening, FEFF `brd`.
    pub broadening: Real,
}

/// Electron-gas parameters produced by FEFF `SFCONV/ppset`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SfconvPlasmaParameters {
    /// Fermi momentum, FEFF `qf`.
    pub fermi_momentum: Real,
    /// Fermi energy, FEFF `ef`.
    pub fermi_energy: Real,
    /// Plasma frequency, FEFF `omp`.
    pub plasma_frequency: Real,
}

/// Limiting momentum values produced by FEFF `SFCONV/qlimits.f90`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SfconvQLimits {
    /// Number of active limiting values, FEFF `nq`.
    pub count: usize,
    /// First limiting value, FEFF `q1`.
    pub q1: Real,
    /// Second limiting value, FEFF `q2`.
    pub q2: Real,
    /// Third limiting value, FEFF `q3`.
    pub q3: Real,
}

/// Result from FEFF `SFCONV/grater.f90` adaptive quadrature.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SfconvAdaptiveIntegral {
    /// Accumulated real integral value.
    pub value: Real,
    /// FEFF `error`: accumulated absolute difference between local estimates.
    pub estimated_error: Real,
    /// FEFF `numcal`: number of integrand evaluations.
    pub evaluations: usize,
    /// FEFF `maxns`: maximum number of active regions on the stack.
    pub max_regions: usize,
}

/// Shared pole/plasma context for FEFF `SFCONV/mksat.f90` helpers.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SfconvSatelliteContext {
    /// Plasma frequency, FEFF `omp`.
    pub plasma_frequency: Real,
    /// Pole energy, FEFF `ompl`.
    pub pole_energy: Real,
    /// Pole dispersion parameter, FEFF `adisp`.
    pub dispersion_parameter: Real,
    /// Bare photoelectron kinetic energy, FEFF `ek`.
    pub photoelectron_energy: Real,
    /// Global relative accuracy parameter, FEFF `acc`.
    pub accuracy: Real,
}

/// Shared electron-gas context for FEFF `SFCONV/senergies.f90` beta helpers.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SfconvSelfEnergyContext {
    /// Fermi energy, FEFF `ef`.
    pub fermi_energy: Real,
    /// Fermi momentum, FEFF `qf`.
    pub fermi_momentum: Real,
    /// Plasma frequency, FEFF `omp`.
    pub plasma_frequency: Real,
    /// Active pole energy, FEFF `ompl`.
    pub pole_energy: Real,
    /// Photoelectron quasiparticle energy, FEFF `ekp`.
    pub quasiparticle_energy: Real,
    /// Photoelectron momentum, FEFF `pk`.
    pub photoelectron_momentum: Real,
    /// Global relative accuracy parameter, FEFF `acc`.
    pub accuracy: Real,
    /// Pole dispersion parameter, FEFF `adisp`.
    pub dispersion_parameter: Real,
    /// Include below-Fermi contributions, FEFF common block `belowqf`.
    pub include_below_fermi: bool,
}

/// FEFF `SFCONV/mksat.f90` self-energy state from common block `energies`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SfconvSatelliteSelfEnergy {
    /// Real part of the on-shell self energy, FEFF `se`.
    pub on_shell_real: Real,
    /// Quasiparticle broadening, FEFF `width`.
    pub width: Real,
    /// Real part of the renormalization constant, FEFF `z1`.
    pub renormalization_real: Real,
    /// Imaginary part of the renormalization constant, FEFF `z1i`.
    pub renormalization_imag: Real,
    /// Real part of the self energy at the current energy, FEFF `se2`.
    pub off_shell_real: Real,
    /// Imaginary part of the self energy at the current energy, FEFF `xise`.
    pub off_shell_imag: Real,
}

/// Result from an integrated FEFF `SFCONV/mksat.f90` satellite helper.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SfconvSatelliteIntegral {
    /// Accumulated satellite value.
    pub value: Real,
    /// Sum of FEFF `grater` local error estimates.
    pub estimated_error: Real,
    /// Total integrand evaluations across FEFF `grater` calls.
    pub evaluations: usize,
    /// Maximum active FEFF `grater` stack size across calls.
    pub max_regions: usize,
}

/// Magnitude and phase produced by FEFF `SFCONV/sfconvsub.f90`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SfconvConvolution {
    /// Magnitude of the convoluted signal, FEFF `cchi`.
    pub amplitude: Real,
    /// Phase of the convoluted signal, FEFF `phase`.
    pub phase: Real,
}

/// Uniform spectral function produced by FEFF `SFCONV/interpsf.f90`.
#[derive(Debug, Clone, PartialEq)]
pub struct SfconvSpectralInterpolation {
    /// Uniform energy grid, FEFF `wpts`.
    pub energy: RealVec,
    /// Interpolated spectral function, FEFF `cspec`.
    pub spectral_function: RealVec,
}

/// Error returned by SFCONV helper kernels.
#[derive(Debug, Clone, Copy, PartialEq, Error)]
pub enum SfconvError {
    /// FEFF `mkrmu` smooths rows 20 and 21, so shorter inputs are unsupported.
    #[error("SFCONV {name} count {actual} is below minimum {minimum}")]
    CountTooSmall {
        name: &'static str,
        actual: usize,
        minimum: usize,
    },
    /// Active rows must fit in each input array.
    #[error("SFCONV active row count {active_len} exceeds {field} length {len}")]
    ActiveCountOutOfRange {
        field: &'static str,
        active_len: usize,
        len: usize,
    },
    /// Two related arrays must have the same length.
    #[error("SFCONV {left} length {left_len} does not match {right} length {right_len}")]
    LengthMismatch {
        left: &'static str,
        left_len: usize,
        right: &'static str,
        right_len: usize,
    },
    /// Fixed-size FEFF helper arrays must have the expected number of slots.
    #[error("SFCONV {field} count {actual} does not match expected count {expected}")]
    CountMismatch {
        field: &'static str,
        actual: usize,
        expected: usize,
    },
    /// Scalar values must be finite.
    #[error("SFCONV {field} must be finite, got {value}")]
    NonFiniteScalar { field: &'static str, value: Real },
    /// Scalar values that appear in denominators must be positive.
    #[error("SFCONV {field} must be positive, got {value}")]
    NonPositiveScalar { field: &'static str, value: Real },
    /// Array values must be finite.
    #[error("SFCONV {field} row {row} must be finite, got {value}")]
    NonFiniteValue {
        field: &'static str,
        row: usize,
        value: Real,
    },
    /// The energy grid must be strictly increasing to avoid FEFF's pole division.
    #[error("SFCONV {field} row {row} must increase, got {current} after {previous}")]
    NonIncreasingEnergy {
        field: &'static str,
        row: usize,
        previous: Real,
        current: Real,
    },
    /// FEFF one-based pole selectors must fit the input arrays.
    #[error("SFCONV {field} index {index} is outside 1..={len}")]
    IndexOutOfRange {
        field: &'static str,
        index: usize,
        len: usize,
    },
    /// The asymmetric branch divides by the real quasiparticle weight.
    #[error("SFCONV asymmetric phase requires a nonzero real quasiparticle weight")]
    ZeroAsymmetricWeight,
    /// The asymmetric branch needs a nonzero plasma-frequency scale.
    #[error("SFCONV asymmetric phase requires a nonzero plasma frequency")]
    ZeroPlasmaFrequency,
    /// FEFF normalizes by the total spectral weight.
    #[error("SFCONV normalization weight must be finite and nonzero, got {value}")]
    InvalidNormalization { value: Real },
    /// The transformed value must be finite.
    #[error("SFCONV transformed row {row} must be finite, got {value}")]
    NonFiniteResult { row: usize, value: Real },
    /// A square-root radicand must stay non-negative.
    #[error("SFCONV {field} radicand must be non-negative, got {value}")]
    NegativeRadicand { field: &'static str, value: Real },
    /// FEFF formula denominator is singular for this input.
    #[error("SFCONV denominator {field} is zero")]
    ZeroDenominator { field: &'static str },
    /// Cubic root solving failed while finding FEFF pole limits.
    #[error("SFCONV pole-limit root solve failed: {source}")]
    RootSolve { source: RootError },
    /// Integration tolerances must be strictly positive.
    #[error("SFCONV tolerance {field} must be positive, got {value}")]
    NonPositiveTolerance { field: &'static str, value: Real },
    /// FEFF integration bounds must form a finite increasing interval.
    #[error("SFCONV integration interval must increase: lower={lower}, upper={upper}")]
    InvalidIntegrationInterval { lower: Real, upper: Real },
    /// FEFF `grater` stores at most 20 explicit split points.
    #[error("SFCONV integration received {count} split points; maximum is {max}")]
    TooManySingularities { count: usize, max: usize },
    /// Explicit split points must be finite, ordered, and inside the interval.
    #[error("invalid SFCONV split point {index}: {value}")]
    InvalidSingularity { index: usize, value: Real },
    /// FEFF `grater` exhausted its fixed region stack.
    #[error("SFCONV adaptive integration exceeded {max_regions} active regions")]
    TooManyIntegrationRegions { max_regions: usize },
}

/// Port of `SFCONV/mkrmu.f90`: discrete Kramers-Kronig transform.
///
/// FEFF integrates `(xmu - xmu0) / (w_i - w_j)` with endpoint/centered energy
/// widths, divides by `pi`, then averages rows 20 and 21 to smooth the legacy
/// phase handoff. The returned array contains exactly `active_len` rows.
pub fn sfconv_kramers_kronig_real_part(
    input: SfconvKramersKronigInput<'_>,
) -> Result<RealVec, SfconvError> {
    validate_count_at_least("active_len", input.active_len, 21)?;
    validate_active_len("imaginary", input.active_len, input.imaginary.len())?;
    validate_active_len(
        "reference_imaginary",
        input.active_len,
        input.reference_imaginary.len(),
    )?;
    validate_active_len("energy", input.active_len, input.energy.len())?;

    for row in 0..input.active_len {
        validate_finite_value("imaginary", row, input.imaginary[row])?;
        validate_finite_value("reference_imaginary", row, input.reference_imaginary[row])?;
        validate_finite_value("energy", row, input.energy[row])?;
        if row > 0 && input.energy[row] <= input.energy[row - 1] {
            return Err(SfconvError::NonIncreasingEnergy {
                field: "energy",
                row,
                previous: input.energy[row - 1],
                current: input.energy[row],
            });
        }
    }

    let mut real_part = Array1::<Real>::zeros(input.active_len);
    for target in 0..input.active_len {
        let mut sum = 0.0;
        for source in 0..input.active_len {
            if source == target {
                continue;
            }
            let width = integration_width(input.energy, input.active_len, source);
            let numerator = input.imaginary[source] - input.reference_imaginary[source];
            sum += width * numerator / (input.energy[source] - input.energy[target]);
        }
        let value = sum / std::f64::consts::PI;
        if !value.is_finite() {
            return Err(SfconvError::NonFiniteResult { row: target, value });
        }
        real_part[target] = value;
    }

    let smoothed = 0.5 * (real_part[19] + real_part[20]);
    real_part[19] = smoothed;
    real_part[20] = smoothed;

    Ok(real_part)
}

/// Port of `SFCONV/plset.f90`: select one epsilon-inverse pole.
///
/// `pole_index_1based` follows FEFF's one-based `ipl` convention. The input
/// arrays correspond to `plengy`, `plwt`, and `plbrd`, and must have matching
/// lengths.
pub fn sfconv_select_pole(
    pole_index_1based: usize,
    energy: ArrayView1<'_, Real>,
    weight: ArrayView1<'_, Real>,
    broadening: ArrayView1<'_, Real>,
) -> Result<SfconvPole, SfconvError> {
    validate_count_at_least("poles", energy.len(), 1)?;
    validate_matching_lengths("energy", energy.len(), "weight", weight.len())?;
    validate_matching_lengths("energy", energy.len(), "broadening", broadening.len())?;
    validate_finite_array("energy", energy)?;
    validate_finite_array("weight", weight)?;
    validate_finite_array("broadening", broadening)?;

    if pole_index_1based == 0 || pole_index_1based > energy.len() {
        return Err(SfconvError::IndexOutOfRange {
            field: "pole",
            index: pole_index_1based,
            len: energy.len(),
        });
    }
    let index = pole_index_1based - 1;
    Ok(SfconvPole {
        energy: energy[index],
        weight: weight[index],
        broadening: broadening[index],
    })
}

/// Port of `SFCONV/ppset`: electron-gas parameters for a Wigner-Seitz radius.
pub fn sfconv_plasma_parameters(
    wigner_seitz_radius: Real,
) -> Result<SfconvPlasmaParameters, SfconvError> {
    validate_positive_scalar("wigner_seitz_radius", wigner_seitz_radius)?;

    let pi = std::f64::consts::PI;
    let fermi_momentum = (9.0 * pi / 4.0).powf(1.0 / 3.0) / wigner_seitz_radius;
    let fermi_energy = fermi_momentum * fermi_momentum / 2.0;
    let concentration = 3.0 / (4.0 * pi * wigner_seitz_radius.powi(3));
    let plasma_frequency = (4.0 * pi * concentration).sqrt();
    Ok(SfconvPlasmaParameters {
        fermi_momentum,
        fermi_energy,
        plasma_frequency,
    })
}

/// Port of `SFCONV/ppole.f90` `wdisp`: pole dispersion relation.
pub fn sfconv_pole_dispersion(
    momentum: Real,
    pole_energy: Real,
    dispersion_parameter: Real,
) -> Result<Real, SfconvError> {
    validate_dispersion_inputs(momentum, pole_energy, dispersion_parameter)?;
    pole_dispersion_value(momentum, pole_energy, dispersion_parameter)
}

/// Port of `SFCONV/ppole.f90` `dwdq`: first dispersion derivative.
pub fn sfconv_pole_dispersion_derivative(
    momentum: Real,
    pole_energy: Real,
    dispersion_parameter: Real,
) -> Result<Real, SfconvError> {
    validate_dispersion_inputs(momentum, pole_energy, dispersion_parameter)?;
    let dispersion = pole_dispersion_value(momentum, pole_energy, dispersion_parameter)?;
    Ok((momentum.powi(3) + 2.0 * dispersion_parameter * momentum) / (2.0 * dispersion))
}

/// Port of `SFCONV/ppole.f90` `d2wdq2`: second dispersion derivative.
pub fn sfconv_pole_dispersion_second_derivative(
    momentum: Real,
    pole_energy: Real,
    dispersion_parameter: Real,
) -> Result<Real, SfconvError> {
    validate_dispersion_inputs(momentum, pole_energy, dispersion_parameter)?;
    let dispersion = pole_dispersion_value(momentum, pole_energy, dispersion_parameter)?;
    let derivative =
        (momentum.powi(3) + 2.0 * dispersion_parameter * momentum) / (2.0 * dispersion);
    let numerator = (3.0 * momentum.powi(2) + 2.0 * dispersion_parameter) * dispersion
        - (momentum.powi(3) + 2.0 * dispersion_parameter * momentum) * derivative;
    Ok(numerator / (2.0 * dispersion.powi(2)))
}

/// Port of `SFCONV/ppole.f90` `qdisp`: inverse pole dispersion relation.
pub fn sfconv_inverse_pole_dispersion(
    energy: Real,
    pole_energy: Real,
    dispersion_parameter: Real,
) -> Result<Real, SfconvError> {
    validate_finite_scalar("energy", energy)?;
    validate_positive_scalar("pole_energy", pole_energy)?;
    validate_finite_scalar("dispersion_parameter", dispersion_parameter)?;

    let discriminant = dispersion_parameter.powi(2) + energy.powi(2) - pole_energy.powi(2);
    if discriminant >= 0.0 {
        let radicand = -2.0 * dispersion_parameter + 2.0 * discriminant.sqrt();
        if radicand >= 0.0 {
            return Ok(radicand.sqrt());
        }
    }
    Ok(0.0)
}

/// Port of `SFCONV/ppole.f90` `vpp2`: squared pole-coupling potential.
pub fn sfconv_coupling_potential_squared(
    momentum: Real,
    plasma_frequency: Real,
    pole_energy: Real,
    dispersion_parameter: Real,
) -> Result<Real, SfconvError> {
    validate_positive_scalar("momentum", momentum.abs())?;
    validate_positive_scalar("plasma_frequency", plasma_frequency)?;
    let dispersion = sfconv_pole_dispersion(momentum, pole_energy, dispersion_parameter)?;
    Ok(2.0 * std::f64::consts::PI * plasma_frequency.powi(2) / (momentum.powi(2) * dispersion))
}

/// Port of `SFCONV/qlimits.f90`: momentum limits for pole-loss inequalities.
pub fn sfconv_q_limits(
    energy: Real,
    photoelectron_momentum: Real,
    pole_energy: Real,
    dispersion_parameter: Real,
    upper_limit: Real,
) -> Result<SfconvQLimits, SfconvError> {
    validate_finite_scalar("energy", energy)?;
    validate_positive_scalar("photoelectron_momentum", photoelectron_momentum)?;
    validate_positive_scalar("pole_energy", pole_energy)?;
    validate_finite_scalar("dispersion_parameter", dispersion_parameter)?;
    validate_positive_scalar("upper_limit", upper_limit)?;

    sfconv_q_limits_with_upper(
        energy,
        photoelectron_momentum,
        pole_energy,
        dispersion_parameter,
        upper_limit,
    )
}

fn sfconv_q_limits_with_upper(
    energy: Real,
    photoelectron_momentum: Real,
    pole_energy: Real,
    dispersion_parameter: Real,
    upper_limit: Real,
) -> Result<SfconvQLimits, SfconvError> {
    let a = photoelectron_momentum;
    let b = energy + dispersion_parameter - 3.0 * photoelectron_momentum.powi(2) / 2.0;
    let c = photoelectron_momentum.powi(3) - 2.0 * energy * photoelectron_momentum;
    let d = pole_energy.powi(2) - energy.powi(2) + energy * photoelectron_momentum.powi(2)
        - photoelectron_momentum.powi(4) / 4.0;
    let roots =
        real_polynomial_roots([a, b, c, d]).map_err(|source| SfconvError::RootSolve { source })?;
    let values = roots.into_inner();

    if roots.real_root_count() == 3 {
        let root0 = values[0].re;
        let root1 = values[1].re;
        let root2 = values[2].re;
        let dev0 = (pole_dispersion_value(root0, pole_energy, dispersion_parameter)?
            + (root0 - photoelectron_momentum).powi(2) / 2.0
            - energy)
            .abs();
        let dev1 = (pole_dispersion_value(root1, pole_energy, dispersion_parameter)?
            + (root1 - photoelectron_momentum).powi(2) / 2.0
            - energy)
            .abs();
        let dev2 = (pole_dispersion_value(root2, pole_energy, dispersion_parameter)?
            + (root2 - photoelectron_momentum).powi(2) / 2.0
            - energy)
            .abs();
        let (q1, q2, q3) = if dev0 > dev1 && dev0 > dev2 {
            (
                root1.abs().min(root2.abs()),
                root1.abs().max(root2.abs()),
                root0.abs(),
            )
        } else if dev1 > dev2 {
            (
                root0.abs().min(root2.abs()),
                root0.abs().max(root2.abs()),
                root1.abs(),
            )
        } else {
            (
                root0.abs().min(root1.abs()),
                root0.abs().max(root1.abs()),
                root2.abs(),
            )
        };
        Ok(SfconvQLimits {
            count: 3,
            q1: q1.min(upper_limit),
            q2: q2.min(upper_limit),
            q3,
        })
    } else {
        let imag0 = values[0].im.abs();
        let imag1 = values[1].im.abs();
        let imag2 = values[2].im.abs();
        let q3 = if imag0 < imag1 && imag0 < imag2 {
            values[0].re.abs()
        } else if imag1 < imag2 {
            values[1].re.abs()
        } else {
            values[2].re.abs()
        };
        Ok(SfconvQLimits {
            count: 1,
            q1: 0.0,
            q2: 0.0,
            q3,
        })
    }
}

/// Port of `SFCONV/ppole.f90` `qthresh`: plasmon-loss onset momentum.
pub fn sfconv_plasmon_threshold_momentum(
    pole_energy: Real,
    dispersion_parameter: Real,
    fermi_energy: Real,
    fermi_momentum: Real,
) -> Result<Real, SfconvError> {
    validate_positive_scalar("pole_energy", pole_energy)?;
    validate_finite_scalar("dispersion_parameter", dispersion_parameter)?;
    validate_positive_scalar("fermi_energy", fermi_energy)?;
    validate_positive_scalar("fermi_momentum", fermi_momentum)?;

    let roots = real_polynomial_roots([
        1.0,
        -3.0 * dispersion_parameter,
        3.0 * dispersion_parameter.powi(2) - 27.0 * pole_energy.powi(2) / 4.0,
        -dispersion_parameter.powi(3),
    ])
    .map_err(|source| SfconvError::RootSolve { source })?;
    let qthresh1 = if roots.real_root_count() == 1 {
        let sorted = roots_sorted_by_imag_descending(roots.into_inner());
        sorted[1].re
    } else {
        roots
            .roots()
            .iter()
            .map(|root| root.re)
            .fold(f64::NEG_INFINITY, Real::max)
    };
    let qthresh1 = if qthresh1 > 0.0 { qthresh1.sqrt() } else { 0.0 };

    let b = 1.5 * fermi_momentum + dispersion_parameter / fermi_momentum;
    let c = fermi_momentum.powi(2) + 2.0 * dispersion_parameter;
    let d = fermi_momentum.powi(3) / 4.0
        + dispersion_parameter * fermi_momentum
        + pole_energy.powi(2) / fermi_momentum;
    let roots_a = real_polynomial_roots([1.0, b, c, d])
        .map_err(|source| SfconvError::RootSolve { source })?;
    let values_a = roots_a.into_inner();
    let q01 = if roots_a.real_root_count() == 1 {
        roots_sorted_by_imag_descending(values_a)[1].re
    } else {
        let selected = select_threshold_root(values_a, |root| {
            let xfact = threshold_factor(dispersion_parameter, pole_energy, root)?;
            Ok(root - fermi_momentum - checked_sqrt("qthresh test", 2.0 * xfact)?)
        })?;
        selected.re
    };

    let roots_b = real_polynomial_roots([1.0, -b, c, -d])
        .map_err(|source| SfconvError::RootSolve { source })?;
    let values_b = roots_b.into_inner();
    let q02 = if roots_b.real_root_count() == 1 {
        roots_sorted_by_imag_descending(values_b)[1].re
    } else {
        // FEFF selects the index using the second cubic, but returns from the
        // first root array. Preserve that historical behavior.
        let index = select_threshold_root_index(values_b, |root| {
            let xfact = threshold_factor(dispersion_parameter, pole_energy, root)?;
            Ok(root + fermi_momentum - checked_sqrt("qthresh test", 2.0 * xfact)?)
        })?;
        values_a[index].re
    };

    let qthresh2 = q01.abs().min(q02.abs());
    let upper_limit = 1000.0 * fermi_momentum;
    let energy1 = qthresh1.powi(2) / 2.0;
    let limits_a = sfconv_q_limits(
        energy1,
        qthresh1,
        pole_energy,
        dispersion_parameter,
        upper_limit,
    )?;
    let _q0a =
        sfconv_inverse_pole_dispersion(energy1 - fermi_energy, pole_energy, dispersion_parameter)?;

    let energy2 = qthresh2.powi(2) / 2.0;
    let limits_b = sfconv_q_limits(
        energy2,
        qthresh2,
        pole_energy,
        dispersion_parameter,
        upper_limit,
    )?;
    let q0b =
        sfconv_inverse_pole_dispersion(energy2 - fermi_energy, pole_energy, dispersion_parameter)?;

    if limits_a.count == 0 || (limits_a.q1 - limits_a.q2).abs() < (limits_b.q1 - q0b).abs() {
        Ok(qthresh1)
    } else {
        Ok(qthresh2)
    }
}

/// Port of `SFCONV/senergies.f90` `exchange`.
///
/// Computes the Hartree-Fock exchange potential for a free electron gas at
/// photoelectron momentum `momentum`.
pub fn sfconv_free_electron_exchange(
    momentum: Real,
    fermi_momentum: Real,
) -> Result<Real, SfconvError> {
    validate_positive_scalar("momentum", momentum)?;
    validate_positive_scalar("fermi_momentum", fermi_momentum)?;

    let value = if momentum == fermi_momentum {
        -fermi_momentum / std::f64::consts::PI
    } else {
        let ratio = (momentum + fermi_momentum) / (momentum - fermi_momentum);
        validate_nonzero_denominator("exchange logarithm", ratio)?;
        -(fermi_momentum
            + ((fermi_momentum.powi(2) - momentum.powi(2)) / (2.0 * momentum)) * ratio.abs().ln())
            / std::f64::consts::PI
    };
    finite_result("free electron exchange", value)
}

/// Port of `SFCONV/senergies.f90` `beta`.
///
/// FEFF uses this extrinsic beta function as the analytic imaginary
/// self-energy contribution for the active pole.
pub fn sfconv_extrinsic_beta(
    energy: Real,
    context: SfconvSelfEnergyContext,
) -> Result<Real, SfconvError> {
    validate_finite_scalar("self-energy energy", energy)?;
    validate_self_energy_context(context)?;

    let pole_energy = context.pole_energy;
    let dispersion_parameter = context.dispersion_parameter;
    let fermi_limited_energy =
        (energy + context.quasiparticle_energy - context.fermi_energy).max(pole_energy);
    let qh =
        sfconv_inverse_pole_dispersion(fermi_limited_energy, pole_energy, dispersion_parameter)?;
    let q0 = sfconv_inverse_pole_dispersion(
        (context.fermi_energy - energy - context.quasiparticle_energy).max(pole_energy),
        pole_energy,
        dispersion_parameter,
    )?;
    let limits = sfconv_q_limits_with_upper(
        energy + context.quasiparticle_energy,
        context.photoelectron_momentum,
        pole_energy,
        dispersion_parameter,
        qh,
    )?;

    let above_fermi = if limits.count == 3 {
        let q1 = checked_sqrt(
            "beta q1",
            limits.q1.powi(2) + context.accuracy * pole_energy,
        )?;
        let q2 = checked_sqrt(
            "beta q2",
            limits.q2.powi(2) + context.accuracy * pole_energy,
        )?;
        let wq1 = sfconv_pole_dispersion(q1, pole_energy, dispersion_parameter)?;
        let wq2 = sfconv_pole_dispersion(q2, pole_energy, dispersion_parameter)?;
        beta_prefactor(context)
            * beta_log_argument(q2, wq2, q1, wq1, pole_energy, dispersion_parameter)?.ln()
    } else {
        0.0
    };

    let below_fermi = if limits.q3 < q0 && context.include_below_fermi {
        let q0 = checked_sqrt("beta q0", q0.powi(2) + context.accuracy * pole_energy)?;
        let q3 = checked_sqrt(
            "beta q3",
            limits.q3.powi(2) + context.accuracy * pole_energy,
        )?;
        let wq0 = sfconv_pole_dispersion(q0, pole_energy, dispersion_parameter)?;
        let wq3 = sfconv_pole_dispersion(q3, pole_energy, dispersion_parameter)?;
        beta_prefactor(context)
            * beta_log_argument(q0, wq0, q3, wq3, pole_energy, dispersion_parameter)?.ln()
    } else {
        0.0
    };

    finite_result("extrinsic beta", above_fermi - below_fermi)
}

/// Port of `SFCONV/senergies.f90` `xienergies`.
pub fn sfconv_imaginary_self_energy(
    energy: Real,
    context: SfconvSelfEnergyContext,
) -> Result<Real, SfconvError> {
    finite_result(
        "imaginary self energy",
        -std::f64::consts::PI * sfconv_extrinsic_beta(energy, context)?,
    )
}

/// Port of `SFCONV/senergies.f90` `findsing`.
pub fn sfconv_find_singularities(
    lower: Real,
    upper: Real,
    candidates: ArrayView1<'_, Real>,
) -> Result<RealVec, SfconvError> {
    validate_finite_scalar("singularity lower bound", lower)?;
    validate_finite_scalar("singularity upper bound", upper)?;
    let mut singularities = candidates
        .iter()
        .enumerate()
        .filter_map(|(index, &candidate)| {
            if !candidate.is_finite() {
                return Some(Err(SfconvError::NonFiniteValue {
                    field: "singularity candidate",
                    row: index,
                    value: candidate,
                }));
            }
            let in_forward_interval = candidate > lower && candidate < upper;
            let in_reverse_interval = candidate < lower && candidate > upper;
            (in_forward_interval || in_reverse_interval).then_some(Ok(candidate))
        })
        .collect::<Result<Vec<_>, _>>()?;
    if singularities.len() > SFCONV_GRATER_MAX_SINGULARITIES {
        return Err(SfconvError::TooManySingularities {
            count: singularities.len(),
            max: SFCONV_GRATER_MAX_SINGULARITIES,
        });
    }
    singularities.sort_by(|left, right| left.total_cmp(right));
    Ok(Array1::from_vec(singularities))
}

/// Port of `SFCONV/grater.f90`: adaptive real quadrature with split points.
///
/// `singularities` are FEFF `xsing`: ordered real split points inserted
/// between `lower` and `upper` before the adaptive stack starts. The returned
/// diagnostics mirror FEFF `error`, `numcal`, and `maxns`.
pub fn sfconv_grater_integrate(
    mut integrand: impl FnMut(Real) -> Result<Real, SfconvError>,
    lower: Real,
    upper: Real,
    absolute_tolerance: Real,
    relative_tolerance: Real,
    singularities: &[Real],
) -> Result<SfconvAdaptiveIntegral, SfconvError> {
    validate_grater_input(
        lower,
        upper,
        absolute_tolerance,
        relative_tolerance,
        singularities,
    )?;

    let mut xleft = vec![0.0; SFCONV_GRATER_MAX_REGIONS];
    let mut fval = vec![[0.0; 3]; SFCONV_GRATER_MAX_REGIONS];
    let mut nstack = singularities.len() + 1;
    let mut max_regions = nstack;
    let mut estimated_error = 0.0;
    let mut value_total = 0.0;

    xleft[0] = lower;
    xleft[singularities.len() + 1] = upper;
    for (index, &singularity) in singularities.iter().enumerate() {
        xleft[index + 1] = singularity;
    }

    for region in 0..nstack {
        let delta = xleft[region + 1] - xleft[region];
        for point in 0..3 {
            fval[region][point] = eval_grater_integrand(
                &mut integrand,
                xleft[region] + delta * SFCONV_GRATER_DX[point],
                region * 3 + point,
            )?;
        }
    }
    let mut evaluations = nstack * 3;
    let total_interval = upper - lower;

    loop {
        if nstack + 3 >= SFCONV_GRATER_MAX_REGIONS {
            return Err(SfconvError::TooManyIntegrationRegions {
                max_regions: SFCONV_GRATER_MAX_REGIONS,
            });
        }

        let region = nstack - 1;
        let delta = xleft[region + 1] - xleft[region];
        xleft[region + 3] = xleft[region + 1];
        xleft[region + 1] = xleft[region] + delta * SFCONV_GRATER_DX[0] * 2.0;
        xleft[region + 2] = xleft[region + 3] - delta * SFCONV_GRATER_DX[0] * 2.0;
        fval[region + 2][1] = fval[region][2];
        fval[region + 1][1] = fval[region][1];
        fval[region][1] = fval[region][0];

        let mut weight_index = 0;
        let mut high_order = 0.0;
        let mut low_order = 0.0;
        for current_region in region..=region + 2 {
            let sub_delta = xleft[current_region + 1] - xleft[current_region];
            fval[current_region][0] = eval_grater_integrand(
                &mut integrand,
                xleft[current_region] + SFCONV_GRATER_DX[0] * sub_delta,
                evaluations,
            )?;
            evaluations += 1;
            fval[current_region][2] = eval_grater_integrand(
                &mut integrand,
                xleft[current_region] + SFCONV_GRATER_DX[2] * sub_delta,
                evaluations,
            )?;
            evaluations += 1;
            for point in 0..3 {
                high_order += SFCONV_GRATER_WT9[weight_index] * fval[current_region][point] * delta;
                low_order += fval[current_region][point] * SFCONV_GRATER_WT[point] * sub_delta;
                weight_index += 1;
            }
        }

        let difference = (high_order - low_order).abs();
        let fraction = delta / total_interval;
        let at_singularity = fraction <= 1.0e-8;
        if difference <= absolute_tolerance * fraction
            || difference <= relative_tolerance * high_order.abs()
            || (at_singularity && (fraction <= 1.0e-15 || difference <= absolute_tolerance * 0.1))
        {
            value_total += high_order;
            estimated_error += difference.abs();
            nstack -= 1;
            if nstack == 0 {
                return Ok(SfconvAdaptiveIntegral {
                    value: value_total,
                    estimated_error,
                    evaluations,
                    max_regions,
                });
            }
        } else {
            nstack += 2;
            max_regions = max_regions.max(nstack);
        }
    }
}

/// Port of `SFCONV/mksat.f90` `xmkesat`.
///
/// This is the extrinsic satellite with the quasiparticle pole subtracted and
/// quasiparticle broadening removed.
pub fn sfconv_extrinsic_satellite_debroadened(
    energy: Real,
    context: SfconvSatelliteContext,
    self_energy: SfconvSatelliteSelfEnergy,
) -> Result<Real, SfconvError> {
    validate_finite_scalar("satellite energy", energy)?;
    validate_satellite_context(context)?;
    validate_satellite_self_energy(self_energy)?;
    validate_nonzero_denominator("satellite energy", energy)?;

    let renormalization_magnitude = checked_hypot(
        "satellite renormalization",
        self_energy.renormalization_real,
        self_energy.renormalization_imag,
    )?;
    validate_nonzero_denominator("satellite renormalization", renormalization_magnitude)?;

    let width_difference = self_energy.width - self_energy.off_shell_imag;
    let energy_difference = energy + self_energy.on_shell_real - self_energy.off_shell_real;
    let denominator = energy_difference.powi(2) + width_difference.powi(2);
    validate_nonzero_denominator("extrinsic satellite", denominator)?;

    let total = -width_difference / denominator;
    let main = -self_energy.renormalization_imag
        / (energy * std::f64::consts::PI * renormalization_magnitude)
        * (-(energy / (2.0 * context.plasma_frequency)).powi(2)).exp();
    finite_result(
        "extrinsic satellite",
        total / (std::f64::consts::PI * renormalization_magnitude) - main,
    )
}

/// Port of `SFCONV/mksat.f90` `xmkgwext`.
///
/// This is the full-broadening extrinsic satellite including quasiparticle
/// contributions.
pub fn sfconv_extrinsic_satellite_broadened(
    energy: Real,
    self_energy: SfconvSatelliteSelfEnergy,
) -> Result<Real, SfconvError> {
    validate_finite_scalar("satellite energy", energy)?;
    validate_satellite_self_energy(self_energy)?;
    let energy_difference = energy + self_energy.on_shell_real - self_energy.off_shell_real;
    let denominator =
        std::f64::consts::PI * (energy_difference.powi(2) + self_energy.off_shell_imag.powi(2));
    validate_nonzero_denominator("broadened extrinsic satellite", denominator)?;
    finite_result(
        "broadened extrinsic satellite",
        self_energy.off_shell_imag / denominator,
    )
}

/// Port of `SFCONV/mksat.f90` `xintxsat`.
pub fn sfconv_interference_satellite_integrand(
    momentum: Real,
    energy: Real,
    width: Real,
    context: SfconvSatelliteContext,
) -> Result<Real, SfconvError> {
    validate_positive_scalar("momentum", momentum)?;
    validate_finite_scalar("satellite energy", energy)?;
    validate_positive_scalar("satellite width", width)?;
    validate_satellite_context(context)?;

    let dispersion =
        sfconv_pole_dispersion(momentum, context.pole_energy, context.dispersion_parameter)?;
    validate_nonzero_denominator("pole dispersion", dispersion)?;
    let coupling = sfconv_coupling_potential_squared(
        momentum,
        context.plasma_frequency,
        context.pole_energy,
        context.dispersion_parameter,
    )?;
    let tolerance = 0.2 * context.plasma_frequency;
    let energy_delta = context.photoelectron_energy - energy;
    let lorentzian =
        width / (std::f64::consts::PI * ((energy - dispersion).powi(2) + width.powi(2)));

    let factor = if energy_delta >= 0.0 {
        let wave_number = checked_sqrt("interference wave number", 2.0 * energy_delta)?;
        validate_nonzero_denominator("interference wave number", wave_number)?;
        let numerator = (dispersion - momentum.powi(2) / 2.0 + wave_number * momentum).powi(2)
            + tolerance.powi(2);
        let denominator = (dispersion - momentum.powi(2) / 2.0 - wave_number * momentum).powi(2)
            + tolerance.powi(2);
        validate_nonzero_denominator("interference logarithm", denominator)?;
        (numerator / denominator).ln() / 2.0 / wave_number
    } else {
        let wave_number = checked_sqrt("interference evanescent wave number", -2.0 * energy_delta)?;
        validate_nonzero_denominator("interference evanescent wave number", wave_number)?;
        let denominator = dispersion - momentum.powi(2) / 2.0;
        validate_nonzero_denominator("interference arctangent", denominator)?;
        (wave_number * momentum / denominator).atan() / wave_number
    };

    finite_result(
        "interference satellite integrand",
        momentum * coupling * lorentzian * factor / dispersion,
    )
}

/// Port of `SFCONV/mksat.f90` `xintisat`.
pub fn sfconv_intrinsic_satellite_integrand(
    momentum: Real,
    energy: Real,
    width: Real,
    context: SfconvSatelliteContext,
) -> Result<Real, SfconvError> {
    validate_positive_scalar("momentum", momentum)?;
    validate_finite_scalar("satellite energy", energy)?;
    validate_positive_scalar("satellite width", width)?;
    validate_satellite_context(context)?;

    let dispersion =
        sfconv_pole_dispersion(momentum, context.pole_energy, context.dispersion_parameter)?;
    validate_nonzero_denominator("pole dispersion", dispersion)?;
    let coupling = sfconv_coupling_potential_squared(
        momentum,
        context.plasma_frequency,
        context.pole_energy,
        context.dispersion_parameter,
    )?;
    let lorentzian =
        width / (((energy - dispersion).powi(2) + width.powi(2)) * std::f64::consts::PI);
    finite_result(
        "intrinsic satellite integrand",
        momentum.powi(2) * coupling * lorentzian / dispersion.powi(2),
    )
}

/// Port of `SFCONV/mksat.f90` `xmkxsat`.
pub fn sfconv_interference_satellite(
    energy: Real,
    width: Real,
    context: SfconvSatelliteContext,
) -> Result<SfconvSatelliteIntegral, SfconvError> {
    validate_finite_scalar("satellite energy", energy)?;
    validate_positive_scalar("satellite width", width)?;
    validate_satellite_context(context)?;
    let q2 = checked_sqrt(
        "interference satellite q2",
        (2.0 * (energy - context.pole_energy)).max(width),
    )?;
    validate_nonzero_denominator("interference satellite q2", q2)?;
    let qwidth = 10.0 * width / q2;
    let qmin = 0.0_f64.max(q2 - qwidth);
    let qmax = q2 + qwidth;
    let first = integrate_mksat_range(qmin, q2, context, |momentum, context| {
        sfconv_interference_satellite_integrand(momentum, energy, width, context)
    })?;
    let second = integrate_mksat_range(q2, qmax, context, |momentum, context| {
        sfconv_interference_satellite_integrand(momentum, energy, width, context)
    })?;
    combine_satellite_integrals(first, second, (2.0 * std::f64::consts::PI).powi(2))
}

/// Port of `SFCONV/mksat.f90` `xmkisat`.
pub fn sfconv_intrinsic_satellite(
    energy: Real,
    width: Real,
    context: SfconvSatelliteContext,
) -> Result<SfconvSatelliteIntegral, SfconvError> {
    validate_finite_scalar("satellite energy", energy)?;
    validate_positive_scalar("satellite width", width)?;
    validate_satellite_context(context)?;
    let q2 = if energy - context.pole_energy > width {
        checked_sqrt(
            "intrinsic satellite q2",
            2.0 * (energy - context.pole_energy),
        )?
    } else {
        checked_sqrt("intrinsic satellite q2", 2.0 * width)?
    };
    validate_nonzero_denominator("intrinsic satellite q2", q2)?;
    let qwidth = 10.0 * q2.min(width / q2);
    let qmax = q2 + qwidth;
    let first = integrate_mksat_range(0.0, q2, context, |momentum, context| {
        sfconv_intrinsic_satellite_integrand(momentum, energy, width, context)
    })?;
    let second = integrate_mksat_range(q2, qmax, context, |momentum, context| {
        sfconv_intrinsic_satellite_integrand(momentum, energy, width, context)
    })?;
    combine_satellite_integrals(first, second, 2.0 * std::f64::consts::PI.powi(2))
}

/// Port of `SFCONV/mksat.f90` `xintak`.
pub fn sfconv_interference_quasiparticle_integrand(
    momentum: Real,
    photoelectron_momentum: Real,
    context: SfconvSatelliteContext,
) -> Result<Real, SfconvError> {
    validate_positive_scalar("momentum", momentum)?;
    validate_positive_scalar("photoelectron_momentum", photoelectron_momentum)?;
    validate_satellite_context(context)?;

    let dispersion =
        sfconv_pole_dispersion(momentum, context.pole_energy, context.dispersion_parameter)?;
    validate_nonzero_denominator("pole dispersion", dispersion)?;
    let coupling = sfconv_coupling_potential_squared(
        momentum,
        context.plasma_frequency,
        context.pole_energy,
        context.dispersion_parameter,
    )?;
    let epsilon = 0.1_f64;
    let numerator = (dispersion + momentum.powi(2) / 2.0 + photoelectron_momentum * momentum)
        .powi(2)
        + (context.pole_energy * epsilon).powi(2);
    let denominator = (dispersion + momentum.powi(2) / 2.0 - photoelectron_momentum * momentum)
        .powi(2)
        + (context.pole_energy * epsilon).powi(2);
    validate_nonzero_denominator("quasiparticle logarithm", denominator)?;
    let log_factor = (numerator / denominator).ln() / 2.0;
    finite_result(
        "interference quasiparticle integrand",
        momentum * coupling * log_factor
            / (dispersion * photoelectron_momentum * 4.0 * std::f64::consts::PI.powi(2)),
    )
}

/// Port of `SFCONV/mksat.f90` `xmkak`.
pub fn sfconv_interference_quasiparticle(
    energy: Real,
    upper_energy: Real,
    context: SfconvSatelliteContext,
) -> Result<SfconvSatelliteIntegral, SfconvError> {
    validate_finite_scalar("satellite energy", energy)?;
    validate_finite_scalar("satellite upper energy", upper_energy)?;
    validate_satellite_context(context)?;
    if energy <= 0.0 {
        return Ok(SfconvSatelliteIntegral {
            value: 0.0,
            estimated_error: 0.0,
            evaluations: 0,
            max_regions: 0,
        });
    }
    let absolute_tolerance =
        checked_sqrt("quasiparticle tolerance", context.plasma_frequency)? * context.accuracy;
    let upper_momentum = checked_sqrt("quasiparticle upper momentum", 2.0 * upper_energy)?;
    let photoelectron_momentum = checked_sqrt(
        "quasiparticle photoelectron momentum",
        2.0 * context.photoelectron_energy,
    )?;
    validate_nonzero_denominator(
        "quasiparticle photoelectron momentum",
        photoelectron_momentum,
    )?;
    let integral = sfconv_grater_integrate(
        |momentum| {
            sfconv_interference_quasiparticle_integrand(momentum, photoelectron_momentum, context)
        },
        absolute_tolerance,
        upper_momentum,
        absolute_tolerance,
        context.accuracy,
        &[],
    )?;
    Ok(SfconvSatelliteIntegral {
        value: integral.value,
        estimated_error: integral.estimated_error,
        evaluations: integral.evaluations,
        max_regions: integral.max_regions,
    })
}

/// Port of `SFCONV/interpsf.f90`: interpolate spectral function to a uniform grid.
///
/// FEFF builds the scalar spectral function from rows 2, 5, and 4 of
/// `spectf` as `spectf(2,j) + spectf(5,j) - 2*spectf(4,j)`, then linearly
/// interpolates that combination from the minimal input grid to `output_len`
/// uniformly spaced points spanning the same energy range.
pub fn sfconv_interpolate_spectral_function(
    input: SfconvSpectralInterpolationInput<'_>,
) -> Result<SfconvSpectralInterpolation, SfconvError> {
    validate_count_at_least("output_len", input.output_len, 2)?;
    validate_count_at_least("energy", input.energy.len(), 2)?;
    validate_count_exact("spectral_function rows", input.spectral_function.nrows(), 8)?;
    validate_matching_lengths(
        "energy",
        input.energy.len(),
        "spectral_function columns",
        input.spectral_function.ncols(),
    )?;
    validate_finite_array("energy", input.energy)?;
    validate_strictly_increasing("energy", input.energy)?;
    validate_finite_spectral_rows(input.spectral_function)?;

    let last_input = input.energy.len() - 1;
    let first_energy = input.energy[0];
    let last_energy = input.energy[last_input];
    let step = (last_energy - first_energy) / (input.output_len as Real - 1.0);
    let mut energy = Array1::<Real>::zeros(input.output_len);
    let mut spectral_function = Array1::<Real>::zeros(input.output_len);

    energy[0] = first_energy;
    spectral_function[0] = combined_spectral_function(input.spectral_function, 0);
    energy[input.output_len - 1] = last_energy;
    spectral_function[input.output_len - 1] =
        combined_spectral_function(input.spectral_function, last_input);

    let mut lower = 0usize;
    for output in 1..(input.output_len - 1) {
        let output_energy = first_energy + step * output as Real;
        energy[output] = output_energy;

        while lower + 1 < last_input && output_energy >= input.energy[lower + 1] {
            lower += 1;
        }

        let upper = lower + 1;
        if !(input.energy[lower]..input.energy[upper]).contains(&output_energy) {
            return Err(SfconvError::NonFiniteResult {
                row: output,
                value: output_energy,
            });
        }
        let low = combined_spectral_function(input.spectral_function, lower);
        let high = combined_spectral_function(input.spectral_function, upper);
        let fraction =
            (output_energy - input.energy[lower]) / (input.energy[upper] - input.energy[lower]);
        spectral_function[output] = low + (high - low) * fraction;
    }

    Ok(SfconvSpectralInterpolation {
        energy,
        spectral_function,
    })
}

/// Port of `SFCONV/sfconvsub.f90`: spectral-function convolution.
///
/// The kernel integrates a signal over the spectral function, optionally
/// applying FEFF's available-energy cutoff and asymmetric quasiparticle phase
/// branch. Diagnostic file emission from the Fortran routine is intentionally
/// kept out of this pure numerical helper.
pub fn sfconv_convolve(
    input: SfconvConvolutionInput<'_>,
) -> Result<SfconvConvolution, SfconvError> {
    validate_convolution_input(input)?;

    let weights = input.weights;
    let pi = std::f64::consts::PI;
    let mut real_convolution = 0.0;
    let mut imag_convolution = 0.0;
    let quasiparticle_magnitude = if input.asymmetric_phase {
        weights[0]
    } else {
        weights[0].hypot(weights[1])
    };
    let quasiparticle_phase = if weights[0] != 0.0 && !input.asymmetric_phase {
        (weights[1] / weights[0]).atan()
    } else {
        0.0
    };
    let quasiparticle_reduction = if !input.cutoff {
        1.0
    } else if input.photoelectron_energy - input.chemical_potential != 0.0 {
        input
            .core_hole_lifetime
            .atan2(input.chemical_potential - input.photoelectron_energy)
            / pi
    } else {
        0.5
    };
    let quasiparticle_weight = quasiparticle_reduction * (quasiparticle_magnitude + weights[2]);
    let mut normalization = quasiparticle_weight;

    let mut cutoff_spectral_function = Array1::<Real>::zeros(input.spectral_function.len());
    for row in 0..input.spectral_function.len() {
        let width = integration_width(input.spectral_energy, input.spectral_function.len(), row);
        let excitation_energy = input.spectral_energy[row];
        let available_energy = input.photoelectron_energy - excitation_energy;
        let cutoff_weight = cutoff_weight(
            input.cutoff,
            available_energy,
            input.chemical_potential,
            input.core_hole_lifetime,
        );

        let mut value = if !input.cutoff {
            input.spectral_function[row]
        } else if excitation_energy >= 0.0 {
            input.spectral_function[row] * cutoff_weight
        } else {
            (input.spectral_function[row] * cutoff_weight).max(0.0)
        };
        if input.asymmetric_phase {
            let half_width = 0.5 * width;
            let smoothing = 3.0 * width;
            let log_ratio = (((excitation_energy + half_width).powi(2) + smoothing.powi(2))
                / ((excitation_energy - half_width).powi(2) + smoothing.powi(2)))
            .ln();
            value -= quasiparticle_reduction
                * (weights[1] / (pi * quasiparticle_magnitude * width))
                * log_ratio
                * (-(excitation_energy / (2.0 * input.plasma_frequency)).powi(2)).exp()
                / 2.0;
        }
        cutoff_spectral_function[row] = value;
        normalization += value * width;
    }
    if !normalization.is_finite() || normalization == 0.0 {
        return Err(SfconvError::InvalidNormalization {
            value: normalization,
        });
    }

    for row in 0..input.spectral_function.len() {
        let width = integration_width(input.spectral_energy, input.spectral_function.len(), row);
        let excitation_energy = input.spectral_energy[row];
        let available_energy = input.photoelectron_energy - excitation_energy;
        let signal = interpolated_signal(input, available_energy)?;
        if row > 0 && row + 1 < input.spectral_function.len() {
            let left_midpoint = 0.5 * (excitation_energy + input.spectral_energy[row - 1]);
            let right_midpoint = 0.5 * (excitation_energy + input.spectral_energy[row + 1]);
            if left_midpoint < 0.0 && right_midpoint >= 0.0 {
                real_convolution += quasiparticle_weight * signal;
            }
        }
        real_convolution += cutoff_spectral_function[row] * width * signal;
    }

    let stored_real = real_convolution;
    real_convolution =
        stored_real * quasiparticle_phase.cos() - imag_convolution * quasiparticle_phase.sin();
    imag_convolution =
        imag_convolution * quasiparticle_phase.cos() + stored_real * quasiparticle_phase.sin();
    real_convolution /= normalization;
    imag_convolution /= normalization;

    let amplitude = real_convolution.hypot(imag_convolution);
    let phase = imag_convolution.atan2(real_convolution);
    if !amplitude.is_finite() {
        return Err(SfconvError::NonFiniteResult {
            row: 0,
            value: amplitude,
        });
    }
    if !phase.is_finite() {
        return Err(SfconvError::NonFiniteResult {
            row: 1,
            value: phase,
        });
    }

    Ok(SfconvConvolution { amplitude, phase })
}

fn validate_convolution_input(input: SfconvConvolutionInput<'_>) -> Result<(), SfconvError> {
    validate_finite_scalar("photoelectron_energy", input.photoelectron_energy)?;
    validate_finite_scalar("chemical_potential", input.chemical_potential)?;
    validate_finite_scalar("core_hole_lifetime", input.core_hole_lifetime)?;
    validate_finite_scalar("plasma_frequency", input.plasma_frequency)?;
    validate_count_at_least("spectral_function", input.spectral_function.len(), 2)?;
    validate_count_at_least("signal", input.signal.len(), 2)?;
    validate_matching_lengths(
        "spectral_energy",
        input.spectral_energy.len(),
        "spectral_function",
        input.spectral_function.len(),
    )?;
    validate_matching_lengths(
        "signal_energy",
        input.signal_energy.len(),
        "signal",
        input.signal.len(),
    )?;
    validate_count_exact("weights", input.weights.len(), 8)?;
    validate_finite_array("spectral_energy", input.spectral_energy)?;
    validate_finite_array("spectral_function", input.spectral_function)?;
    validate_finite_array("signal_energy", input.signal_energy)?;
    validate_finite_array("signal", input.signal)?;
    validate_finite_array("weights", input.weights)?;
    validate_strictly_increasing("spectral_energy", input.spectral_energy)?;
    validate_strictly_increasing("signal_energy", input.signal_energy)?;
    if input.asymmetric_phase && input.weights[0] == 0.0 {
        return Err(SfconvError::ZeroAsymmetricWeight);
    }
    if input.asymmetric_phase && input.plasma_frequency == 0.0 {
        return Err(SfconvError::ZeroPlasmaFrequency);
    }
    Ok(())
}

fn validate_grater_input(
    lower: Real,
    upper: Real,
    absolute_tolerance: Real,
    relative_tolerance: Real,
    singularities: &[Real],
) -> Result<(), SfconvError> {
    validate_finite_scalar("grater lower", lower)?;
    validate_finite_scalar("grater upper", upper)?;
    if upper <= lower {
        return Err(SfconvError::InvalidIntegrationInterval { lower, upper });
    }
    validate_positive_tolerance("abr", absolute_tolerance)?;
    validate_positive_tolerance("rlr", relative_tolerance)?;
    if singularities.len() > SFCONV_GRATER_MAX_SINGULARITIES {
        return Err(SfconvError::TooManySingularities {
            count: singularities.len(),
            max: SFCONV_GRATER_MAX_SINGULARITIES,
        });
    }

    let mut previous = lower;
    for (index, &singularity) in singularities.iter().enumerate() {
        if !singularity.is_finite()
            || singularity <= lower
            || singularity >= upper
            || singularity <= previous
        {
            return Err(SfconvError::InvalidSingularity {
                index,
                value: singularity,
            });
        }
        previous = singularity;
    }
    Ok(())
}

fn validate_positive_tolerance(field: &'static str, value: Real) -> Result<(), SfconvError> {
    validate_finite_scalar(field, value)?;
    if value > 0.0 {
        Ok(())
    } else {
        Err(SfconvError::NonPositiveTolerance { field, value })
    }
}

fn eval_grater_integrand(
    integrand: &mut impl FnMut(Real) -> Result<Real, SfconvError>,
    argument: Real,
    row: usize,
) -> Result<Real, SfconvError> {
    validate_finite_scalar("grater argument", argument)?;
    let value = integrand(argument)?;
    if value.is_finite() {
        Ok(value)
    } else {
        Err(SfconvError::NonFiniteValue {
            field: "grater integrand",
            row,
            value,
        })
    }
}

fn validate_satellite_context(context: SfconvSatelliteContext) -> Result<(), SfconvError> {
    validate_positive_scalar("plasma_frequency", context.plasma_frequency)?;
    validate_positive_scalar("pole_energy", context.pole_energy)?;
    validate_finite_scalar("dispersion_parameter", context.dispersion_parameter)?;
    validate_positive_scalar("photoelectron_energy", context.photoelectron_energy)?;
    validate_positive_tolerance("accuracy", context.accuracy)
}

fn validate_self_energy_context(context: SfconvSelfEnergyContext) -> Result<(), SfconvError> {
    validate_positive_scalar("fermi_energy", context.fermi_energy)?;
    validate_positive_scalar("fermi_momentum", context.fermi_momentum)?;
    validate_positive_scalar("plasma_frequency", context.plasma_frequency)?;
    validate_positive_scalar("pole_energy", context.pole_energy)?;
    validate_finite_scalar("quasiparticle_energy", context.quasiparticle_energy)?;
    validate_positive_scalar("photoelectron_momentum", context.photoelectron_momentum)?;
    validate_positive_tolerance("accuracy", context.accuracy)?;
    validate_finite_scalar("dispersion_parameter", context.dispersion_parameter)
}

fn validate_satellite_self_energy(
    self_energy: SfconvSatelliteSelfEnergy,
) -> Result<(), SfconvError> {
    validate_finite_scalar("on_shell_real", self_energy.on_shell_real)?;
    validate_finite_scalar("satellite width", self_energy.width)?;
    validate_finite_scalar("renormalization_real", self_energy.renormalization_real)?;
    validate_finite_scalar("renormalization_imag", self_energy.renormalization_imag)?;
    validate_finite_scalar("off_shell_real", self_energy.off_shell_real)?;
    validate_finite_scalar("off_shell_imag", self_energy.off_shell_imag)
}

fn checked_hypot(field: &'static str, left: Real, right: Real) -> Result<Real, SfconvError> {
    validate_finite_scalar(field, left)?;
    validate_finite_scalar(field, right)?;
    let value = left.hypot(right);
    if value.is_finite() {
        Ok(value)
    } else {
        Err(SfconvError::NonFiniteScalar { field, value })
    }
}

fn validate_nonzero_denominator(field: &'static str, value: Real) -> Result<(), SfconvError> {
    validate_finite_scalar(field, value)?;
    if value == 0.0 {
        Err(SfconvError::ZeroDenominator { field })
    } else {
        Ok(())
    }
}

fn finite_result(field: &'static str, value: Real) -> Result<Real, SfconvError> {
    if value.is_finite() {
        Ok(value)
    } else {
        Err(SfconvError::NonFiniteScalar { field, value })
    }
}

fn beta_prefactor(context: SfconvSelfEnergyContext) -> Real {
    context.plasma_frequency.powi(2)
        / (4.0 * std::f64::consts::PI * context.photoelectron_momentum * context.pole_energy)
}

fn beta_log_argument(
    numerator_momentum: Real,
    numerator_dispersion: Real,
    denominator_momentum: Real,
    denominator_dispersion: Real,
    pole_energy: Real,
    dispersion_parameter: Real,
) -> Result<Real, SfconvError> {
    let numerator_denominator = pole_energy
        + numerator_dispersion
        + dispersion_parameter * numerator_momentum.powi(2) / (2.0 * pole_energy);
    let denominator_denominator = pole_energy
        + denominator_dispersion
        + dispersion_parameter * denominator_momentum.powi(2) / (2.0 * pole_energy);
    validate_nonzero_denominator("beta numerator", numerator_denominator)?;
    validate_nonzero_denominator("beta denominator", denominator_momentum)?;
    validate_nonzero_denominator("beta denominator", denominator_denominator)?;
    let argument = numerator_momentum.powi(2) / numerator_denominator * denominator_denominator
        / denominator_momentum.powi(2);
    validate_positive_scalar("beta logarithm", argument)?;
    Ok(argument)
}

fn integrate_mksat_range(
    lower: Real,
    upper: Real,
    context: SfconvSatelliteContext,
    mut integrand: impl FnMut(Real, SfconvSatelliteContext) -> Result<Real, SfconvError>,
) -> Result<SfconvAdaptiveIntegral, SfconvError> {
    sfconv_grater_integrate(
        |momentum| integrand(momentum, context),
        lower,
        upper,
        context.plasma_frequency * context.accuracy,
        context.accuracy,
        &[],
    )
}

fn combine_satellite_integrals(
    first: SfconvAdaptiveIntegral,
    second: SfconvAdaptiveIntegral,
    normalization: Real,
) -> Result<SfconvSatelliteIntegral, SfconvError> {
    validate_nonzero_denominator("satellite normalization", normalization)?;
    let value = finite_result(
        "satellite integral",
        (first.value + second.value) / normalization,
    )?;
    Ok(SfconvSatelliteIntegral {
        value,
        estimated_error: (first.estimated_error + second.estimated_error) / normalization.abs(),
        evaluations: first.evaluations + second.evaluations,
        max_regions: first.max_regions.max(second.max_regions),
    })
}

fn validate_dispersion_inputs(
    momentum: Real,
    pole_energy: Real,
    dispersion_parameter: Real,
) -> Result<(), SfconvError> {
    validate_finite_scalar("momentum", momentum)?;
    validate_positive_scalar("pole_energy", pole_energy)?;
    validate_finite_scalar("dispersion_parameter", dispersion_parameter)
}

fn pole_dispersion_value(
    momentum: Real,
    pole_energy: Real,
    dispersion_parameter: Real,
) -> Result<Real, SfconvError> {
    let radicand =
        pole_energy.powi(2) + dispersion_parameter * momentum.powi(2) + momentum.powi(4) / 4.0;
    checked_sqrt("pole_dispersion", radicand)
}

fn checked_sqrt(field: &'static str, value: Real) -> Result<Real, SfconvError> {
    if !value.is_finite() {
        return Err(SfconvError::NonFiniteScalar { field, value });
    }
    if value < 0.0 {
        return Err(SfconvError::NegativeRadicand { field, value });
    }
    Ok(value.sqrt())
}

fn threshold_factor(
    dispersion_parameter: Real,
    pole_energy: Real,
    root: Real,
) -> Result<Real, SfconvError> {
    let radicand =
        dispersion_parameter.powi(2) + (root.powi(2) / 2.0).powi(2) - pole_energy.powi(2);
    Ok(checked_sqrt("qthresh factor", radicand)? - dispersion_parameter)
}

fn roots_sorted_by_imag_descending(mut roots: [crate::Complex; 3]) -> [crate::Complex; 3] {
    loop {
        let mut swaps = 0;
        for index in 0..2 {
            if roots[index].im < roots[index + 1].im {
                roots.swap(index, index + 1);
                swaps += 1;
            }
        }
        if swaps == 0 {
            return roots;
        }
    }
}

fn select_threshold_root<F>(
    roots: [crate::Complex; 3],
    score: F,
) -> Result<crate::Complex, SfconvError>
where
    F: FnMut(Real) -> Result<Real, SfconvError>,
{
    let index = select_threshold_root_index(roots, score)?;
    Ok(roots[index])
}

fn select_threshold_root_index<F>(
    roots: [crate::Complex; 3],
    mut score: F,
) -> Result<usize, SfconvError>
where
    F: FnMut(Real) -> Result<Real, SfconvError>,
{
    let test0 = score(roots[0].re)?;
    let test1 = score(roots[1].re)?;
    let test2 = score(roots[2].re)?;
    if test0 < test1 && test0 < test2 {
        Ok(0)
    } else if test1 < test2 {
        Ok(1)
    } else {
        Ok(2)
    }
}

fn cutoff_weight(
    cutoff: bool,
    available_energy: Real,
    chemical_potential: Real,
    gamma: Real,
) -> Real {
    if !cutoff {
        1.0
    } else if available_energy - chemical_potential != 0.0 {
        gamma.atan2(chemical_potential - available_energy) / std::f64::consts::PI
    } else {
        0.5
    }
}

fn interpolated_signal(
    input: SfconvConvolutionInput<'_>,
    available_energy: Real,
) -> Result<Real, SfconvError> {
    let last = input.signal.len() - 1;
    if available_energy > input.signal_energy[last] {
        return Ok(input.signal[last]);
    }
    if available_energy <= input.signal_energy[0] {
        let amplitude = input.signal[0];
        let delta = input.chemical_potential - input.signal_energy[0];
        let lambda = delta.powi(2)
            / (std::f64::consts::PI
                * amplitude.abs()
                * (delta.powi(2) + input.core_hole_lifetime.powi(2)));
        let signal = amplitude * (lambda * (available_energy - input.signal_energy[0])).exp();
        if signal.is_finite() {
            return Ok(signal);
        }
        return Err(SfconvError::NonFiniteResult {
            row: 2,
            value: signal,
        });
    }

    for row in 0..last {
        if available_energy > input.signal_energy[row]
            && available_energy <= input.signal_energy[row + 1]
        {
            let fraction = (available_energy - input.signal_energy[row])
                / (input.signal_energy[row + 1] - input.signal_energy[row]);
            return Ok(input.signal[row] + (input.signal[row + 1] - input.signal[row]) * fraction);
        }
    }

    Err(SfconvError::NonFiniteResult {
        row: 3,
        value: available_energy,
    })
}

fn integration_width(energy: ArrayView1<'_, Real>, active_len: usize, row: usize) -> Real {
    if row == 0 {
        energy[1] - energy[0]
    } else if row + 1 == active_len {
        energy[active_len - 1] - energy[active_len - 2]
    } else {
        0.5 * (energy[row + 1] - energy[row - 1])
    }
}

fn combined_spectral_function(spectral_function: ArrayView2<'_, Real>, column: usize) -> Real {
    spectral_function[(1, column)] + spectral_function[(4, column)]
        - 2.0 * spectral_function[(3, column)]
}

fn validate_finite_spectral_rows(
    spectral_function: ArrayView2<'_, Real>,
) -> Result<(), SfconvError> {
    for &row in &[1, 3, 4] {
        for column in 0..spectral_function.ncols() {
            validate_finite_value(
                "spectral_function",
                column,
                spectral_function[(row, column)],
            )?;
        }
    }
    Ok(())
}

fn validate_matching_lengths(
    left: &'static str,
    left_len: usize,
    right: &'static str,
    right_len: usize,
) -> Result<(), SfconvError> {
    if left_len == right_len {
        Ok(())
    } else {
        Err(SfconvError::LengthMismatch {
            left,
            left_len,
            right,
            right_len,
        })
    }
}

fn validate_count_exact(
    field: &'static str,
    actual: usize,
    expected: usize,
) -> Result<(), SfconvError> {
    if actual == expected {
        Ok(())
    } else {
        Err(SfconvError::CountMismatch {
            field,
            actual,
            expected,
        })
    }
}

fn validate_count_at_least(
    name: &'static str,
    actual: usize,
    minimum: usize,
) -> Result<(), SfconvError> {
    if actual < minimum {
        Err(SfconvError::CountTooSmall {
            name,
            actual,
            minimum,
        })
    } else {
        Ok(())
    }
}

fn validate_active_len(
    field: &'static str,
    active_len: usize,
    len: usize,
) -> Result<(), SfconvError> {
    if active_len > len {
        Err(SfconvError::ActiveCountOutOfRange {
            field,
            active_len,
            len,
        })
    } else {
        Ok(())
    }
}

fn validate_finite_scalar(field: &'static str, value: Real) -> Result<(), SfconvError> {
    if value.is_finite() {
        Ok(())
    } else {
        Err(SfconvError::NonFiniteScalar { field, value })
    }
}

fn validate_positive_scalar(field: &'static str, value: Real) -> Result<(), SfconvError> {
    validate_finite_scalar(field, value)?;
    if value > 0.0 {
        Ok(())
    } else {
        Err(SfconvError::NonPositiveScalar { field, value })
    }
}

fn validate_finite_array(
    field: &'static str,
    values: ArrayView1<'_, Real>,
) -> Result<(), SfconvError> {
    for (row, value) in values.iter().copied().enumerate() {
        validate_finite_value(field, row, value)?;
    }
    Ok(())
}

fn validate_finite_value(field: &'static str, row: usize, value: Real) -> Result<(), SfconvError> {
    if value.is_finite() {
        Ok(())
    } else {
        Err(SfconvError::NonFiniteValue { field, row, value })
    }
}

fn validate_strictly_increasing(
    field: &'static str,
    values: ArrayView1<'_, Real>,
) -> Result<(), SfconvError> {
    for row in 1..values.len() {
        if values[row] <= values[row - 1] {
            return Err(SfconvError::NonIncreasingEnergy {
                field,
                row,
                previous: values[row - 1],
                current: values[row],
            });
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use ndarray::{Array1, Array2, ShapeBuilder, array};

    use crate::Real;

    use super::{
        SfconvAdaptiveIntegral, SfconvConvolutionInput, SfconvError, SfconvKramersKronigInput,
        SfconvPole, SfconvQLimits, SfconvSatelliteContext, SfconvSatelliteSelfEnergy,
        SfconvSelfEnergyContext, SfconvSpectralInterpolationInput, sfconv_convolve,
        sfconv_coupling_potential_squared, sfconv_extrinsic_beta,
        sfconv_extrinsic_satellite_broadened, sfconv_extrinsic_satellite_debroadened,
        sfconv_find_singularities, sfconv_free_electron_exchange, sfconv_grater_integrate,
        sfconv_imaginary_self_energy, sfconv_interference_quasiparticle,
        sfconv_interference_quasiparticle_integrand, sfconv_interference_satellite,
        sfconv_interference_satellite_integrand, sfconv_interpolate_spectral_function,
        sfconv_intrinsic_satellite, sfconv_intrinsic_satellite_integrand,
        sfconv_inverse_pole_dispersion, sfconv_kramers_kronig_real_part, sfconv_plasma_parameters,
        sfconv_plasmon_threshold_momentum, sfconv_pole_dispersion,
        sfconv_pole_dispersion_derivative, sfconv_pole_dispersion_second_derivative,
        sfconv_q_limits, sfconv_select_pole,
    };

    #[test]
    fn kramers_kronig_real_part_matches_feff_mkrmu_reference() -> Result<(), SfconvError> {
        let (imaginary, reference_imaginary, energy) = mkrmu_reference_inputs(25);

        let real_part = sfconv_kramers_kronig_real_part(SfconvKramersKronigInput {
            imaginary: imaginary.view(),
            reference_imaginary: reference_imaginary.view(),
            energy: energy.view(),
            active_len: 25,
        })?;

        let expected = [
            0.653_321_127_749_770_8,
            0.750_003_058_275_569_8,
            0.770_088_761_144_957_1,
            0.744_953_602_096_770_5,
            0.685_875_097_053_667_7,
            0.599_956_814_602_449_9,
            0.492_993_575_338_788_3,
            0.370_329_818_936_448_6,
            0.237_144_234_118_930_07,
            0.098_519_596_973_469_21,
            -0.040_581_567_325_286_456,
            -0.175_385_521_001_154_32,
            -0.301_395_336_623_902_3,
            -0.414_483_981_972_534_94,
            -0.510_982_552_336_513_5,
            -0.587_755_578_520_523_2,
            -0.642_255_441_484_044_2,
            -0.672_546_008_587_787_2,
            -0.677_279_884_911_601_4,
            -0.631_242_351_812_862_9,
            -0.631_242_351_812_862_9,
            -0.530_174_264_181_443_8,
            -0.422_544_809_832_420_15,
            -0.273_383_187_221_121_7,
            -0.036_668_636_491_773_95,
        ];
        for (actual, expected) in real_part.iter().zip(expected) {
            assert_close(*actual, expected, 1.0e-13);
        }
        Ok(())
    }

    #[test]
    fn kramers_kronig_real_part_rejects_invalid_inputs() {
        let (imaginary, reference_imaginary, energy) = mkrmu_reference_inputs(21);

        assert!(matches!(
            sfconv_kramers_kronig_real_part(SfconvKramersKronigInput {
                imaginary: imaginary.view(),
                reference_imaginary: reference_imaginary.view(),
                energy: energy.view(),
                active_len: 20,
            }),
            Err(SfconvError::CountTooSmall {
                name: "active_len",
                ..
            })
        ));
        assert!(matches!(
            sfconv_kramers_kronig_real_part(SfconvKramersKronigInput {
                imaginary: imaginary.view(),
                reference_imaginary: reference_imaginary.view(),
                energy: energy.view(),
                active_len: 22,
            }),
            Err(SfconvError::ActiveCountOutOfRange {
                field: "imaginary",
                ..
            })
        ));

        let mut bad_imaginary = imaginary.clone();
        bad_imaginary[3] = f64::NAN;
        assert!(matches!(
            sfconv_kramers_kronig_real_part(SfconvKramersKronigInput {
                imaginary: bad_imaginary.view(),
                reference_imaginary: reference_imaginary.view(),
                energy: energy.view(),
                active_len: 21,
            }),
            Err(SfconvError::NonFiniteValue {
                field: "imaginary",
                row: 3,
                ..
            })
        ));

        let mut bad_energy = energy.clone();
        bad_energy[5] = bad_energy[4];
        assert!(matches!(
            sfconv_kramers_kronig_real_part(SfconvKramersKronigInput {
                imaginary: imaginary.view(),
                reference_imaginary: reference_imaginary.view(),
                energy: bad_energy.view(),
                active_len: 21,
            }),
            Err(SfconvError::NonIncreasingEnergy { row: 5, .. })
        ));
    }

    #[test]
    fn selects_pole_parameters_matches_feff_plset_reference() -> Result<(), SfconvError> {
        let (energy, weight, broadening) = plset_reference_inputs();

        assert_pole_close(
            sfconv_select_pole(3, energy.view(), weight.view(), broadening.view())?,
            SfconvPole {
                energy: 0.495,
                weight: 0.46,
                broadening: 0.048,
            },
        );
        assert_pole_close(
            sfconv_select_pole(5, energy.view(), weight.view(), broadening.view())?,
            SfconvPole {
                energy: 0.975,
                weight: 0.600_000_000_000_000_1,
                broadening: 0.1,
            },
        );
        Ok(())
    }

    #[test]
    fn selects_pole_parameters_rejects_invalid_inputs() {
        let (energy, weight, broadening) = plset_reference_inputs();

        assert!(matches!(
            sfconv_select_pole(0, energy.view(), weight.view(), broadening.view()),
            Err(SfconvError::IndexOutOfRange {
                field: "pole",
                index: 0,
                len: 5,
            })
        ));
        assert!(matches!(
            sfconv_select_pole(6, energy.view(), weight.view(), broadening.view()),
            Err(SfconvError::IndexOutOfRange {
                field: "pole",
                index: 6,
                len: 5,
            })
        ));

        let short_weight = Array1::from_iter(weight.iter().copied().take(4));
        assert!(matches!(
            sfconv_select_pole(1, energy.view(), short_weight.view(), broadening.view()),
            Err(SfconvError::LengthMismatch {
                left: "energy",
                right: "weight",
                ..
            })
        ));

        let mut bad_energy = energy.clone();
        bad_energy[2] = f64::NAN;
        assert!(matches!(
            sfconv_select_pole(3, bad_energy.view(), weight.view(), broadening.view()),
            Err(SfconvError::NonFiniteValue {
                field: "energy",
                row: 2,
                ..
            })
        ));
    }

    #[test]
    fn plasma_parameters_match_feff_ppset_reference() -> Result<(), SfconvError> {
        let first = sfconv_plasma_parameters(2.35)?;
        assert_close(first.fermi_momentum, 0.816_663_103_267_026_7, 1.0e-15);
        assert_close(first.fermi_energy, 0.333_469_312_118_865_2, 1.0e-15);
        assert_close(first.plasma_frequency, 0.480_793_772_651_942_2, 1.0e-15);

        let second = sfconv_plasma_parameters(0.95)?;
        assert_close(second.fermi_momentum, 2.020_166_623_871_066, 1.0e-15);
        assert_close(second.fermi_energy, 2.040_536_594_101_310_7, 1.0e-15);
        assert_close(second.plasma_frequency, 1.870_575_403_449_765_5, 1.0e-15);
        Ok(())
    }

    #[test]
    fn plasma_parameters_reject_invalid_radius() {
        assert_eq!(
            sfconv_plasma_parameters(0.0),
            Err(SfconvError::NonPositiveScalar {
                field: "wigner_seitz_radius",
                value: 0.0,
            })
        );
        assert!(matches!(
            sfconv_plasma_parameters(f64::NAN),
            Err(SfconvError::NonFiniteScalar {
                field: "wigner_seitz_radius",
                ..
            })
        ));
    }

    #[test]
    fn pole_dispersion_helpers_match_feff_ppole_reference() -> Result<(), SfconvError> {
        let pole_energy = 0.47;
        let dispersion_parameter = 0.28;
        let plasma_frequency = 0.62;

        assert_close(
            sfconv_pole_dispersion(0.35, pole_energy, dispersion_parameter)?,
            0.508_872_835_293_848_2,
            1.0e-15,
        );
        assert_close(
            sfconv_pole_dispersion_derivative(0.35, pole_energy, dispersion_parameter)?,
            0.234_709_915_161_871_29,
            1.0e-15,
        );
        assert_close(
            sfconv_pole_dispersion_second_derivative(0.35, pole_energy, dispersion_parameter)?,
            0.803_071_469_689_919_9,
            1.0e-15,
        );
        assert_close(
            sfconv_inverse_pole_dispersion(0.80, pole_energy, dispersion_parameter)?,
            0.922_319_683_172_048_9,
            1.0e-15,
        );
        assert_close(
            sfconv_coupling_potential_squared(
                0.35,
                plasma_frequency,
                pole_energy,
                dispersion_parameter,
            )?,
            38.745_198_544_546_376,
            1.0e-14,
        );

        assert_close(
            sfconv_pole_dispersion(1.70, pole_energy, dispersion_parameter)?,
            1.765_821_338_641_030_2,
            1.0e-15,
        );
        assert_close(
            sfconv_pole_dispersion_derivative(1.70, pole_energy, dispersion_parameter)?,
            1.660_700_284_807_318_7,
            1.0e-15,
        );
        assert_close(
            sfconv_pole_dispersion_second_derivative(1.70, pole_energy, dispersion_parameter)?,
            1.051_677_496_133_378_6,
            1.0e-15,
        );
        assert_close(
            sfconv_inverse_pole_dispersion(0.30, pole_energy, dispersion_parameter)?,
            0.0,
            0.0,
        );
        assert_close(
            sfconv_coupling_potential_squared(
                1.70,
                plasma_frequency,
                pole_energy,
                dispersion_parameter,
            )?,
            0.473_280_535_773_200_1,
            1.0e-15,
        );
        Ok(())
    }

    #[test]
    fn pole_dispersion_helpers_reject_invalid_inputs() {
        assert!(matches!(
            sfconv_pole_dispersion(f64::NAN, 0.47, 0.28),
            Err(SfconvError::NonFiniteScalar {
                field: "momentum",
                ..
            })
        ));
        assert_eq!(
            sfconv_pole_dispersion(0.35, 0.0, 0.28),
            Err(SfconvError::NonPositiveScalar {
                field: "pole_energy",
                value: 0.0,
            })
        );
        assert_eq!(
            sfconv_coupling_potential_squared(0.0, 0.62, 0.47, 0.28),
            Err(SfconvError::NonPositiveScalar {
                field: "momentum",
                value: 0.0,
            })
        );
        assert!(matches!(
            sfconv_pole_dispersion(1.0, 0.47, -10.0),
            Err(SfconvError::NegativeRadicand {
                field: "pole_dispersion",
                ..
            })
        ));
    }

    #[test]
    fn q_limits_match_feff_qlimits_reference() -> Result<(), SfconvError> {
        assert_q_limits_close(
            sfconv_q_limits(1.15, 1.05, 0.47, 0.28, 12.0)?,
            SfconvQLimits {
                count: 3,
                q1: 0.112_905_963_336_969_05,
                q2: 1.252_615_998_981_518,
                q3: 0.926_614_797_549_310_8,
            },
            1.0e-14,
        );
        assert_q_limits_close(
            sfconv_q_limits(0.55, 0.92, 0.47, 0.28, 3.0)?,
            SfconvQLimits {
                count: 1,
                q1: 0.0,
                q2: 0.0,
                q3: 0.590_402_885_211_133_4,
            },
            1.0e-14,
        );
        assert_q_limits_close(
            sfconv_q_limits(2.40, 0.60, 0.47, 0.28, 0.75)?,
            SfconvQLimits {
                count: 3,
                q1: 0.75,
                q2: 0.75,
                q3: 4.179_832_657_474_71,
            },
            1.0e-14,
        );
        Ok(())
    }

    #[test]
    fn q_limits_reject_invalid_inputs() {
        assert!(matches!(
            sfconv_q_limits(1.15, f64::NAN, 0.47, 0.28, 12.0),
            Err(SfconvError::NonFiniteScalar {
                field: "photoelectron_momentum",
                ..
            })
        ));
        assert_eq!(
            sfconv_q_limits(1.15, 0.0, 0.47, 0.28, 12.0),
            Err(SfconvError::NonPositiveScalar {
                field: "photoelectron_momentum",
                value: 0.0,
            })
        );
        assert_eq!(
            sfconv_q_limits(1.15, 1.05, 0.47, 0.28, 0.0),
            Err(SfconvError::NonPositiveScalar {
                field: "upper_limit",
                value: 0.0,
            })
        );
    }

    #[test]
    fn plasmon_threshold_momentum_matches_feff_qthresh_reference() -> Result<(), SfconvError> {
        assert_close(
            sfconv_plasmon_threshold_momentum(0.47, 0.28, 0.42, 0.88)?,
            0.972_154_268_542_323_2,
            1.0e-14,
        );
        assert_close(
            sfconv_plasmon_threshold_momentum(0.75, 0.31, 0.55, 1.05)?,
            1.230_338_193_805_480_7,
            1.0e-14,
        );
        Ok(())
    }

    #[test]
    fn plasmon_threshold_momentum_rejects_invalid_inputs() {
        assert_eq!(
            sfconv_plasmon_threshold_momentum(0.0, 0.28, 0.42, 0.88),
            Err(SfconvError::NonPositiveScalar {
                field: "pole_energy",
                value: 0.0,
            })
        );
        assert_eq!(
            sfconv_plasmon_threshold_momentum(0.47, 0.28, 0.0, 0.88),
            Err(SfconvError::NonPositiveScalar {
                field: "fermi_energy",
                value: 0.0,
            })
        );
    }

    #[test]
    fn senergies_beta_helpers_match_feff_reference() -> Result<(), SfconvError> {
        let lowq0_context = senergies_reference_context(false);
        assert_close(
            sfconv_free_electron_exchange(1.0, lowq0_context.fermi_momentum)?,
            -std::f64::consts::FRAC_1_PI,
            1.0e-15,
        );
        assert_close(
            sfconv_free_electron_exchange(1.35, lowq0_context.fermi_momentum)?,
            -0.133_662_411_513_184_28,
            1.0e-15,
        );
        assert_close(
            sfconv_extrinsic_beta(0.36, lowq0_context)?,
            0.287_008_463_933_952_74,
            1.0e-14,
        );
        assert_close(
            sfconv_extrinsic_beta(0.95, lowq0_context)?,
            0.099_242_494_271_372_31,
            1.0e-14,
        );
        assert_close(
            sfconv_imaginary_self_energy(0.36, lowq0_context)?,
            -0.901_663_681_812_997,
            1.0e-14,
        );

        let lowq1_context = senergies_reference_context(true);
        assert_close(sfconv_extrinsic_beta(-0.20, lowq1_context)?, 0.0, 0.0);
        assert_close(
            sfconv_extrinsic_beta(0.36, lowq1_context)?,
            0.287_008_463_933_952_74,
            1.0e-14,
        );
        assert_close(
            sfconv_imaginary_self_energy(-0.20, lowq1_context)?,
            0.0,
            0.0,
        );
        Ok(())
    }

    #[test]
    fn finds_senergies_split_points_like_feff() -> Result<(), SfconvError> {
        let candidates = array![0.90, 0.20, 1.40, 0.70, -0.10];

        let forward = sfconv_find_singularities(0.15, 1.00, candidates.view())?;
        assert_real_slice_close(&forward, &[0.20, 0.70, 0.90], 0.0);

        let reverse = sfconv_find_singularities(1.00, 0.15, candidates.view())?;
        assert_real_slice_close(&reverse, &[0.20, 0.70, 0.90], 0.0);

        let empty = sfconv_find_singularities(0.15, 0.15, candidates.view())?;
        assert!(empty.is_empty());
        Ok(())
    }

    #[test]
    fn senergies_helpers_reject_invalid_inputs() {
        let context = senergies_reference_context(false);
        assert_eq!(
            sfconv_free_electron_exchange(0.0, 1.0),
            Err(SfconvError::NonPositiveScalar {
                field: "momentum",
                value: 0.0,
            })
        );
        assert!(matches!(
            sfconv_extrinsic_beta(
                0.36,
                SfconvSelfEnergyContext {
                    photoelectron_momentum: 0.0,
                    ..context
                },
            ),
            Err(SfconvError::NonPositiveScalar {
                field: "photoelectron_momentum",
                ..
            })
        ));
        assert!(matches!(
            sfconv_find_singularities(0.0, 1.0, array![0.2, f64::NAN].view()),
            Err(SfconvError::NonFiniteValue {
                field: "singularity candidate",
                row: 1,
                ..
            })
        ));
    }

    #[test]
    fn grater_integrate_matches_feff_reference() -> Result<(), SfconvError> {
        assert_integral_close(
            sfconv_grater_integrate(
                |x| Ok(x.powi(4) - 2.0 * x + 1.0),
                -0.25,
                1.75,
                1.0e-6,
                1.0e-6,
                &[],
            )?,
            SfconvAdaptiveIntegral {
                value: 2.282_812_623_992_166_7,
                estimated_error: 1.651_258_862_978_011_2e-8,
                evaluations: 9,
                max_regions: 1,
            },
            1.0e-14,
        );

        assert_integral_close(
            sfconv_grater_integrate(
                |x| Ok((5.0 * x).sin() / (1.0 + x * x)),
                0.0,
                4.0,
                1.0e-6,
                1.0e-6,
                &[],
            )?,
            SfconvAdaptiveIntegral {
                value: 0.214_866_405_696_591,
                estimated_error: 2.960_202_197_766_978_5e-7,
                evaluations: 135,
                max_regions: 6,
            },
            1.0e-13,
        );

        assert_integral_close(
            sfconv_grater_integrate(
                |x| Ok((x - 0.3).abs() + 0.25 * (x - 0.8).abs()),
                -1.0,
                2.0,
                1.0e-6,
                1.0e-6,
                &[0.3, 0.8],
            )?,
            SfconvAdaptiveIntegral {
                value: 2.874_999_978_367_709_4,
                estimated_error: 1.071_163_531_207_730_6e-7,
                evaluations: 27,
                max_regions: 3,
            },
            1.0e-14,
        );
        Ok(())
    }

    #[test]
    fn grater_integrate_rejects_invalid_inputs() {
        assert_eq!(
            sfconv_grater_integrate(Ok, 1.0, 1.0, 1.0e-6, 1.0e-6, &[]),
            Err(SfconvError::InvalidIntegrationInterval {
                lower: 1.0,
                upper: 1.0,
            })
        );
        assert_eq!(
            sfconv_grater_integrate(Ok, 0.0, 1.0, 0.0, 1.0e-6, &[]),
            Err(SfconvError::NonPositiveTolerance {
                field: "abr",
                value: 0.0,
            })
        );
        assert_eq!(
            sfconv_grater_integrate(Ok, 0.0, 1.0, 1.0e-6, 1.0e-6, &[0.5, 0.4]),
            Err(SfconvError::InvalidSingularity {
                index: 1,
                value: 0.4,
            })
        );
        assert!(matches!(
            sfconv_grater_integrate(|_| Ok(f64::NAN), 0.0, 1.0, 1.0e-6, 1.0e-6, &[]),
            Err(SfconvError::NonFiniteValue {
                field: "grater integrand",
                ..
            })
        ));
    }

    #[test]
    fn mksat_helpers_match_feff_reference() -> Result<(), SfconvError> {
        let context = mksat_reference_context();
        let self_energy = mksat_reference_self_energy();

        assert_close(
            sfconv_extrinsic_satellite_debroadened(0.36, context, self_energy)?,
            -0.044_294_665_346_589_21,
            1.0e-14,
        );
        assert_close(
            sfconv_extrinsic_satellite_broadened(0.36, self_energy)?,
            0.039_176_601_376_466_56,
            1.0e-14,
        );
        assert_close(
            sfconv_interference_satellite_integrand(0.55, 0.32, 0.045, context)?,
            4.656_810_436_207_971,
            1.0e-13,
        );
        assert_close(
            sfconv_intrinsic_satellite_integrand(0.55, 0.32, 0.045, context)?,
            2.780_182_754_299_514_3,
            1.0e-13,
        );
        assert_close(
            sfconv_interference_satellite_integrand(0.55, 0.95, 0.045, context)?,
            1.568_981_693_763_851_9,
            1.0e-13,
        );

        let interference = sfconv_interference_satellite(0.75, 0.045, context)?;
        assert_close(interference.value, 0.742_287_519_666_663_1, 1.0e-12);
        assert!(interference.evaluations > 0);
        assert!(interference.max_regions > 0);

        let intrinsic = sfconv_intrinsic_satellite(0.75, 0.045, context)?;
        assert_close(intrinsic.value, 0.496_852_311_955_514_77, 1.0e-12);
        assert!(intrinsic.evaluations > 0);
        assert!(intrinsic.max_regions > 0);

        let quasiparticle = sfconv_interference_quasiparticle(0.35, 2.40, context)?;
        assert_close(quasiparticle.value, 0.882_200_373_088_965_2, 1.0e-12);
        assert!(quasiparticle.evaluations > 0);
        assert!(quasiparticle.max_regions > 0);

        assert_close(
            sfconv_interference_quasiparticle(-0.01, 2.40, context)?.value,
            0.0,
            0.0,
        );
        assert_close(
            sfconv_interference_quasiparticle_integrand(0.55, (2.0_f64 * 0.85).sqrt(), context)?,
            0.886_179_631_715_177_2,
            1.0e-13,
        );
        Ok(())
    }

    #[test]
    fn mksat_helpers_reject_invalid_inputs() {
        let context = mksat_reference_context();
        let self_energy = mksat_reference_self_energy();
        assert_eq!(
            sfconv_extrinsic_satellite_debroadened(0.0, context, self_energy),
            Err(SfconvError::ZeroDenominator {
                field: "satellite energy",
            })
        );
        assert!(matches!(
            sfconv_interference_satellite_integrand(0.0, 0.32, 0.045, context),
            Err(SfconvError::NonPositiveScalar {
                field: "momentum",
                ..
            })
        ));
        assert!(matches!(
            sfconv_intrinsic_satellite(0.75, 0.0, context),
            Err(SfconvError::NonPositiveScalar {
                field: "satellite width",
                ..
            })
        ));
        assert!(matches!(
            sfconv_interference_quasiparticle(0.35, -1.0, context),
            Err(SfconvError::NegativeRadicand { .. })
        ));
    }

    #[test]
    fn interpolates_spectral_function_matches_feff_interpsf_reference() -> Result<(), SfconvError> {
        let (energy, spectral_function) = interpsf_reference_inputs();
        let interpolation =
            sfconv_interpolate_spectral_function(SfconvSpectralInterpolationInput {
                energy: energy.view(),
                spectral_function: spectral_function.view(),
                output_len: 13,
            })?;

        let expected_energy = [
            -2.0,
            -1.727_590_833_333_333_4,
            -1.455_181_666_666_666_8,
            -1.182_772_5,
            -0.910_363_333_333_333_4,
            -0.637_954_166_666_666_8,
            -0.365_545,
            -0.093_135_833_333_333_42,
            0.179_273_333_333_333_17,
            0.451_682_5,
            0.724_091_666_666_666_4,
            0.996_500_833_333_333_2,
            1.268_91,
        ];
        let expected_spectral_function = [
            -0.03,
            -0.035_578_048_005_086_65,
            -0.040_441_264_512_519_18,
            -0.044_809_714_285_714_24,
            -0.048_809_091_974_223_85,
            -0.052_519_432_577_500_3,
            -0.055_996_334_265_299_72,
            -0.059_278_128_963_028_02,
            -0.062_395_108_746_383_016,
            -0.065_369_121_964_238_19,
            -0.068_218_832_777_920_12,
            -0.070_958_429_921_906_93,
            -0.073_599_999_999_999_89,
        ];

        assert_real_slice_close(&interpolation.energy, &expected_energy, 1.0e-15);
        assert_real_slice_close(
            &interpolation.spectral_function,
            &expected_spectral_function,
            1.0e-14,
        );
        Ok(())
    }

    #[test]
    fn interpolates_spectral_function_rejects_invalid_inputs() {
        let (energy, spectral_function) = interpsf_reference_inputs();

        assert!(matches!(
            sfconv_interpolate_spectral_function(SfconvSpectralInterpolationInput {
                energy: energy.view(),
                spectral_function: spectral_function.view(),
                output_len: 1,
            }),
            Err(SfconvError::CountTooSmall {
                name: "output_len",
                ..
            })
        ));

        let short_rows =
            Array2::from_shape_fn((7, spectral_function.ncols()).f(), |(row, column)| {
                spectral_function[(row, column)]
            });
        assert!(matches!(
            sfconv_interpolate_spectral_function(SfconvSpectralInterpolationInput {
                energy: energy.view(),
                spectral_function: short_rows.view(),
                output_len: 13,
            }),
            Err(SfconvError::CountMismatch {
                field: "spectral_function rows",
                actual: 7,
                expected: 8,
            })
        ));

        let short_energy = Array1::from_iter(energy.iter().copied().take(100));
        assert!(matches!(
            sfconv_interpolate_spectral_function(SfconvSpectralInterpolationInput {
                energy: short_energy.view(),
                spectral_function: spectral_function.view(),
                output_len: 13,
            }),
            Err(SfconvError::LengthMismatch {
                left: "energy",
                right: "spectral_function columns",
                ..
            })
        ));

        let mut bad_energy = energy.clone();
        bad_energy[10] = bad_energy[9];
        assert!(matches!(
            sfconv_interpolate_spectral_function(SfconvSpectralInterpolationInput {
                energy: bad_energy.view(),
                spectral_function: spectral_function.view(),
                output_len: 13,
            }),
            Err(SfconvError::NonIncreasingEnergy { row: 10, .. })
        ));
    }

    #[test]
    fn convolve_matches_feff_sfconvsub_reference() -> Result<(), SfconvError> {
        let reference = sfconvsub_reference_inputs();

        let cutoff_phase = sfconv_convolve(SfconvConvolutionInput {
            photoelectron_energy: 1.35,
            chemical_potential: 0.15,
            core_hole_lifetime: 0.08,
            signal_energy: reference.signal_energy.view(),
            signal: reference.signal.view(),
            spectral_energy: reference.spectral_energy.view(),
            spectral_function: reference.spectral_function.view(),
            weights: reference.weights.view(),
            asymmetric_phase: false,
            cutoff: true,
            plasma_frequency: 0.55,
        })?;
        assert_close(cutoff_phase.amplitude, 0.404_768_834_000_475_8, 1.0e-14);
        assert_close(cutoff_phase.phase, 0.244_978_663_126_864_14, 1.0e-14);

        let no_cutoff_phase = sfconv_convolve(SfconvConvolutionInput {
            cutoff: false,
            ..sfconv_reference_input(
                reference.signal_energy.view(),
                reference.signal.view(),
                reference.spectral_energy.view(),
                reference.spectral_function.view(),
                reference.weights.view(),
            )
        })?;
        assert_close(no_cutoff_phase.amplitude, 0.405_036_447_280_840_4, 1.0e-14);
        assert_close(no_cutoff_phase.phase, 0.244_978_663_126_864_14, 1.0e-14);

        let asym_cutoff = sfconv_convolve(SfconvConvolutionInput {
            asymmetric_phase: true,
            ..sfconv_reference_input(
                reference.signal_energy.view(),
                reference.signal.view(),
                reference.spectral_energy.view(),
                reference.spectral_function.view(),
                reference.weights.view(),
            )
        })?;
        assert_close(asym_cutoff.amplitude, 0.394_308_834_584_619_57, 1.0e-14);
        assert_close(asym_cutoff.phase, 0.0, 1.0e-14);
        Ok(())
    }

    #[test]
    fn convolve_rejects_invalid_inputs() {
        let reference = sfconvsub_reference_inputs();

        let short_signal = array![0.62, 0.82, 0.74, 0.48, 0.22];
        assert!(matches!(
            sfconv_convolve(SfconvConvolutionInput {
                signal: short_signal.view(),
                ..sfconv_reference_input(
                    reference.signal_energy.view(),
                    reference.signal.view(),
                    reference.spectral_energy.view(),
                    reference.spectral_function.view(),
                    reference.weights.view(),
                )
            }),
            Err(SfconvError::LengthMismatch {
                left: "signal_energy",
                ..
            })
        ));

        let bad_spectral_energy = array![-0.18, -0.04, 0.0, 0.0, 0.31, 0.55, 0.82];
        assert!(matches!(
            sfconv_convolve(SfconvConvolutionInput {
                spectral_energy: bad_spectral_energy.view(),
                ..sfconv_reference_input(
                    reference.signal_energy.view(),
                    reference.signal.view(),
                    reference.spectral_energy.view(),
                    reference.spectral_function.view(),
                    reference.weights.view(),
                )
            }),
            Err(SfconvError::NonIncreasingEnergy {
                field: "spectral_energy",
                row: 3,
                ..
            })
        ));

        let zero_asym_weight = array![0.0, 0.18, 0.11, 0.0, 0.0, 0.0, 0.0, 0.0];
        assert!(matches!(
            sfconv_convolve(SfconvConvolutionInput {
                weights: zero_asym_weight.view(),
                asymmetric_phase: true,
                ..sfconv_reference_input(
                    reference.signal_energy.view(),
                    reference.signal.view(),
                    reference.spectral_energy.view(),
                    reference.spectral_function.view(),
                    reference.weights.view(),
                )
            }),
            Err(SfconvError::ZeroAsymmetricWeight)
        ));
    }

    fn mkrmu_reference_inputs(count: usize) -> (Array1<Real>, Array1<Real>, Array1<Real>) {
        let indices = (1..=count).map(|index| index as Real);
        let imaginary = Array1::from_iter(
            indices
                .clone()
                .map(|index| (0.17 * index).sin() + 0.01 * index),
        );
        let reference_imaginary =
            Array1::from_iter(indices.clone().map(|index| 0.2 * (0.11 * index).cos()));
        let energy = Array1::from_iter((0..count).map(|index| {
            let index = index as Real;
            0.05 * index + 0.002 * index * index
        }));
        (imaginary, reference_imaginary, energy)
    }

    fn plset_reference_inputs() -> (Array1<Real>, Array1<Real>, Array1<Real>) {
        let energy = Array1::from_shape_fn(5, |index| {
            let i = index as Real + 1.0;
            0.12 * i + 0.015 * i * i
        });
        let weight = Array1::from_shape_fn(5, |index| {
            let i = index as Real + 1.0;
            0.25 + 0.07 * i
        });
        let broadening = Array1::from_shape_fn(5, |index| {
            let i = index as Real + 1.0;
            0.01 * i + 0.002 * i * i
        });
        (energy, weight, broadening)
    }

    fn interpsf_reference_inputs() -> (Array1<Real>, Array2<Real>) {
        let count = 110usize;
        let energy = Array1::from_shape_fn(count, |index| {
            let i = index as Real;
            -2.0 + 0.018 * i + 0.000_11 * i * i
        });
        let spectral_function = Array2::from_shape_fn((8, count).f(), |(row, column)| {
            let fortran_row = row as Real + 1.0;
            let i = column as Real;
            0.03 * fortran_row + 0.002 * i + 0.000_4 * fortran_row * i + 0.000_01 * i * i
        });
        (energy, spectral_function)
    }

    struct SfconvSubReference {
        spectral_energy: Array1<Real>,
        spectral_function: Array1<Real>,
        signal_energy: Array1<Real>,
        signal: Array1<Real>,
        weights: Array1<Real>,
    }

    fn sfconvsub_reference_inputs() -> SfconvSubReference {
        SfconvSubReference {
            spectral_energy: array![-0.18, -0.04, 0.0, 0.12, 0.31, 0.55, 0.82],
            spectral_function: array![0.05, 0.18, 0.30, 0.23, 0.14, 0.07, 0.02],
            signal_energy: array![0.40, 0.72, 0.95, 1.22, 1.58, 1.95],
            signal: array![0.62, 0.82, 0.74, 0.48, 0.22, 0.12],
            weights: array![0.72, 0.18, 0.11, 0.0, 0.0, 0.0, 0.0, 0.0],
        }
    }

    fn sfconv_reference_input<'a>(
        signal_energy: ndarray::ArrayView1<'a, Real>,
        signal: ndarray::ArrayView1<'a, Real>,
        spectral_energy: ndarray::ArrayView1<'a, Real>,
        spectral_function: ndarray::ArrayView1<'a, Real>,
        weights: ndarray::ArrayView1<'a, Real>,
    ) -> SfconvConvolutionInput<'a> {
        SfconvConvolutionInput {
            photoelectron_energy: 1.35,
            chemical_potential: 0.15,
            core_hole_lifetime: 0.08,
            signal_energy,
            signal,
            spectral_energy,
            spectral_function,
            weights,
            asymmetric_phase: false,
            cutoff: true,
            plasma_frequency: 0.55,
        }
    }

    fn assert_close(actual: Real, expected: Real, tolerance: Real) {
        assert!(
            (actual - expected).abs() <= tolerance * expected.abs().max(1.0),
            "{actual} != {expected}"
        );
    }

    fn assert_real_slice_close(actual: &Array1<Real>, expected: &[Real], tolerance: Real) {
        assert_eq!(actual.len(), expected.len());
        for (&actual, &expected) in actual.iter().zip(expected) {
            assert_close(actual, expected, tolerance);
        }
    }

    fn assert_pole_close(actual: SfconvPole, expected: SfconvPole) {
        assert_close(actual.energy, expected.energy, 1.0e-15);
        assert_close(actual.weight, expected.weight, 1.0e-15);
        assert_close(actual.broadening, expected.broadening, 1.0e-15);
    }

    fn assert_q_limits_close(actual: SfconvQLimits, expected: SfconvQLimits, tolerance: Real) {
        assert_eq!(actual.count, expected.count);
        assert_close(actual.q1, expected.q1, tolerance);
        assert_close(actual.q2, expected.q2, tolerance);
        assert_close(actual.q3, expected.q3, tolerance);
    }

    fn assert_integral_close(
        actual: SfconvAdaptiveIntegral,
        expected: SfconvAdaptiveIntegral,
        tolerance: Real,
    ) {
        assert_close(actual.value, expected.value, tolerance);
        assert_close(
            actual.estimated_error,
            expected.estimated_error,
            tolerance.max(1.0e-12),
        );
        assert_eq!(actual.evaluations, expected.evaluations);
        assert_eq!(actual.max_regions, expected.max_regions);
    }

    fn mksat_reference_context() -> SfconvSatelliteContext {
        SfconvSatelliteContext {
            plasma_frequency: 0.62,
            pole_energy: 0.47,
            dispersion_parameter: 0.28,
            photoelectron_energy: 0.85,
            accuracy: 1.0e-4,
        }
    }

    fn mksat_reference_self_energy() -> SfconvSatelliteSelfEnergy {
        SfconvSatelliteSelfEnergy {
            on_shell_real: 0.12,
            width: 0.08,
            renormalization_real: 0.82,
            renormalization_imag: 0.06,
            off_shell_real: 0.03,
            off_shell_imag: 0.025,
        }
    }

    fn senergies_reference_context(include_below_fermi: bool) -> SfconvSelfEnergyContext {
        SfconvSelfEnergyContext {
            fermi_energy: 0.50,
            fermi_momentum: 1.00,
            plasma_frequency: 0.62,
            pole_energy: 0.47,
            quasiparticle_energy: 0.91,
            photoelectron_momentum: (2.0_f64 * 0.85).sqrt(),
            accuracy: 1.0e-4,
            dispersion_parameter: 0.28,
            include_below_fermi,
        }
    }
}
