//! Phase-shift utilities ported from FEFF common routines.
//!
//! FEFF uses `pijump` to remove discontinuous `2*pi` jumps from scattering
//! phases before path and spectrum assembly. The routines here keep the same
//! nearest-branch selection while returning explicit errors for invalid input.

use ndarray::ArrayView1;
use thiserror::Error;

use crate::{Complex, Real, RealVec};

const FINE_STRUCTURE_ALPHA: Real = 1.0 / 137.03598956;

/// Error returned by phase-processing routines.
#[derive(Debug, Clone, Copy, PartialEq, Error)]
#[non_exhaustive]
pub enum PhaseError {
    /// FEFF phase unwrapping requires finite floating-point inputs.
    #[error("phase values must be finite, got phase={phase} previous={previous}")]
    NonFinite { phase: Real, previous: Real },
    /// Complex phase helpers require finite real and imaginary parts.
    #[error("{argument} must be finite, got {value:?}")]
    NonFiniteComplex {
        argument: &'static str,
        value: Complex,
    },
    /// Real phase-amplitude inputs must be finite.
    #[error("{argument} must be finite, got {value}")]
    NonFiniteReal { argument: &'static str, value: Real },
    /// The FEFF complex arctangent formula is singular at `+i` and `-i`.
    #[error("complex arctangent is singular for {value:?}")]
    SingularComplexArctangent { value: Complex },
    /// `phamp` cannot divide by the relativistic small-component factor.
    #[error("phase-amplitude relativistic factor is zero for ck={ck:?}")]
    ZeroRelativisticFactor { ck: Complex },
}

/// Complex amplitude and phase returned by FEFF `atan2c` and `phamp`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ComplexAmplitudePhase {
    /// Complex amplitude `A`.
    pub amplitude: Complex,
    /// Complex phase `phi`.
    pub phase: Complex,
}

/// Remove a discontinuous `2*pi` jump from `phase` relative to `previous`.
///
/// This ports FEFF's `COMMON/pijump.f90`: it considers the current phase
/// difference and the neighboring branches shifted by `jump * 2*pi`, then
/// chooses the branch with the smallest absolute phase difference.
pub fn remove_phase_jump(phase: Real, previous: Real) -> Result<Real, PhaseError> {
    if !(phase.is_finite() && previous.is_finite()) {
        return Err(PhaseError::NonFinite { phase, previous });
    }

    let delta = phase - previous;
    let jump = ((delta.abs() + std::f64::consts::PI) / std::f64::consts::TAU).trunc();
    let candidates = [
        delta,
        delta - jump * std::f64::consts::TAU,
        delta + jump * std::f64::consts::TAU,
    ];
    let best = candidates
        .iter()
        .copied()
        .fold(candidates[0], |best, candidate| {
            if (best.abs() - candidate.abs()).abs() <= 0.01 || candidate.abs() < best.abs() {
                candidate
            } else {
                best
            }
        });
    Ok(previous + best)
}

/// Remove `2*pi` jumps from a phase sequence.
///
/// The first phase is preserved and each later phase is unwrapped against the
/// previous unwrapped value.
pub fn remove_phase_jumps(phases: impl IntoIterator<Item = Real>) -> Result<Vec<Real>, PhaseError> {
    let mut phases = phases.into_iter();
    let Some(first) = phases.next() else {
        return Ok(Vec::new());
    };
    if !first.is_finite() {
        return Err(PhaseError::NonFinite {
            phase: first,
            previous: first,
        });
    }

    let mut previous = first;
    let mut unwrapped = vec![first];
    for phase in phases {
        previous = remove_phase_jump(phase, previous)?;
        unwrapped.push(previous);
    }
    Ok(unwrapped)
}

/// Remove `2*pi` jumps from an `ndarray` phase vector.
pub fn remove_phase_jumps_array(phases: ArrayView1<'_, Real>) -> Result<RealVec, PhaseError> {
    remove_phase_jumps(phases.iter().copied()).map(RealVec::from_vec)
}

/// Port of FEFF `atancc`: complex arctangent by the legacy real formula.
///
/// FEFF uses this helper inside `atan2c` instead of the library complex
/// arctangent. The singular points `+i` and `-i` return an error rather than
/// producing an infinite imaginary phase.
pub fn complex_atan(value: Complex) -> Result<Complex, PhaseError> {
    ensure_complex_finite("value", value)?;

    let xx = value.re;
    let yy = value.im;
    let alpha = if xx != 0.0 {
        let alpha_base = 1.0 - xx * xx - yy * yy;
        let alpha_ratio =
            ((alpha_base * alpha_base + 4.0 * xx * xx).sqrt() - alpha_base) / (2.0 * xx);
        alpha_ratio.atan()
    } else {
        0.0
    };

    let numerator = xx * xx + (yy + 1.0) * (yy + 1.0);
    let denominator = xx * xx + (yy - 1.0) * (yy - 1.0);
    if numerator == 0.0 || denominator == 0.0 {
        return Err(PhaseError::SingularComplexArctangent { value });
    }
    let beta = (numerator / denominator).ln() / 4.0;
    let phase = Complex::new(alpha, beta);
    ensure_complex_finite("phase", phase)?;
    Ok(phase)
}

/// Port of FEFF `atan2c`.
///
/// Returns `amplitude` and `phase` such that `a = amplitude*cos(phase)` and
/// `b = amplitude*sin(phase)`, matching FEFF's branch and sign adjustment.
pub fn complex_atan2_amplitude_phase(
    a: Complex,
    b: Complex,
) -> Result<ComplexAmplitudePhase, PhaseError> {
    ensure_complex_finite("a", a)?;
    ensure_complex_finite("b", b)?;

    let mut result = if a.norm() + b.norm() == 0.0 {
        ComplexAmplitudePhase {
            amplitude: Complex::new(0.0, 0.0),
            phase: Complex::new(0.0, 0.0),
        }
    } else if a.norm() > b.norm() {
        let phase = complex_atan(b / a)?;
        ComplexAmplitudePhase {
            amplitude: a / phase.cos(),
            phase,
        }
    } else {
        let phase = Complex::new(std::f64::consts::FRAC_PI_2, 0.0) - complex_atan(a / b)?;
        ComplexAmplitudePhase {
            amplitude: b / phase.sin(),
            phase,
        }
    };

    if result.amplitude.re < 0.0 {
        result.amplitude = -result.amplitude;
        result.phase += Complex::new(std::f64::consts::PI, 0.0);
    }
    ensure_complex_finite("amplitude", result.amplitude)?;
    ensure_complex_finite("phase", result.phase)?;
    Ok(result)
}

/// Port of FEFF `phamp`: muffin-tin phase shift and amplitude.
///
/// The arguments correspond to the FEFF radial values at the muffin-tin radius:
/// `pu`, `qu`, wave number `ck`, spherical Bessel/Neumann values `jl`, `nl`,
/// and their derivatives `jlp`, `nlp`. The returned [`ComplexAmplitudePhase`]
/// stores FEFF's `amp` and `ph` outputs.
#[allow(clippy::too_many_arguments)]
pub fn muffin_tin_phase_amplitude(
    rmt: Real,
    pu: Complex,
    qu: Complex,
    ck: Complex,
    jl: Complex,
    nl: Complex,
    jlp: Complex,
    nlp: Complex,
    ikap: i32,
) -> Result<ComplexAmplitudePhase, PhaseError> {
    ensure_real_finite("rmt", rmt)?;
    ensure_complex_finite("pu", pu)?;
    ensure_complex_finite("qu", qu)?;
    ensure_complex_finite("ck", ck)?;
    ensure_complex_finite("jl", jl)?;
    ensure_complex_finite("nl", nl)?;
    ensure_complex_finite("jlp", jlp)?;
    ensure_complex_finite("nlp", nlp)?;

    let sign = if ikap < 0 { -1.0 } else { 1.0 };
    let xkr = ck * rmt;
    let scaled_ck = ck * FINE_STRUCTURE_ALPHA;
    let one = Complex::new(1.0, 0.0);
    let factor = scaled_ck * sign / (one + (one + scaled_ck * scaled_ck).sqrt());
    if factor.norm() == 0.0 {
        return Err(PhaseError::ZeroRelativisticFactor { ck });
    }

    let common = ck * xkr * sign;
    let a = common * (pu * nlp - qu * nl / factor);
    let b = -common * (qu * jl / factor - pu * jlp);
    complex_atan2_amplitude_phase(a, b)
}

fn ensure_complex_finite(argument: &'static str, value: Complex) -> Result<(), PhaseError> {
    if !(value.re.is_finite() && value.im.is_finite()) {
        return Err(PhaseError::NonFiniteComplex { argument, value });
    }
    Ok(())
}

fn ensure_real_finite(argument: &'static str, value: Real) -> Result<(), PhaseError> {
    if !value.is_finite() {
        return Err(PhaseError::NonFiniteReal { argument, value });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use ndarray::array;

    use super::*;

    #[test]
    fn removes_positive_two_pi_jump() -> Result<(), PhaseError> {
        let previous = 0.1;
        let phase = previous + std::f64::consts::TAU + 0.2;
        let unwrapped = remove_phase_jump(phase, previous)?;
        assert!((unwrapped - 0.3).abs() < 1.0e-12);
        Ok(())
    }

    #[test]
    fn removes_negative_two_pi_jump() -> Result<(), PhaseError> {
        let previous = 0.1;
        let phase = previous - std::f64::consts::TAU - 0.4;
        let unwrapped = remove_phase_jump(phase, previous)?;
        assert!((unwrapped + 0.3).abs() < 1.0e-12);
        Ok(())
    }

    #[test]
    fn unwraps_ndarray_phase_vector() -> Result<(), PhaseError> {
        let phases = array![
            0.0,
            0.2,
            std::f64::consts::TAU + 0.4,
            std::f64::consts::TAU + 0.6,
        ];
        let unwrapped = remove_phase_jumps_array(phases.view())?;
        assert_eq!(unwrapped.len(), 4);
        assert!((unwrapped[0] - 0.0).abs() < 1.0e-12);
        assert!((unwrapped[1] - 0.2).abs() < 1.0e-12);
        assert!((unwrapped[2] - 0.4).abs() < 1.0e-12);
        assert!((unwrapped[3] - 0.6).abs() < 1.0e-12);
        Ok(())
    }

    #[test]
    fn rejects_non_finite_phase() {
        let err = remove_phase_jump(f64::NAN, 0.0);
        assert!(matches!(err, Err(PhaseError::NonFinite { .. })));
    }

    #[test]
    fn complex_atan_matches_feff_reference() -> Result<(), PhaseError> {
        assert_complex_close(
            complex_atan(Complex::new(0.75, -0.25))?,
            Complex::new(0.6629088318340162, -0.15899719167999918),
        );
        assert_complex_close(
            complex_atan(Complex::new(-0.5, 0.8))?,
            Complex::new(-0.7306184000104763, 0.6219440230539883),
        );
        Ok(())
    }

    #[test]
    fn complex_atan2_matches_feff_reference() -> Result<(), PhaseError> {
        let first =
            complex_atan2_amplitude_phase(Complex::new(1.25, -0.5), Complex::new(-0.75, 0.9))?;
        assert_complex_close(
            first.amplitude,
            Complex::new(1.3918811641621127, -0.9339877810492372),
        );
        assert_complex_close(
            first.phase,
            Complex::new(-0.7067734207144984, 0.25565008797151506),
        );

        let second =
            complex_atan2_amplitude_phase(Complex::new(0.1, 0.2), Complex::new(2.0, -0.4))?;
        assert_complex_close(
            second.amplitude,
            Complex::new(1.9908542558314004, -0.39179161292962883),
        );
        assert_complex_close(
            second.phase,
            Complex::new(1.5416323649318944, -0.10607638164251979),
        );

        let zero = complex_atan2_amplitude_phase(Complex::new(0.0, 0.0), Complex::new(0.0, 0.0))?;
        assert_eq!(zero.amplitude, Complex::new(0.0, 0.0));
        assert_eq!(zero.phase, Complex::new(0.0, 0.0));
        Ok(())
    }

    #[test]
    fn muffin_tin_phase_amplitude_matches_feff_reference() -> Result<(), PhaseError> {
        let first = muffin_tin_phase_amplitude(
            1.7,
            Complex::new(0.8, 0.2),
            Complex::new(-0.3, 0.4),
            Complex::new(1.1, 0.15),
            Complex::new(0.9, -0.1),
            Complex::new(-0.2, 0.7),
            Complex::new(0.4, 0.3),
            Complex::new(-0.6, 0.25),
            -2,
        )?;
        assert_complex_close(
            first.phase,
            Complex::new(2.094408693607371, -0.7485440329584997),
        );
        assert_complex_close(
            first.amplitude,
            Complex::new(60.75929219190526, -187.0456205085373),
        );

        let second = muffin_tin_phase_amplitude(
            0.85,
            Complex::new(-0.45, 0.65),
            Complex::new(0.25, -0.15),
            Complex::new(0.8, -0.05),
            Complex::new(0.55, 0.35),
            Complex::new(-0.15, -0.45),
            Complex::new(0.3, -0.25),
            Complex::new(0.7, 0.1),
            1,
        )?;
        assert_complex_close(
            second.phase,
            Complex::new(5.298647485949331, 0.35631223860898137),
        );
        assert_complex_close(
            second.amplitude,
            Complex::new(38.36296826901459, 7.081449225896745),
        );
        Ok(())
    }

    #[test]
    fn complex_atan_rejects_branch_singularities() {
        assert!(matches!(
            complex_atan(Complex::new(0.0, 1.0)),
            Err(PhaseError::SingularComplexArctangent { .. })
        ));
        assert!(matches!(
            complex_atan(Complex::new(0.0, -1.0)),
            Err(PhaseError::SingularComplexArctangent { .. })
        ));
    }

    fn assert_complex_close(actual: Complex, expected: Complex) {
        assert!(
            (actual - expected).norm() < 1.0e-12,
            "actual={actual:?}, expected={expected:?}, diff={}",
            (actual - expected).norm()
        );
    }
}
