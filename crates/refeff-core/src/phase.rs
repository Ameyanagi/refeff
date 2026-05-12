//! Phase-shift utilities ported from FEFF common routines.
//!
//! FEFF uses `pijump` to remove discontinuous `2*pi` jumps from scattering
//! phases before path and spectrum assembly. The routines here keep the same
//! nearest-branch selection while returning explicit errors for invalid input.

use ndarray::ArrayView1;
use thiserror::Error;

use crate::{Real, RealVec};

/// Error returned by phase-processing routines.
#[derive(Debug, Clone, Copy, PartialEq, Error)]
pub enum PhaseError {
    /// FEFF phase unwrapping requires finite floating-point inputs.
    #[error("phase values must be finite, got phase={phase} previous={previous}")]
    NonFinite { phase: Real, previous: Real },
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
}
