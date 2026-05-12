//! Self-energy singularity helpers ported from FEFF.
//!
//! `SELF/fndsng.f90` finds real singularities of the Hedin-Lundqvist
//! self-energy integrands by solving the FEFF cubic and quadratic equations,
//! filtering roots to the integration window, and sorting the accepted values.

use std::cmp::Ordering;

use crate::{Complex, Real, RootError, cubic_zeros, quadratic_zeros};

const SINGULARITY_TOLERANCE: Real = 1.0e-4;

/// FEFF self-energy integrand selector used by `FndSng`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SingularityFunction {
    /// FEFF `iFcn = 1`: solve both equation families.
    First,
    /// FEFF `iFcn = 2`: solve only the cubic equation family.
    Second,
}

/// Error returned by self-energy singularity helpers.
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum SelfEnergyError {
    /// FEFF only defines singularity selectors `1` and `2`.
    #[error("invalid self-energy singularity function {value}; expected 1 or 2")]
    InvalidFunction { value: i32 },
    /// Inputs must have finite real and imaginary parts.
    #[error("self-energy input {name} must be finite, got {value:?}")]
    NonFiniteComplex { name: &'static str, value: Complex },
    /// Inputs must be finite real values.
    #[error("self-energy input {name} must be finite, got {value}")]
    NonFiniteReal { name: &'static str, value: Real },
    /// Polynomial root solving failed.
    #[error(transparent)]
    Root(#[from] RootError),
}

impl TryFrom<i32> for SingularityFunction {
    type Error = SelfEnergyError;

    fn try_from(value: i32) -> Result<Self, Self::Error> {
        match value {
            1 => Ok(Self::First),
            2 => Ok(Self::Second),
            value => Err(SelfEnergyError::InvalidFunction { value }),
        }
    }
}

/// Port of FEFF `FndSng`: find real self-energy integrand singularities.
///
/// `limits` are the lower and upper integration limits. `dp_parameters`
/// corresponds to FEFF `DPPar(1:4)`, with `DPPar(1)` as `Wp/EFermi` and
/// `DPPar(3)` as `Energy/EFermi`. `complex_parameters` corresponds to
/// `CPar(1:2)`, where `CPar(1)` is `ck/kFermi`.
pub fn find_self_energy_singularities(
    limits: [Complex; 2],
    dp_parameters: [Real; 4],
    complex_parameters: [Complex; 2],
    function: SingularityFunction,
) -> Result<Vec<Real>, SelfEnergyError> {
    ensure_finite_complex("lower limit", limits[0])?;
    ensure_finite_complex("upper limit", limits[1])?;
    ensure_finite_complex("CPar(1)", complex_parameters[0])?;
    ensure_finite_complex("CPar(2)", complex_parameters[1])?;
    for (index, &value) in dp_parameters.iter().enumerate() {
        ensure_finite_real(dp_name(index), value)?;
    }

    let k = complex_parameters[0];
    let energy = dp_parameters[2];
    let plasma = dp_parameters[0];
    let lower = limits[0].re;
    let upper = limits[1].re;
    let mut singularities = Vec::new();

    let base_cubic = [
        4.0 * k,
        2.0 * (3.0 * k * k - energy - 2.0 / 3.0),
        4.0 * k * (k * k - energy),
        (k * k - energy) * (k * k - energy) - plasma * plasma,
    ];

    let plus_roots = cubic_zeros(base_cubic)?;
    singularities.extend(
        plus_roots
            .roots()
            .iter()
            .copied()
            .filter(|&root| accepts_cubic_root(k, energy, plasma, root, true, lower, upper))
            .map(|root| root.re),
    );

    let minus_roots = cubic_zeros([-base_cubic[0], base_cubic[1], -base_cubic[2], base_cubic[3]])?;
    singularities.extend(
        minus_roots
            .roots()
            .iter()
            .copied()
            .filter(|&root| accepts_cubic_root(k, energy, plasma, root, false, lower, upper))
            .map(|root| root.re),
    );

    if function == SingularityFunction::First {
        let roots = quadratic_zeros([
            Complex::new(1.0, 0.0),
            Complex::new(4.0 / 3.0, 0.0),
            Complex::new(plasma * plasma, 0.0),
        ])?;
        for &root in roots.roots() {
            if root.im.abs() <= SINGULARITY_TOLERANCE {
                let square_root = root.sqrt();
                let positive = square_root.re;
                let negative = -square_root.re;
                if positive >= lower && positive <= upper {
                    singularities.push(positive);
                }
                if negative >= lower && negative <= upper {
                    singularities.push(negative);
                }
            }
        }
    }

    sort_like_feff(&mut singularities);
    Ok(singularities)
}

fn accepts_cubic_root(
    k: Complex,
    energy: Real,
    plasma: Real,
    root: Complex,
    positive_branch: bool,
    lower: Real,
    upper: Real,
) -> bool {
    let radical = (root * root * root * root + root * root * (4.0 / 3.0) + plasma * plasma).sqrt();
    let test = if positive_branch {
        ((k + root) * (k + root) - energy + radical).norm()
    } else {
        ((k - root) * (k - root) - energy - radical).norm()
    };

    test < SINGULARITY_TOLERANCE
        && root.re >= lower
        && root.re <= upper
        && root.im.abs() <= SINGULARITY_TOLERANCE
}

fn sort_like_feff(values: &mut [Real]) {
    values.sort_by(|left, right| {
        if left < right {
            Ordering::Less
        } else if left > right {
            Ordering::Greater
        } else {
            Ordering::Equal
        }
    });
}

fn ensure_finite_complex(name: &'static str, value: Complex) -> Result<(), SelfEnergyError> {
    if value.re.is_finite() && value.im.is_finite() {
        Ok(())
    } else {
        Err(SelfEnergyError::NonFiniteComplex { name, value })
    }
}

fn ensure_finite_real(name: &'static str, value: Real) -> Result<(), SelfEnergyError> {
    if value.is_finite() {
        Ok(())
    } else {
        Err(SelfEnergyError::NonFiniteReal { name, value })
    }
}

fn dp_name(index: usize) -> &'static str {
    match index {
        0 => "DPPar(1)",
        1 => "DPPar(2)",
        2 => "DPPar(3)",
        _ => "DPPar(4)",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn finds_no_singularities_for_feff_empty_window_case() -> Result<(), SelfEnergyError> {
        let values = find_self_energy_singularities(
            [Complex::new(0.0, 0.0), Complex::new(4.0, 0.0)],
            [0.5, 0.01, 1.0, 0.0],
            [Complex::new(1.0, 0.0), Complex::new(0.0, 0.0)],
            SingularityFunction::Second,
        )?;

        assert!(values.is_empty());
        Ok(())
    }

    #[test]
    fn finds_feff_cubic_singularities() -> Result<(), SelfEnergyError> {
        let values = find_self_energy_singularities(
            [Complex::new(-2.0, 0.0), Complex::new(2.0, 0.0)],
            [0.35, 0.02, 0.8, 0.0],
            [Complex::new(0.7, 0.0), Complex::new(0.0, 0.0)],
            SingularityFunction::Second,
        )?;

        assert_real_slice_close(
            &values,
            &[
                -0.5702425768022866,
                -0.5421244103394355,
                -0.03049911884380313,
            ],
        );
        Ok(())
    }

    #[test]
    fn finds_feff_first_function_singularities() -> Result<(), SelfEnergyError> {
        let values = find_self_energy_singularities(
            [Complex::new(-2.0, 0.0), Complex::new(2.0, 0.0)],
            [0.35, 0.02, 0.8, 0.0],
            [Complex::new(0.7, 0.0), Complex::new(0.0, 0.0)],
            SingularityFunction::First,
        )?;

        assert_real_slice_close(
            &values,
            &[
                -0.5702425768022866,
                -0.5421244103394355,
                -0.03049911884380313,
                0.0,
                -0.0,
                0.0,
                -0.0,
            ],
        );
        Ok(())
    }

    #[test]
    fn rejects_invalid_singularity_inputs() {
        assert_eq!(
            SingularityFunction::try_from(3),
            Err(SelfEnergyError::InvalidFunction { value: 3 })
        );
        assert!(matches!(
            find_self_energy_singularities(
                [Complex::new(f64::NAN, 0.0), Complex::new(2.0, 0.0)],
                [0.35, 0.02, 0.8, 0.0],
                [Complex::new(0.7, 0.0), Complex::new(0.0, 0.0)],
                SingularityFunction::Second,
            ),
            Err(SelfEnergyError::NonFiniteComplex {
                name: "lower limit",
                ..
            })
        ));
    }

    fn assert_real_slice_close(actual: &[Real], expected: &[Real]) {
        assert_eq!(actual.len(), expected.len());
        for (&actual, &expected) in actual.iter().zip(expected.iter()) {
            assert!(
                (actual - expected).abs() < 1.0e-12,
                "actual={actual} expected={expected}"
            );
            if expected == 0.0 {
                assert_eq!(actual.to_bits(), expected.to_bits());
            }
        }
    }
}
