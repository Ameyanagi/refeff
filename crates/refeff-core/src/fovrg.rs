//! FEFF FOVRG numerical helpers.
//!
//! These routines cover small pieces of the relativistic radial solver that can
//! be validated independently of the full `dfovrg` integration path.

use ndarray::{Array1, ArrayView1};
use thiserror::Error;

use crate::{Complex, ComplexVec, Real};

// `diff.f90` uses unsuffixed Fortran real literals in these stencils. Preserve
// their default-real rounding before widening to the Rust `Real` type.
const F77_REAL_HALF: Real = 0.5_f32 as Real;
const F77_REAL_ONE_POINT_TWO: Real = 1.2_f32 as Real;
const F77_REAL_ONE_POINT_FIVE: Real = 1.5_f32 as Real;
const F77_REAL_TWO: Real = 2.0_f32 as Real;
const F77_REAL_TWO_POINT_FOUR_FIVE: Real = 2.45_f32 as Real;
const F77_REAL_THREE_POINT_SEVEN_FIVE: Real = 3.75_f32 as Real;
const F77_REAL_SIX: Real = 6.0_f32 as Real;
const F77_REAL_SIX_AND_TWO_THIRDS: Real = 6.666_666_5_f32 as Real;
const F77_REAL_SEVEN_POINT_FIVE: Real = 7.5_f32 as Real;
const F77_REAL_EIGHT: Real = 8.0_f32 as Real;
const F77_REAL_TWELVE: Real = 12.0_f32 as Real;
const F77_REAL_ONE_SIXTH: Real = 0.166_666_67_f32 as Real;

/// Inputs for FEFF `FOVRG/diff.f90`.
#[derive(Debug, Clone, Copy)]
pub struct FovrgC3DerivativeInput<'a> {
    /// Complex potential values `v`.
    pub potential: ArrayView1<'a, Complex>,
    /// Radial grid `dr`.
    pub radii: ArrayView1<'a, Real>,
    /// Relativistic kappa `kap`.
    pub kappa: i32,
    /// Speed of light `cl`.
    pub speed_of_light: Real,
    /// Logarithmic grid step `dx`.
    pub delta: Real,
    /// Number of active radial rows `n`.
    pub active_len: usize,
}

/// Error returned by FOVRG helper kernels.
#[derive(Debug, Clone, Copy, PartialEq, Error)]
pub enum FovrgError {
    /// FEFF `diff` uses rows 1..=8 in the first two one-sided stencils.
    #[error("FOVRG {name} count {actual} is below minimum {minimum}")]
    CountTooSmall {
        name: &'static str,
        actual: usize,
        minimum: usize,
    },
    /// Active rows must fit in every input array.
    #[error("FOVRG active row count {active_len} exceeds {field} length {len}")]
    ActiveCountOutOfRange {
        field: &'static str,
        active_len: usize,
        len: usize,
    },
    /// Scalar inputs must be finite.
    #[error("FOVRG {name} must be finite, got {value}")]
    NonFiniteInput { name: &'static str, value: Real },
    /// Divisor-like scalar inputs must be nonzero.
    #[error("FOVRG {name} must be nonzero")]
    ZeroInput { name: &'static str },
    /// Radii must be positive.
    #[error("FOVRG radius row {row} must be positive, got {value}")]
    NonPositiveRadius { row: usize, value: Real },
    /// Complex potential values must be finite.
    #[error("FOVRG potential row {row} must be finite, got {value}")]
    NonFinitePotential { row: usize, value: Complex },
    /// Output values must remain finite.
    #[error("FOVRG derivative row {row} must be finite, got {value}")]
    NonFiniteResult { row: usize, value: Complex },
}

/// Port of `FOVRG/diff.f90`: C3 radial derivative term.
///
/// FEFF first differentiates `v(r) * r^2` with one-sided boundary stencils and
/// a centered fourth-order interior stencil, then returns
/// `(d(v*r^2)/dx - 2*v*r^2) / r * (kap+1) / cl`.
pub fn fovrg_c3_derivative(input: FovrgC3DerivativeInput<'_>) -> Result<ComplexVec, FovrgError> {
    validate_count_at_least("active_len", input.active_len, 8)?;
    validate_active_len("potential", input.active_len, input.potential.len())?;
    validate_active_len("radii", input.active_len, input.radii.len())?;
    validate_nonzero_finite("delta", input.delta)?;
    validate_nonzero_finite("speed_of_light", input.speed_of_light)?;

    for row in 0..input.active_len {
        validate_radius(row, input.radii[row])?;
        validate_potential(row, input.potential[row])?;
    }

    let vt = Array1::from_iter(
        (0..input.active_len).map(|row| input.potential[row] * input.radii[row].powi(2)),
    );
    let mut derivative = Array1::<Complex>::zeros(input.active_len);

    derivative[0] = ((F77_REAL_SIX * vt[1]
        + F77_REAL_SIX_AND_TWO_THIRDS * vt[3]
        + F77_REAL_ONE_POINT_TWO * vt[5])
        - (F77_REAL_TWO_POINT_FOUR_FIVE * vt[0]
            + F77_REAL_SEVEN_POINT_FIVE * vt[2]
            + F77_REAL_THREE_POINT_SEVEN_FIVE * vt[4]
            + F77_REAL_ONE_SIXTH * vt[6]))
        / input.delta;
    derivative[1] = ((F77_REAL_SIX * vt[2]
        + F77_REAL_SIX_AND_TWO_THIRDS * vt[4]
        + F77_REAL_ONE_POINT_TWO * vt[6])
        - (F77_REAL_TWO_POINT_FOUR_FIVE * vt[1]
            + F77_REAL_SEVEN_POINT_FIVE * vt[3]
            + F77_REAL_THREE_POINT_SEVEN_FIVE * vt[5]
            + F77_REAL_ONE_SIXTH * vt[7]))
        / input.delta;

    for row in 2..input.active_len - 2 {
        derivative[row] = ((vt[row - 2] + F77_REAL_EIGHT * vt[row + 1])
            - (F77_REAL_EIGHT * vt[row - 1] + vt[row + 2]))
            / F77_REAL_TWELVE
            / input.delta;
    }

    let last = input.active_len - 1;
    derivative[last - 1] = (vt[last] - vt[last - 2]) / (F77_REAL_TWO * input.delta);
    derivative[last] = (F77_REAL_HALF * vt[last - 2] - F77_REAL_TWO * vt[last - 1]
        + F77_REAL_ONE_POINT_FIVE * vt[last])
        / input.delta;

    let scale = ((input.kappa as f32 + 1.0_f32) as Real) / input.speed_of_light;
    let mut output = Array1::<Complex>::zeros(input.active_len);
    for row in 0..input.active_len {
        let value = (derivative[row] - F77_REAL_TWO * vt[row]) / input.radii[row] * scale;
        if !value.re.is_finite() || !value.im.is_finite() {
            return Err(FovrgError::NonFiniteResult { row, value });
        }
        output[row] = value;
    }
    Ok(output)
}

fn validate_count_at_least(
    name: &'static str,
    actual: usize,
    minimum: usize,
) -> Result<(), FovrgError> {
    if actual < minimum {
        Err(FovrgError::CountTooSmall {
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
) -> Result<(), FovrgError> {
    if active_len > len {
        Err(FovrgError::ActiveCountOutOfRange {
            field,
            active_len,
            len,
        })
    } else {
        Ok(())
    }
}

fn validate_nonzero_finite(name: &'static str, value: Real) -> Result<(), FovrgError> {
    if !value.is_finite() {
        Err(FovrgError::NonFiniteInput { name, value })
    } else if value == 0.0 {
        Err(FovrgError::ZeroInput { name })
    } else {
        Ok(())
    }
}

fn validate_radius(row: usize, value: Real) -> Result<(), FovrgError> {
    if !value.is_finite() {
        Err(FovrgError::NonFiniteInput {
            name: "radius",
            value,
        })
    } else if value <= 0.0 {
        Err(FovrgError::NonPositiveRadius { row, value })
    } else {
        Ok(())
    }
}

fn validate_potential(row: usize, value: Complex) -> Result<(), FovrgError> {
    if value.re.is_finite() && value.im.is_finite() {
        Ok(())
    } else {
        Err(FovrgError::NonFinitePotential { row, value })
    }
}

#[cfg(test)]
mod tests {
    use ndarray::Array1;

    use crate::{Complex, Real};

    use super::{FovrgC3DerivativeInput, FovrgError, fovrg_c3_derivative};

    #[test]
    fn c3_derivative_matches_feff_diff_reference() -> Result<(), FovrgError> {
        let (potential, radii) = diff_reference_inputs(10);

        let derivative = fovrg_c3_derivative(FovrgC3DerivativeInput {
            potential: potential.view(),
            radii: radii.view(),
            kappa: -2,
            speed_of_light: 137.035_999_084,
            delta: 0.0375,
            active_len: 10,
        })?;

        let expected = [
            (-0.011_975_827_006_405_27, -0.011_279_195_671_167_455),
            (-0.016_505_394_195_758_99, -0.008_884_114_730_822_418),
            (-0.020_242_542_448_345_43, -0.005_647_908_958_998_54),
            (-0.022_839_291_155_546_27, -0.001_659_964_058_354_706_8),
            (-0.024_047_315_082_090_202, 0.002_950_607_669_371_263_3),
            (-0.023_683_648_659_231_31, 0.008_014_885_042_325_136),
            (-0.021_663_526_338_827_583, 0.013_330_188_602_550_464),
            (-0.018_012_853_921_219_218, 0.018_667_556_473_840_063),
            (-0.012_457_714_462_626_513, 0.023_984_332_127_499_31),
            (-0.007_300_598_102_380_937, 0.028_056_048_903_698_883),
        ];
        for (actual, (expected_re, expected_im)) in derivative.iter().zip(expected) {
            assert_complex_close(*actual, expected_re, expected_im, 1.0e-13);
        }
        Ok(())
    }

    #[test]
    fn c3_derivative_rejects_invalid_inputs() {
        let (potential, radii) = diff_reference_inputs(8);

        assert!(matches!(
            fovrg_c3_derivative(FovrgC3DerivativeInput {
                potential: potential.view(),
                radii: radii.view(),
                kappa: -2,
                speed_of_light: 137.035_999_084,
                delta: 0.0375,
                active_len: 7,
            }),
            Err(FovrgError::CountTooSmall {
                name: "active_len",
                ..
            })
        ));
        assert!(matches!(
            fovrg_c3_derivative(FovrgC3DerivativeInput {
                potential: potential.view(),
                radii: radii.view(),
                kappa: -2,
                speed_of_light: 137.035_999_084,
                delta: 0.0375,
                active_len: 9,
            }),
            Err(FovrgError::ActiveCountOutOfRange {
                field: "potential",
                ..
            })
        ));
        assert!(matches!(
            fovrg_c3_derivative(FovrgC3DerivativeInput {
                potential: potential.view(),
                radii: radii.view(),
                kappa: -2,
                speed_of_light: 137.035_999_084,
                delta: 0.0,
                active_len: 8,
            }),
            Err(FovrgError::ZeroInput { name: "delta" })
        ));

        let mut bad_radii = radii.clone();
        bad_radii[3] = 0.0;
        assert!(matches!(
            fovrg_c3_derivative(FovrgC3DerivativeInput {
                potential: potential.view(),
                radii: bad_radii.view(),
                kappa: -2,
                speed_of_light: 137.035_999_084,
                delta: 0.0375,
                active_len: 8,
            }),
            Err(FovrgError::NonPositiveRadius { row: 3, .. })
        ));

        let mut bad_potential = potential.clone();
        bad_potential[2] = Complex::new(f64::NAN, 0.0);
        assert!(matches!(
            fovrg_c3_derivative(FovrgC3DerivativeInput {
                potential: bad_potential.view(),
                radii: radii.view(),
                kappa: -2,
                speed_of_light: 137.035_999_084,
                delta: 0.0375,
                active_len: 8,
            }),
            Err(FovrgError::NonFinitePotential { row: 2, .. })
        ));
    }

    fn diff_reference_inputs(count: usize) -> (Array1<Complex>, Array1<Real>) {
        let potential = Array1::from_iter((1..=count).map(|index| {
            let index = index as Real;
            Complex::new(
                (0.21 * index).sin() + 0.03 * index,
                (0.17 * index).cos() - 0.02 * index,
            )
        }));
        let radii = Array1::from_iter((1..=count).map(|index| {
            let index = index as Real;
            0.15 + 0.04 * index + 0.001 * index * index
        }));
        (potential, radii)
    }

    fn assert_complex_close(
        actual: Complex,
        expected_re: Real,
        expected_im: Real,
        tolerance: Real,
    ) {
        assert_close(actual.re, expected_re, tolerance);
        assert_close(actual.im, expected_im, tolerance);
    }

    fn assert_close(actual: Real, expected: Real, tolerance: Real) {
        assert!(
            (actual - expected).abs() <= tolerance * expected.abs().max(1.0),
            "{actual} != {expected}"
        );
    }
}
