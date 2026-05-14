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

/// Inputs for FEFF `FOVRG/yzktec.f90`.
#[derive(Debug, Clone, Copy)]
pub struct FovrgYkZkTransformInput<'a> {
    /// Tabulated source function `f`.
    pub source: ArrayView1<'a, Complex>,
    /// Origin development coefficients `af` for [`FovrgYkZkTransformInput::source`].
    pub source_coefficients: ArrayView1<'a, Complex>,
    /// Radial grid `dr`.
    pub radii: ArrayView1<'a, Real>,
    /// Initial origin power `ap`; FEFF uses only its real part before overwriting it.
    pub initial_power: Complex,
    /// Logarithmic radial step `h`.
    pub step: Real,
    /// Multipole order `k`.
    pub angular_momentum: usize,
    /// Number of active origin coefficients `nd`.
    pub coefficient_count: usize,
    /// Number of active source samples `np`; FEFF clamps this to `idim - 1`.
    pub source_len: usize,
    /// Active radial capacity `idim`.
    pub active_len: usize,
    /// Optional tail correction `dyzk`.
    pub tail_correction: Complex,
}

/// Output from FEFF `FOVRG/yzktec.f90`.
#[derive(Debug, Clone, PartialEq)]
pub struct FovrgYkZkTransform {
    /// Transformed `yk` values, zero-filled after [`FovrgYkZkTransform::computed_len`].
    pub yk: ComplexVec,
    /// Intermediate `zk` values, zero-filled after [`FovrgYkZkTransform::computed_len`].
    pub zk: ComplexVec,
    /// Mutated `af` development coefficients for `yk`.
    pub yk_coefficients: ComplexVec,
    /// Development coefficients `ag` for `zk`.
    pub zk_coefficients: ComplexVec,
    /// FEFF output scalar `ap`, the leading origin constant for `yk`.
    pub origin_constant: Complex,
    /// Number of meaningful radial rows, equivalent to clamped `np + 1`.
    pub computed_len: usize,
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
    /// Counts that are later converted to FEFF integer exponents must fit.
    #[error("FOVRG {name} count {actual} exceeds maximum {maximum}")]
    CountTooLarge {
        name: &'static str,
        actual: usize,
        maximum: usize,
    },
    /// Scalar inputs must be finite.
    #[error("FOVRG {name} must be finite, got {value}")]
    NonFiniteInput { name: &'static str, value: Real },
    /// Positive scalar inputs must be finite and greater than zero.
    #[error("FOVRG {name} must be positive, got {value}")]
    NonPositiveInput { name: &'static str, value: Real },
    /// Divisor-like scalar inputs must be nonzero.
    #[error("FOVRG {name} must be nonzero")]
    ZeroInput { name: &'static str },
    /// FEFF formulas with a zero denominator are reported instead of evaluated.
    #[error("FOVRG denominator {name} is zero")]
    ZeroDenominator { name: &'static str },
    /// Radii must be positive.
    #[error("FOVRG radius row {row} must be positive, got {value}")]
    NonPositiveRadius { row: usize, value: Real },
    /// Complex inputs must be finite.
    #[error("FOVRG {name} row {row} must be finite, got {value}")]
    NonFiniteComplexInput {
        name: &'static str,
        row: usize,
        value: Complex,
    },
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

/// Port of `FOVRG/yzktec.f90`: build the radial `yk` and `zk` exchange kernels.
///
/// FEFF evaluates
/// `zk(r) = r^-k * integral(0..r, f(u) * u^k du)` and then
/// `yk(r) = zk(r) + r^(k+1) * integral(r..infinity, f(u) * u^(-k-1) du)`.
/// The first integration runs outward on the logarithmic radial mesh and the
/// second runs backward from FEFF's clamped `np + 1` endpoint.
pub fn fovrg_yk_zk_transform(
    input: FovrgYkZkTransformInput<'_>,
) -> Result<FovrgYkZkTransform, FovrgError> {
    validate_count_at_least("active_len", input.active_len, 2)?;
    validate_count_at_least("source_len", input.source_len, 1)?;
    validate_count_at_least("coefficient_count", input.coefficient_count, 1)?;
    validate_active_len("source", input.active_len, input.source.len())?;
    validate_active_len("radii", input.active_len, input.radii.len())?;
    validate_active_len(
        "source_coefficients",
        input.coefficient_count,
        input.source_coefficients.len(),
    )?;
    validate_positive_finite("step", input.step)?;
    validate_complex_input("initial_power", 0, input.initial_power)?;
    validate_complex_input("tail_correction", 0, input.tail_correction)?;
    if input.angular_momentum > i32::MAX as usize - 1 {
        return Err(FovrgError::CountTooLarge {
            name: "angular_momentum",
            actual: input.angular_momentum,
            maximum: i32::MAX as usize - 1,
        });
    }

    let source_len = input.source_len.min(input.active_len - 1);
    let computed_len = source_len + 1;
    for row in 0..source_len {
        validate_complex_input("source", row, input.source[row])?;
        validate_radius(row, input.radii[row])?;
    }
    for coefficient in 0..input.coefficient_count {
        validate_complex_input(
            "source_coefficients",
            coefficient,
            input.source_coefficients[coefficient],
        )?;
    }

    let k = input.angular_momentum;
    let k_real = k as Real;
    let k_i32 = k as i32;
    let k_plus_one_i32 = (k + 1) as i32;
    let singular_tolerance = 1.0e-5_f32 as Real;
    let mut yk = Array1::<Complex>::zeros(input.active_len);
    let mut zk = Array1::<Complex>::zeros(input.active_len);
    let mut yk_coefficients = Array1::<Complex>::from_iter(
        (0..input.coefficient_count).map(|row| input.source_coefficients[row]),
    );
    let mut zk_coefficients = Array1::<Complex>::zeros(input.coefficient_count);
    for row in 0..source_len {
        yk[row] = input.source[row];
    }

    let mut power = input.initial_power.re;
    let mut origin_constant = Complex::new(0.0, 0.0);
    for coefficient in 0..input.coefficient_count {
        power += 1.0;
        let zk_denominator = power + k_real;
        validate_nonzero_denominator("yk_zk_origin_zk", zk_denominator)?;
        zk_coefficients[coefficient] = yk_coefficients[coefficient] / zk_denominator;

        if yk_coefficients[coefficient] != Complex::new(0.0, 0.0) {
            let radius_power = input.radii[0].powf(power);
            zk[0] += zk_coefficients[coefficient] * radius_power;

            let yk_denominator = power - k_real - 1.0;
            if yk_denominator.abs() <= singular_tolerance {
                yk_coefficients[coefficient] = Complex::new(0.0, 0.0);
                power -= 1.0;
            } else {
                yk_coefficients[coefficient] =
                    ((k + k + 1) as Real) * zk_coefficients[coefficient] / yk_denominator;
            }
            origin_constant += yk_coefficients[coefficient] * radius_power;
        }
    }

    for row in 0..source_len {
        yk[row] *= input.radii[row];
    }

    let hk = input.step * k_real;
    let e = (-input.step).exp();
    let ehk = e.powi(k_i32);
    let b1 = if k == 0 {
        input.step / 2.0
    } else {
        (ehk - 1.0 + hk) / (hk * k_real)
    };
    let b0 = input.step - (1.0 + hk) * b1;
    for row in 0..source_len {
        zk[row + 1] = zk[row] * ehk + b0 * yk[row] + b1 * yk[row + 1];
    }

    yk[source_len] = zk[source_len] + input.tail_correction;
    let backward_ehk = ehk * e;
    let backward_hk = hk + input.step;
    let backward_order = (k + k + 1) as Real;
    let backward_b1 =
        backward_order * (backward_ehk - 1.0 + backward_hk) / (backward_hk * (k_real + 1.0));
    let backward_b0 = backward_order * input.step - (1.0 + backward_hk) * backward_b1;
    for row in (0..source_len).rev() {
        yk[row] = yk[row + 1] * backward_ehk + backward_b0 * zk[row + 1] + backward_b1 * zk[row];
    }

    origin_constant = (origin_constant + yk[0]) / input.radii[0].powi(k_plus_one_i32);
    validate_complex_result("origin_constant", 0, origin_constant)?;
    for row in 0..computed_len {
        validate_complex_result("yk", row, yk[row])?;
        validate_complex_result("zk", row, zk[row])?;
    }
    for row in 0..input.coefficient_count {
        validate_complex_result("yk_coefficients", row, yk_coefficients[row])?;
        validate_complex_result("zk_coefficients", row, zk_coefficients[row])?;
    }

    Ok(FovrgYkZkTransform {
        yk,
        zk,
        yk_coefficients,
        zk_coefficients,
        origin_constant,
        computed_len,
    })
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

fn validate_positive_finite(name: &'static str, value: Real) -> Result<(), FovrgError> {
    if !value.is_finite() {
        Err(FovrgError::NonFiniteInput { name, value })
    } else if value <= 0.0 {
        Err(FovrgError::NonPositiveInput { name, value })
    } else {
        Ok(())
    }
}

fn validate_nonzero_denominator(name: &'static str, value: Real) -> Result<(), FovrgError> {
    if value == 0.0 {
        Err(FovrgError::ZeroDenominator { name })
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

fn validate_complex_input(
    name: &'static str,
    row: usize,
    value: Complex,
) -> Result<(), FovrgError> {
    if value.re.is_finite() && value.im.is_finite() {
        Ok(())
    } else {
        Err(FovrgError::NonFiniteComplexInput { name, row, value })
    }
}

fn validate_potential(row: usize, value: Complex) -> Result<(), FovrgError> {
    if value.re.is_finite() && value.im.is_finite() {
        Ok(())
    } else {
        Err(FovrgError::NonFinitePotential { row, value })
    }
}

fn validate_complex_result(
    name: &'static str,
    row: usize,
    value: Complex,
) -> Result<(), FovrgError> {
    if value.re.is_finite() && value.im.is_finite() {
        Ok(())
    } else {
        Err(FovrgError::NonFiniteComplexInput { name, row, value })
    }
}

#[cfg(test)]
mod tests {
    use ndarray::Array1;

    use crate::{Complex, Real};

    use super::{
        FovrgC3DerivativeInput, FovrgError, FovrgYkZkTransformInput, fovrg_c3_derivative,
        fovrg_yk_zk_transform,
    };

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

    #[test]
    fn yk_zk_transform_matches_feff_yzktec_reference() -> Result<(), FovrgError> {
        let (source, coefficients, radii) = yzktec_reference_inputs(12);

        let transform = fovrg_yk_zk_transform(FovrgYkZkTransformInput {
            source: source.view(),
            source_coefficients: coefficients.view(),
            radii: radii.view(),
            initial_power: Complex::new(1.35, -0.25),
            step: 0.0725,
            angular_momentum: 2,
            coefficient_count: 6,
            source_len: 9,
            active_len: 12,
            tail_correction: Complex::new(0.011, -0.006),
        })?;

        assert_eq!(transform.computed_len, 10);
        assert_complex_close(
            transform.origin_constant,
            1_069.293_326_934_643,
            639.337_203_837_502_8,
            1.0e-12,
        );

        let expected_rows = [
            (
                0.006_376_970_423_953_328,
                0.003_747_109_936_537_645_4,
                0.000_019_115_876_398_023_115,
                0.000_002_615_603_860_575_636,
            ),
            (
                0.007_841_326_927_116_237,
                0.004_425_503_213_295_339,
                0.000_415_175_421_810_819_7,
                0.001_186_221_311_123_577_3,
            ),
            (
                0.009_454_062_278_996_728,
                0.004_817_609_696_528_203,
                0.001_052_233_690_642_138_8,
                0.002_225_420_005_754_274_7,
            ),
            (
                0.011_156_498_748_891_856,
                0.004_912_703_002_968_925,
                0.001_915_624_422_266_479,
                0.003_118_393_016_633_964_7,
            ),
            (
                0.012_883_154_525_001_68,
                0.004_698_896_965_378_377,
                0.002_982_924_829_137_327,
                0.003_859_683_726_441_837_7,
            ),
            (
                0.014_563_357_943_144_598,
                0.004_164_285_902_400_606,
                0.004_223_307_649_668_978,
                0.004_440_459_445_284_031,
            ),
            (
                0.016_123_447_951_845_19,
                0.003_298_256_791_962_156,
                0.005_597_243_449_987_172_5,
                0.004_848_768_666_445_236,
            ),
            (
                0.017_489_549_856_229_16,
                0.002_093_015_402_338_084,
                0.007_056_612_425_756_153,
                0.005_069_801_813_782_371,
            ),
            (
                0.018_590_890_204_374_55,
                0.000_545_375_511_912_808,
                0.008_545_277_387_035_53,
                0.005_086_162_856_970_115_5,
            ),
            (
                0.019_630_800_153_888_66,
                -0.001_305_325_902_639_564_2,
                0.008_630_800_153_888_66,
                0.004_694_674_097_360_436,
            ),
        ];
        for (row, (yk_re, yk_im, zk_re, zk_im)) in expected_rows.into_iter().enumerate() {
            assert_complex_close(transform.yk[row], yk_re, yk_im, 1.0e-13);
            assert_complex_close(transform.zk[row], zk_re, zk_im, 1.0e-13);
        }

        let expected_coefficients = [
            (
                -1.824_158_963_244_542,
                -0.246_122_633_186_553_6,
                0.237_140_665_221_790_4,
                0.031_995_942_314_251_964,
            ),
            (
                2.794_098_740_012_050_3,
                0.730_272_609_187_755_4,
                0.195_586_911_800_843_6,
                0.051_119_082_643_142_88,
            ),
            (
                0.609_454_103_153_871_8,
                0.232_241_030_552_876_95,
                0.164_552_607_851_545_35,
                0.062_705_078_249_276_76,
            ),
            (
                0.297_530_519_518_787_1,
                0.147_284_129_112_308_2,
                0.139_839_344_173_829_93,
                0.069_223_540_682_784_84,
            ),
            (
                0.178_046_974_447_949_95,
                0.107_477_058_743_461_04,
                0.119_291_472_880_126_45,
                0.072_009_629_358_118_89,
            ),
            (
                0.116_898_830_661_045_85,
                0.082_624_380_349_051_94,
                0.101_701_982_675_109_88,
                0.071_883_210_903_675_18,
            ),
        ];
        for (row, (yk_re, yk_im, zk_re, zk_im)) in expected_coefficients.into_iter().enumerate() {
            assert_complex_close(transform.yk_coefficients[row], yk_re, yk_im, 1.0e-13);
            assert_complex_close(transform.zk_coefficients[row], zk_re, zk_im, 1.0e-13);
        }
        Ok(())
    }

    #[test]
    fn yk_zk_transform_rejects_invalid_inputs() {
        let (source, coefficients, radii) = yzktec_reference_inputs(12);

        assert!(matches!(
            fovrg_yk_zk_transform(FovrgYkZkTransformInput {
                source: source.view(),
                source_coefficients: coefficients.view(),
                radii: radii.view(),
                initial_power: Complex::new(1.35, 0.0),
                step: 0.0725,
                angular_momentum: 2,
                coefficient_count: 6,
                source_len: 9,
                active_len: 1,
                tail_correction: Complex::new(0.0, 0.0),
            }),
            Err(FovrgError::CountTooSmall {
                name: "active_len",
                ..
            })
        ));
        assert!(matches!(
            fovrg_yk_zk_transform(FovrgYkZkTransformInput {
                source: source.view(),
                source_coefficients: coefficients.view(),
                radii: radii.view(),
                initial_power: Complex::new(1.35, 0.0),
                step: 0.0,
                angular_momentum: 2,
                coefficient_count: 6,
                source_len: 9,
                active_len: 12,
                tail_correction: Complex::new(0.0, 0.0),
            }),
            Err(FovrgError::NonPositiveInput { name: "step", .. })
        ));
        assert!(matches!(
            fovrg_yk_zk_transform(FovrgYkZkTransformInput {
                source: source.view(),
                source_coefficients: coefficients.view(),
                radii: radii.view(),
                initial_power: Complex::new(1.35, 0.0),
                step: 0.0725,
                angular_momentum: 2,
                coefficient_count: 11,
                source_len: 9,
                active_len: 12,
                tail_correction: Complex::new(0.0, 0.0),
            }),
            Err(FovrgError::ActiveCountOutOfRange {
                field: "source_coefficients",
                ..
            })
        ));

        let mut bad_source = source.clone();
        bad_source[3] = Complex::new(0.0, Real::NAN);
        assert!(matches!(
            fovrg_yk_zk_transform(FovrgYkZkTransformInput {
                source: bad_source.view(),
                source_coefficients: coefficients.view(),
                radii: radii.view(),
                initial_power: Complex::new(1.35, 0.0),
                step: 0.0725,
                angular_momentum: 2,
                coefficient_count: 6,
                source_len: 9,
                active_len: 12,
                tail_correction: Complex::new(0.0, 0.0),
            }),
            Err(FovrgError::NonFiniteComplexInput {
                name: "source",
                row: 3,
                ..
            })
        ));

        let mut bad_radii = radii.clone();
        bad_radii[0] = -1.0;
        assert!(matches!(
            fovrg_yk_zk_transform(FovrgYkZkTransformInput {
                source: source.view(),
                source_coefficients: coefficients.view(),
                radii: bad_radii.view(),
                initial_power: Complex::new(1.35, 0.0),
                step: 0.0725,
                angular_momentum: 2,
                coefficient_count: 6,
                source_len: 9,
                active_len: 12,
                tail_correction: Complex::new(0.0, 0.0),
            }),
            Err(FovrgError::NonPositiveRadius { row: 0, .. })
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

    fn yzktec_reference_inputs(count: usize) -> (Array1<Complex>, Array1<Complex>, Array1<Real>) {
        let step = 0.0725;
        let source = Array1::from_iter((1..=count).map(|index| {
            let index = index as Real;
            Complex::new(
                (0.19 * index).sin() + 0.02 * index,
                (0.11 * index).cos() - 0.03 * index,
            )
        }));
        let coefficients = Array1::from_iter((1..=10).map(|index| {
            let index = index as Real;
            Complex::new(
                0.04 * index + (0.13 * index).cos(),
                -0.03 * index + (0.17 * index).sin(),
            )
        }));
        let radii = Array1::from_iter((1..=count).map(|index| {
            let index = index as Real;
            0.018 * (step * (index - 1.0)).exp()
        }));
        (source, coefficients, radii)
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
