//! FEFF FOVRG numerical helpers.
//!
//! These routines cover small pieces of the relativistic radial solver that can
//! be validated independently of the full `dfovrg` integration path.

use ndarray::{Array1, Array2, ArrayView1};

use crate::{
    Complex, ComplexVec, Real, RealMat,
    angular::wigner_3j,
    bessel::{besjh, besjn},
};

// `diff.f90` uses unsuffixed Fortran real literals in these stencils. Preserve
// their default-real rounding before widening to the Rust `Real` type.
const F77_REAL_HALF: Real = 0.5_f32 as Real;
const F77_REAL_ONE_POINT_TWO: Real = 1.2_f32 as Real;
const F77_REAL_ONE_POINT_FIVE: Real = 1.5_f32 as Real;
const F77_REAL_TWO: Real = 2.0_f32 as Real;
const F77_REAL_TWO_POINT_FOUR_FIVE: Real = 2.45_f32 as Real;
const F77_REAL_THREE_POINT_THREE: Real = 3.3_f32 as Real;
const F77_REAL_THREE_POINT_SEVEN_FIVE: Real = 3.75_f32 as Real;
const F77_REAL_FOUR_POINT_TWO: Real = 4.2_f32 as Real;
const F77_REAL_SIX: Real = 6.0_f32 as Real;
const F77_REAL_SIX_AND_TWO_THIRDS: Real = 6.666_666_5_f32 as Real;
const F77_REAL_SEVEN_POINT_FIVE: Real = 7.5_f32 as Real;
const F77_REAL_SEVEN_POINT_EIGHT: Real = 7.8_f32 as Real;
const F77_REAL_EIGHT: Real = 8.0_f32 as Real;
const F77_REAL_TWELVE: Real = 12.0_f32 as Real;
const F77_REAL_ONE_SIXTH: Real = 0.166_666_67_f32 as Real;
const F77_REAL_FOURTEEN_OVER_FORTY_FIVE: Real = (14.0_f32 as Real) / (45.0_f32 as Real);
const F77_REAL_TWENTY_FOUR_OVER_FORTY_FIVE: Real = (24.0_f32 as Real) / (45.0_f32 as Real);
const F77_REAL_SIXTY_FOUR_OVER_FORTY_FIVE: Real = (64.0_f32 as Real) / (45.0_f32 as Real);
const FOVRG_INT_OUT_HISTORY: usize = 6;
const FOVRG_INT_OUT_TEST: Real = 1.0e5;
const FOVRG_ANGULAR_COEFFICIENT_SLOTS: usize = 5;
const FOVRG_ORIGIN_COEFFICIENTS: usize = 10;
const FEFF_ALPHA_INVERSE: Real = 137.03598956;
const FEFF_FINE_STRUCTURE_ALPHA: Real = 1.0 / FEFF_ALPHA_INVERSE;
const FEFF_WFIRDC_SPEED_OF_LIGHT: Real = 137.0373;
const FOVRG_BOUND_ORBITAL_THRESHOLD: Real = 1.0e-11;
const FOVRG_WKB_MINIMUM_COUNT: usize = 10;

mod internals;
mod types;

use internals::*;

pub use types::*;

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

/// Port of `FOVRG/aprdep.f90`: real polynomial product coefficient.
///
/// Returns the coefficient for power `coefficient_count - 1` in the product of
/// two real origin-development polynomials.
pub fn fovrg_real_product_coefficient(
    left_coefficients: ArrayView1<'_, Real>,
    right_coefficients: ArrayView1<'_, Real>,
    coefficient_count: usize,
) -> Result<Real, FovrgError> {
    validate_count_at_least("coefficient_count", coefficient_count, 1)?;
    validate_active_len(
        "left_coefficients",
        coefficient_count,
        left_coefficients.len(),
    )?;
    validate_active_len(
        "right_coefficients",
        coefficient_count,
        right_coefficients.len(),
    )?;
    for coefficient in 0..coefficient_count {
        validate_real_input(
            "left_coefficients",
            coefficient,
            left_coefficients[coefficient],
        )?;
        validate_real_input(
            "right_coefficients",
            coefficient,
            right_coefficients[coefficient],
        )?;
    }

    let coefficient =
        real_product_coefficient(left_coefficients, right_coefficients, coefficient_count);
    validate_real_result("real_product_coefficient", 0, coefficient)?;
    Ok(coefficient)
}

/// Port of `FOVRG/aprdec.f90`: complex-real polynomial product coefficient.
///
/// Returns the coefficient for power `coefficient_count - 1` in the product of
/// a complex origin-development polynomial and a real one.
pub fn fovrg_complex_real_product_coefficient(
    complex_coefficients: ArrayView1<'_, Complex>,
    real_coefficients: ArrayView1<'_, Real>,
    coefficient_count: usize,
) -> Result<Complex, FovrgError> {
    validate_count_at_least("coefficient_count", coefficient_count, 1)?;
    validate_active_len(
        "complex_coefficients",
        coefficient_count,
        complex_coefficients.len(),
    )?;
    validate_active_len(
        "real_coefficients",
        coefficient_count,
        real_coefficients.len(),
    )?;
    for coefficient in 0..coefficient_count {
        validate_complex_input(
            "complex_coefficients",
            coefficient,
            complex_coefficients[coefficient],
        )?;
        validate_real_input(
            "real_coefficients",
            coefficient,
            real_coefficients[coefficient],
        )?;
    }

    let coefficient = complex_real_product_coefficient(
        complex_coefficients,
        real_coefficients,
        coefficient_count,
    );
    validate_complex_result("complex_real_product_coefficient", 0, coefficient)?;
    Ok(coefficient)
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

/// Port of `FOVRG/yzkrdc.f90`: construct exchange source terms and `yk/zk`.
///
/// FEFF forms `f = cg_i * ps + cp_i * qs`, builds origin coefficients from the
/// products of the large/small development polynomials, and then delegates the
/// radial integrations to `yzktec`.
pub fn fovrg_yk_zk_exchange(
    input: FovrgYkZkExchangeInput<'_>,
) -> Result<FovrgYkZkTransform, FovrgError> {
    validate_count_at_least("active_len", input.active_len, 2)?;
    validate_count_at_least("source_len", input.source_len, 1)?;
    validate_count_at_least("orbital_len", input.orbital_len, 1)?;
    validate_count_at_least("coefficient_count", input.coefficient_count, 1)?;
    validate_active_len(
        "large_component",
        input.active_len,
        input.large_component.len(),
    )?;
    validate_active_len(
        "small_component",
        input.active_len,
        input.small_component.len(),
    )?;
    validate_active_len(
        "partner_large_component",
        input.active_len,
        input.partner_large_component.len(),
    )?;
    validate_active_len(
        "partner_small_component",
        input.active_len,
        input.partner_small_component.len(),
    )?;
    validate_active_len("radii", input.active_len, input.radii.len())?;
    validate_active_len(
        "large_coefficients",
        input.coefficient_count,
        input.large_coefficients.len(),
    )?;
    validate_active_len(
        "small_coefficients",
        input.coefficient_count,
        input.small_coefficients.len(),
    )?;
    validate_active_len(
        "partner_large_coefficients",
        input.coefficient_count,
        input.partner_large_coefficients.len(),
    )?;
    validate_active_len(
        "partner_small_coefficients",
        input.coefficient_count,
        input.partner_small_coefficients.len(),
    )?;
    validate_positive_finite("step", input.step)?;
    validate_finite("orbital_power", input.orbital_power)?;
    validate_finite("partner_power", input.partner_power)?;

    let source_len = input
        .orbital_len
        .min(input.source_len)
        .min(input.active_len - 1);
    for row in 0..source_len {
        validate_real_input("large_component", row, input.large_component[row])?;
        validate_real_input("small_component", row, input.small_component[row])?;
        validate_complex_input(
            "partner_large_component",
            row,
            input.partner_large_component[row],
        )?;
        validate_complex_input(
            "partner_small_component",
            row,
            input.partner_small_component[row],
        )?;
    }
    for coefficient in 0..input.coefficient_count {
        validate_real_input(
            "large_coefficients",
            coefficient,
            input.large_coefficients[coefficient],
        )?;
        validate_real_input(
            "small_coefficients",
            coefficient,
            input.small_coefficients[coefficient],
        )?;
        validate_complex_input(
            "partner_large_coefficients",
            coefficient,
            input.partner_large_coefficients[coefficient],
        )?;
        validate_complex_input(
            "partner_small_coefficients",
            coefficient,
            input.partner_small_coefficients[coefficient],
        )?;
    }

    let source = Array1::from_iter((0..input.active_len).map(|row| {
        if row < source_len {
            input.large_component[row] * input.partner_large_component[row]
                + input.small_component[row] * input.partner_small_component[row]
        } else {
            Complex::new(0.0, 0.0)
        }
    }));
    let source_coefficients = Array1::from_iter((1..=input.coefficient_count).map(|count| {
        complex_real_product_coefficient(
            input.partner_large_coefficients,
            input.large_coefficients,
            count,
        ) + complex_real_product_coefficient(
            input.partner_small_coefficients,
            input.small_coefficients,
            count,
        )
    }));

    fovrg_yk_zk_transform(FovrgYkZkTransformInput {
        source: source.view(),
        source_coefficients: source_coefficients.view(),
        radii: input.radii,
        initial_power: Complex::new(input.orbital_power + input.partner_power, 0.0),
        step: input.step,
        angular_momentum: input.angular_momentum,
        coefficient_count: input.coefficient_count,
        source_len,
        active_len: input.active_len,
        tail_correction: Complex::new(0.0, 0.0),
    })
}

/// Port of `FOVRG/dsordc.f90`: complex radial overlap integral.
///
/// FEFF forms `hg = dg * cg_j + dp * cp_j`, integrates `hg(r) * r` over the
/// logarithmic radial mesh with its Simpson stencil, and adds the analytic
/// origin contribution from the product of the large/small development
/// coefficients.
pub fn fovrg_overlap_integral(input: FovrgOverlapIntegralInput<'_>) -> Result<Complex, FovrgError> {
    validate_count_at_least("active_len", input.active_len, 3)?;
    validate_count_at_least("coefficient_count", input.coefficient_count, 1)?;
    if input.active_len.is_multiple_of(2) {
        return Err(FovrgError::CountMustBeOdd {
            name: "active_len",
            actual: input.active_len,
        });
    }
    validate_active_len(
        "large_integrand",
        input.active_len,
        input.large_integrand.len(),
    )?;
    validate_active_len(
        "small_integrand",
        input.active_len,
        input.small_integrand.len(),
    )?;
    validate_active_len(
        "large_component",
        input.active_len,
        input.large_component.len(),
    )?;
    validate_active_len(
        "small_component",
        input.active_len,
        input.small_component.len(),
    )?;
    validate_active_len("radii", input.active_len, input.radii.len())?;
    validate_active_len(
        "large_integrand_coefficients",
        input.coefficient_count,
        input.large_integrand_coefficients.len(),
    )?;
    validate_active_len(
        "small_integrand_coefficients",
        input.coefficient_count,
        input.small_integrand_coefficients.len(),
    )?;
    validate_active_len(
        "large_coefficients",
        input.coefficient_count,
        input.large_coefficients.len(),
    )?;
    validate_active_len(
        "small_coefficients",
        input.coefficient_count,
        input.small_coefficients.len(),
    )?;
    validate_finite("integrand_power", input.integrand_power)?;
    validate_finite("orbital_power", input.orbital_power)?;
    validate_positive_finite("step", input.step)?;

    for row in 0..input.active_len {
        validate_complex_input("large_integrand", row, input.large_integrand[row])?;
        validate_complex_input("small_integrand", row, input.small_integrand[row])?;
        validate_real_input("large_component", row, input.large_component[row])?;
        validate_real_input("small_component", row, input.small_component[row])?;
        validate_radius(row, input.radii[row])?;
    }
    for coefficient in 0..input.coefficient_count {
        validate_complex_input(
            "large_integrand_coefficients",
            coefficient,
            input.large_integrand_coefficients[coefficient],
        )?;
        validate_complex_input(
            "small_integrand_coefficients",
            coefficient,
            input.small_integrand_coefficients[coefficient],
        )?;
        validate_real_input(
            "large_coefficients",
            coefficient,
            input.large_coefficients[coefficient],
        )?;
        validate_real_input(
            "small_coefficients",
            coefficient,
            input.small_coefficients[coefficient],
        )?;
    }

    let mixed_integrand = Array1::from_iter((0..input.active_len).map(|row| {
        (input.large_integrand[row] * input.large_component[row]
            + input.small_integrand[row] * input.small_component[row])
            * input.radii[row]
    }));

    let simpson_sum = (1..input.active_len - 1)
        .step_by(2)
        .fold(Complex::new(0.0, 0.0), |sum, row| {
            sum + mixed_integrand[row] + mixed_integrand[row] + mixed_integrand[row + 1]
        });
    let mut integral = input.step
        * (simpson_sum + simpson_sum + mixed_integrand[0] - mixed_integrand[input.active_len - 1])
        / 3.0;

    let mut origin_power = input.integrand_power + input.orbital_power;
    for coefficient in 1..=input.coefficient_count {
        origin_power += 1.0;
        validate_nonzero_denominator("overlap_origin_power", origin_power)?;
        let origin_coefficient = complex_real_product_coefficient(
            input.large_integrand_coefficients,
            input.large_coefficients,
            coefficient,
        ) + complex_real_product_coefficient(
            input.small_integrand_coefficients,
            input.small_coefficients,
            coefficient,
        );
        integral += origin_coefficient * input.radii[0].powf(origin_power) / origin_power;
    }
    validate_complex_result("overlap_integral", 0, integral)?;
    Ok(integral)
}

/// Port of `FOVRG/ortdac.f90`: Schmidt orthogonalization against bound orbitals.
///
/// FEFF walks the bound orbitals in order, skips orbitals whose kappa differs
/// from `ikap` or whose occupation is not positive, computes the current
/// overlap with `dsordc`, and subtracts that overlap from both the radial
/// target arrays and their origin development coefficients.
pub fn fovrg_schmidt_orthogonalize(
    input: FovrgOrthogonalizationInput<'_>,
) -> Result<FovrgOrthogonalization, FovrgError> {
    validate_count_at_least("active_len", input.active_len, 3)?;
    validate_count_at_least("coefficient_count", input.coefficient_count, 1)?;
    if input.active_len.is_multiple_of(2) {
        return Err(FovrgError::CountMustBeOdd {
            name: "active_len",
            actual: input.active_len,
        });
    }
    validate_active_len(
        "target_large_component",
        input.active_len,
        input.target_large_component.len(),
    )?;
    validate_active_len(
        "target_small_component",
        input.active_len,
        input.target_small_component.len(),
    )?;
    validate_active_len("radii", input.active_len, input.radii.len())?;
    validate_active_len(
        "target_large_coefficients",
        input.coefficient_count,
        input.target_large_coefficients.len(),
    )?;
    validate_active_len(
        "target_small_coefficients",
        input.coefficient_count,
        input.target_small_coefficients.len(),
    )?;
    validate_matrix_rows(
        "bound_large_components",
        input.active_len,
        input.bound_large_components.shape()[0],
    )?;
    validate_matrix_rows(
        "bound_small_components",
        input.active_len,
        input.bound_small_components.shape()[0],
    )?;
    validate_matrix_cols(
        "bound_large_components",
        input.bound_orbital_count,
        input.bound_large_components.shape()[1],
    )?;
    validate_matrix_cols(
        "bound_small_components",
        input.bound_orbital_count,
        input.bound_small_components.shape()[1],
    )?;
    validate_matrix_rows(
        "bound_large_coefficients",
        input.coefficient_count,
        input.bound_large_coefficients.shape()[0],
    )?;
    validate_matrix_rows(
        "bound_small_coefficients",
        input.coefficient_count,
        input.bound_small_coefficients.shape()[0],
    )?;
    validate_matrix_cols(
        "bound_large_coefficients",
        input.bound_orbital_count,
        input.bound_large_coefficients.shape()[1],
    )?;
    validate_matrix_cols(
        "bound_small_coefficients",
        input.bound_orbital_count,
        input.bound_small_coefficients.shape()[1],
    )?;
    validate_active_len(
        "electron_counts",
        input.bound_orbital_count,
        input.electron_counts.len(),
    )?;
    validate_active_len("kappa", input.bound_orbital_count, input.kappa.len())?;
    validate_active_len(
        "orbital_powers",
        input.bound_orbital_count,
        input.orbital_powers.len(),
    )?;
    validate_nonzero_kappa("target_kappa", 0, input.target_kappa)?;
    validate_finite("target_power", input.target_power)?;
    validate_positive_finite("step", input.step)?;

    for row in 0..input.active_len {
        validate_complex_input(
            "target_large_component",
            row,
            input.target_large_component[row],
        )?;
        validate_complex_input(
            "target_small_component",
            row,
            input.target_small_component[row],
        )?;
        validate_radius(row, input.radii[row])?;
        for orbital in 0..input.bound_orbital_count {
            validate_real_input(
                "bound_large_components",
                row,
                input.bound_large_components[(row, orbital)],
            )?;
            validate_real_input(
                "bound_small_components",
                row,
                input.bound_small_components[(row, orbital)],
            )?;
        }
    }
    for coefficient in 0..input.coefficient_count {
        validate_complex_input(
            "target_large_coefficients",
            coefficient,
            input.target_large_coefficients[coefficient],
        )?;
        validate_complex_input(
            "target_small_coefficients",
            coefficient,
            input.target_small_coefficients[coefficient],
        )?;
        for orbital in 0..input.bound_orbital_count {
            validate_real_input(
                "bound_large_coefficients",
                coefficient,
                input.bound_large_coefficients[(coefficient, orbital)],
            )?;
            validate_real_input(
                "bound_small_coefficients",
                coefficient,
                input.bound_small_coefficients[(coefficient, orbital)],
            )?;
        }
    }
    for orbital in 0..input.bound_orbital_count {
        validate_real_input("electron_counts", orbital, input.electron_counts[orbital])?;
        validate_nonzero_kappa("kappa", orbital, input.kappa[orbital])?;
        validate_real_input("orbital_powers", orbital, input.orbital_powers[orbital])?;
    }

    let mut large_component = input.target_large_component.to_owned();
    let mut small_component = input.target_small_component.to_owned();
    let mut large_coefficients = input.target_large_coefficients.to_owned();
    let mut small_coefficients = input.target_small_coefficients.to_owned();
    let mut overlaps = Array1::<Complex>::zeros(input.bound_orbital_count);

    for orbital in 0..input.bound_orbital_count {
        if input.kappa[orbital] != input.target_kappa || input.electron_counts[orbital] <= 0.0 {
            continue;
        }

        let overlap = fovrg_overlap_integral(FovrgOverlapIntegralInput {
            large_integrand: large_component.view(),
            small_integrand: small_component.view(),
            large_integrand_coefficients: large_coefficients.view(),
            small_integrand_coefficients: small_coefficients.view(),
            large_component: input.bound_large_components.column(orbital),
            small_component: input.bound_small_components.column(orbital),
            large_coefficients: input.bound_large_coefficients.column(orbital),
            small_coefficients: input.bound_small_coefficients.column(orbital),
            radii: input.radii,
            integrand_power: input.target_power,
            orbital_power: input.orbital_powers[orbital],
            step: input.step,
            coefficient_count: input.coefficient_count,
            active_len: input.active_len,
        })?;
        overlaps[orbital] = overlap;

        for row in 0..input.active_len {
            large_component[row] -= overlap * input.bound_large_components[(row, orbital)];
            small_component[row] -= overlap * input.bound_small_components[(row, orbital)];
        }
        for coefficient in 0..input.coefficient_count {
            large_coefficients[coefficient] -=
                overlap * input.bound_large_coefficients[(coefficient, orbital)];
            small_coefficients[coefficient] -=
                overlap * input.bound_small_coefficients[(coefficient, orbital)];
        }
    }

    for row in 0..input.active_len {
        validate_complex_result("large_component", row, large_component[row])?;
        validate_complex_result("small_component", row, small_component[row])?;
    }
    for coefficient in 0..input.coefficient_count {
        validate_complex_result(
            "large_coefficients",
            coefficient,
            large_coefficients[coefficient],
        )?;
        validate_complex_result(
            "small_coefficients",
            coefficient,
            small_coefficients[coefficient],
        )?;
    }
    for orbital in 0..input.bound_orbital_count {
        validate_complex_result("overlaps", orbital, overlaps[orbital])?;
    }

    Ok(FovrgOrthogonalization {
        large_component,
        small_component,
        large_coefficients,
        small_coefficients,
        overlaps,
    })
}

/// Port of `FOVRG/muatcc.f90`: angular coefficients for exchange coupling.
///
/// FEFF builds `afgkc(ikap, orbital, index)` for every target kappa. This
/// helper returns the single target-kappa row consumed by `potex`, indexed as
/// `(orbital, index)` with FEFF's fixed five coefficient slots.
pub fn fovrg_angular_coefficients(
    input: FovrgAngularCoefficientsInput<'_>,
) -> Result<RealMat, FovrgError> {
    validate_count_at_least("bound_orbital_count", input.bound_orbital_count, 1)?;
    validate_active_len(
        "electron_counts",
        input.bound_orbital_count,
        input.electron_counts.len(),
    )?;
    validate_active_len(
        "valence_counts",
        input.bound_orbital_count,
        input.valence_counts.len(),
    )?;
    validate_active_len("kappa", input.bound_orbital_count, input.kappa.len())?;
    validate_nonzero_kappa("target_kappa", 0, input.target_kappa)?;

    for orbital in 0..input.bound_orbital_count {
        validate_real_input("electron_counts", orbital, input.electron_counts[orbital])?;
        validate_real_input("valence_counts", orbital, input.valence_counts[orbital])?;
        validate_nonzero_kappa("kappa", orbital, input.kappa[orbital])?;
    }

    let target_j = target_j_value(input.target_kappa);
    let target_j_i32 = fovrg_usize_to_i32("target_j", target_j)?;
    let mut coefficients =
        Array2::<Real>::zeros((input.bound_orbital_count, FOVRG_ANGULAR_COEFFICIENT_SLOTS));

    for orbital in 0..input.bound_orbital_count {
        let bound_j = target_j_value(input.kappa[orbital]);
        let max_multipole = (target_j + bound_j) / 2;
        let mut min_multipole = target_j.abs_diff(bound_j) / 2;
        if (input.target_kappa < 0) != (input.kappa[orbital] < 0) {
            min_multipole += 1;
        }
        let required_slots = (max_multipole - min_multipole) / 2 + 1;
        if required_slots > FOVRG_ANGULAR_COEFFICIENT_SLOTS {
            return Err(FovrgError::CountTooLarge {
                name: "angular_coefficient_slots",
                actual: required_slots,
                maximum: FOVRG_ANGULAR_COEFFICIENT_SLOTS,
            });
        }
        if input.valence_counts[orbital] > 0.0 {
            continue;
        }

        let bound_j_i32 = fovrg_usize_to_i32("bound_j", bound_j)?;
        let mut multipole = min_multipole;
        while multipole <= max_multipole {
            let angular_index = (multipole - min_multipole) / 2;
            let doubled_multipole = multipole.checked_mul(2).ok_or(FovrgError::CountTooLarge {
                name: "doubled_multipole",
                actual: multipole,
                maximum: usize::MAX / 2,
            })?;
            let wigner = wigner_3j(
                target_j_i32,
                fovrg_usize_to_i32("doubled_multipole", doubled_multipole)?,
                bound_j_i32,
                1,
                0,
                2,
            )
            .map_err(|source| FovrgError::AngularCoefficient { source })?;
            let coefficient = input.electron_counts[orbital] * wigner * wigner;
            validate_real_result("angular_coefficients", orbital, coefficient)?;
            coefficients[(orbital, angular_index)] = coefficient;
            multipole += 2;
        }
    }

    Ok(coefficients)
}

/// Port of FEFF `FOVRG/dfovrg.f90` `flatv`: exact flat-potential propagation.
///
/// For a constant potential between two radii, FEFF solves the Dirac equation
/// analytically with spherical Bessel and Neumann functions. The returned
/// components are the values at `end_radius` implied by the initial components
/// at `start_radius`.
pub fn fovrg_flat_potential_propagate(
    input: FovrgFlatPotentialInput,
) -> Result<FovrgFlatPotentialPropagation, FovrgError> {
    validate_positive_finite("start_radius", input.start_radius)?;
    validate_positive_finite("end_radius", input.end_radius)?;
    validate_complex_input("large_component", 0, input.large_component)?;
    validate_complex_input("small_component", 0, input.small_component)?;
    validate_complex_input("energy", 0, input.energy)?;
    validate_complex_input("average_potential", 0, input.average_potential)?;
    validate_nonzero_kappa("kappa", 0, input.kappa)?;

    let energy_offset = input.energy - input.average_potential;
    let alpha_wave_offset = FEFF_FINE_STRUCTURE_ALPHA * energy_offset;
    let wave_number = (2.0 * energy_offset + alpha_wave_offset * alpha_wave_offset).sqrt();
    let start_argument = wave_number * input.start_radius;

    let (sign, large_l, small_l) = if input.kappa < 0 {
        let large_l = input.kappa.unsigned_abs() as usize - 1;
        (-1.0, large_l, large_l + 1)
    } else {
        let large_l = input.kappa as usize;
        (1.0, large_l, large_l - 1)
    };
    let max_l = large_l.max(small_l);
    let alpha_wave = wave_number * FEFF_FINE_STRUCTURE_ALPHA;
    let factor = sign * alpha_wave / (1.0 + (1.0 + alpha_wave * alpha_wave).sqrt());
    validate_nonzero_complex_denominator("flat_potential_factor", factor)?;

    let start_bessel = besjn(start_argument, max_l)
        .map_err(|source| FovrgError::FlatPotentialBessel { source })?;
    let amplitude_j = sign
        * wave_number
        * start_argument
        * (input.large_component * start_bessel.y[small_l]
            - input.small_component * start_bessel.y[large_l] / factor);
    let amplitude_y = sign
        * wave_number
        * start_argument
        * (input.small_component * start_bessel.j[large_l] / factor
            - input.large_component * start_bessel.j[small_l]);

    let end_argument = wave_number * input.end_radius;
    let end_bessel =
        besjn(end_argument, max_l).map_err(|source| FovrgError::FlatPotentialBessel { source })?;
    let large_component = input.end_radius
        * (end_bessel.j[large_l] * amplitude_j + end_bessel.y[large_l] * amplitude_y);
    let small_component = input.end_radius
        * factor
        * (end_bessel.j[small_l] * amplitude_j + end_bessel.y[small_l] * amplitude_y);

    validate_complex_result("flat_large_component", 0, large_component)?;
    validate_complex_result("flat_small_component", 0, small_component)?;
    Ok(FovrgFlatPotentialPropagation {
        large_component,
        small_component,
    })
}

/// Port of FEFF `FOVRG/intout.f90`: outward Dirac radial integration.
///
/// FEFF starts with a six-point Runge-Kutta bootstrap, converts those
/// derivatives to Milne history values, then advances the inhomogeneous Dirac
/// system with predictor-corrector iterations. Rows after `last_index` are
/// zero-filled like FEFF's `max0+1:np` cleanup.
pub fn fovrg_outward_integrate(
    input: FovrgOutwardIntegrationInput<'_>,
) -> Result<FovrgOutwardIntegration, FovrgError> {
    validate_outward_integration_input(&input)?;

    let ccl = input.speed_of_light + input.speed_of_light;
    let kappa = input.kappa as Real;
    let energy_over_light = input.energy / input.speed_of_light;
    let exp_half_step = (input.step / 2.0).exp();
    let mut large_component = Array1::<Complex>::zeros(input.active_len);
    let mut small_component = Array1::<Complex>::zeros(input.active_len);
    large_component[input.start_index] = input.initial_large_component;
    small_component[input.start_index] = input.initial_small_component;

    let mut large_derivative = [Complex::new(0.0, 0.0); FOVRG_INT_OUT_HISTORY];
    let mut small_derivative = [Complex::new(0.0, 0.0); FOVRG_INT_OUT_HISTORY];
    let mut current = input.start_index;
    let mut history_index = 0usize;
    let mut difficult_iterations = 0usize;

    let (f, g, c3) = fovrg_outward_grid_terms(&input, current, energy_over_light, ccl)?;
    large_derivative[history_index] = input.step
        * (g * small_component[current] - kappa * large_component[current]
            + input.small_exchange[current]);
    small_derivative[history_index] = input.step
        * (kappa * small_component[current]
            - (f - c3) * large_component[current]
            - input.large_exchange[current]);

    while current < input.last_index {
        let midpoint =
            fovrg_outward_midpoint_terms(&input, current, energy_over_light, ccl, exp_half_step)?;
        let mut large_trial = large_component[current] + 0.5 * large_derivative[history_index];
        let mut small_trial = small_component[current] + 0.5 * small_derivative[history_index];
        let large_derivative_2 =
            input.step * (midpoint.g * small_trial - kappa * large_trial + midpoint.small_exchange);
        let small_derivative_2 = input.step
            * (kappa * small_trial
                - (midpoint.f - midpoint.c3) * large_trial
                - midpoint.large_exchange);
        large_trial += F77_REAL_HALF * (large_derivative_2 - large_derivative[history_index]);
        small_trial += F77_REAL_HALF * (small_derivative_2 - small_derivative[history_index]);
        let large_derivative_3 =
            input.step * (midpoint.g * small_trial - kappa * large_trial + midpoint.small_exchange);
        let small_derivative_3 = input.step
            * (kappa * small_trial
                - (midpoint.f - midpoint.c3) * large_trial
                - midpoint.large_exchange);
        large_trial += large_derivative_3 - F77_REAL_HALF * large_derivative_2;
        small_trial += small_derivative_3 - F77_REAL_HALF * small_derivative_2;

        current += 1;
        history_index += 1;
        let (f, g, c3) = fovrg_outward_grid_terms(&input, current, energy_over_light, ccl)?;
        let large_derivative_4 =
            input.step * (g * small_trial - kappa * large_trial + input.small_exchange[current]);
        let small_derivative_4 = input.step
            * (kappa * small_trial - (f - c3) * large_trial - input.large_exchange[current]);
        large_component[current] = large_component[current - 1]
            + (large_derivative[history_index - 1]
                + F77_REAL_TWO * (large_derivative_2 + large_derivative_3)
                + large_derivative_4)
                / F77_REAL_SIX;
        small_component[current] = small_component[current - 1]
            + (small_derivative[history_index - 1]
                + F77_REAL_TWO * (small_derivative_2 + small_derivative_3)
                + small_derivative_4)
                / F77_REAL_SIX;
        large_derivative[history_index] = input.step
            * (g * small_component[current] - kappa * large_component[current]
                + input.small_exchange[current]);
        small_derivative[history_index] = input.step
            * (kappa * small_component[current]
                - (f - c3) * large_component[current]
                - input.large_exchange[current]);

        if history_index + 1 >= FOVRG_INT_OUT_HISTORY {
            break;
        }
    }

    if current < input.last_index {
        for row in 0..FOVRG_INT_OUT_HISTORY {
            large_derivative[row] /= input.step;
            small_derivative[row] /= input.step;
        }

        let a1 = input.step * F77_REAL_THREE_POINT_THREE;
        let a2 = -input.step * F77_REAL_FOUR_POINT_TWO;
        let a3 = input.step * F77_REAL_SEVEN_POINT_EIGHT;
        let a4 = input.step * F77_REAL_FOURTEEN_OVER_FORTY_FIVE;
        let a5 = input.step * F77_REAL_SIXTY_FOUR_OVER_FORTY_FIVE;
        let a6 = input.step * F77_REAL_TWENTY_FOUR_OVER_FORTY_FIVE;

        for row in (input.start_index + FOVRG_INT_OUT_HISTORY - 1)..input.last_index {
            let mut predicted_large = large_component[row - 5]
                + a1 * (large_derivative[5] + large_derivative[1])
                + a2 * (large_derivative[4] + large_derivative[2])
                + a3 * large_derivative[3];
            let mut predicted_small = small_component[row - 5]
                + a1 * (small_derivative[5] + small_derivative[1])
                + a2 * (small_derivative[4] + small_derivative[2])
                + a3 * small_derivative[3];
            let corrected_large_base = large_component[row - 3]
                + a4 * large_derivative[2]
                + a5 * (large_derivative[5] + large_derivative[3])
                + a6 * large_derivative[4];
            let corrected_small_base = small_component[row - 3]
                + a4 * small_derivative[2]
                + a5 * (small_derivative[5] + small_derivative[3])
                + a6 * small_derivative[4];

            large_derivative.copy_within(1..FOVRG_INT_OUT_HISTORY, 0);
            small_derivative.copy_within(1..FOVRG_INT_OUT_HISTORY, 0);

            let next = row + 1;
            let (f, g, c3) = fovrg_outward_grid_terms(&input, next, energy_over_light, ccl)?;
            let mut retry_count = 0usize;
            loop {
                large_derivative[5] =
                    g * predicted_small - kappa * predicted_large + input.small_exchange[next];
                small_derivative[5] = kappa * predicted_small
                    - (f - c3) * predicted_large
                    - input.large_exchange[next];
                large_component[next] = corrected_large_base + a4 * large_derivative[5];
                small_component[next] = corrected_small_base + a4 * small_derivative[5];

                let large_failed = (FOVRG_INT_OUT_TEST * (large_component[next] - predicted_large))
                    .norm()
                    > large_component[next].norm();
                let small_failed = (FOVRG_INT_OUT_TEST * (small_component[next] - predicted_small))
                    .norm()
                    > small_component[next].norm();
                if large_failed || small_failed {
                    if retry_count < 40 {
                        predicted_large = large_component[next];
                        predicted_small = small_component[next];
                        retry_count += 1;
                    } else {
                        difficult_iterations += 1;
                        break;
                    }
                } else {
                    break;
                }
            }
        }
    }

    for row in input.last_index + 1..input.active_len {
        large_component[row] = Complex::new(0.0, 0.0);
        small_component[row] = Complex::new(0.0, 0.0);
    }
    for row in 0..input.active_len {
        validate_complex_result("large_component", row, large_component[row])?;
        validate_complex_result("small_component", row, small_component[row])?;
    }

    Ok(FovrgOutwardIntegration {
        large_component,
        small_component,
        difficult_iterations,
    })
}

/// Port of FEFF `FOVRG/solout.f90`: regular solution integrated outward.
///
/// FEFF builds the origin power-series coefficients, integrates from the
/// origin through `min(jri, iwkb)` with `intout`, then uses exact flat-potential
/// propagation to reach `max0`. The returned arrays follow FEFF's zero-fill
/// convention after `last_index`.
pub fn fovrg_outgoing_solution(
    input: FovrgOutgoingSolutionInput<'_>,
) -> Result<FovrgOutgoingSolution, FovrgError> {
    validate_outgoing_solution_input(&input)?;

    let mut large_coefficients = Array1::<Complex>::zeros(
        input
            .potential_coefficients
            .len()
            .max(input.coefficient_count),
    );
    let mut small_coefficients = Array1::<Complex>::zeros(
        input
            .potential_coefficients
            .len()
            .max(input.coefficient_count),
    );
    let mut initial_small_coefficient = input.initial_small_coefficient;
    if input.potential_coefficients[0].re < 0.0 {
        if input.kappa > 0 {
            validate_nonzero_complex_denominator(
                "solout_point_nucleus_large_denominator",
                input.potential_coefficients[0],
            )?;
            initial_small_coefficient = -input.initial_large_coefficient
                * (input.kappa as Real + input.origin_power)
                / input.potential_coefficients[0];
        } else if input.kappa < 0 {
            let denominator = input.kappa as Real - input.origin_power;
            validate_nonzero_denominator("solout_point_nucleus_small_denominator", denominator)?;
            initial_small_coefficient =
                -input.initial_large_coefficient * input.potential_coefficients[0] / denominator;
        }
    }

    large_coefficients[0] = input.initial_large_coefficient;
    small_coefficients[0] = initial_small_coefficient;
    for coefficient in 1..input.coefficient_count {
        large_coefficients[coefficient] = input.large_exchange_coefficients[coefficient - 1];
        small_coefficients[coefficient] = input.small_exchange_coefficients[coefficient - 1];
    }

    let energy_over_light = input.energy / input.speed_of_light;
    if input.c3_scale == 0 {
        fovrg_desclaux_origin_series(
            input,
            &mut large_coefficients,
            &mut small_coefficients,
            energy_over_light,
        )?;
    } else {
        fovrg_relativistic_origin_series(
            input,
            &mut large_coefficients,
            &mut small_coefficients,
            energy_over_light,
        )?;
    }

    let (initial_large_component, initial_small_component) =
        fovrg_origin_components(input, large_coefficients.view(), small_coefficients.view());
    let flat_start_index = input.radial_match_index.min(input.wkb_index);
    let integrated = fovrg_outward_integrate(FovrgOutwardIntegrationInput {
        initial_large_component,
        initial_small_component,
        energy: input.energy,
        potential: input.potential,
        potential_coefficients: input.potential_coefficients,
        large_exchange: input.large_exchange,
        small_exchange: input.small_exchange,
        c3_potential: input.c3_potential,
        radii: input.radii,
        speed_of_light: input.speed_of_light,
        step: input.step,
        kappa: input.kappa,
        c3_scale: input.c3_scale,
        start_index: 0,
        last_index: flat_start_index,
        active_len: input.active_len,
    })?;
    let mut large_component = integrated.large_component;
    let mut small_component = integrated.small_component;

    for row in flat_start_index..input.last_index {
        let average_potential = fovrg_solout_average_potential(input, row)?;
        let propagated = fovrg_flat_potential_propagate(FovrgFlatPotentialInput {
            start_radius: input.radii[row],
            end_radius: input.radii[row + 1],
            large_component: large_component[row],
            small_component: small_component[row],
            energy: input.energy,
            average_potential,
            kappa: input.kappa,
        })?;
        large_component[row + 1] = propagated.large_component;
        small_component[row + 1] = propagated.small_component;
    }
    for row in input.last_index + 1..input.active_len {
        large_component[row] = Complex::new(0.0, 0.0);
        small_component[row] = Complex::new(0.0, 0.0);
    }

    for row in 0..input.active_len {
        validate_complex_result("large_component", row, large_component[row])?;
        validate_complex_result("small_component", row, small_component[row])?;
    }
    for coefficient in 0..input.coefficient_count {
        validate_complex_result(
            "large_coefficients",
            coefficient,
            large_coefficients[coefficient],
        )?;
        validate_complex_result(
            "small_coefficients",
            coefficient,
            small_coefficients[coefficient],
        )?;
    }

    Ok(FovrgOutgoingSolution {
        large_component,
        small_component,
        large_coefficients,
        small_coefficients,
        difficult_iterations: integrated.difficult_iterations,
    })
}

/// Port of FEFF `FOVRG/solin.f90`: irregular solution integrated inward.
///
/// FEFF seeds the outer region from spherical Hankel functions, propagates
/// through the flat-potential interval, then integrates the inhomogeneous
/// Dirac system inward to the first radial point. Coefficients after the first
/// origin term are zero-filled to match `solin`.
pub fn fovrg_inward_solution(
    input: FovrgInwardSolutionInput<'_>,
) -> Result<FovrgInwardSolution, FovrgError> {
    validate_inward_solution_input(&input)?;

    let ccl = input.speed_of_light + input.speed_of_light;
    let flat_start_index = input.radial_match_index.min(input.wkb_index);
    let mut derivative_energy = input.energy / input.speed_of_light;
    let mut large_component = Array1::<Complex>::zeros(input.active_len);
    let mut small_component = Array1::<Complex>::zeros(input.active_len);
    let mut large_coefficients = Array1::<Complex>::zeros(input.coefficient_count);
    let mut small_coefficients = Array1::<Complex>::zeros(input.coefficient_count);
    let mut large_derivative = [Complex::new(0.0, 0.0); FOVRG_INT_OUT_HISTORY];
    let mut small_derivative = [Complex::new(0.0, 0.0); FOVRG_INT_OUT_HISTORY];

    let match_potential = input.speed_of_light * input.potential[input.radial_match_index + 1];
    let energy_offset = input.energy - match_potential;
    let alpha_wave_offset = FEFF_FINE_STRUCTURE_ALPHA * energy_offset;
    let wave_number = (2.0 * energy_offset + alpha_wave_offset * alpha_wave_offset).sqrt();
    let large_l = if input.kappa < 0 {
        input.kappa.unsigned_abs() as usize - 1
    } else {
        input.kappa as usize
    };
    let small_l = if input.kappa < 0 {
        large_l + 1
    } else {
        large_l - 1
    };
    let max_l = large_l.max(small_l);
    let sign = if input.kappa > 0 { 1.0 } else { -1.0 };
    let alpha_wave = wave_number * FEFF_FINE_STRUCTURE_ALPHA;
    let factor = sign * alpha_wave / (1.0 + (1.0 + alpha_wave * alpha_wave).sqrt());
    let normalization_denominator = (1.0 + factor * factor).sqrt();
    validate_nonzero_complex_denominator("inward_hankel_normalization", normalization_denominator)?;
    let normalization = Complex::new(1.0, 0.0) / normalization_denominator;

    for row in input.radial_match_index..=input.last_index {
        let argument = wave_number * input.radii[row];
        let hankel =
            besjh(argument, max_l).map_err(|source| FovrgError::FlatPotentialBessel { source })?;
        large_component[row] = hankel.h[large_l] * input.radii[row] * normalization;
        small_component[row] = hankel.h[small_l] * input.radii[row] * normalization * factor;

        if let Some(history_slot) = fovrg_inward_history_slot(flat_start_index, row) {
            let (large, small) = fovrg_inward_derivatives(
                &input,
                row,
                derivative_energy,
                ccl,
                false,
                large_component[row],
                small_component[row],
            )?;
            large_derivative[history_slot] = large;
            small_derivative[history_slot] = small;
        }
    }

    for row in (flat_start_index..input.radial_match_index).rev() {
        let mut average_potential = fovrg_solin_average_potential(input, row)?;
        if input.c3_scale > 0 {
            let radius_average = (input.radii[row] + input.radii[row + 1]) / 2.0;
            derivative_energy = radius_average.powi(3)
                * (ccl + (input.energy - average_potential) / input.speed_of_light).powi(2);
            validate_nonzero_complex_denominator("solin_c3_flat_denominator", derivative_energy)?;
            average_potential += (input.c3_scale as Real) * input.speed_of_light
                / derivative_energy
                * (input.c3_potential[row] + input.c3_potential[row + 1])
                / 2.0;
        }

        let propagated = fovrg_flat_potential_propagate(FovrgFlatPotentialInput {
            start_radius: input.radii[row + 1],
            end_radius: input.radii[row],
            large_component: large_component[row + 1],
            small_component: small_component[row + 1],
            energy: input.energy,
            average_potential,
            kappa: input.kappa,
        })?;
        large_component[row] = propagated.large_component;
        small_component[row] = propagated.small_component;

        if let Some(history_slot) = fovrg_inward_history_slot(flat_start_index, row) {
            let (large, small) = fovrg_inward_derivatives(
                &input,
                row,
                derivative_energy,
                ccl,
                true,
                large_component[row],
                small_component[row],
            )?;
            large_derivative[history_slot] = large;
            small_derivative[history_slot] = small;
        }
    }

    let a1 = input.step * F77_REAL_THREE_POINT_THREE;
    let a2 = -input.step * F77_REAL_FOUR_POINT_TWO;
    let a3 = input.step * F77_REAL_SEVEN_POINT_EIGHT;
    let a4 = input.step * F77_REAL_FOURTEEN_OVER_FORTY_FIVE;
    let a5 = input.step * F77_REAL_SIXTY_FOUR_OVER_FORTY_FIVE;
    let a6 = input.step * F77_REAL_TWENTY_FOUR_OVER_FORTY_FIVE;
    let mut difficult_iterations = 0usize;

    for row in (1..=flat_start_index).rev() {
        let mut predicted_large = large_component[row + 5]
            + a1 * (large_derivative[5] + large_derivative[1])
            + a2 * (large_derivative[4] + large_derivative[2])
            + a3 * large_derivative[3];
        let mut predicted_small = small_component[row + 5]
            + a1 * (small_derivative[5] + small_derivative[1])
            + a2 * (small_derivative[4] + small_derivative[2])
            + a3 * small_derivative[3];
        let corrected_large_base = large_component[row + 3]
            + a4 * large_derivative[2]
            + a5 * (large_derivative[5] + large_derivative[3])
            + a6 * large_derivative[4];
        let corrected_small_base = small_component[row + 3]
            + a4 * small_derivative[2]
            + a5 * (small_derivative[5] + small_derivative[3])
            + a6 * small_derivative[4];

        large_derivative.copy_within(1..FOVRG_INT_OUT_HISTORY, 0);
        small_derivative.copy_within(1..FOVRG_INT_OUT_HISTORY, 0);

        let next = row - 1;
        let mut retry_count = 0usize;
        loop {
            let (large, small) = fovrg_inward_derivatives(
                &input,
                next,
                derivative_energy,
                ccl,
                true,
                predicted_large,
                predicted_small,
            )?;
            large_derivative[5] = large;
            small_derivative[5] = small;
            large_component[next] = corrected_large_base + a4 * large_derivative[5];
            small_component[next] = corrected_small_base + a4 * small_derivative[5];

            let large_failed = (FOVRG_INT_OUT_TEST * (large_component[next] - predicted_large))
                .norm()
                > large_component[next].norm();
            let small_failed = (FOVRG_INT_OUT_TEST * (small_component[next] - predicted_small))
                .norm()
                > small_component[next].norm();
            if large_failed || small_failed {
                if retry_count < 40 {
                    predicted_large = large_component[next];
                    predicted_small = small_component[next];
                    retry_count += 1;
                } else {
                    difficult_iterations += 1;
                    break;
                }
            } else {
                break;
            }
        }
    }

    for row in input.last_index + 1..input.active_len {
        large_component[row] = Complex::new(0.0, 0.0);
        small_component[row] = Complex::new(0.0, 0.0);
    }

    let origin_scale = input.radii[0].powf(-input.origin_power);
    large_coefficients[0] = large_component[0] * origin_scale;
    small_coefficients[0] = small_component[0] * origin_scale;

    for row in 0..input.active_len {
        validate_complex_result("large_component", row, large_component[row])?;
        validate_complex_result("small_component", row, small_component[row])?;
    }
    for coefficient in 0..input.coefficient_count {
        validate_complex_result(
            "large_coefficients",
            coefficient,
            large_coefficients[coefficient],
        )?;
        validate_complex_result(
            "small_coefficients",
            coefficient,
            small_coefficients[coefficient],
        )?;
    }

    Ok(FovrgInwardSolution {
        large_component,
        small_component,
        large_coefficients,
        small_coefficients,
        difficult_iterations,
    })
}

/// Port of FEFF `FOVRG/wfirdc.f90`: initial photoelectron orbital assembly.
///
/// FEFF builds the point-nucleus mesh, derives origin powers and normalization
/// factors for the bound orbitals plus the target photoelectron, constructs the
/// direct potential development, then solves the regular or irregular radial
/// Dirac equation through `solout` or `solin`.
pub fn fovrg_initial_photoelectron(
    input: FovrgInitialPhotoelectronInput<'_>,
) -> Result<FovrgInitialPhotoelectron, FovrgError> {
    validate_initial_photoelectron_input(&input)?;

    let nuclear_potential = fovrg_nuclear_potential(FovrgNuclearPotentialInput {
        nuclear_charge: input.nuclear_charge,
        step: input.step,
        first_radius_times_charge: input.nuclear_charge * (-8.8_f64).exp(),
        radial_count: input.active_len,
        coefficient_count: FOVRG_ORIGIN_COEFFICIENTS,
    })?;
    let first_radius = nuclear_potential.radii[0];
    let target_index = input.orbital_count - 1;
    let relativistic_shift = (input.nuclear_charge / input.speed_of_light).powi(2);

    let mut origin_powers = Array1::<Real>::zeros(input.orbital_count);
    let mut normalization = Array1::<Real>::zeros(input.orbital_count);
    for orbital in 0..input.orbital_count {
        let kappa = input.kappa[orbital];
        let mut radicand = (kappa * kappa) as Real - relativistic_shift;
        if orbital == target_index {
            radicand += (kappa + 1) as Real * input.c3_scale as Real;
        }
        if radicand < 0.0 {
            return Err(FovrgError::NegativeRadicand {
                name: "origin_power",
                row: orbital,
                value: radicand,
            });
        }

        origin_powers[orbital] = radicand.sqrt();
        normalization[orbital] =
            first_radius.powf(origin_powers[orbital] - kappa.unsigned_abs() as Real);
    }
    if input.irregular {
        origin_powers[target_index] = -origin_powers[target_index];
        validate_nonzero_finite("target_normalization", normalization[target_index])?;
        normalization[target_index] = 1.0 / normalization[target_index];
    }

    let direct_potential = Array1::from_iter((0..input.active_len).map(|row| {
        if row < input.radial_match_index {
            input.exchange_correlation_potential[row] / input.speed_of_light
        } else {
            input.exchange_correlation_potential[input.radial_match_index + 1]
                / input.speed_of_light
        }
    }));
    let zero_exchange = Array1::<Complex>::zeros(input.active_len);
    let zero_exchange_coefficients = Array1::<Complex>::zeros(FOVRG_ORIGIN_COEFFICIENTS);
    let potential_development = fovrg_potential_development(FovrgPotentialDevelopmentInput {
        nuclear_coefficients: nuclear_potential.development_coefficients.view(),
        large_coefficients: input.bound_large_coefficients,
        small_coefficients: input.bound_small_coefficients,
        electron_counts: input.electron_counts,
        kappa: input.kappa,
        normalization: normalization.view(),
        radii: nuclear_potential.radii.view(),
        speed_of_light: input.speed_of_light,
        coefficient_count: input.coefficient_count,
        orbital_count: input.orbital_count,
    })?;
    let mut potential_coefficients = potential_development.potential_coefficients;
    let nucleus_row = nuclear_potential.nucleus_index - 1;
    potential_coefficients[1] += (input.exchange_correlation_potential[nucleus_row]
        - nuclear_potential.potential[nucleus_row])
        / input.speed_of_light;

    let retained_len = fovrg_photoelectron_retained_len(input.step, input.active_len)?;
    let mut orbital_lengths = input.orbital_lengths.to_owned();
    orbital_lengths[target_index] = orbital_lengths[target_index].min(retained_len);
    let target_last_index = orbital_lengths[target_index] - 1;
    let (initial_large_coefficient, initial_small_coefficient) = if input.irregular {
        (
            input.initial_large_coefficient,
            input.initial_small_coefficient,
        )
    } else {
        fovrg_regular_initial_coefficients(
            input.nuclear_charge,
            input.speed_of_light,
            input.kappa[target_index],
            origin_powers[target_index],
        )?
    };

    let solution = if input.irregular {
        let solution = fovrg_inward_solution(FovrgInwardSolutionInput {
            initial_large_coefficient,
            initial_small_coefficient,
            energy: input.energy,
            origin_power: origin_powers[target_index],
            kappa: input.kappa[target_index],
            muffin_tin_radius: input.muffin_tin_radius,
            potential: direct_potential.view(),
            large_exchange: zero_exchange.view(),
            small_exchange: zero_exchange.view(),
            c3_potential: input.c3_potential,
            radii: nuclear_potential.radii.view(),
            speed_of_light: input.speed_of_light,
            step: input.step,
            c3_scale: input.c3_scale,
            radial_match_index: input.radial_match_index,
            last_index: target_last_index,
            wkb_index: input.wkb_index,
            coefficient_count: input.coefficient_count,
            active_len: input.active_len,
        })?;
        let mut large_coefficients = Array1::<Complex>::zeros(FOVRG_ORIGIN_COEFFICIENTS);
        let mut small_coefficients = Array1::<Complex>::zeros(FOVRG_ORIGIN_COEFFICIENTS);
        for coefficient in 0..solution.large_coefficients.len() {
            large_coefficients[coefficient] = solution.large_coefficients[coefficient];
            small_coefficients[coefficient] = solution.small_coefficients[coefficient];
        }
        FovrgInitialPhotoelectronRadialSolution {
            large_component: solution.large_component,
            small_component: solution.small_component,
            large_coefficients,
            small_coefficients,
            difficult_iterations: solution.difficult_iterations,
        }
    } else {
        let solution = fovrg_outgoing_solution(FovrgOutgoingSolutionInput {
            initial_large_coefficient,
            initial_small_coefficient,
            energy: input.energy,
            origin_power: origin_powers[target_index],
            kappa: input.kappa[target_index],
            muffin_tin_radius: input.muffin_tin_radius,
            potential: direct_potential.view(),
            potential_coefficients: potential_coefficients.view(),
            large_exchange: zero_exchange.view(),
            small_exchange: zero_exchange.view(),
            large_exchange_coefficients: zero_exchange_coefficients.view(),
            small_exchange_coefficients: zero_exchange_coefficients.view(),
            c3_potential: input.c3_potential,
            radii: nuclear_potential.radii.view(),
            speed_of_light: input.speed_of_light,
            step: input.step,
            c3_scale: input.c3_scale,
            radial_match_index: input.radial_match_index,
            last_index: target_last_index,
            wkb_index: input.wkb_index,
            coefficient_count: input.coefficient_count,
            active_len: input.active_len,
        })?;
        FovrgInitialPhotoelectronRadialSolution {
            large_component: solution.large_component,
            small_component: solution.small_component,
            large_coefficients: solution.large_coefficients,
            small_coefficients: solution.small_coefficients,
            difficult_iterations: solution.difficult_iterations,
        }
    };

    for row in 0..input.active_len {
        validate_complex_result(
            "photoelectron_large_component",
            row,
            solution.large_component[row],
        )?;
        validate_complex_result(
            "photoelectron_small_component",
            row,
            solution.small_component[row],
        )?;
    }
    for coefficient in 0..FOVRG_ORIGIN_COEFFICIENTS {
        validate_complex_result(
            "photoelectron_large_coefficients",
            coefficient,
            solution.large_coefficients[coefficient],
        )?;
        validate_complex_result(
            "photoelectron_small_coefficients",
            coefficient,
            solution.small_coefficients[coefficient],
        )?;
    }

    Ok(FovrgInitialPhotoelectron {
        large_component: solution.large_component,
        small_component: solution.small_component,
        large_coefficients: solution.large_coefficients,
        small_coefficients: solution.small_coefficients,
        origin_powers,
        normalization,
        orbital_lengths,
        nuclear_potential,
        direct_potential,
        potential_coefficients,
        retained_len,
        target_last_index,
        difficult_iterations: solution.difficult_iterations,
    })
}

/// Port of the orbital bookkeeping in FEFF `FOVRG/inmuac.f90`.
///
/// FEFF normally calls `getorb` before this point to obtain occupations and
/// quantum numbers. This helper covers the deterministic `inmuac` work after
/// that: find each bound orbital's last tabulated row, flag open shells, count
/// target-kappa matches, subtract valence occupations for exchange, and append
/// the photoelectron kappa slot.
pub fn fovrg_orbital_setup(
    input: FovrgOrbitalSetupInput<'_>,
) -> Result<FovrgOrbitalSetup, FovrgError> {
    validate_orbital_setup_input(&input)?;

    let mut orbital_lengths = Array1::<usize>::zeros(input.bound_orbital_count + 1);
    let mut kappa = Array1::<i32>::zeros(input.bound_orbital_count + 1);
    let mut core_counts = Array1::<Real>::zeros(input.bound_orbital_count);
    let mut open_shell = Array1::<bool>::from_elem(input.bound_orbital_count, false);
    let mut matching_kappa_count = 0usize;

    for orbital in 0..input.bound_orbital_count {
        let length = (0..input.active_len)
            .rev()
            .find(|&row| {
                input.bound_large_components[(row, orbital)].abs() >= FOVRG_BOUND_ORBITAL_THRESHOLD
                    || input.bound_small_components[(row, orbital)].abs()
                        >= FOVRG_BOUND_ORBITAL_THRESHOLD
            })
            .map_or(0, |row| row + 1);
        validate_count_at_least("orbital_length", length, 1)?;
        orbital_lengths[orbital] = length;

        kappa[orbital] = input.kappa[orbital];
        core_counts[orbital] = input.electron_counts[orbital] - input.valence_counts[orbital];
        validate_real_result("core_counts", orbital, core_counts[orbital])?;
        open_shell[orbital] =
            input.electron_counts[orbital] < 2.0 * input.kappa[orbital].unsigned_abs() as Real;
        if input.target_kappa == input.kappa[orbital] {
            matching_kappa_count += 1;
        }
    }
    kappa[input.bound_orbital_count] = input.target_kappa;

    Ok(FovrgOrbitalSetup {
        orbital_lengths,
        kappa,
        core_counts,
        open_shell,
        matching_kappa_count,
    })
}

/// Port of FEFF `FOVRG/dfovrg.f90`: Dirac photoelectron radial solver.
///
/// The solver first flattens the interstitial potentials, computes FEFF's WKB
/// switch point, builds the initial photoelectron orbital through
/// [`fovrg_initial_photoelectron`], and optionally runs the nonlocal exchange
/// update loop before returning the muffin-tin values.
pub fn fovrg_dirac_solver(
    input: FovrgDiracSolverInput<'_>,
) -> Result<FovrgDiracSolution, FovrgError> {
    let active_len = fovrg_dirac_active_len(input.step, input.radii.len())?;
    validate_dirac_solver_input(&input, active_len)?;

    let coefficient_count = if input.irregular { 2 } else { 3 };
    let target_len = input
        .target_last_index
        .checked_add(1)
        .ok_or(FovrgError::CountTooLarge {
            name: "target_last_index",
            actual: input.target_last_index,
            maximum: usize::MAX - 1,
        })?;
    let mut wkb_index = fovrg_dirac_wkb_index(input.energy, input.step, target_len, active_len)?;

    let mut exchange_correlation_potential = input.exchange_correlation_potential.to_owned();
    let mut valence_exchange_correlation_potential =
        input.valence_exchange_correlation_potential.to_owned();
    let flat_potential = exchange_correlation_potential[input.radial_match_index + 1];
    for row in input.radial_match_index + 1..exchange_correlation_potential.len() {
        exchange_correlation_potential[row] = flat_potential;
        valence_exchange_correlation_potential[row] = flat_potential;
    }

    let orbital_setup = fovrg_orbital_setup(FovrgOrbitalSetupInput {
        bound_large_components: input.bound_large_components,
        bound_small_components: input.bound_small_components,
        electron_counts: input.electron_counts,
        valence_counts: input.valence_counts,
        kappa: input.kappa,
        target_kappa: input.target_kappa,
        active_len,
        bound_orbital_count: input.bound_orbital_count,
    })?;
    let mut orbital_lengths = orbital_setup.orbital_lengths.clone();
    orbital_lengths[input.bound_orbital_count] = target_len;
    if wkb_index + 1 >= target_len.saturating_sub(1) {
        wkb_index = active_len - 1;
    }

    let c3_potential = fovrg_dirac_c3_potential(&input, exchange_correlation_potential.view())?;

    let initial = fovrg_initial_photoelectron(FovrgInitialPhotoelectronInput {
        energy: input.energy,
        bound_large_coefficients: input.bound_large_coefficients,
        bound_small_coefficients: input.bound_small_coefficients,
        electron_counts: input.electron_counts,
        kappa: orbital_setup.kappa.view(),
        orbital_lengths: orbital_lengths.view(),
        exchange_correlation_potential: exchange_correlation_potential.view(),
        c3_potential: c3_potential.view(),
        initial_large_coefficient: input.muffin_tin_large_component,
        initial_small_coefficient: input.muffin_tin_small_component,
        nuclear_charge: input.atomic_number,
        muffin_tin_radius: input.muffin_tin_radius,
        step: input.step,
        speed_of_light: FEFF_WFIRDC_SPEED_OF_LIGHT,
        c3_scale: input.c3_scale,
        irregular: input.irregular,
        radial_match_index: input.radial_match_index,
        wkb_index,
        coefficient_count,
        orbital_count: input.bound_orbital_count + 1,
        active_len,
    })?;

    let target_index = input.bound_orbital_count;
    let mut large_component = initial.large_component.clone();
    let mut small_component = initial.small_component.clone();
    let mut large_coefficients =
        fovrg_zero_extended_coefficients(initial.large_coefficients.view());
    let mut small_coefficients =
        fovrg_zero_extended_coefficients(initial.small_coefficients.view());
    let mut direct_potential = initial.direct_potential.clone();
    let mut potential_coefficients = initial.potential_coefficients.clone();
    let mut large_exchange = Array1::<Complex>::zeros(active_len);
    let mut small_exchange = Array1::<Complex>::zeros(active_len);
    let mut large_exchange_coefficients = Array1::<Complex>::zeros(coefficient_count);
    let mut small_exchange_coefficients = Array1::<Complex>::zeros(coefficient_count);
    let mut difficult_iterations = initial.difficult_iterations;
    let mut iteration_count = 0usize;

    if input.exchange_cycle_count != 0 {
        potential_coefficients[1] += (valence_exchange_correlation_potential[0]
            - exchange_correlation_potential[0])
            / FEFF_WFIRDC_SPEED_OF_LIGHT;
        for row in 0..=wkb_index {
            direct_potential[row] =
                valence_exchange_correlation_potential[row] / FEFF_WFIRDC_SPEED_OF_LIGHT;
        }

        let angular_coefficients = fovrg_angular_coefficients(FovrgAngularCoefficientsInput {
            electron_counts: orbital_setup.core_counts.view(),
            valence_counts: input.valence_counts,
            kappa: input.kappa,
            target_kappa: input.target_kappa,
            bound_orbital_count: input.bound_orbital_count,
        })?;
        let radial_output_count = (input.radial_match_index + 1).min(wkb_index + 1);

        for _ in 0..=input.exchange_cycle_count {
            iteration_count += 1;
            let exchange = fovrg_exchange_potential(FovrgExchangePotentialInput {
                target_large_component: large_component.view(),
                target_small_component: small_component.view(),
                target_large_coefficients: large_coefficients.view(),
                target_small_coefficients: small_coefficients.view(),
                bound_large_components: input.bound_large_components,
                bound_small_components: input.bound_small_components,
                bound_large_coefficients: input.bound_large_coefficients,
                bound_small_coefficients: input.bound_small_coefficients,
                angular_coefficients: angular_coefficients.view(),
                orbital_powers: initial.origin_powers.view(),
                kappa: input.kappa,
                orbital_lengths: initial.orbital_lengths.view(),
                normalization: initial.normalization.view(),
                radii: initial.nuclear_potential.radii.view(),
                target_power: initial.origin_powers[target_index],
                target_kappa: input.target_kappa,
                target_normalization: initial.normalization[target_index],
                speed_of_light: FEFF_WFIRDC_SPEED_OF_LIGHT,
                step: input.step,
                coefficient_count,
                source_len: initial.retained_len,
                active_len,
                radial_output_count,
                bound_orbital_count: input.bound_orbital_count,
            })?;
            large_exchange = exchange.large_potential;
            small_exchange = exchange.small_potential;
            large_exchange_coefficients = exchange.large_coefficients;
            small_exchange_coefficients = exchange.small_coefficients;

            if input.irregular {
                let solution = fovrg_inward_solution(FovrgInwardSolutionInput {
                    initial_large_coefficient: input.muffin_tin_large_component,
                    initial_small_coefficient: input.muffin_tin_small_component,
                    energy: input.energy,
                    origin_power: initial.origin_powers[target_index],
                    kappa: input.target_kappa,
                    muffin_tin_radius: input.muffin_tin_radius,
                    potential: direct_potential.view(),
                    large_exchange: large_exchange.view(),
                    small_exchange: small_exchange.view(),
                    c3_potential: c3_potential.view(),
                    radii: initial.nuclear_potential.radii.view(),
                    speed_of_light: FEFF_WFIRDC_SPEED_OF_LIGHT,
                    step: input.step,
                    c3_scale: input.c3_scale,
                    radial_match_index: input.radial_match_index,
                    last_index: initial.target_last_index,
                    wkb_index,
                    coefficient_count,
                    active_len,
                })?;
                difficult_iterations += solution.difficult_iterations;
                large_component = solution.large_component;
                small_component = solution.small_component;
                large_coefficients =
                    fovrg_zero_extended_coefficients(solution.large_coefficients.view());
                small_coefficients =
                    fovrg_zero_extended_coefficients(solution.small_coefficients.view());
            } else {
                let solution = fovrg_outgoing_solution(FovrgOutgoingSolutionInput {
                    initial_large_coefficient: large_coefficients[0],
                    initial_small_coefficient: small_coefficients[0],
                    energy: input.energy,
                    origin_power: initial.origin_powers[target_index],
                    kappa: input.target_kappa,
                    muffin_tin_radius: input.muffin_tin_radius,
                    potential: direct_potential.view(),
                    potential_coefficients: potential_coefficients.view(),
                    large_exchange: large_exchange.view(),
                    small_exchange: small_exchange.view(),
                    large_exchange_coefficients: large_exchange_coefficients.view(),
                    small_exchange_coefficients: small_exchange_coefficients.view(),
                    c3_potential: c3_potential.view(),
                    radii: initial.nuclear_potential.radii.view(),
                    speed_of_light: FEFF_WFIRDC_SPEED_OF_LIGHT,
                    step: input.step,
                    c3_scale: input.c3_scale,
                    radial_match_index: input.radial_match_index,
                    last_index: initial.target_last_index,
                    wkb_index,
                    coefficient_count,
                    active_len,
                })?;
                difficult_iterations += solution.difficult_iterations;
                large_component = solution.large_component;
                small_component = solution.small_component;
                large_coefficients =
                    fovrg_zero_extended_coefficients(solution.large_coefficients.view());
                small_coefficients =
                    fovrg_zero_extended_coefficients(solution.small_coefficients.view());
            }
        }
    }

    let (muffin_tin_large_component, muffin_tin_small_component) = if input.irregular {
        (
            input.muffin_tin_large_component,
            input.muffin_tin_small_component,
        )
    } else {
        let propagated = fovrg_flat_potential_propagate(FovrgFlatPotentialInput {
            start_radius: input.radii[input.radial_match_index],
            end_radius: input.muffin_tin_radius,
            large_component: large_component[input.radial_match_index],
            small_component: small_component[input.radial_match_index],
            energy: input.energy,
            average_potential: exchange_correlation_potential[input.radial_match_index + 1],
            kappa: input.target_kappa,
        })?;
        (propagated.large_component, propagated.small_component)
    };

    for row in 0..active_len {
        validate_complex_result("dirac_large_component", row, large_component[row])?;
        validate_complex_result("dirac_small_component", row, small_component[row])?;
        validate_complex_result("dirac_direct_potential", row, direct_potential[row])?;
        validate_complex_result("dirac_c3_potential", row, c3_potential[row])?;
    }
    for coefficient in 0..FOVRG_ORIGIN_COEFFICIENTS {
        validate_complex_result(
            "dirac_large_coefficients",
            coefficient,
            large_coefficients[coefficient],
        )?;
        validate_complex_result(
            "dirac_small_coefficients",
            coefficient,
            small_coefficients[coefficient],
        )?;
    }

    Ok(FovrgDiracSolution {
        large_component,
        small_component,
        large_coefficients,
        small_coefficients,
        muffin_tin_large_component,
        muffin_tin_small_component,
        exchange_correlation_potential,
        valence_exchange_correlation_potential,
        direct_potential,
        potential_coefficients,
        large_exchange,
        small_exchange,
        large_exchange_coefficients,
        small_exchange_coefficients,
        c3_potential,
        origin_powers: initial.origin_powers,
        normalization: initial.normalization,
        orbital_lengths: initial.orbital_lengths,
        active_len,
        retained_len: initial.retained_len,
        wkb_index,
        target_last_index: initial.target_last_index,
        iteration_count,
        difficult_iterations,
    })
}

/// Port of `FOVRG/potex.f90`: exchange-potential accumulation.
///
/// FEFF loops over bound orbitals and allowed multipoles, obtains the `yk`
/// exchange kernel from `yzkrdc`, accumulates the radial exchange potentials
/// `eg/ep`, updates their origin development coefficients `ceg/cep`, and
/// finally divides retained rows and coefficients by `cl`.
pub fn fovrg_exchange_potential(
    input: FovrgExchangePotentialInput<'_>,
) -> Result<FovrgExchangePotential, FovrgError> {
    validate_count_at_least("active_len", input.active_len, 2)?;
    validate_count_at_least("source_len", input.source_len, 1)?;
    validate_count_at_least("coefficient_count", input.coefficient_count, 1)?;
    validate_active_len(
        "target_large_component",
        input.active_len,
        input.target_large_component.len(),
    )?;
    validate_active_len(
        "target_small_component",
        input.active_len,
        input.target_small_component.len(),
    )?;
    validate_active_len("radii", input.active_len, input.radii.len())?;
    validate_active_len(
        "target_large_coefficients",
        input.coefficient_count,
        input.target_large_coefficients.len(),
    )?;
    validate_active_len(
        "target_small_coefficients",
        input.coefficient_count,
        input.target_small_coefficients.len(),
    )?;
    validate_matrix_rows(
        "bound_large_components",
        input.active_len,
        input.bound_large_components.shape()[0],
    )?;
    validate_matrix_rows(
        "bound_small_components",
        input.active_len,
        input.bound_small_components.shape()[0],
    )?;
    validate_matrix_cols(
        "bound_large_components",
        input.bound_orbital_count,
        input.bound_large_components.shape()[1],
    )?;
    validate_matrix_cols(
        "bound_small_components",
        input.bound_orbital_count,
        input.bound_small_components.shape()[1],
    )?;
    validate_matrix_rows(
        "bound_large_coefficients",
        input.coefficient_count,
        input.bound_large_coefficients.shape()[0],
    )?;
    validate_matrix_rows(
        "bound_small_coefficients",
        input.coefficient_count,
        input.bound_small_coefficients.shape()[0],
    )?;
    validate_matrix_cols(
        "bound_large_coefficients",
        input.bound_orbital_count,
        input.bound_large_coefficients.shape()[1],
    )?;
    validate_matrix_cols(
        "bound_small_coefficients",
        input.bound_orbital_count,
        input.bound_small_coefficients.shape()[1],
    )?;
    validate_matrix_rows(
        "angular_coefficients",
        input.bound_orbital_count,
        input.angular_coefficients.shape()[0],
    )?;
    validate_active_len(
        "orbital_powers",
        input.bound_orbital_count,
        input.orbital_powers.len(),
    )?;
    validate_active_len("kappa", input.bound_orbital_count, input.kappa.len())?;
    validate_active_len(
        "orbital_lengths",
        input.bound_orbital_count,
        input.orbital_lengths.len(),
    )?;
    validate_active_len(
        "normalization",
        input.bound_orbital_count,
        input.normalization.len(),
    )?;
    validate_active_len(
        "radial_output_count",
        input.radial_output_count,
        input.active_len,
    )?;
    validate_nonzero_kappa("target_kappa", 0, input.target_kappa)?;
    validate_finite("target_power", input.target_power)?;
    validate_nonzero_finite("target_normalization", input.target_normalization)?;
    validate_nonzero_finite("speed_of_light", input.speed_of_light)?;
    validate_positive_finite("step", input.step)?;

    for row in 0..input.active_len {
        validate_complex_input(
            "target_large_component",
            row,
            input.target_large_component[row],
        )?;
        validate_complex_input(
            "target_small_component",
            row,
            input.target_small_component[row],
        )?;
        validate_radius(row, input.radii[row])?;
        for orbital in 0..input.bound_orbital_count {
            validate_real_input(
                "bound_large_components",
                row,
                input.bound_large_components[(row, orbital)],
            )?;
            validate_real_input(
                "bound_small_components",
                row,
                input.bound_small_components[(row, orbital)],
            )?;
        }
    }
    for coefficient in 0..input.coefficient_count {
        validate_complex_input(
            "target_large_coefficients",
            coefficient,
            input.target_large_coefficients[coefficient],
        )?;
        validate_complex_input(
            "target_small_coefficients",
            coefficient,
            input.target_small_coefficients[coefficient],
        )?;
        for orbital in 0..input.bound_orbital_count {
            validate_real_input(
                "bound_large_coefficients",
                coefficient,
                input.bound_large_coefficients[(coefficient, orbital)],
            )?;
            validate_real_input(
                "bound_small_coefficients",
                coefficient,
                input.bound_small_coefficients[(coefficient, orbital)],
            )?;
        }
    }
    for orbital in 0..input.bound_orbital_count {
        validate_nonzero_kappa("kappa", orbital, input.kappa[orbital])?;
        validate_real_input("orbital_powers", orbital, input.orbital_powers[orbital])?;
        validate_real_input("normalization", orbital, input.normalization[orbital])?;
        validate_count_at_least("orbital_length", input.orbital_lengths[orbital], 1)?;
        for index in 0..input.angular_coefficients.shape()[1] {
            validate_real_input(
                "angular_coefficients",
                orbital,
                input.angular_coefficients[(orbital, index)],
            )?;
        }
    }

    let target_j = target_j_value(input.target_kappa);
    let mut large_potential = Array1::<Complex>::zeros(input.active_len);
    let mut small_potential = Array1::<Complex>::zeros(input.active_len);
    let mut large_coefficients = Array1::<Complex>::zeros(input.coefficient_count);
    let mut small_coefficients = Array1::<Complex>::zeros(input.coefficient_count);

    for orbital in 0..input.bound_orbital_count {
        let bound_j = target_j_value(input.kappa[orbital]);
        let max_multipole = (bound_j + target_j) / 2;
        let mut multipole = bound_j.abs_diff(max_multipole);
        if (input.kappa[orbital] < 0) != (input.target_kappa < 0) {
            multipole += 1;
        }
        let min_multipole = multipole;

        while multipole <= max_multipole {
            let angular_index = (multipole - min_multipole) / 2;
            validate_matrix_cols(
                "angular_coefficients",
                angular_index + 1,
                input.angular_coefficients.shape()[1],
            )?;
            let angular_coefficient = input.angular_coefficients[(orbital, angular_index)];
            if angular_coefficient != 0.0 {
                let transform = fovrg_yk_zk_exchange(FovrgYkZkExchangeInput {
                    large_component: input.bound_large_components.column(orbital),
                    small_component: input.bound_small_components.column(orbital),
                    large_coefficients: input.bound_large_coefficients.column(orbital),
                    small_coefficients: input.bound_small_coefficients.column(orbital),
                    partner_large_component: input.target_large_component,
                    partner_small_component: input.target_small_component,
                    partner_large_coefficients: input.target_large_coefficients,
                    partner_small_coefficients: input.target_small_coefficients,
                    radii: input.radii,
                    orbital_power: input.orbital_powers[orbital],
                    partner_power: input.target_power,
                    step: input.step,
                    angular_momentum: multipole,
                    coefficient_count: input.coefficient_count,
                    orbital_len: input.orbital_lengths[orbital],
                    source_len: input.source_len,
                    active_len: input.active_len,
                })?;

                for row in 0..input.active_len {
                    large_potential[row] += angular_coefficient
                        * transform.yk[row]
                        * input.bound_large_components[(row, orbital)];
                    small_potential[row] += angular_coefficient
                        * transform.yk[row]
                        * input.bound_small_components[(row, orbital)];
                }

                if let Some(coefficient_start) = exchange_coefficient_start(
                    multipole,
                    input.kappa[orbital],
                    input.target_kappa,
                    input.target_power,
                )
                .filter(|&start| start <= input.coefficient_count)
                {
                    for coefficient in coefficient_start..=input.coefficient_count {
                        let target_row = coefficient - 1;
                        let bound_row = coefficient - coefficient_start;
                        let scale = angular_coefficient
                            * transform.origin_constant
                            * input.normalization[orbital]
                            / input.target_normalization;
                        large_coefficients[target_row] +=
                            input.bound_large_coefficients[(bound_row, orbital)] * scale;
                        small_coefficients[target_row] +=
                            input.bound_small_coefficients[(bound_row, orbital)] * scale;
                    }
                }

                let product_start = 2 * input.kappa[orbital].unsigned_abs() as usize + 1;
                if product_start <= input.coefficient_count {
                    let scale = angular_coefficient * input.normalization[orbital].powi(2);
                    for coefficient in product_start..=input.coefficient_count {
                        let product_count = coefficient + 1 - product_start;
                        large_coefficients[coefficient - 1] -= scale
                            * complex_real_product_coefficient(
                                transform.yk_coefficients.view(),
                                input.bound_large_coefficients.column(orbital),
                                product_count,
                            );
                        small_coefficients[coefficient - 1] -= scale
                            * complex_real_product_coefficient(
                                transform.yk_coefficients.view(),
                                input.bound_small_coefficients.column(orbital),
                                product_count,
                            );
                    }
                }
            }
            multipole += 2;
        }
    }

    for coefficient in 0..input.coefficient_count {
        large_coefficients[coefficient] /= input.speed_of_light;
        small_coefficients[coefficient] /= input.speed_of_light;
        validate_complex_result(
            "large_coefficients",
            coefficient,
            large_coefficients[coefficient],
        )?;
        validate_complex_result(
            "small_coefficients",
            coefficient,
            small_coefficients[coefficient],
        )?;
    }
    for row in 0..input.active_len {
        if row < input.radial_output_count {
            large_potential[row] /= input.speed_of_light;
            small_potential[row] /= input.speed_of_light;
        } else {
            large_potential[row] = Complex::new(0.0, 0.0);
            small_potential[row] = Complex::new(0.0, 0.0);
        }
        validate_complex_result("large_potential", row, large_potential[row])?;
        validate_complex_result("small_potential", row, small_potential[row])?;
    }

    Ok(FovrgExchangePotential {
        large_potential,
        small_potential,
        large_coefficients,
        small_coefficients,
    })
}

/// Port of `FOVRG/nucdec.f90`: point-nucleus radial grid and potential.
///
/// FEFF10 currently resets the nuclear mass to zero inside `nucdec`, so the
/// active branch is the point-nucleus Coulomb potential:
/// `dr(i) = dr1 / dz * exp(hx * (i - 1))`, `dv(i) = -dz / dr(i)`, and
/// `av(1) = -dz` with all remaining development coefficients zero.
pub fn fovrg_nuclear_potential(
    input: FovrgNuclearPotentialInput,
) -> Result<FovrgNuclearPotential, FovrgError> {
    validate_positive_finite("nuclear_charge", input.nuclear_charge)?;
    validate_positive_finite("step", input.step)?;
    validate_positive_finite("first_radius_times_charge", input.first_radius_times_charge)?;
    validate_count_at_least("radial_count", input.radial_count, 1)?;
    validate_count_at_least("coefficient_count", input.coefficient_count, 5)?;

    let first_radius = input.first_radius_times_charge / input.nuclear_charge;
    let mut radii = Array1::<Real>::zeros(input.radial_count);
    let mut potential = Array1::<Real>::zeros(input.radial_count);
    for row in 0..input.radial_count {
        radii[row] = first_radius * (input.step * row as Real).exp();
        validate_radius(row, radii[row])?;

        potential[row] = -input.nuclear_charge / radii[row];
        validate_real_result("nuclear_potential", row, potential[row])?;
    }

    let mut development_coefficients = Array1::<Real>::zeros(input.coefficient_count);
    development_coefficients[0] = -input.nuclear_charge;
    validate_real_result("development_coefficients", 0, development_coefficients[0])?;

    Ok(FovrgNuclearPotential {
        development_coefficients,
        radii,
        potential,
        nucleus_index: 1,
        first_radius_times_charge: input.first_radius_times_charge,
    })
}

/// Port of `FOVRG/potdvp.f90`: potential development coefficients.
///
/// FEFF accumulates bound-orbital density development coefficients from
/// occupied large/small radial polynomials, integrates those coefficients into
/// a local potential expansion, adds the nuclear development, and divides the
/// resulting `av` coefficients by `cl`.
pub fn fovrg_potential_development(
    input: FovrgPotentialDevelopmentInput<'_>,
) -> Result<FovrgPotentialDevelopment, FovrgError> {
    validate_count_at_least("coefficient_count", input.coefficient_count, 1)?;
    validate_count_at_least("orbital_count", input.orbital_count, 1)?;
    validate_count_at_least("nuclear_coefficients", input.nuclear_coefficients.len(), 2)?;
    validate_count_at_least("radii", input.radii.len(), 1)?;
    validate_active_len(
        "nuclear_coefficients",
        input.coefficient_count,
        input.nuclear_coefficients.len(),
    )?;
    validate_matrix_rows(
        "large_coefficients",
        input.coefficient_count,
        input.large_coefficients.shape()[0],
    )?;
    validate_matrix_rows(
        "small_coefficients",
        input.coefficient_count,
        input.small_coefficients.shape()[0],
    )?;
    let bound_orbitals = input.orbital_count - 1;
    validate_matrix_cols(
        "large_coefficients",
        bound_orbitals,
        input.large_coefficients.shape()[1],
    )?;
    validate_matrix_cols(
        "small_coefficients",
        bound_orbitals,
        input.small_coefficients.shape()[1],
    )?;
    validate_active_len(
        "electron_counts",
        bound_orbitals,
        input.electron_counts.len(),
    )?;
    validate_active_len("kappa", bound_orbitals, input.kappa.len())?;
    validate_active_len("normalization", bound_orbitals, input.normalization.len())?;
    validate_nonzero_finite("speed_of_light", input.speed_of_light)?;
    validate_radius(0, input.radii[0])?;
    if input.coefficient_count > i32::MAX as usize - 1 {
        return Err(FovrgError::CountTooLarge {
            name: "coefficient_count",
            actual: input.coefficient_count,
            maximum: i32::MAX as usize - 1,
        });
    }

    for coefficient in 0..input.nuclear_coefficients.len() {
        validate_real_input(
            "nuclear_coefficients",
            coefficient,
            input.nuclear_coefficients[coefficient],
        )?;
    }
    for orbital in 0..bound_orbitals {
        validate_real_input("electron_counts", orbital, input.electron_counts[orbital])?;
        validate_real_input("normalization", orbital, input.normalization[orbital])?;
        validate_nonzero_kappa("kappa", orbital, input.kappa[orbital])?;
        for coefficient in 0..input.coefficient_count {
            validate_real_input(
                "large_coefficients",
                coefficient,
                input.large_coefficients[(coefficient, orbital)],
            )?;
            validate_real_input(
                "small_coefficients",
                coefficient,
                input.small_coefficients[(coefficient, orbital)],
            )?;
        }
    }

    let mut density_coefficients = Array1::<Real>::zeros(input.coefficient_count);
    for orbital in 0..bound_orbitals {
        let kappa_abs = input.kappa[orbital].unsigned_abs() as usize;
        let leading_power = kappa_abs.saturating_mul(2);
        let product_count = input.coefficient_count + 2;
        if leading_power >= product_count {
            continue;
        }
        let max_product_order = product_count - leading_power;
        for product_order in 1..=max_product_order {
            let density_row = leading_power - 2 + product_order;
            density_coefficients[density_row - 1] += input.electron_counts[orbital]
                * (real_product_coefficient(
                    input.large_coefficients.column(orbital),
                    input.large_coefficients.column(orbital),
                    product_order,
                ) + real_product_coefficient(
                    input.small_coefficients.column(orbital),
                    input.small_coefficients.column(orbital),
                    product_order,
                ))
                * input.normalization[orbital].powi(2);
        }
    }

    let mut origin_correction = 0.0;
    for coefficient in 1..=input.coefficient_count {
        let row = coefficient - 1;
        density_coefficients[row] /= (coefficient + 2) as Real * (coefficient + 1) as Real;
        origin_correction +=
            density_coefficients[row] * input.radii[0].powi(coefficient as i32 + 1);
    }

    let mut potential_coefficients = Array1::from_iter(
        input
            .nuclear_coefficients
            .iter()
            .copied()
            .map(|value| Complex::new(value, 0.0)),
    );
    for coefficient in 1..=input.coefficient_count {
        let potential_row = coefficient + 3;
        if potential_row <= input.coefficient_count {
            potential_coefficients[potential_row - 1] -= density_coefficients[coefficient - 1];
        }
    }
    potential_coefficients[1] += origin_correction;
    for row in 0..potential_coefficients.len() {
        potential_coefficients[row] /= input.speed_of_light;
        validate_complex_result("potential_coefficients", row, potential_coefficients[row])?;
    }
    for row in 0..density_coefficients.len() {
        validate_real_result("density_coefficients", row, density_coefficients[row])?;
    }
    validate_real_result("origin_correction", 0, origin_correction)?;

    Ok(FovrgPotentialDevelopment {
        potential_coefficients,
        density_coefficients,
        origin_correction,
    })
}

#[cfg(test)]
mod tests;
