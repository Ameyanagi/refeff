//! SCREEN radial-solution setup helpers.

use crate::Complex;

use super::constants::{SCREEN_ALPHA_INVERSE, SCREEN_FINE_STRUCTURE_ALPHA};
use super::types::*;
use super::validation::{
    complex32_result, validate_count_at_least, validate_finite_complex_input, validate_positive,
    validate_result_finite_complex,
};

/// Port the per-energy setup shared by `screensub.f90` and `chi_crpa.f90`.
///
/// For each contour point, FEFF computes `p2 = em(ie) - eref`,
/// `ck = sqrt(2*p2 + (p2*alphfs)**2)`, converts `ck` to the single-precision
/// `cks(1)` value passed into FMS, forms `xkmt = rmt * ck`, and chooses the
/// number of Dirac correction cycles from `mod(ixc0, 10)`.
pub fn screen_energy_state(
    input: ScreenEnergyStateInput,
) -> Result<ScreenEnergyState, ScreenError> {
    validate_finite_complex_input("energy", input.energy)?;
    validate_finite_complex_input("reference_energy", input.reference_energy)?;
    validate_positive("muffin_tin_radius", input.muffin_tin_radius)?;

    let kinetic_energy = input.energy - input.reference_energy;
    validate_result_finite_complex("kinetic_energy", kinetic_energy)?;
    let alpha_scaled = kinetic_energy * SCREEN_FINE_STRUCTURE_ALPHA;
    let wave_number = (kinetic_energy * 2.0 + alpha_scaled * alpha_scaled).sqrt();
    validate_result_finite_complex("wave_number", wave_number)?;
    let muffin_tin_argument = wave_number * input.muffin_tin_radius;
    validate_result_finite_complex("muffin_tin_argument", muffin_tin_argument)?;
    let fms_wave_number = complex32_result("fms_wave_number", wave_number)?;
    let dirac_cycle_count = if input.exchange_selector % 10 < 5 {
        0
    } else {
        3
    };

    Ok(ScreenEnergyState {
        kinetic_energy,
        wave_number,
        fms_wave_number,
        muffin_tin_argument,
        dirac_cycle_count,
    })
}

/// Port the angular-momentum selector from SCREEN `getph.f90`.
///
/// FEFF starts from the requested phase-shift `lmaxsc`, caps it by the global
/// `lx`, then applies historical light-element overrides: elements up to Be
/// use `lmax = 2`, and H/He use `lmax = 1`. The overrides intentionally replace
/// the previous cap, matching the original assignment order.
pub fn screen_getph_lmax(
    atomic_number: usize,
    requested_lmax: usize,
    angular_capacity_lx: usize,
) -> Result<usize, ScreenError> {
    validate_count_at_least("atomic_number", atomic_number, 1)?;

    let mut lmax = requested_lmax.min(angular_capacity_lx);
    if atomic_number <= 4 {
        lmax = 2;
    }
    if atomic_number <= 2 {
        lmax = 1;
    }
    Ok(lmax)
}

/// Port the SCREEN/CRPA radial-solution normalization scalar setup.
///
/// After `phamp`, `screensub.f90` and `chi_crpa.f90` compute the relativistic
/// lower-component factor, `dum1`, and the regular-solution scale `xfnorm`.
/// FEFF sets `xfnorm` to zero when the phase amplitude is exactly zero; this
/// helper preserves that branch and validates all finite complex results.
pub fn screen_solution_normalization(
    input: ScreenSolutionNormalizationInput,
) -> Result<ScreenSolutionNormalization, ScreenError> {
    validate_finite_complex_input("wave_number", input.wave_number)?;
    validate_finite_complex_input("phase_amplitude", input.phase_amplitude)?;

    let one = Complex::new(1.0, 0.0);
    let zero = Complex::new(0.0, 0.0);
    let alpha_scaled = input.wave_number * SCREEN_FINE_STRUCTURE_ALPHA;
    let lower_denominator = one + (one + alpha_scaled * alpha_scaled).sqrt();
    validate_result_finite_complex("small_component_denominator", lower_denominator)?;
    if lower_denominator == zero {
        return Err(ScreenError::ZeroComplexResult {
            name: "small_component_denominator",
        });
    }

    let small_component_factor = -alpha_scaled / lower_denominator;
    validate_result_finite_complex("small_component_factor", small_component_factor)?;
    let scale_denominator = (one + small_component_factor * small_component_factor).sqrt();
    validate_result_finite_complex("relativistic_scale_denominator", scale_denominator)?;
    if scale_denominator == zero {
        return Err(ScreenError::ZeroComplexResult {
            name: "relativistic_scale_denominator",
        });
    }

    let relativistic_scale = one / scale_denominator;
    validate_result_finite_complex("relativistic_scale", relativistic_scale)?;
    let regular_solution_scale = if input.phase_amplitude == zero {
        zero
    } else {
        let scale = relativistic_scale / input.phase_amplitude;
        validate_result_finite_complex("regular_solution_scale", scale)?;
        scale
    };

    Ok(ScreenSolutionNormalization {
        small_component_factor,
        relativistic_scale,
        regular_solution_scale,
    })
}

/// Port the irregular muffin-tin boundary values from SCREEN `screensub.f90`.
///
/// Before calling `dfovrg` with `irr = 1`, FEFF initializes the irregular
/// solution from either the standing-wave `N*cos(ph0)+J*sin(ph0)` expression or
/// the outgoing-Hankel branch selected by `irrh == 1`. This helper computes the
/// two complex boundary values while reusing the same relativistic `factor` and
/// `dum1` terms as [`screen_solution_normalization`].
pub fn screen_irregular_initial_condition(
    input: ScreenIrregularInitialConditionInput,
) -> Result<ScreenIrregularInitialCondition, ScreenError> {
    validate_positive("muffin_tin_radius", input.muffin_tin_radius)?;
    validate_finite_complex_input("phase_shift", input.phase_shift)?;
    validate_finite_complex_input("wave_number", input.wave_number)?;
    validate_finite_complex_input("bessel_j_l", input.bessel_j_l)?;
    validate_finite_complex_input("bessel_j_l_plus_1", input.bessel_j_l_plus_1)?;
    validate_finite_complex_input("neumann_l", input.neumann_l)?;
    validate_finite_complex_input("neumann_l_plus_1", input.neumann_l_plus_1)?;

    let normalization = screen_solution_normalization(ScreenSolutionNormalizationInput {
        wave_number: input.wave_number,
        phase_amplitude: Complex::new(1.0, 0.0),
    })?;
    let relativistic_scale = normalization.relativistic_scale;
    let small_component_factor = normalization.small_component_factor;

    let radius_scale = input.muffin_tin_radius * relativistic_scale;
    let (large_component, small_component) = if input.use_hankel_boundary {
        validate_finite_complex_input("hankel_l", input.hankel_l)?;
        validate_finite_complex_input("hankel_l_plus_1", input.hankel_l_plus_1)?;
        let phase_factor = (Complex::new(0.0, 1.0) * input.phase_shift).exp();
        validate_result_finite_complex("irregular_phase_factor", phase_factor)?;
        (
            input.hankel_l * phase_factor * radius_scale,
            input.hankel_l_plus_1 * phase_factor * radius_scale * small_component_factor,
        )
    } else {
        let cos_phase = input.phase_shift.cos();
        let sin_phase = input.phase_shift.sin();
        validate_result_finite_complex("irregular_cos_phase", cos_phase)?;
        validate_result_finite_complex("irregular_sin_phase", sin_phase)?;
        (
            (input.neumann_l * cos_phase + input.bessel_j_l * sin_phase) * radius_scale,
            (input.neumann_l_plus_1 * cos_phase + input.bessel_j_l_plus_1 * sin_phase)
                * radius_scale
                * small_component_factor,
        )
    };

    validate_result_finite_complex("irregular_large_component", large_component)?;
    validate_result_finite_complex("irregular_small_component", small_component)?;
    Ok(ScreenIrregularInitialCondition {
        large_component,
        small_component,
    })
}

/// Port the irregular-solution Wronskian rescaling from SCREEN `screensub.f90`.
///
/// After the irregular `dfovrg` pass, FEFF computes a complex Wronskian at
/// `jri`, inverts it with the photoelectron wave number, and multiplies both
/// irregular radial components by `exp(i*ph0) * qu`. A zero Wronskian follows
/// the original branch and yields a zero scale instead of dividing.
pub fn screen_irregular_wronskian_scale(
    input: ScreenIrregularWronskianScaleInput,
) -> Result<ScreenIrregularWronskianScale, ScreenError> {
    validate_finite_complex_input("phase_shift", input.phase_shift)?;
    validate_finite_complex_input("wave_number", input.wave_number)?;
    validate_finite_complex_input("regular_large_at_match", input.regular_large_at_match)?;
    validate_finite_complex_input("regular_small_at_match", input.regular_small_at_match)?;
    validate_finite_complex_input("irregular_large_at_match", input.irregular_large_at_match)?;
    validate_finite_complex_input("irregular_small_at_match", input.irregular_small_at_match)?;

    let phase_factor = (Complex::new(0.0, 1.0) * input.phase_shift).exp();
    validate_result_finite_complex("wronskian_phase_factor", phase_factor)?;
    let denominator = 2.0
        * SCREEN_ALPHA_INVERSE
        * phase_factor
        * (input.irregular_large_at_match * input.regular_small_at_match
            - input.regular_large_at_match * input.irregular_small_at_match);
    validate_result_finite_complex("wronskian_denominator", denominator)?;

    let zero = Complex::new(0.0, 0.0);
    let reciprocal_wave_scale = if denominator == zero {
        zero
    } else {
        if input.wave_number == zero {
            return Err(ScreenError::ZeroComplexResult {
                name: "wave_number",
            });
        }
        let value = Complex::new(1.0, 0.0) / denominator / input.wave_number;
        validate_result_finite_complex("wronskian_reciprocal_wave_scale", value)?;
        value
    };
    let irregular_solution_scale = phase_factor * reciprocal_wave_scale;
    validate_result_finite_complex(
        "wronskian_irregular_solution_scale",
        irregular_solution_scale,
    )?;

    Ok(ScreenIrregularWronskianScale {
        phase_factor,
        denominator,
        reciprocal_wave_scale,
        irregular_solution_scale,
    })
}

/// Port the exact radial continuation from SCREEN `screensub.f90`.
///
/// After the regular and irregular `dfovrg` solutions are normalized, FEFF
/// overwrites rows `jri:ilast` with exact free-particle combinations evaluated
/// at `xck = ck * ri(j)`. This scalar helper computes one such row from the
/// already-evaluated Bessel, Neumann, and Hankel values; the caller owns the
/// radial loop and special-function evaluation.
pub fn screen_exact_radial_continuation(
    input: ScreenExactRadialContinuationInput,
) -> Result<ScreenExactRadialContinuation, ScreenError> {
    validate_positive("radius", input.radius)?;
    validate_finite_complex_input("phase_shift", input.phase_shift)?;
    validate_finite_complex_input("wave_number", input.wave_number)?;
    validate_finite_complex_input("bessel_j_l", input.bessel_j_l)?;
    validate_finite_complex_input("bessel_j_l_plus_1", input.bessel_j_l_plus_1)?;
    validate_finite_complex_input("neumann_l", input.neumann_l)?;
    validate_finite_complex_input("neumann_l_plus_1", input.neumann_l_plus_1)?;
    validate_finite_complex_input("hankel_l", input.hankel_l)?;
    validate_finite_complex_input("hankel_l_plus_1", input.hankel_l_plus_1)?;

    let normalization = screen_solution_normalization(ScreenSolutionNormalizationInput {
        wave_number: input.wave_number,
        phase_amplitude: Complex::new(1.0, 0.0),
    })?;
    let relativistic_scale = normalization.relativistic_scale;
    let small_component_factor = normalization.small_component_factor;
    let radius_scale = input.radius * relativistic_scale;
    let cos_phase = input.phase_shift.cos();
    let sin_phase = input.phase_shift.sin();
    let phase_factor = (Complex::new(0.0, 1.0) * input.phase_shift).exp();
    validate_result_finite_complex("exact_continuation_cos_phase", cos_phase)?;
    validate_result_finite_complex("exact_continuation_sin_phase", sin_phase)?;
    validate_result_finite_complex("exact_continuation_phase_factor", phase_factor)?;

    let regular_large_component =
        (input.bessel_j_l * cos_phase - input.neumann_l * sin_phase) * radius_scale;
    let regular_small_component = (input.bessel_j_l_plus_1 * cos_phase
        - input.neumann_l_plus_1 * sin_phase)
        * radius_scale
        * small_component_factor;
    let irregular_large_component = input.hankel_l * phase_factor * radius_scale;
    let irregular_small_component =
        input.hankel_l_plus_1 * phase_factor * radius_scale * small_component_factor;

    validate_result_finite_complex("exact_regular_large_component", regular_large_component)?;
    validate_result_finite_complex("exact_regular_small_component", regular_small_component)?;
    validate_result_finite_complex("exact_irregular_large_component", irregular_large_component)?;
    validate_result_finite_complex("exact_irregular_small_component", irregular_small_component)?;

    Ok(ScreenExactRadialContinuation {
        regular_large_component,
        regular_small_component,
        irregular_large_component,
        irregular_small_component,
    })
}
