//! SCREEN radial-solution setup helpers.

use ndarray::{Array1, Array2, Array3, Axis};

use crate::{
    Complex, FovrgDiracSolution, XsphRegularPhaseInput, besjh, besjn, fovrg_dirac_solver,
    xsph_regular_phase,
};

use super::constants::{SCREEN_ALPHA_INVERSE, SCREEN_FINE_STRUCTURE_ALPHA};
use super::types::*;
use super::validation::{
    complex32_result, validate_active_count, validate_count_at_least,
    validate_finite_complex_input, validate_positive, validate_result_finite_complex,
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

/// Generate exact free-particle continuation rows for one SCREEN radial channel.
///
/// FEFF evaluates `besjn`/`besjh` at `ck*ri(j)` and overwrites rows
/// `jri:ilast` with the exact continuation formulas. The returned array keeps
/// the full active length so it can be passed directly to
/// [`screen_radial_channel_assembly`].
pub fn screen_exact_radial_continuation_tail(
    input: ScreenExactRadialContinuationTailInput<'_>,
) -> Result<ScreenExactRadialContinuationTail, ScreenError> {
    validate_count_at_least("active_count", input.active_count, 1)?;
    validate_count_at_least(
        "radial_match_index_1based",
        input.radial_match_index_1based,
        1,
    )?;
    if input.radial_match_index_1based > input.active_count {
        return Err(ScreenError::RadialBoundOutOfRange {
            name: "radial_match_index_1based",
            value: input.radial_match_index_1based,
            capacity: input.active_count,
        });
    }
    validate_active_count(input.active_count, input.radii.len())?;
    validate_finite_complex_input("phase_shift", input.phase_shift)?;
    validate_finite_complex_input("wave_number", input.wave_number)?;
    let max_l = input
        .angular_momentum
        .checked_add(1)
        .ok_or(ScreenError::IndexSizeOverflow {
            name: "angular_momentum",
        })?;

    let mut rows = Array1::from_elem(
        input.active_count,
        ScreenExactRadialContinuation {
            regular_large_component: Complex::new(0.0, 0.0),
            regular_small_component: Complex::new(0.0, 0.0),
            irregular_large_component: Complex::new(0.0, 0.0),
            irregular_small_component: Complex::new(0.0, 0.0),
        },
    );
    let start_index = input.radial_match_index_1based - 1;
    for index in start_index..input.active_count {
        let radius = input.radii[index];
        validate_positive("radius", radius)?;
        let argument = input.wave_number * radius;
        validate_result_finite_complex("exact_continuation_argument", argument)?;
        let bessel = besjn(argument, max_l)?;
        let hankel = besjh(argument, max_l)?;
        rows[index] = screen_exact_radial_continuation(ScreenExactRadialContinuationInput {
            radius,
            phase_shift: input.phase_shift,
            wave_number: input.wave_number,
            bessel_j_l: bessel.j[input.angular_momentum],
            neumann_l: bessel.y[input.angular_momentum],
            bessel_j_l_plus_1: bessel.j[max_l],
            neumann_l_plus_1: bessel.y[max_l],
            hankel_l: hankel.h[input.angular_momentum],
            hankel_l_plus_1: hankel.h[max_l],
        })?;
    }

    Ok(ScreenExactRadialContinuationTail {
        rows,
        start_index_1based: input.radial_match_index_1based,
    })
}

/// Assemble one SCREEN radial channel from raw regular and irregular solves.
///
/// This mirrors the post-`dfovrg` handoff in `SCREEN/screensub.f90`: the
/// regular solution is multiplied by FEFF `xfnorm`, the irregular solution is
/// Wronskian-normalized at `jri`, and rows `jri:ilast` are optionally
/// overwritten by exact free-particle continuation values.
pub fn screen_radial_channel_assembly(
    input: ScreenRadialChannelAssemblyInput<'_>,
) -> Result<ScreenRadialChannelAssembly, ScreenError> {
    validate_count_at_least("active_count", input.active_count, 1)?;
    validate_count_at_least(
        "radial_match_index_1based",
        input.radial_match_index_1based,
        1,
    )?;
    if input.radial_match_index_1based > input.active_count {
        return Err(ScreenError::RadialBoundOutOfRange {
            name: "radial_match_index_1based",
            value: input.radial_match_index_1based,
            capacity: input.active_count,
        });
    }
    validate_active_count(input.active_count, input.regular_large.len())?;
    validate_active_count(input.active_count, input.regular_small.len())?;
    validate_active_count(input.active_count, input.irregular_large.len())?;
    validate_active_count(input.active_count, input.irregular_small.len())?;
    if let Some(exact) = input.exact_continuation {
        validate_active_count(input.active_count, exact.len())?;
    }

    let normalization = screen_solution_normalization(ScreenSolutionNormalizationInput {
        wave_number: input.wave_number,
        phase_amplitude: input.phase_amplitude,
    })?;

    let mut regular_large = Array1::zeros(input.active_count);
    let mut regular_small = Array1::zeros(input.active_count);
    for index in 0..input.active_count {
        let large = input.regular_large[index];
        let small = input.regular_small[index];
        validate_finite_complex_input("regular_large", large)?;
        validate_finite_complex_input("regular_small", small)?;
        regular_large[index] = large * normalization.regular_solution_scale;
        regular_small[index] = small * normalization.regular_solution_scale;
        validate_result_finite_complex("assembled_regular_large", regular_large[index])?;
        validate_result_finite_complex("assembled_regular_small", regular_small[index])?;
    }

    let match_index = input.radial_match_index_1based - 1;
    let irregular_wronskian_scale =
        screen_irregular_wronskian_scale(ScreenIrregularWronskianScaleInput {
            phase_shift: input.phase_shift,
            wave_number: input.wave_number,
            regular_large_at_match: regular_large[match_index],
            regular_small_at_match: regular_small[match_index],
            irregular_large_at_match: input.irregular_large[match_index],
            irregular_small_at_match: input.irregular_small[match_index],
        })?;

    let mut irregular_large = Array1::zeros(input.active_count);
    let mut irregular_small = Array1::zeros(input.active_count);
    for index in 0..input.active_count {
        let large = input.irregular_large[index];
        let small = input.irregular_small[index];
        validate_finite_complex_input("irregular_large", large)?;
        validate_finite_complex_input("irregular_small", small)?;
        irregular_large[index] = large * irregular_wronskian_scale.irregular_solution_scale;
        irregular_small[index] = small * irregular_wronskian_scale.irregular_solution_scale;
        validate_result_finite_complex("assembled_irregular_large", irregular_large[index])?;
        validate_result_finite_complex("assembled_irregular_small", irregular_small[index])?;
    }

    if let Some(exact) = input.exact_continuation {
        for index in match_index..input.active_count {
            let continuation = exact[index];
            validate_finite_complex_input(
                "exact_regular_large",
                continuation.regular_large_component,
            )?;
            validate_finite_complex_input(
                "exact_regular_small",
                continuation.regular_small_component,
            )?;
            validate_finite_complex_input(
                "exact_irregular_large",
                continuation.irregular_large_component,
            )?;
            validate_finite_complex_input(
                "exact_irregular_small",
                continuation.irregular_small_component,
            )?;
            regular_large[index] = continuation.regular_large_component;
            regular_small[index] = continuation.regular_small_component;
            irregular_large[index] = continuation.irregular_large_component;
            irregular_small[index] = continuation.irregular_small_component;
        }
    }

    Ok(ScreenRadialChannelAssembly {
        regular_large,
        regular_small,
        irregular_large,
        irregular_small,
        normalization,
        irregular_wronskian_scale,
    })
}

/// Drive one SCREEN angular channel through raw FOVRG solves and assembly.
///
/// This helper joins the production call sequence from `screensub.f90`: solve
/// the regular Dirac radial channel, derive the irregular muffin-tin boundary
/// condition from the phase and free-particle functions, solve the irregular
/// channel, overwrite the exterior rows with exact continuation values, and
/// return the normalized radial components consumed by SCREEN response kernels.
pub fn screen_fovrg_channel_assembly(
    input: ScreenFovrgChannelAssemblyInput<'_>,
) -> Result<ScreenFovrgChannelAssembly, ScreenError> {
    validate_screen_fovrg_channel_solvers(
        input.regular_solver,
        input.irregular_solver,
        input.radial_match_index_1based,
        input.active_count,
    )?;
    validate_finite_complex_input("phase_shift", input.phase_shift)?;
    validate_finite_complex_input("phase_amplitude", input.phase_amplitude)?;
    validate_finite_complex_input("wave_number", input.wave_number)?;

    let regular_solution = fovrg_dirac_solver(input.regular_solver)?;
    screen_fovrg_channel_assembly_with_regular(input, regular_solution)
}

/// Drive one SCREEN angular channel and recover `phamp` from the regular solve.
///
/// This variant is the source-backed radial driver shape: it does not require a
/// cached or sidecar phase-amplitude table. The regular FOVRG muffin-tin values
/// are matched with the same FEFF `phamp` helper used by XSPH, and the recovered
/// phase/amplitude then feed the irregular branch and SCREEN normalization.
pub fn screen_fovrg_matched_channel_assembly(
    input: ScreenFovrgMatchedChannelAssemblyInput<'_>,
) -> Result<ScreenFovrgChannelAssembly, ScreenError> {
    validate_screen_fovrg_channel_solvers(
        input.regular_solver,
        input.irregular_solver,
        input.radial_match_index_1based,
        input.active_count,
    )?;
    validate_finite_complex_input("wave_number", input.wave_number)?;

    let regular_solution = fovrg_dirac_solver(input.regular_solver)?;
    let phase = xsph_regular_phase(XsphRegularPhaseInput {
        muffin_tin_radius: input.regular_solver.muffin_tin_radius,
        wave_number: input.wave_number,
        regular_large_at_muffin_tin: regular_solution.muffin_tin_large_component,
        regular_small_at_muffin_tin: regular_solution.muffin_tin_small_component,
        kappa: input.regular_solver.target_kappa,
    })?;
    let explicit = ScreenFovrgChannelAssemblyInput {
        regular_solver: input.regular_solver,
        irregular_solver: input.irregular_solver,
        phase_shift: phase.phase_shift,
        phase_amplitude: phase.phase_amplitude,
        wave_number: input.wave_number,
        angular_momentum: input.angular_momentum,
        radial_match_index_1based: input.radial_match_index_1based,
        active_count: input.active_count,
        use_hankel_boundary: input.use_hankel_boundary,
    };
    screen_fovrg_channel_assembly_with_regular(explicit, regular_solution)
}

fn screen_fovrg_channel_assembly_with_regular(
    input: ScreenFovrgChannelAssemblyInput<'_>,
    regular_solution: FovrgDiracSolution,
) -> Result<ScreenFovrgChannelAssembly, ScreenError> {
    validate_count_at_least("active_count", input.active_count, 1)?;
    validate_count_at_least(
        "radial_match_index_1based",
        input.radial_match_index_1based,
        1,
    )?;
    if input.radial_match_index_1based > input.active_count {
        return Err(ScreenError::RadialBoundOutOfRange {
            name: "radial_match_index_1based",
            value: input.radial_match_index_1based,
            capacity: input.active_count,
        });
    }
    let bessel_order =
        input
            .angular_momentum
            .checked_add(1)
            .ok_or(ScreenError::IndexSizeOverflow {
                name: "angular_momentum",
            })?;
    let muffin_tin_argument = input.wave_number * input.regular_solver.muffin_tin_radius;
    validate_result_finite_complex("fovrg_channel_muffin_tin_argument", muffin_tin_argument)?;
    let bessel = besjn(muffin_tin_argument, bessel_order)?;
    let hankel = besjh(muffin_tin_argument, bessel_order)?;
    let irregular_initial_condition =
        screen_irregular_initial_condition(ScreenIrregularInitialConditionInput {
            muffin_tin_radius: input.regular_solver.muffin_tin_radius,
            phase_shift: input.phase_shift,
            wave_number: input.wave_number,
            bessel_j_l: bessel.j[input.angular_momentum],
            neumann_l: bessel.y[input.angular_momentum],
            bessel_j_l_plus_1: bessel.j[bessel_order],
            neumann_l_plus_1: bessel.y[bessel_order],
            hankel_l: hankel.h[input.angular_momentum],
            hankel_l_plus_1: hankel.h[bessel_order],
            use_hankel_boundary: input.use_hankel_boundary,
        })?;

    let irregular_solver = crate::FovrgDiracSolverInput {
        muffin_tin_large_component: irregular_initial_condition.large_component,
        muffin_tin_small_component: irregular_initial_condition.small_component,
        ..input.irregular_solver
    };
    let irregular_solution = fovrg_dirac_solver(irregular_solver)?;
    let exact_continuation =
        screen_exact_radial_continuation_tail(ScreenExactRadialContinuationTailInput {
            radii: input.regular_solver.radii,
            phase_shift: input.phase_shift,
            wave_number: input.wave_number,
            angular_momentum: input.angular_momentum,
            radial_match_index_1based: input.radial_match_index_1based,
            active_count: input.active_count,
        })?;

    let assembled = screen_radial_channel_assembly(ScreenRadialChannelAssemblyInput {
        regular_large: regular_solution.large_component.view(),
        regular_small: regular_solution.small_component.view(),
        irregular_large: irregular_solution.large_component.view(),
        irregular_small: irregular_solution.small_component.view(),
        exact_continuation: Some(exact_continuation.rows.view()),
        phase_shift: input.phase_shift,
        phase_amplitude: input.phase_amplitude,
        wave_number: input.wave_number,
        radial_match_index_1based: input.radial_match_index_1based,
        active_count: input.active_count,
    })?;

    Ok(ScreenFovrgChannelAssembly {
        phase_shift: input.phase_shift,
        phase_amplitude: input.phase_amplitude,
        regular_solution,
        irregular_initial_condition,
        irregular_solution,
        exact_continuation,
        assembled,
    })
}

fn validate_screen_fovrg_channel_solvers(
    regular_solver: crate::FovrgDiracSolverInput<'_>,
    irregular_solver: crate::FovrgDiracSolverInput<'_>,
    radial_match_index_1based: usize,
    active_count: usize,
) -> Result<(), ScreenError> {
    validate_count_at_least("active_count", active_count, 1)?;
    validate_count_at_least("radial_match_index_1based", radial_match_index_1based, 1)?;
    if radial_match_index_1based > active_count {
        return Err(ScreenError::RadialBoundOutOfRange {
            name: "radial_match_index_1based",
            value: radial_match_index_1based,
            capacity: active_count,
        });
    }
    if regular_solver.irregular {
        return Err(ScreenError::FovrgSolverBranchMismatch {
            name: "regular_solver",
            expected: false,
            actual: regular_solver.irregular,
        });
    }
    if !irregular_solver.irregular {
        return Err(ScreenError::FovrgSolverBranchMismatch {
            name: "irregular_solver",
            expected: true,
            actual: irregular_solver.irregular,
        });
    }

    let regular_match_index_1based =
        regular_solver
            .radial_match_index
            .checked_add(1)
            .ok_or(ScreenError::IndexSizeOverflow {
                name: "regular_solver.radial_match_index",
            })?;
    if regular_match_index_1based != radial_match_index_1based {
        return Err(ScreenError::FovrgSolverMatchIndexMismatch {
            name: "regular_solver",
            solver_index_1based: regular_match_index_1based,
            radial_match_index_1based,
        });
    }

    let irregular_match_index_1based = irregular_solver.radial_match_index.checked_add(1).ok_or(
        ScreenError::IndexSizeOverflow {
            name: "irregular_solver.radial_match_index",
        },
    )?;
    if irregular_match_index_1based != radial_match_index_1based {
        return Err(ScreenError::FovrgSolverMatchIndexMismatch {
            name: "irregular_solver",
            solver_index_1based: irregular_match_index_1based,
            radial_match_index_1based,
        });
    }
    if regular_solver.radii.len() != irregular_solver.radii.len() {
        return Err(ScreenError::FovrgSolverRadialGridMismatch {
            regular_len: regular_solver.radii.len(),
            irregular_len: irregular_solver.radii.len(),
        });
    }
    if regular_solver.muffin_tin_radius != irregular_solver.muffin_tin_radius {
        return Err(ScreenError::FovrgSolverMuffinTinRadiusMismatch {
            regular: regular_solver.muffin_tin_radius,
            irregular: irregular_solver.muffin_tin_radius,
        });
    }
    Ok(())
}

/// Drive prepared SCREEN FOVRG channel inputs over a contour/angular grid.
///
/// The input solver slices follow FEFF's nested loop order `(energy, l)`.
/// Each channel is solved through [`screen_fovrg_channel_assembly`], then the
/// normalized radial components are packed as `(energy, radial, l)` cubes for
/// the SCREEN response assembly kernels.
pub fn screen_fovrg_cube_assembly(
    input: ScreenFovrgCubeAssemblyInput<'_>,
) -> Result<ScreenFovrgCubeAssembly, ScreenError> {
    let (energy_count, angular_count) = input.phase_shifts.dim();
    validate_count_at_least("energy_count", energy_count, 1)?;
    validate_count_at_least("angular_count", angular_count, 1)?;
    validate_count_at_least("active_count", input.active_count, 1)?;
    validate_count_at_least(
        "radial_match_index_1based",
        input.radial_match_index_1based,
        1,
    )?;
    validate_matrix_shape(
        "phase_amplitudes",
        input.phase_amplitudes.dim(),
        energy_count,
        angular_count,
    )?;
    validate_count_at_least("wave_numbers", input.wave_numbers.len(), energy_count)?;
    let expected_solver_count =
        energy_count
            .checked_mul(angular_count)
            .ok_or(ScreenError::IndexSizeOverflow {
                name: "screen_fovrg_solver_grid",
            })?;
    if input.regular_solvers.len() != expected_solver_count
        || input.irregular_solvers.len() != expected_solver_count
    {
        return Err(ScreenError::FovrgSolverCountMismatch {
            expected: expected_solver_count,
            regular: input.regular_solvers.len(),
            irregular: input.irregular_solvers.len(),
        });
    }

    let mut regular_large = Array3::zeros((energy_count, input.active_count, angular_count));
    let mut regular_small = Array3::zeros((energy_count, input.active_count, angular_count));
    let mut irregular_large = Array3::zeros((energy_count, input.active_count, angular_count));
    let mut irregular_small = Array3::zeros((energy_count, input.active_count, angular_count));
    let mut irregular_initial_large = Array2::zeros((energy_count, angular_count));
    let mut irregular_initial_small = Array2::zeros((energy_count, angular_count));
    let mut regular_iteration_counts = Array2::zeros((energy_count, angular_count));
    let mut irregular_iteration_counts = Array2::zeros((energy_count, angular_count));
    let mut difficult_iterations = Array2::zeros((energy_count, angular_count));

    for energy_index in 0..energy_count {
        for angular_momentum in 0..angular_count {
            let solver_index =
                screen_fovrg_solver_index(energy_index, angular_momentum, angular_count)?;
            let channel = screen_fovrg_channel_assembly(ScreenFovrgChannelAssemblyInput {
                regular_solver: input.regular_solvers[solver_index],
                irregular_solver: input.irregular_solvers[solver_index],
                phase_shift: input.phase_shifts[(energy_index, angular_momentum)],
                phase_amplitude: input.phase_amplitudes[(energy_index, angular_momentum)],
                wave_number: input.wave_numbers[energy_index],
                angular_momentum,
                radial_match_index_1based: input.radial_match_index_1based,
                active_count: input.active_count,
                use_hankel_boundary: input.use_hankel_boundary,
            })?;

            irregular_initial_large[(energy_index, angular_momentum)] =
                channel.irregular_initial_condition.large_component;
            irregular_initial_small[(energy_index, angular_momentum)] =
                channel.irregular_initial_condition.small_component;
            regular_iteration_counts[(energy_index, angular_momentum)] =
                channel.regular_solution.iteration_count;
            irregular_iteration_counts[(energy_index, angular_momentum)] =
                channel.irregular_solution.iteration_count;
            difficult_iterations[(energy_index, angular_momentum)] =
                channel.regular_solution.difficult_iterations
                    + channel.irregular_solution.difficult_iterations;
            for radial_index in 0..input.active_count {
                regular_large[(energy_index, radial_index, angular_momentum)] =
                    channel.assembled.regular_large[radial_index];
                regular_small[(energy_index, radial_index, angular_momentum)] =
                    channel.assembled.regular_small[radial_index];
                irregular_large[(energy_index, radial_index, angular_momentum)] =
                    channel.assembled.irregular_large[radial_index];
                irregular_small[(energy_index, radial_index, angular_momentum)] =
                    channel.assembled.irregular_small[radial_index];
            }
        }
    }

    Ok(ScreenFovrgCubeAssembly {
        radial_cubes: ScreenRadialCubeAssembly {
            regular_large,
            regular_small,
            irregular_large,
            irregular_small,
        },
        irregular_initial_large,
        irregular_initial_small,
        regular_iteration_counts,
        irregular_iteration_counts,
        difficult_iterations,
    })
}

/// Drive matched SCREEN FOVRG solves over a contour/angular grid.
///
/// This is the source radial-solver grid shape used before SCREEN response
/// assembly. The regular FOVRG pass in each channel is matched with FEFF
/// `phamp` to recover both `ph0` and `temp`; those values then drive the
/// irregular pass and the response-ready radial cube packing.
pub fn screen_fovrg_matched_cube_assembly(
    input: ScreenFovrgMatchedCubeAssemblyInput<'_>,
) -> Result<ScreenFovrgMatchedCubeAssembly, ScreenError> {
    validate_count_at_least("energy_count", input.wave_numbers.len(), 1)?;
    validate_count_at_least("angular_count", input.angular_count, 1)?;
    validate_count_at_least("active_count", input.active_count, 1)?;
    let energy_count = input.wave_numbers.len();
    let expected_solver_count =
        energy_count
            .checked_mul(input.angular_count)
            .ok_or(ScreenError::IndexSizeOverflow {
                name: "screen_fovrg_matched_solver_grid",
            })?;
    if input.regular_solvers.len() != expected_solver_count
        || input.irregular_solvers.len() != expected_solver_count
    {
        return Err(ScreenError::FovrgSolverCountMismatch {
            expected: expected_solver_count,
            regular: input.regular_solvers.len(),
            irregular: input.irregular_solvers.len(),
        });
    }

    let mut regular_large = Array3::zeros((energy_count, input.active_count, input.angular_count));
    let mut regular_small = Array3::zeros((energy_count, input.active_count, input.angular_count));
    let mut irregular_large =
        Array3::zeros((energy_count, input.active_count, input.angular_count));
    let mut irregular_small =
        Array3::zeros((energy_count, input.active_count, input.angular_count));
    let mut phase_shifts = Array2::zeros((energy_count, input.angular_count));
    let mut phase_amplitudes = Array2::zeros((energy_count, input.angular_count));
    let mut irregular_initial_large = Array2::zeros((energy_count, input.angular_count));
    let mut irregular_initial_small = Array2::zeros((energy_count, input.angular_count));
    let mut regular_iteration_counts = Array2::zeros((energy_count, input.angular_count));
    let mut irregular_iteration_counts = Array2::zeros((energy_count, input.angular_count));
    let mut difficult_iterations = Array2::zeros((energy_count, input.angular_count));

    for energy_index in 0..energy_count {
        for angular_momentum in 0..input.angular_count {
            let solver_index =
                screen_fovrg_solver_index(energy_index, angular_momentum, input.angular_count)?;
            let channel =
                screen_fovrg_matched_channel_assembly(ScreenFovrgMatchedChannelAssemblyInput {
                    regular_solver: input.regular_solvers[solver_index],
                    irregular_solver: input.irregular_solvers[solver_index],
                    wave_number: input.wave_numbers[energy_index],
                    angular_momentum,
                    radial_match_index_1based: input.radial_match_index_1based,
                    active_count: input.active_count,
                    use_hankel_boundary: input.use_hankel_boundary,
                })?;

            phase_shifts[(energy_index, angular_momentum)] = channel.phase_shift;
            phase_amplitudes[(energy_index, angular_momentum)] = channel.phase_amplitude;
            irregular_initial_large[(energy_index, angular_momentum)] =
                channel.irregular_initial_condition.large_component;
            irregular_initial_small[(energy_index, angular_momentum)] =
                channel.irregular_initial_condition.small_component;
            regular_iteration_counts[(energy_index, angular_momentum)] =
                channel.regular_solution.iteration_count;
            irregular_iteration_counts[(energy_index, angular_momentum)] =
                channel.irregular_solution.iteration_count;
            difficult_iterations[(energy_index, angular_momentum)] =
                channel.regular_solution.difficult_iterations
                    + channel.irregular_solution.difficult_iterations;
            for radial_index in 0..input.active_count {
                regular_large[(energy_index, radial_index, angular_momentum)] =
                    channel.assembled.regular_large[radial_index];
                regular_small[(energy_index, radial_index, angular_momentum)] =
                    channel.assembled.regular_small[radial_index];
                irregular_large[(energy_index, radial_index, angular_momentum)] =
                    channel.assembled.irregular_large[radial_index];
                irregular_small[(energy_index, radial_index, angular_momentum)] =
                    channel.assembled.irregular_small[radial_index];
            }
        }
    }

    Ok(ScreenFovrgMatchedCubeAssembly {
        solved: ScreenFovrgCubeAssembly {
            radial_cubes: ScreenRadialCubeAssembly {
                regular_large,
                regular_small,
                irregular_large,
                irregular_small,
            },
            irregular_initial_large,
            irregular_initial_small,
            regular_iteration_counts,
            irregular_iteration_counts,
            difficult_iterations,
        },
        phase_shifts,
        phase_amplitudes,
    })
}

/// Assemble normalized SCREEN radial cubes over contour energy and angular channels.
///
/// The raw inputs are the per-channel regular and irregular `dfovrg` outputs
/// with shape `(energy, radial, angular)`. This helper applies
/// [`screen_radial_channel_assembly`] to each `(energy,l)` channel and returns
/// the large/small regular and irregular cubes needed before response-slice
/// assembly.
pub fn screen_radial_cube_assembly(
    input: ScreenRadialCubeAssemblyInput<'_>,
) -> Result<ScreenRadialCubeAssembly, ScreenError> {
    let (energy_count, radial_count, angular_count) = input.regular_large.dim();
    validate_count_at_least("energy_count", energy_count, 1)?;
    validate_count_at_least("angular_count", angular_count, 1)?;
    validate_count_at_least("active_count", input.active_count, 1)?;
    validate_count_at_least(
        "radial_match_index_1based",
        input.radial_match_index_1based,
        1,
    )?;
    validate_cube_shape(
        "regular_large",
        input.regular_large.dim(),
        energy_count,
        input.active_count,
        angular_count,
    )?;
    validate_count_at_least("regular_large_radial", radial_count, input.active_count)?;
    validate_cube_shape(
        "regular_small",
        input.regular_small.dim(),
        energy_count,
        input.active_count,
        angular_count,
    )?;
    validate_cube_shape(
        "irregular_large",
        input.irregular_large.dim(),
        energy_count,
        input.active_count,
        angular_count,
    )?;
    validate_cube_shape(
        "irregular_small",
        input.irregular_small.dim(),
        energy_count,
        input.active_count,
        angular_count,
    )?;
    if let Some(exact) = input.exact_continuation {
        validate_cube_shape(
            "exact_continuation",
            exact.dim(),
            energy_count,
            input.active_count,
            angular_count,
        )?;
    }
    validate_matrix_shape(
        "phase_shifts",
        input.phase_shifts.dim(),
        energy_count,
        angular_count,
    )?;
    validate_matrix_shape(
        "phase_amplitudes",
        input.phase_amplitudes.dim(),
        energy_count,
        angular_count,
    )?;
    validate_count_at_least("wave_numbers", input.wave_numbers.len(), energy_count)?;

    let mut regular_large = Array3::zeros((energy_count, input.active_count, angular_count));
    let mut regular_small = Array3::zeros((energy_count, input.active_count, angular_count));
    let mut irregular_large = Array3::zeros((energy_count, input.active_count, angular_count));
    let mut irregular_small = Array3::zeros((energy_count, input.active_count, angular_count));

    for energy_index in 0..energy_count {
        let regular_large_energy = input.regular_large.index_axis(Axis(0), energy_index);
        let regular_small_energy = input.regular_small.index_axis(Axis(0), energy_index);
        let irregular_large_energy = input.irregular_large.index_axis(Axis(0), energy_index);
        let irregular_small_energy = input.irregular_small.index_axis(Axis(0), energy_index);

        for angular_momentum in 0..angular_count {
            let channel = if let Some(exact) = input.exact_continuation {
                let exact_energy = exact.index_axis(Axis(0), energy_index);
                let exact_channel = exact_energy.index_axis(Axis(1), angular_momentum);
                screen_radial_channel_assembly(ScreenRadialChannelAssemblyInput {
                    regular_large: regular_large_energy.index_axis(Axis(1), angular_momentum),
                    regular_small: regular_small_energy.index_axis(Axis(1), angular_momentum),
                    irregular_large: irregular_large_energy.index_axis(Axis(1), angular_momentum),
                    irregular_small: irregular_small_energy.index_axis(Axis(1), angular_momentum),
                    exact_continuation: Some(exact_channel),
                    phase_shift: input.phase_shifts[(energy_index, angular_momentum)],
                    phase_amplitude: input.phase_amplitudes[(energy_index, angular_momentum)],
                    wave_number: input.wave_numbers[energy_index],
                    radial_match_index_1based: input.radial_match_index_1based,
                    active_count: input.active_count,
                })?
            } else {
                screen_radial_channel_assembly(ScreenRadialChannelAssemblyInput {
                    regular_large: regular_large_energy.index_axis(Axis(1), angular_momentum),
                    regular_small: regular_small_energy.index_axis(Axis(1), angular_momentum),
                    irregular_large: irregular_large_energy.index_axis(Axis(1), angular_momentum),
                    irregular_small: irregular_small_energy.index_axis(Axis(1), angular_momentum),
                    exact_continuation: None,
                    phase_shift: input.phase_shifts[(energy_index, angular_momentum)],
                    phase_amplitude: input.phase_amplitudes[(energy_index, angular_momentum)],
                    wave_number: input.wave_numbers[energy_index],
                    radial_match_index_1based: input.radial_match_index_1based,
                    active_count: input.active_count,
                })?
            };

            for radial_index in 0..input.active_count {
                regular_large[(energy_index, radial_index, angular_momentum)] =
                    channel.regular_large[radial_index];
                regular_small[(energy_index, radial_index, angular_momentum)] =
                    channel.regular_small[radial_index];
                irregular_large[(energy_index, radial_index, angular_momentum)] =
                    channel.irregular_large[radial_index];
                irregular_small[(energy_index, radial_index, angular_momentum)] =
                    channel.irregular_small[radial_index];
            }
        }
    }

    Ok(ScreenRadialCubeAssembly {
        regular_large,
        regular_small,
        irregular_large,
        irregular_small,
    })
}

fn validate_cube_shape(
    name: &'static str,
    (energies, radial, angular): (usize, usize, usize),
    required_energies: usize,
    required_radial: usize,
    required_angular: usize,
) -> Result<(), ScreenError> {
    validate_count_at_least(name, energies, required_energies)?;
    validate_count_at_least(name, radial, required_radial)?;
    validate_count_at_least(name, angular, required_angular)
}

fn validate_matrix_shape(
    name: &'static str,
    (rows, columns): (usize, usize),
    required_rows: usize,
    required_columns: usize,
) -> Result<(), ScreenError> {
    validate_count_at_least(name, rows, required_rows)?;
    validate_count_at_least(name, columns, required_columns)
}

fn screen_fovrg_solver_index(
    energy_index: usize,
    angular_momentum: usize,
    angular_count: usize,
) -> Result<usize, ScreenError> {
    energy_index
        .checked_mul(angular_count)
        .and_then(|offset| offset.checked_add(angular_momentum))
        .ok_or(ScreenError::IndexSizeOverflow {
            name: "screen_fovrg_solver_index",
        })
}
