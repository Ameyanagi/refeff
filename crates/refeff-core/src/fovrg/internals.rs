use super::*;

#[derive(Debug, Clone, Copy)]
pub(super) struct FovrgOutwardMidpoint {
    pub(super) f: Complex,
    pub(super) g: Complex,
    pub(super) c3: Complex,
    pub(super) large_exchange: Complex,
    pub(super) small_exchange: Complex,
}

#[derive(Debug, Clone)]
pub(super) struct FovrgInitialPhotoelectronRadialSolution {
    pub(super) large_component: ComplexVec,
    pub(super) small_component: ComplexVec,
    pub(super) large_coefficients: ComplexVec,
    pub(super) small_coefficients: ComplexVec,
    pub(super) difficult_iterations: usize,
}

pub(super) fn validate_dirac_solver_input(
    input: &FovrgDiracSolverInput<'_>,
    active_len: usize,
) -> Result<(), FovrgError> {
    validate_count_at_least("active_len", active_len, FOVRG_WKB_MINIMUM_COUNT)?;
    validate_count_at_least("bound_orbital_count", input.bound_orbital_count, 1)?;
    let coefficient_count = if input.irregular { 2 } else { 3 };
    validate_active_len("radii", active_len, input.radii.len())?;
    validate_active_len(
        "exchange_correlation_potential",
        active_len,
        input.exchange_correlation_potential.len(),
    )?;
    validate_active_len(
        "valence_exchange_correlation_potential",
        input.exchange_correlation_potential.len(),
        input.valence_exchange_correlation_potential.len(),
    )?;
    validate_matrix_rows(
        "bound_large_components",
        active_len,
        input.bound_large_components.shape()[0],
    )?;
    validate_matrix_rows(
        "bound_small_components",
        active_len,
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
        coefficient_count,
        input.bound_large_coefficients.shape()[0],
    )?;
    validate_matrix_rows(
        "bound_small_coefficients",
        coefficient_count,
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
    validate_active_len(
        "valence_counts",
        input.bound_orbital_count,
        input.valence_counts.len(),
    )?;
    validate_active_len("kappa", input.bound_orbital_count, input.kappa.len())?;
    validate_positive_finite("atomic_number", input.atomic_number)?;
    validate_positive_finite("step", input.step)?;
    validate_finite("muffin_tin_radius", input.muffin_tin_radius)?;
    validate_nonzero_kappa("target_kappa", 0, input.target_kappa)?;
    validate_complex_input("energy", 0, input.energy)?;
    validate_complex_input(
        "muffin_tin_large_component",
        0,
        input.muffin_tin_large_component,
    )?;
    validate_complex_input(
        "muffin_tin_small_component",
        0,
        input.muffin_tin_small_component,
    )?;

    if input.radial_match_index + 1 >= active_len {
        return Err(FovrgError::ActiveCountOutOfRange {
            field: "radial_match_index",
            active_len: input.radial_match_index + 2,
            len: active_len,
        });
    }
    validate_count_at_least("c3_rows", input.radial_match_index + 1, 8)?;

    for row in 0..active_len {
        validate_radius(row, input.radii[row])?;
    }
    for row in 0..input.exchange_correlation_potential.len() {
        validate_complex_input(
            "exchange_correlation_potential",
            row,
            input.exchange_correlation_potential[row],
        )?;
        validate_complex_input(
            "valence_exchange_correlation_potential",
            row,
            input.valence_exchange_correlation_potential[row],
        )?;
    }
    for row in 0..active_len {
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
    for coefficient in 0..coefficient_count {
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
        validate_real_input("valence_counts", orbital, input.valence_counts[orbital])?;
        validate_nonzero_kappa("kappa", orbital, input.kappa[orbital])?;
    }

    Ok(())
}

pub(super) fn validate_orbital_setup_input(
    input: &FovrgOrbitalSetupInput<'_>,
) -> Result<(), FovrgError> {
    validate_count_at_least("active_len", input.active_len, 1)?;
    validate_count_at_least("bound_orbital_count", input.bound_orbital_count, 1)?;
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

    for row in 0..input.active_len {
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
    for orbital in 0..input.bound_orbital_count {
        validate_real_input("electron_counts", orbital, input.electron_counts[orbital])?;
        validate_real_input("valence_counts", orbital, input.valence_counts[orbital])?;
        validate_nonzero_kappa("kappa", orbital, input.kappa[orbital])?;
    }
    Ok(())
}

pub(super) fn fovrg_dirac_active_len(step: Real, capacity_len: usize) -> Result<usize, FovrgError> {
    validate_count_at_least("radii", capacity_len, 1)?;
    validate_positive_finite("step", step)?;
    let rounded = fovrg_fortran_nint_positive("idim", 12.5 / step)?;
    let mut active_len = rounded.checked_add(1).ok_or(FovrgError::CountTooLarge {
        name: "idim",
        actual: rounded,
        maximum: usize::MAX - 1,
    })?;
    active_len = active_len.min(capacity_len);
    if active_len.is_multiple_of(2) {
        active_len = active_len.saturating_sub(1);
    }
    validate_count_at_least("active_len", active_len, 1)?;
    Ok(active_len)
}

pub(super) fn fovrg_dirac_wkb_index(
    energy: Complex,
    step: Real,
    target_len: usize,
    active_len: usize,
) -> Result<usize, FovrgError> {
    validate_count_at_least("active_len", active_len, FOVRG_WKB_MINIMUM_COUNT)?;
    validate_count_at_least("target_len", target_len, 1)?;
    let wave_term = 2.0 * energy + (energy / FEFF_ALPHA_INVERSE).powi(2);
    let wave_norm = wave_term.norm().sqrt();
    validate_positive_finite("wkb_wave_norm", wave_norm)?;

    let rwkb = 0.5 / step / wave_norm;
    validate_positive_finite("rwkb", rwkb)?;
    let raw_count = ((rwkb.ln() + 8.8) / step + 2.0).trunc();
    validate_finite("wkb_count", raw_count)?;
    if raw_count >= usize::MAX as Real {
        return Err(FovrgError::CountTooLarge {
            name: "wkb_count",
            actual: usize::MAX,
            maximum: usize::MAX - 1,
        });
    }

    let mut wkb_count = if raw_count <= 0.0 {
        0
    } else {
        raw_count as usize
    };
    wkb_count = wkb_count.min(active_len).max(FOVRG_WKB_MINIMUM_COUNT);
    if wkb_count >= target_len.saturating_sub(1) {
        wkb_count = active_len;
    }
    Ok(wkb_count - 1)
}

pub(super) fn fovrg_dirac_c3_potential(
    input: &FovrgDiracSolverInput<'_>,
    exchange_correlation_potential: ArrayView1<'_, Complex>,
) -> Result<ComplexVec, FovrgError> {
    let c3_active_len = input.radial_match_index + 1;
    let derivative = fovrg_c3_derivative(FovrgC3DerivativeInput {
        potential: exchange_correlation_potential,
        radii: input.radii,
        kappa: input.target_kappa,
        speed_of_light: FEFF_ALPHA_INVERSE,
        delta: input.step,
        active_len: c3_active_len,
    })?;
    let mut c3_potential =
        Array1::<Complex>::zeros(fovrg_dirac_active_len(input.step, input.radii.len())?);
    for row in 0..input.radial_match_index {
        c3_potential[row] = derivative[row];
    }
    Ok(c3_potential)
}

pub(super) fn fovrg_zero_extended_coefficients(
    coefficients: ArrayView1<'_, Complex>,
) -> ComplexVec {
    let mut extended = Array1::<Complex>::zeros(FOVRG_ORIGIN_COEFFICIENTS);
    let copy_len = coefficients.len().min(FOVRG_ORIGIN_COEFFICIENTS);
    for coefficient in 0..copy_len {
        extended[coefficient] = coefficients[coefficient];
    }
    extended
}

pub(super) fn fovrg_fortran_nint_positive(
    name: &'static str,
    value: Real,
) -> Result<usize, FovrgError> {
    validate_finite(name, value)?;
    if value < 0.0 {
        return Err(FovrgError::NonPositiveInput { name, value });
    }
    let rounded = (value + 0.5).floor();
    if rounded >= usize::MAX as Real {
        Err(FovrgError::CountTooLarge {
            name,
            actual: usize::MAX,
            maximum: usize::MAX - 1,
        })
    } else {
        Ok(rounded as usize)
    }
}

pub(super) fn validate_initial_photoelectron_input(
    input: &FovrgInitialPhotoelectronInput<'_>,
) -> Result<(), FovrgError> {
    validate_count_at_least("active_len", input.active_len, 1)?;
    validate_count_at_least("coefficient_count", input.coefficient_count, 1)?;
    validate_count_at_least("orbital_count", input.orbital_count, 1)?;
    validate_active_len(
        "exchange_correlation_potential",
        input.active_len,
        input.exchange_correlation_potential.len(),
    )?;
    validate_active_len("c3_potential", input.active_len, input.c3_potential.len())?;
    validate_active_len("kappa", input.orbital_count, input.kappa.len())?;
    validate_active_len(
        "orbital_lengths",
        input.orbital_count,
        input.orbital_lengths.len(),
    )?;
    let bound_orbital_count = input.orbital_count - 1;
    validate_active_len(
        "electron_counts",
        bound_orbital_count,
        input.electron_counts.len(),
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
        bound_orbital_count,
        input.bound_large_coefficients.shape()[1],
    )?;
    validate_matrix_cols(
        "bound_small_coefficients",
        bound_orbital_count,
        input.bound_small_coefficients.shape()[1],
    )?;
    validate_positive_finite("nuclear_charge", input.nuclear_charge)?;
    validate_positive_finite("speed_of_light", input.speed_of_light)?;
    validate_positive_finite("step", input.step)?;
    validate_finite("muffin_tin_radius", input.muffin_tin_radius)?;
    validate_complex_input("energy", 0, input.energy)?;
    validate_complex_input(
        "initial_large_coefficient",
        0,
        input.initial_large_coefficient,
    )?;
    validate_complex_input(
        "initial_small_coefficient",
        0,
        input.initial_small_coefficient,
    )?;

    if input.radial_match_index >= input.active_len {
        return Err(FovrgError::ActiveCountOutOfRange {
            field: "radial_match_index",
            active_len: input.radial_match_index + 1,
            len: input.active_len,
        });
    }
    if input.radial_match_index + 1 >= input.active_len {
        return Err(FovrgError::ActiveCountOutOfRange {
            field: "radial_match_index",
            active_len: input.radial_match_index + 2,
            len: input.active_len,
        });
    }
    if input.wkb_index >= input.active_len {
        return Err(FovrgError::ActiveCountOutOfRange {
            field: "wkb_index",
            active_len: input.wkb_index + 1,
            len: input.active_len,
        });
    }

    for row in 0..input.active_len {
        validate_complex_input(
            "exchange_correlation_potential",
            row,
            input.exchange_correlation_potential[row],
        )?;
        validate_complex_input("c3_potential", row, input.c3_potential[row])?;
    }
    for orbital in 0..input.orbital_count {
        validate_nonzero_kappa("kappa", orbital, input.kappa[orbital])?;
    }
    let target_index = input.orbital_count - 1;
    validate_count_at_least(
        "target_orbital_length",
        input.orbital_lengths[target_index],
        1,
    )?;
    for orbital in 0..bound_orbital_count {
        validate_real_input("electron_counts", orbital, input.electron_counts[orbital])?;
        for coefficient in 0..input.coefficient_count {
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
    Ok(())
}

pub(super) fn fovrg_photoelectron_retained_len(
    step: Real,
    active_len: usize,
) -> Result<usize, FovrgError> {
    let retained = 1.0 + (8.8 + 10.0_f64.ln()) / step;
    validate_finite("photoelectron_retained_len", retained)?;
    if retained >= usize::MAX as Real {
        return Err(FovrgError::CountTooLarge {
            name: "photoelectron_retained_len",
            actual: usize::MAX,
            maximum: usize::MAX - 1,
        });
    }
    Ok((retained.trunc() as usize).min(active_len))
}

pub(super) fn fovrg_regular_initial_coefficients(
    nuclear_charge: Real,
    speed_of_light: Real,
    kappa: i32,
    origin_power: Real,
) -> Result<(Complex, Complex), FovrgError> {
    let large = Complex::new(1.0, 0.0);
    let small = if kappa < 0 {
        let denominator = speed_of_light * (kappa as Real - origin_power);
        validate_nonzero_denominator("photoelectron_initial_small_denominator", denominator)?;
        large * nuclear_charge / denominator
    } else {
        large * speed_of_light * (kappa as Real + origin_power) / nuclear_charge
    };
    validate_complex_result("photoelectron_initial_large", 0, large)?;
    validate_complex_result("photoelectron_initial_small", 0, small)?;
    Ok((large, small))
}

pub(super) fn validate_outward_integration_input(
    input: &FovrgOutwardIntegrationInput<'_>,
) -> Result<(), FovrgError> {
    validate_count_at_least("active_len", input.active_len, 1)?;
    validate_active_len("potential", input.active_len, input.potential.len())?;
    validate_active_len(
        "large_exchange",
        input.active_len,
        input.large_exchange.len(),
    )?;
    validate_active_len(
        "small_exchange",
        input.active_len,
        input.small_exchange.len(),
    )?;
    validate_active_len("c3_potential", input.active_len, input.c3_potential.len())?;
    validate_active_len("radii", input.active_len, input.radii.len())?;
    validate_count_at_least(
        "potential_coefficients",
        input.potential_coefficients.len(),
        4,
    )?;
    if input.start_index >= input.active_len {
        return Err(FovrgError::ActiveCountOutOfRange {
            field: "start_index",
            active_len: input.start_index + 1,
            len: input.active_len,
        });
    }
    if input.last_index >= input.active_len {
        return Err(FovrgError::ActiveCountOutOfRange {
            field: "last_index",
            active_len: input.last_index + 1,
            len: input.active_len,
        });
    }
    if input.start_index > input.last_index {
        return Err(FovrgError::InvalidRange {
            name: "outward_integration",
            start: input.start_index,
            end: input.last_index,
        });
    }
    validate_nonzero_kappa("kappa", 0, input.kappa)?;
    validate_nonzero_finite("speed_of_light", input.speed_of_light)?;
    validate_nonzero_finite("step", input.step)?;
    validate_complex_input("initial_large_component", 0, input.initial_large_component)?;
    validate_complex_input("initial_small_component", 0, input.initial_small_component)?;
    validate_complex_input("energy", 0, input.energy)?;
    for row in 0..input.active_len {
        validate_complex_input("potential", row, input.potential[row])?;
        validate_complex_input("large_exchange", row, input.large_exchange[row])?;
        validate_complex_input("small_exchange", row, input.small_exchange[row])?;
        validate_complex_input("c3_potential", row, input.c3_potential[row])?;
        validate_radius(row, input.radii[row])?;
    }
    for row in 0..input.potential_coefficients.len() {
        validate_complex_input(
            "potential_coefficients",
            row,
            input.potential_coefficients[row],
        )?;
    }
    Ok(())
}

pub(super) fn fovrg_outward_grid_terms(
    input: &FovrgOutwardIntegrationInput<'_>,
    row: usize,
    energy_over_light: Complex,
    ccl: Real,
) -> Result<(Complex, Complex, Complex), FovrgError> {
    let f = (energy_over_light - input.potential[row]) * input.radii[row];
    let g = f + ccl * input.radii[row];
    let c3 = fovrg_outward_c3(input.c3_scale, input.c3_potential[row], g)?;
    Ok((f, g, c3))
}

pub(super) fn fovrg_outward_midpoint_terms(
    input: &FovrgOutwardIntegrationInput<'_>,
    row: usize,
    energy_over_light: Complex,
    ccl: Real,
    exp_half_step: Real,
) -> Result<FovrgOutwardMidpoint, FovrgError> {
    let next = row + 1;
    let radius = input.radii[row];
    let next_radius = input.radii[next];
    let half_radius = radius * exp_half_step;
    let interval = next_radius - radius;
    validate_nonzero_denominator("outward_radius_interval", interval)?;
    let left_weight = (next_radius - half_radius) / interval;
    let right_weight = (half_radius - radius) / interval;

    let (potential, c3_potential) = if input.potential_coefficients[0].re < 0.0
        && input.start_index == 0
    {
        let left_potential = input.potential[row] - input.potential_coefficients[0] / radius;
        let right_potential = input.potential[next] - input.potential_coefficients[0] / next_radius;
        let potential = left_potential * left_weight
            + right_potential * right_weight
            + input.potential_coefficients[0] / half_radius;
        let c3_potential = (input.c3_potential[row] * left_weight * radius
            + input.c3_potential[next] * right_weight * next_radius)
            / half_radius;
        (potential, c3_potential)
    } else if input.start_index == 0 {
        let left_radius_sq = radius * radius;
        let right_radius_sq = next_radius * next_radius;
        let half_radius_sq = half_radius * half_radius;
        let left_potential =
            input.potential[row] - input.potential_coefficients[3] * left_radius_sq;
        let right_potential =
            input.potential[next] - input.potential_coefficients[3] * right_radius_sq;
        let potential = (left_potential * (next_radius - half_radius)
            + right_potential * (half_radius - radius))
            / interval
            + input.potential_coefficients[3] * half_radius_sq;
        let c3_potential = (input.c3_potential[row] * left_weight / left_radius_sq
            + input.c3_potential[next] * right_weight / right_radius_sq)
            * half_radius_sq;
        (potential, c3_potential)
    } else {
        (
            input.potential[row] * left_weight + input.potential[next] * right_weight,
            input.c3_potential[row] * left_weight + input.c3_potential[next] * right_weight,
        )
    };

    let large_exchange =
        input.large_exchange[row] * left_weight + input.large_exchange[next] * right_weight;
    let small_exchange =
        input.small_exchange[row] * left_weight + input.small_exchange[next] * right_weight;
    let f = (energy_over_light - potential) * half_radius;
    let g = f + ccl * half_radius;
    let c3 = fovrg_outward_c3(input.c3_scale, c3_potential, g)?;

    Ok(FovrgOutwardMidpoint {
        f,
        g,
        c3,
        large_exchange,
        small_exchange,
    })
}

pub(super) fn fovrg_outward_c3(
    c3_scale: i32,
    c3_potential: Complex,
    denominator: Complex,
) -> Result<Complex, FovrgError> {
    if c3_scale == 0 {
        Ok(Complex::new(0.0, 0.0))
    } else {
        validate_nonzero_complex_denominator("outward_c3_denominator", denominator)?;
        Ok((c3_scale as Real) * c3_potential / denominator.powi(2))
    }
}

pub(super) fn validate_outgoing_solution_input(
    input: &FovrgOutgoingSolutionInput<'_>,
) -> Result<(), FovrgError> {
    validate_count_at_least("active_len", input.active_len, 1)?;
    validate_count_at_least("coefficient_count", input.coefficient_count, 1)?;
    if input.c3_scale != 0 {
        validate_count_at_least("coefficient_count", input.coefficient_count, 3)?;
    }
    validate_active_len("potential", input.active_len, input.potential.len())?;
    validate_active_len(
        "large_exchange",
        input.active_len,
        input.large_exchange.len(),
    )?;
    validate_active_len(
        "small_exchange",
        input.active_len,
        input.small_exchange.len(),
    )?;
    validate_active_len("c3_potential", input.active_len, input.c3_potential.len())?;
    validate_active_len("radii", input.active_len, input.radii.len())?;
    validate_active_len(
        "potential_coefficients",
        input.coefficient_count.max(4),
        input.potential_coefficients.len(),
    )?;
    validate_active_len(
        "large_exchange_coefficients",
        input.coefficient_count.saturating_sub(1),
        input.large_exchange_coefficients.len(),
    )?;
    validate_active_len(
        "small_exchange_coefficients",
        input.coefficient_count.saturating_sub(1),
        input.small_exchange_coefficients.len(),
    )?;
    validate_nonzero_kappa("kappa", 0, input.kappa)?;
    validate_finite("origin_power", input.origin_power)?;
    validate_finite("muffin_tin_radius", input.muffin_tin_radius)?;
    validate_nonzero_finite("speed_of_light", input.speed_of_light)?;
    validate_nonzero_finite("step", input.step)?;
    validate_complex_input(
        "initial_large_coefficient",
        0,
        input.initial_large_coefficient,
    )?;
    validate_complex_input(
        "initial_small_coefficient",
        0,
        input.initial_small_coefficient,
    )?;
    validate_complex_input("energy", 0, input.energy)?;

    if input.radial_match_index >= input.active_len {
        return Err(FovrgError::ActiveCountOutOfRange {
            field: "radial_match_index",
            active_len: input.radial_match_index + 1,
            len: input.active_len,
        });
    }
    if input.wkb_index >= input.active_len {
        return Err(FovrgError::ActiveCountOutOfRange {
            field: "wkb_index",
            active_len: input.wkb_index + 1,
            len: input.active_len,
        });
    }
    if input.last_index >= input.active_len {
        return Err(FovrgError::ActiveCountOutOfRange {
            field: "last_index",
            active_len: input.last_index + 1,
            len: input.active_len,
        });
    }
    let flat_start_index = input.radial_match_index.min(input.wkb_index);
    if flat_start_index > input.last_index {
        return Err(FovrgError::InvalidRange {
            name: "outgoing_solution",
            start: flat_start_index,
            end: input.last_index,
        });
    }
    if input.wkb_index < input.last_index && input.wkb_index + 2 >= input.active_len {
        return Err(FovrgError::ActiveCountOutOfRange {
            field: "wkb_index",
            active_len: input.wkb_index + 3,
            len: input.active_len,
        });
    }

    for row in 0..input.active_len {
        validate_complex_input("potential", row, input.potential[row])?;
        validate_complex_input("large_exchange", row, input.large_exchange[row])?;
        validate_complex_input("small_exchange", row, input.small_exchange[row])?;
        validate_complex_input("c3_potential", row, input.c3_potential[row])?;
        validate_radius(row, input.radii[row])?;
    }
    for coefficient in 0..input.potential_coefficients.len() {
        validate_complex_input(
            "potential_coefficients",
            coefficient,
            input.potential_coefficients[coefficient],
        )?;
    }
    for coefficient in 0..input.large_exchange_coefficients.len() {
        validate_complex_input(
            "large_exchange_coefficients",
            coefficient,
            input.large_exchange_coefficients[coefficient],
        )?;
    }
    for coefficient in 0..input.small_exchange_coefficients.len() {
        validate_complex_input(
            "small_exchange_coefficients",
            coefficient,
            input.small_exchange_coefficients[coefficient],
        )?;
    }

    Ok(())
}

pub(super) fn fovrg_desclaux_origin_series(
    input: FovrgOutgoingSolutionInput<'_>,
    large_coefficients: &mut ComplexVec,
    small_coefficients: &mut ComplexVec,
    energy_over_light: Complex,
) -> Result<(), FovrgError> {
    let ccl = FEFF_ALPHA_INVERSE + FEFF_ALPHA_INVERSE;
    let kappa = input.kappa as Real;
    for coefficient in 1..input.coefficient_count {
        let k = coefficient as Real;
        let a = input.origin_power + kappa + k;
        let b = input.origin_power - kappa + k;
        let denominator = a * b + input.potential_coefficients[0].powi(2);
        validate_nonzero_complex_denominator("solout_desclaux_denominator", denominator)?;
        let mut f = (energy_over_light + ccl) * small_coefficients[coefficient - 1]
            + small_coefficients[coefficient];
        let mut g = energy_over_light * large_coefficients[coefficient - 1]
            + large_coefficients[coefficient];
        for term in 1..=coefficient {
            f -= input.potential_coefficients[term] * small_coefficients[coefficient - term];
            g -= input.potential_coefficients[term] * large_coefficients[coefficient - term];
        }
        large_coefficients[coefficient] =
            (b * f + input.potential_coefficients[0] * g) / denominator;
        small_coefficients[coefficient] =
            (input.potential_coefficients[0] * f - a * g) / denominator;
    }
    Ok(())
}

pub(super) fn fovrg_relativistic_origin_series(
    input: FovrgOutgoingSolutionInput<'_>,
    large_coefficients: &mut ComplexVec,
    small_coefficients: &mut ComplexVec,
    energy_over_light: Complex,
) -> Result<(), FovrgError> {
    let ccl = FEFF_ALPHA_INVERSE + FEFF_ALPHA_INVERSE;
    let speed_squared = ccl * ccl;
    let two_z = -input.potential_coefficients[0].re * 2.0 * input.speed_of_light;
    let il = if input.kappa > 0 {
        input.kappa + 1
    } else {
        -input.kappa
    } as Real;
    let l0 = il - 1.0;
    large_coefficients[0] = input.initial_large_coefficient;
    if two_z <= 0.0 {
        let denominator = 2.0 * il + 1.0;
        validate_nonzero_denominator("solout_relativistic_il_denominator", denominator)?;
        small_coefficients[0] =
            -energy_over_light / denominator * input.radii[0] * large_coefficients[0];
        large_coefficients[1] = Complex::new(0.0, 0.0);
        small_coefficients[1] = Complex::new(0.0, 0.0);
        large_coefficients[2] = Complex::new(0.0, 0.0);
        small_coefficients[2] = Complex::new(0.0, 0.0);
    } else {
        let rat1 = two_z / ccl;
        let rat2 = rat1 * rat1;
        let rat3 = speed_squared / two_z;
        validate_nonzero_denominator(
            "solout_relativistic_fl2_denominator",
            2.0 * input.origin_power + 1.0,
        )?;
        validate_nonzero_denominator(
            "solout_relativistic_fl1_denominator",
            input.origin_power + 1.0,
        )?;
        small_coefficients[0] = (input.origin_power - il) * rat3 * large_coefficients[0];
        large_coefficients[1] = (3.0 * input.origin_power - rat2)
            / (2.0 * input.origin_power + 1.0)
            * large_coefficients[0];
        small_coefficients[1] = rat3
            * ((input.origin_power - l0) * large_coefficients[1] - large_coefficients[0])
            - small_coefficients[0];
        large_coefficients[2] = ((input.origin_power + 3.0 * il) * large_coefficients[1]
            - 3.0 * l0 * large_coefficients[0]
            + (input.origin_power + il + 3.0) / rat3 * small_coefficients[1])
            / (input.origin_power + 1.0)
            / 4.0;
        small_coefficients[2] = (rat3
            * (2.0 * l0 * (input.origin_power + 2.0 - il) - l0 - rat2)
            * large_coefficients[1]
            - 3.0 * l0 * rat3 * (input.origin_power + 2.0 - il) * large_coefficients[0]
            + (input.origin_power + 3.0 - 2.0 * il - rat2) * small_coefficients[1])
            / (input.origin_power + 1.0)
            / 4.0;
        small_coefficients[0] /= ccl;
        large_coefficients[1] *= rat3;
        small_coefficients[1] *= rat3 / ccl;
        large_coefficients[2] *= rat3 * rat3;
        small_coefficients[2] *= rat3 * rat3 / ccl;
    }
    Ok(())
}

pub(super) fn fovrg_origin_components(
    input: FovrgOutgoingSolutionInput<'_>,
    large_coefficients: ArrayView1<'_, Complex>,
    small_coefficients: ArrayView1<'_, Complex>,
) -> (Complex, Complex) {
    if input.c3_scale == 0 {
        (0..input.coefficient_count).fold(
            (Complex::new(0.0, 0.0), Complex::new(0.0, 0.0)),
            |(large_sum, small_sum), coefficient| {
                let power = input.origin_power + coefficient as Real;
                let radius_power = input.radii[0].powf(power);
                (
                    large_sum + radius_power * large_coefficients[coefficient],
                    small_sum + radius_power * small_coefficients[coefficient],
                )
            },
        )
    } else {
        let radius = input.radii[0];
        let radius_power = radius.powf(input.origin_power);
        (
            radius_power
                * (large_coefficients[0]
                    + radius * (large_coefficients[1] + radius * large_coefficients[2])),
            radius_power
                * (small_coefficients[0]
                    + radius * (small_coefficients[1] + radius * small_coefficients[2])),
        )
    }
}

pub(super) fn fovrg_solout_average_potential(
    input: FovrgOutgoingSolutionInput<'_>,
    row: usize,
) -> Result<Complex, FovrgError> {
    let mut average_potential = if row == input.wkb_index {
        let extrapolated = input.speed_of_light
            * (3.0 * input.potential[input.wkb_index + 1] - input.potential[input.wkb_index + 2])
            / 2.0;
        if input.wkb_index + 1 == input.radial_match_index {
            input.speed_of_light * (input.potential[row] + input.potential[row + 1]) / 2.0
        } else {
            extrapolated
        }
    } else {
        input.speed_of_light * (input.potential[row] + input.potential[row + 1]) / 2.0
    };

    if input.c3_scale > 0 && row < input.radial_match_index {
        let radius_average = (input.radii[row] + input.radii[row + 1]) / 2.0;
        let relativistic = FEFF_ALPHA_INVERSE
            + FEFF_ALPHA_INVERSE
            + (input.energy - average_potential) / input.speed_of_light;
        let denominator = radius_average.powi(3) * relativistic.powi(2);
        validate_nonzero_complex_denominator("solout_c3_flat_denominator", denominator)?;
        average_potential += (input.c3_scale as Real) * input.speed_of_light / denominator
            * (input.c3_potential[row] + input.c3_potential[row + 1])
            / 2.0;
    }

    Ok(average_potential)
}

pub(super) fn validate_inward_solution_input(
    input: &FovrgInwardSolutionInput<'_>,
) -> Result<(), FovrgError> {
    validate_count_at_least("active_len", input.active_len, 1)?;
    validate_count_at_least("coefficient_count", input.coefficient_count, 1)?;
    validate_active_len("potential", input.active_len, input.potential.len())?;
    validate_active_len(
        "large_exchange",
        input.active_len,
        input.large_exchange.len(),
    )?;
    validate_active_len(
        "small_exchange",
        input.active_len,
        input.small_exchange.len(),
    )?;
    validate_active_len("c3_potential", input.active_len, input.c3_potential.len())?;
    validate_active_len("radii", input.active_len, input.radii.len())?;
    validate_nonzero_kappa("kappa", 0, input.kappa)?;
    validate_finite("origin_power", input.origin_power)?;
    validate_finite("muffin_tin_radius", input.muffin_tin_radius)?;
    validate_nonzero_finite("speed_of_light", input.speed_of_light)?;
    validate_nonzero_finite("step", input.step)?;
    validate_complex_input(
        "initial_large_coefficient",
        0,
        input.initial_large_coefficient,
    )?;
    validate_complex_input(
        "initial_small_coefficient",
        0,
        input.initial_small_coefficient,
    )?;
    validate_complex_input("energy", 0, input.energy)?;

    if input.radial_match_index >= input.active_len {
        return Err(FovrgError::ActiveCountOutOfRange {
            field: "radial_match_index",
            active_len: input.radial_match_index + 1,
            len: input.active_len,
        });
    }
    if input.radial_match_index + 1 >= input.active_len {
        return Err(FovrgError::ActiveCountOutOfRange {
            field: "radial_match_index",
            active_len: input.radial_match_index + 2,
            len: input.active_len,
        });
    }
    if input.wkb_index >= input.active_len {
        return Err(FovrgError::ActiveCountOutOfRange {
            field: "wkb_index",
            active_len: input.wkb_index + 1,
            len: input.active_len,
        });
    }
    if input.last_index >= input.active_len {
        return Err(FovrgError::ActiveCountOutOfRange {
            field: "last_index",
            active_len: input.last_index + 1,
            len: input.active_len,
        });
    }
    if input.radial_match_index > input.last_index {
        return Err(FovrgError::InvalidRange {
            name: "inward_solution",
            start: input.radial_match_index,
            end: input.last_index,
        });
    }
    if input.wkb_index < input.radial_match_index && input.wkb_index + 2 >= input.active_len {
        return Err(FovrgError::ActiveCountOutOfRange {
            field: "wkb_index",
            active_len: input.wkb_index + 3,
            len: input.active_len,
        });
    }
    let flat_start_index = input.radial_match_index.min(input.wkb_index);
    if flat_start_index > 0 {
        let history_rows = input.last_index - flat_start_index + 1;
        validate_count_at_least("inward_history_rows", history_rows, FOVRG_INT_OUT_HISTORY)?;
    }

    for row in 0..input.active_len {
        validate_complex_input("potential", row, input.potential[row])?;
        validate_complex_input("large_exchange", row, input.large_exchange[row])?;
        validate_complex_input("small_exchange", row, input.small_exchange[row])?;
        validate_complex_input("c3_potential", row, input.c3_potential[row])?;
        validate_radius(row, input.radii[row])?;
    }

    Ok(())
}

pub(super) fn fovrg_solin_average_potential(
    input: FovrgInwardSolutionInput<'_>,
    row: usize,
) -> Result<Complex, FovrgError> {
    let average_potential = if row == input.wkb_index {
        let extrapolated = input.speed_of_light
            * (3.0 * input.potential[input.wkb_index + 1] - input.potential[input.wkb_index + 2])
            / 2.0;
        if input.wkb_index + 1 == input.radial_match_index {
            input.speed_of_light * (input.potential[row] + input.potential[row + 1]) / 2.0
        } else {
            extrapolated
        }
    } else {
        input.speed_of_light * (input.potential[row] + input.potential[row + 1]) / 2.0
    };
    validate_complex_result("solin_average_potential", row, average_potential)?;
    Ok(average_potential)
}

pub(super) fn fovrg_inward_history_slot(flat_start_index: usize, row: usize) -> Option<usize> {
    let slot = flat_start_index + FOVRG_INT_OUT_HISTORY - 1;
    if row <= slot { Some(slot - row) } else { None }
}

pub(super) fn fovrg_inward_derivatives(
    input: &FovrgInwardSolutionInput<'_>,
    row: usize,
    energy_term: Complex,
    ccl: Real,
    include_exchange: bool,
    large_component: Complex,
    small_component: Complex,
) -> Result<(Complex, Complex), FovrgError> {
    let f = (energy_term - input.potential[row]) * input.radii[row];
    let g = f + ccl * input.radii[row];
    let c3 = fovrg_outward_c3(input.c3_scale, input.c3_potential[row], g)?;
    let (large_exchange, small_exchange) = if include_exchange {
        (input.large_exchange[row], input.small_exchange[row])
    } else {
        (Complex::new(0.0, 0.0), Complex::new(0.0, 0.0))
    };
    let kappa = input.kappa as Real;
    let large_derivative = -(g * small_component - kappa * large_component + small_exchange);
    let small_derivative = -(kappa * small_component - (f - c3) * large_component - large_exchange);
    validate_complex_result("inward_large_derivative", row, large_derivative)?;
    validate_complex_result("inward_small_derivative", row, small_derivative)?;
    Ok((large_derivative, small_derivative))
}

pub(super) fn validate_count_at_least(
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

pub(super) fn target_j_value(kappa: i32) -> usize {
    2 * kappa.unsigned_abs() as usize - 1
}

pub(super) fn exchange_coefficient_start(
    multipole: usize,
    bound_kappa: i32,
    target_kappa: i32,
    target_power: Real,
) -> Option<usize> {
    let bound_abs = i64::from(bound_kappa.unsigned_abs());
    let target_abs = i64::from(target_kappa.unsigned_abs());
    let multipole = multipole as i64;
    let start = if target_power < 0.0 {
        multipole + 1 + bound_abs + target_abs
    } else {
        multipole + 1 + bound_abs - target_abs
    };
    (start >= 1).then_some(start as usize)
}

pub(super) fn validate_active_len(
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

pub(super) fn validate_matrix_rows(
    field: &'static str,
    required: usize,
    available: usize,
) -> Result<(), FovrgError> {
    if required > available {
        Err(FovrgError::ActiveCountOutOfRange {
            field,
            active_len: required,
            len: available,
        })
    } else {
        Ok(())
    }
}

pub(super) fn validate_matrix_cols(
    field: &'static str,
    required: usize,
    available: usize,
) -> Result<(), FovrgError> {
    if required > available {
        Err(FovrgError::ActiveCountOutOfRange {
            field,
            active_len: required,
            len: available,
        })
    } else {
        Ok(())
    }
}

pub(super) fn fovrg_usize_to_i32(name: &'static str, value: usize) -> Result<i32, FovrgError> {
    i32::try_from(value).map_err(|_| FovrgError::CountTooLarge {
        name,
        actual: value,
        maximum: i32::MAX as usize,
    })
}

pub(super) fn validate_finite(name: &'static str, value: Real) -> Result<(), FovrgError> {
    if value.is_finite() {
        Ok(())
    } else {
        Err(FovrgError::NonFiniteInput { name, value })
    }
}

pub(super) fn validate_nonzero_finite(name: &'static str, value: Real) -> Result<(), FovrgError> {
    if !value.is_finite() {
        Err(FovrgError::NonFiniteInput { name, value })
    } else if value == 0.0 {
        Err(FovrgError::ZeroInput { name })
    } else {
        Ok(())
    }
}

pub(super) fn validate_positive_finite(name: &'static str, value: Real) -> Result<(), FovrgError> {
    if !value.is_finite() {
        Err(FovrgError::NonFiniteInput { name, value })
    } else if value <= 0.0 {
        Err(FovrgError::NonPositiveInput { name, value })
    } else {
        Ok(())
    }
}

pub(super) fn validate_nonzero_denominator(
    name: &'static str,
    value: Real,
) -> Result<(), FovrgError> {
    if value == 0.0 {
        Err(FovrgError::ZeroDenominator { name })
    } else {
        Ok(())
    }
}

pub(super) fn validate_nonzero_complex_denominator(
    name: &'static str,
    value: Complex,
) -> Result<(), FovrgError> {
    if value == Complex::new(0.0, 0.0) {
        Err(FovrgError::ZeroDenominator { name })
    } else {
        Ok(())
    }
}

pub(super) fn validate_radius(row: usize, value: Real) -> Result<(), FovrgError> {
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

pub(super) fn validate_nonzero_kappa(
    name: &'static str,
    row: usize,
    value: i32,
) -> Result<(), FovrgError> {
    if value == 0 {
        Err(FovrgError::InvalidQuantumNumber { name, row, value })
    } else {
        Ok(())
    }
}

pub(super) fn validate_real_input(
    name: &'static str,
    row: usize,
    value: Real,
) -> Result<(), FovrgError> {
    if value.is_finite() {
        Ok(())
    } else {
        Err(FovrgError::NonFiniteRealInput { name, row, value })
    }
}

pub(super) fn validate_real_result(
    name: &'static str,
    row: usize,
    value: Real,
) -> Result<(), FovrgError> {
    if value.is_finite() {
        Ok(())
    } else {
        Err(FovrgError::NonFiniteRealInput { name, row, value })
    }
}

pub(super) fn validate_complex_input(
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

pub(super) fn validate_potential(row: usize, value: Complex) -> Result<(), FovrgError> {
    if value.re.is_finite() && value.im.is_finite() {
        Ok(())
    } else {
        Err(FovrgError::NonFinitePotential { row, value })
    }
}

pub(super) fn complex_real_product_coefficient(
    complex_coefficients: ArrayView1<'_, Complex>,
    real_coefficients: ArrayView1<'_, Real>,
    count: usize,
) -> Complex {
    (0..count).fold(Complex::new(0.0, 0.0), |sum, index| {
        sum + complex_coefficients[index] * real_coefficients[count - 1 - index]
    })
}

pub(super) fn real_product_coefficient(
    left_coefficients: ArrayView1<'_, Real>,
    right_coefficients: ArrayView1<'_, Real>,
    count: usize,
) -> Real {
    (0..count).fold(0.0, |sum, index| {
        sum + left_coefficients[index] * right_coefficients[count - 1 - index]
    })
}

pub(super) fn validate_complex_result(
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
