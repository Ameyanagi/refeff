use super::validation::*;
use super::*;

/// Integrate a radial density into a Coulomb potential using FEFF `potslw`.
///
/// This ports `ATOM/potslw.f90`, a four-point integration stencil used by the
/// potential module's Coulomb update. FEFF only defines values through `np`; the
/// Rust result preserves the caller's grid length and zero-fills the inactive
/// tail.
pub fn coulomb_potential_slw(
    input: CoulombPotentialSlwInput<'_>,
) -> Result<CoulombPotentialSlw, GridError> {
    validate_delta(input.delta)?;

    let density_len = input.density.len();
    let radii_len = input.radii.len();
    if density_len != radii_len {
        return Err(GridError::CoulombLengthMismatch {
            density_len,
            radii_len,
        });
    }
    validate_positive_grid_length("density", density_len)?;
    validate_source_len_at_least("active", input.active_len, 3)?;
    ensure_source_length("density", input.active_len, density_len)?;
    validate_component_prefix_values("density", input.density, input.active_len)?;
    validate_positive_radii(input.radii, input.active_len)?;

    let mut potential = Array1::<Real>::zeros(density_len);
    let mut work = Array1::<Real>::zeros(density_len);
    let scale = input.delta / 24.0;
    for index in 0..input.active_len {
        potential[index] = input.density[index] * input.radii[index];
    }

    let grid_ratio = input.delta.exp();
    let grid_ratio2 = grid_ratio * grid_ratio;
    work[1] = input.radii[0] * (input.density[1] - input.density[0] * grid_ratio2)
        / (12.0 * (grid_ratio - 1.0));
    work[0] = potential[0] / 3.0 - work[1] / grid_ratio2;
    work[1] = potential[1] / 3.0 - work[1] * grid_ratio2;

    let last_inner = input.active_len - 2;
    for index in 2..=last_inner {
        work[index] = work[index - 1]
            + scale
                * (13.0 * (potential[index] + potential[index - 1])
                    - (potential[index - 2] + potential[index + 1]));
    }

    work[input.active_len - 1] = work[last_inner];
    potential[last_inner] = work[last_inner];
    potential[input.active_len - 1] = work[last_inner];
    for fortran_i in 3..=last_inner + 1 {
        let index = input.active_len - fortran_i;
        potential[index] = potential[index + 1] / grid_ratio
            + scale
                * (13.0 * (work[index + 1] / grid_ratio + work[index])
                    - (work[index + 2] / grid_ratio2 + work[index - 1] * grid_ratio));
    }
    potential[0] = potential[2] / grid_ratio2
        + input.delta * (work[0] + 4.0 * work[1] / grid_ratio + work[2] / grid_ratio2) / 3.0;

    potential
        .iter_mut()
        .zip(input.radii.iter())
        .take(input.active_len)
        .for_each(|(potential, radius)| *potential /= radius);

    Ok(CoulombPotentialSlw {
        potential,
        active_len: input.active_len,
    })
}

/// Build FEFF's SCMT complex-energy contour from `ecv` to `xmu`.
///
/// This ports `POT/grids.f90`. FEFF first creates a short vertical line above
/// `ecv`, then a real-axis bridge that retains the initial imaginary part, and
/// finally a descending set of points above `xmu`. The Rust version preserves
/// FEFF's count and rounding rules while validating that the caller-provided
/// table sizes are large enough.
pub fn scmt_energy_grid(input: ScmtEnergyGridInput) -> Result<ScmtEnergyGrid, GridError> {
    validate_finite_scalar("core_valence_energy", input.core_valence_energy)?;
    validate_finite_scalar("fermi_energy", input.fermi_energy)?;
    let energy_span = input.fermi_energy - input.core_valence_energy;
    validate_finite_scalar("energy_span", energy_span)?;
    validate_positive_grid_length("energy", input.max_points)?;
    validate_positive_grid_length("step", input.step_count)?;

    let lower_imaginary_count = input
        .step_count
        .checked_add(1)
        .ok_or(GridError::GridLengthOverflow { name: "step" })?
        / 2;
    let upper_imaginary_count = input.step_count - 1;
    let minimum_points = lower_imaginary_count
        .checked_mul(2)
        .and_then(|value| value.checked_add(upper_imaginary_count))
        .ok_or(GridError::OutputGridTooShort {
            required: usize::MAX,
            available: input.max_points,
        })?;
    if input.max_points < minimum_points {
        return Err(GridError::OutputGridTooShort {
            required: minimum_points,
            available: input.max_points,
        });
    }

    let real_axis_max = input.max_points - lower_imaginary_count - upper_imaginary_count;
    let minimum_imaginary = 0.05 / FEFF_HARTREE_EV;
    let mut energies = Array1::<Complex>::zeros(input.max_points);
    let mut steps = Array1::<Real>::zeros(input.step_count);

    for index in 1..=lower_imaginary_count {
        let imaginary = minimum_imaginary * square_index_as_real("step", index)?;
        energies[index - 1] = Complex::new(input.core_valence_energy, imaginary);
    }
    steps[input.step_count - 1] = energies[lower_imaginary_count - 1].im / 4.0;

    let bridge_step_guess = energies[lower_imaginary_count - 1].im / 4.0;
    let rounded_bridge_points = (energy_span / bridge_step_guess).round();
    let mut real_axis_count = if rounded_bridge_points <= 0.0 {
        0
    } else if rounded_bridge_points >= real_axis_max as Real {
        real_axis_max
    } else {
        rounded_bridge_points as usize
    };
    if real_axis_count < lower_imaginary_count {
        real_axis_count = lower_imaginary_count;
    }

    let real_step = energy_span / real_axis_count as Real;
    for index in lower_imaginary_count + 1..=lower_imaginary_count + real_axis_count {
        energies[index - 1] = energies[index - 2] + Complex::new(real_step, 0.0);
    }

    let active_len = lower_imaginary_count + real_axis_count + upper_imaginary_count;
    for index in 1..=upper_imaginary_count {
        let imaginary = minimum_imaginary * square_index_as_real("step", index + 1)? / 4.0;
        steps[index - 1] = imaginary / 4.0;
        energies[active_len - index] = Complex::new(input.fermi_energy, imaginary);
    }

    Ok(ScmtEnergyGrid {
        energies,
        steps,
        active_len,
        lower_imaginary_count,
        real_axis_count,
        upper_imaginary_count,
    })
}
