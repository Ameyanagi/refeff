use super::super::radial::loucks_x;
use super::super::validation::*;
use super::super::*;

/// Average FEFF potential and overlapped density over an interstitial shell.
///
/// This ports `POT/istval.f90`. FEFF integrates `r**3 * value` over the
/// logarithmic Loucks coordinate and divides by `(rws**3 - rmt**3) / 3`, leaving
/// out the common `4*pi` factor in both the integral and the shell volume.
pub fn interstitial_shell_values(
    input: InterstitialShellValuesInput<'_>,
) -> Result<InterstitialShellValues, GridError> {
    if !(input.muffin_tin_radius.is_finite() && input.muffin_tin_radius > 0.0) {
        return Err(GridError::InvalidRadius {
            radius: input.muffin_tin_radius,
        });
    }
    if !(input.wigner_seitz_radius.is_finite() && input.wigner_seitz_radius > 0.0) {
        return Err(GridError::InvalidRadius {
            radius: input.wigner_seitz_radius,
        });
    }
    if input.wigner_seitz_radius <= input.muffin_tin_radius {
        return Err(GridError::InvalidRadiusOrder {
            inner_radius: input.muffin_tin_radius,
            outer_radius: input.wigner_seitz_radius,
        });
    }
    validate_grid_index("muffin_tin", input.muffin_tin_index)?;
    validate_grid_index("wigner_seitz", input.wigner_seitz_index)?;
    if input.wigner_seitz_index < input.muffin_tin_index {
        return Err(GridError::InvalidGridIndexRange {
            lower_index: input.muffin_tin_index,
            upper_index: input.wigner_seitz_index,
        });
    }
    validate_positive_grid_length("total_potential", input.total_potential.len())?;
    validate_positive_grid_length("overlapped_density", input.overlapped_density.len())?;
    validate_component_values("total_potential", input.total_potential)?;
    validate_component_values("overlapped_density", input.overlapped_density)?;

    let required =
        input
            .wigner_seitz_index
            .checked_add(1)
            .ok_or(GridError::GridLengthOverflow {
                name: "interstitial",
            })?;
    ensure_source_length("total_potential", required, input.total_potential.len())?;
    ensure_source_length(
        "overlapped_density",
        required,
        input.overlapped_density.len(),
    )?;

    let shell_volume = (input.wigner_seitz_radius.powi(3) - input.muffin_tin_radius.powi(3)) / 3.0;
    let potential_integral = interstitial_shell_integral(
        input.total_potential,
        input.muffin_tin_radius,
        input.muffin_tin_index,
        input.wigner_seitz_radius,
        input.wigner_seitz_index,
    )?;
    let density_integral = interstitial_shell_integral(
        input.overlapped_density,
        input.muffin_tin_radius,
        input.muffin_tin_index,
        input.wigner_seitz_radius,
        input.wigner_seitz_index,
    )?;

    Ok(InterstitialShellValues {
        interstitial_potential: potential_integral / shell_volume,
        interstitial_density: density_integral / shell_volume,
        shell_volume,
    })
}

fn interstitial_shell_integral(
    values: ArrayView1<'_, Real>,
    muffin_tin_radius: Real,
    muffin_tin_index: usize,
    wigner_seitz_radius: Real,
    wigner_seitz_index: usize,
) -> Result<Real, GridError> {
    let trapezoid_sum = (muffin_tin_index..wigner_seitz_index).try_fold(0.0, |sum, index| {
        let right = radius_cubed_grid_value(values, index + 1, "grid")?;
        let left = radius_cubed_grid_value(values, index, "grid")?;
        Ok::<_, GridError>(sum + 0.5 * (right + left) * LOUCKS_DELTA)
    })?;
    let upper_cap = interstitial_shell_cap(values, wigner_seitz_radius, wigner_seitz_index)?;
    let lower_cap = interstitial_shell_cap(values, muffin_tin_radius, muffin_tin_index)?;
    Ok(trapezoid_sum + upper_cap - lower_cap)
}

fn interstitial_shell_cap(
    values: ArrayView1<'_, Real>,
    radius: Real,
    index: usize,
) -> Result<Real, GridError> {
    let cap_width = radius.ln() - loucks_x(index);
    let ratio = cap_width / LOUCKS_DELTA;
    let left = radius_cubed_grid_value(values, index, "grid")?;
    let right = radius_cubed_grid_value(values, index + 1, "grid")?;
    Ok(0.5 * cap_width * ((2.0 - ratio) * left + ratio * right))
}
