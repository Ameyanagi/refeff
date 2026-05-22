use super::super::radial::{
    fortran_truncated_index, loucks_index_below, loucks_x, radial_index_below, sumax_literal_x,
};
use super::super::validation::*;
use super::super::*;

/// Add one FEFF `sumax` spherical overlap contribution on the Loucks grid.
///
/// This ports `POT/sumax.f90`, used by FEFF's overlapped potential/density
/// setup. The input and accumulated arrays use the fixed Loucks spacing
/// `delta = 0.05`; only grid points through the neighbor distance are updated,
/// matching FEFF's `jtop = ii(rn)` behavior.
pub fn sum_loucks_spherical_overlap(
    input: LoucksSphericalOverlapInput<'_>,
) -> Result<LoucksSphericalOverlap, GridError> {
    if !(input.neighbor_distance.is_finite() && input.neighbor_distance > 0.0) {
        return Err(GridError::InvalidRadius {
            radius: input.neighbor_distance,
        });
    }
    validate_finite_scalar("multiplicity", input.multiplicity)?;

    let source_len = input.source.len();
    let accumulated_len = input.accumulated.len();
    if source_len != accumulated_len {
        return Err(GridError::OverlapLengthMismatch {
            source_len,
            accumulated_len,
        });
    }
    validate_positive_grid_length("source", source_len)?;
    validate_component_values("source", input.source)?;
    validate_component_values("accumulated", input.accumulated)?;

    let cutoff_index = loucks_index_below(SUMAX_WIGNER_SEITZ_RADIUS)?;
    let active_len = loucks_index_below(input.neighbor_distance)?;
    ensure_source_length("source", cutoff_index, source_len)?;
    ensure_source_length("accumulated", active_len, accumulated_len)?;

    let source = input.source.iter().copied().collect::<Vec<_>>();
    let mut accumulated = input.accumulated.iter().copied().collect::<Array1<_>>();
    if active_len == 0 {
        return Ok(LoucksSphericalOverlap {
            accumulated,
            active_len,
        });
    }

    let top_x = loucks_x(cutoff_index);

    for index in 1..=active_len {
        let x = loucks_x(index);
        let radius = x.exp();
        let contribution = sumax_integral_contribution(
            input.neighbor_distance,
            input.multiplicity,
            &source,
            top_x,
            radius,
        )?;
        accumulated[index - 1] += contribution;
    }

    Ok(LoucksSphericalOverlap {
        accumulated,
        active_len,
    })
}

fn sumax_integral_contribution(
    neighbor_distance: Real,
    multiplicity: Real,
    source: &[Real],
    top_x: Real,
    radius: Real,
) -> Result<Real, GridError> {
    let lower_radius = neighbor_distance - radius;
    if lower_radius <= 0.0 {
        return Ok(0.0);
    }

    let lower_x = lower_radius.ln();
    if lower_x >= top_x {
        return Ok(0.0);
    }

    let mut integral = 0.0;
    let mut lower_index =
        fortran_truncated_index(2.0 + 20.0 * (lower_x + SUMAX_LITERAL_OFFSET)).max(1);
    let mut lower_grid_x = sumax_literal_x(lower_index);
    if lower_index >= 2 {
        let cap_width = lower_grid_x - lower_x;
        let lower_value = source_value(source, lower_index, "source")?;
        let previous_value = source_value(source, lower_index - 1, "source")?;
        integral += 0.5
            * cap_width
            * (lower_value * (2.0 - 20.0 * cap_width) * (2.0 * lower_grid_x).exp()
                + 20.0
                    * cap_width
                    * previous_value
                    * (2.0 * (lower_grid_x - SUMAX_LITERAL_DELTA)).exp());
    }

    let upper_x = (neighbor_distance + radius).ln();
    let upper_index = if upper_x >= top_x {
        radial_index_below(SUMAX_WIGNER_SEITZ_RADIUS, LOUCKS_DELTA)?
    } else {
        let index = fortran_truncated_index(1.0 + 20.0 * (upper_x + SUMAX_LITERAL_OFFSET));
        if index < lower_index {
            let near_zero = source_value(source, index, "source")?
                * (2.0 * (lower_grid_x - SUMAX_LITERAL_DELTA)).exp();
            let lower_value =
                source_value(source, lower_index, "source")? * (2.0 * lower_grid_x).exp();
            let upper_interp = near_zero
                + 20.0 * (lower_value - near_zero) * (upper_x - lower_grid_x + SUMAX_LITERAL_DELTA);
            let lower_interp = near_zero
                + 20.0 * (lower_value - near_zero) * (lower_x - lower_grid_x + SUMAX_LITERAL_DELTA);
            integral = 0.5 * (lower_interp + upper_interp) * (upper_x - lower_x);
            return Ok(0.5 * integral * multiplicity / (neighbor_distance * radius));
        }

        let upper_grid_x = sumax_literal_x(index);
        let cap_width = upper_x - upper_grid_x;
        let upper_value = source_value(source, index, "source")?;
        let next_value = source_value(source, index + 1, "source")?;
        integral += 0.5
            * cap_width
            * (upper_value * (2.0 - 20.0 * cap_width) * (2.0 * upper_grid_x).exp()
                + next_value
                    * 20.0
                    * cap_width
                    * (2.0 * (upper_grid_x + SUMAX_LITERAL_DELTA)).exp());
        index
    };

    while upper_index > lower_index {
        let current = source_value(source, lower_index, "source")? * (2.0 * lower_grid_x).exp();
        let next = source_value(source, lower_index + 1, "source")?
            * (2.0 * (lower_grid_x + SUMAX_LITERAL_DELTA)).exp();
        integral += 0.5 * (current + next) * SUMAX_LITERAL_DELTA;
        lower_index += 1;
        if lower_index < upper_index {
            lower_grid_x += SUMAX_LITERAL_DELTA;
        }
    }

    Ok(0.5 * integral * multiplicity / (neighbor_distance * radius))
}
