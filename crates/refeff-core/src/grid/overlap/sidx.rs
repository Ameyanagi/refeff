use super::super::radial::{feff_legacy_loucks_index_below, feff_legacy_loucks_radius};
use super::super::validation::*;
use super::super::*;

/// Locate FEFF overlapped-density tail indices and adjust radii when needed.
///
/// This ports the defined behavior of `POT/sidx.f90`. FEFF scans `rholap`
/// from `imt = ii(rmt)` until the first value at or below `1.0e-5`, then moves
/// the Norman radius inward if its index lies beyond the last positive-density
/// point. The original Fortran leaves `imax` undefined when the first scanned
/// density value is already below cutoff; Rust reports that case as
/// [`GridError::NoActiveDensityTail`].
pub fn overlap_density_indices(
    input: OverlapDensityIndicesInput<'_>,
) -> Result<OverlapDensityIndices, GridError> {
    if !(input.muffin_tin_radius.is_finite() && input.muffin_tin_radius > 0.0) {
        return Err(GridError::InvalidRadius {
            radius: input.muffin_tin_radius,
        });
    }
    if !(input.norman_radius.is_finite() && input.norman_radius > 0.0) {
        return Err(GridError::InvalidRadius {
            radius: input.norman_radius,
        });
    }
    validate_positive_grid_length("overlapped_density", input.overlapped_density.len())?;
    validate_component_values("overlapped_density", input.overlapped_density)?;

    let muffin_tin_index = feff_legacy_loucks_index_below(input.muffin_tin_radius)?;
    let initial_norman_index = feff_legacy_loucks_index_below(input.norman_radius)?;
    validate_grid_index("muffin_tin", muffin_tin_index)?;
    validate_grid_index("norman", initial_norman_index)?;
    ensure_source_length(
        "overlapped_density",
        muffin_tin_index,
        input.overlapped_density.len(),
    )?;

    let mut max_density_index = None;
    for index in muffin_tin_index..=input.overlapped_density.len() {
        if view_value(input.overlapped_density, index, "overlapped_density")? <= SIDX_DENSITY_CUTOFF
        {
            break;
        }
        max_density_index = Some(index);
    }
    let max_density_index = max_density_index.ok_or(GridError::NoActiveDensityTail {
        start_index: muffin_tin_index,
        threshold: SIDX_DENSITY_CUTOFF,
    })?;

    let (norman_index, norman_radius, moved_norman_radius) =
        if initial_norman_index > max_density_index {
            (
                max_density_index,
                feff_legacy_loucks_radius(max_density_index),
                true,
            )
        } else {
            (initial_norman_index, input.norman_radius, false)
        };

    Ok(OverlapDensityIndices {
        max_density_index,
        muffin_tin_index,
        norman_index,
        muffin_tin_radius: input.muffin_tin_radius,
        norman_radius,
        moved_norman_radius,
    })
}
