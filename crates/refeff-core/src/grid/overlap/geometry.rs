use super::super::validation::*;
use super::super::*;

/// Volume of one FEFF spherical-overlap cap from `POT/istprm.f90` `calcvl`.
///
/// `sphere_radius` is the radius of the sphere whose cap is being measured,
/// `other_radius` is the radius of the overlapping sphere, and
/// `center_distance` is the distance between sphere centers. FEFF callers use
/// this only after confirming the spheres overlap; this function preserves the
/// algebraic `calcvl` formula and validates only finite positive inputs.
pub fn sphere_overlap_cap_volume(
    sphere_radius: Real,
    other_radius: Real,
    center_distance: Real,
) -> Result<Real, GridError> {
    validate_positive_finite_scalar("sphere_radius", sphere_radius)?;
    validate_positive_finite_scalar("other_radius", other_radius)?;
    validate_positive_finite_scalar("center_distance", center_distance)?;

    let plane_distance = (sphere_radius.powi(2) - other_radius.powi(2) + center_distance.powi(2))
        / (2.0 * center_distance);
    let cap_height = sphere_radius - plane_distance;
    let volume = PI / 3.0 * cap_height.powi(2) * (3.0 * sphere_radius - cap_height);
    validate_finite_scalar("sphere_overlap_cap_volume", volume)?;
    Ok(volume)
}

/// Total lens volume of two overlapping spheres using FEFF `calcvl` caps.
pub fn sphere_overlap_lens_volume(
    radius_a: Real,
    radius_b: Real,
    center_distance: Real,
) -> Result<Real, GridError> {
    Ok(
        sphere_overlap_cap_volume(radius_a, radius_b, center_distance)?
            + sphere_overlap_cap_volume(radius_b, radius_a, center_distance)?,
    )
}
