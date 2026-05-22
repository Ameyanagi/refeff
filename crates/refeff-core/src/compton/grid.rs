use super::*;

/// Port of FEFF `compton_build_grid`.
///
/// `smax` and `zmax` use `norman_radius` when supplied as zero, matching the
/// Fortran assignments from `rnrm(0)`. The returned grid contains the FEFF
/// q-axis rotation matrix or identity when the rotation angle is negligible.
pub fn compton_build_grid(input: ComptonGridInput) -> Result<ComptonGrid, ComptonError> {
    validate_grid_count("ns", input.ns)?;
    validate_grid_count("nphi", input.nphi)?;
    validate_grid_count("nz", input.nz)?;
    validate_grid_count("nzp", input.nzp)?;
    validate_extent("smax", input.smax)?;
    validate_extent("phimax", input.phimax)?;
    validate_extent("zmax", input.zmax)?;
    validate_extent("zpmax", input.zpmax)?;
    validate_vector("qhat", input.qhat)?;

    let smax = default_extent(input.smax, input.norman_radius, "smax", "norman_radius")?;
    let zmax = default_extent(input.zmax, input.norman_radius, "zmax", "norman_radius")?;
    let s = linspace(0.0, smax, input.ns);
    let phi = linspace(0.0, input.phimax, input.nphi);
    let z = linspace(-zmax, zmax, input.nz);
    let zp = linspace(-input.zpmax, input.zpmax, input.nzp);

    let rotation_axis = compton_rotation_axis_angle([0.0, 0.0, 1.0], input.qhat)?;
    let (rotate, rotation_matrix) = if rotation_axis.theta.abs() > COMPTON_ROTATION_TOLERANCE {
        (
            true,
            compton_rotation_matrix(rotation_axis.axis, rotation_axis.theta)?,
        )
    } else {
        (false, identity_rotation_matrix())
    };

    Ok(ComptonGrid {
        s,
        phi,
        z,
        zp,
        rotate,
        rotation_matrix,
    })
}
