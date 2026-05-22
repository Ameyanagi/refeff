use super::*;

/// Port of FEFF `compton_jzzp`: integrate `rho(r,r')` over the xy plane.
///
/// The callback supplies FEFF's `rhorrp(v, vp)` density matrix value after any
/// q-axis-to-cluster rotation has been applied. The returned matrix has shape
/// `(nz, nzp)` in Fortran-order storage, matching FEFF's `jzzp(gr%nz,gr%nzp)`.
pub fn compton_jzzp<F>(grid: &ComptonGrid, mut density: F) -> Result<RealMat, ComptonError>
where
    F: FnMut(Vector3, Vector3) -> Result<Real, ComptonError>,
{
    validate_grid_for_jzzp(grid)?;
    let rotation = grid.rotation_matrix.view();
    let mut jzzp = Array2::zeros((grid.nz(), grid.nzp()).f());

    for izp in 0..grid.nzp() {
        let zp = grid.zp[izp];
        for iz in 0..grid.nz() {
            let z = grid.z[iz];
            let mut previous_s_integral = 0.0;

            for is in 0..grid.ns() {
                let s = grid.s[is];
                let mut phi_integral = 0.0;
                let mut previous_rho = 0.0;

                for iphi in 0..grid.nphi() {
                    let phi = grid.phi[iphi];
                    let (sin_phi, cos_phi) = phi.sin_cos();
                    let x = s * cos_phi;
                    let y = s * sin_phi;
                    let mut r = [x, y, z];
                    let mut rp = [x, y, zp];
                    if grid.rotate {
                        r = rotate_vector_checked_shape(rotation, r);
                        rp = rotate_vector_checked_shape(rotation, rp);
                    }

                    let mut rho = density(r, rp)?;
                    if !rho.is_finite() {
                        return Err(ComptonError::NonFiniteDensity { value: rho });
                    }
                    rho *= s;

                    if iphi > 0 {
                        let dphi = grid.phi[iphi] - grid.phi[iphi - 1];
                        phi_integral += (previous_rho + rho) * 0.5 * dphi / std::f64::consts::TAU;
                    }
                    previous_rho = rho;
                }

                if is > 0 {
                    let ds = grid.s[is] - grid.s[is - 1];
                    jzzp[(iz, izp)] += (phi_integral + previous_s_integral) * 0.5 * ds;
                }
                previous_s_integral = phi_integral;
            }
        }
    }

    validate_matrix_finite("jzzp", jzzp.view())?;
    Ok(jzzp)
}

/// Port of FEFF `calculate_rhozzp`: build the `rhozzp.dat` diagnostic slice.
///
/// FEFF evaluates `rho(r,r')` at fixed `r = (0, 0, 0.01)` while scanning
/// `r' = (0, 0, z')` from `0.01` through `0.01 + zpmax`, rotating both
/// vectors into cluster coordinates when the COMPTON grid requests it. This
/// helper preserves that calculation but lets callers choose the sample count
/// for tests and diagnostics.
pub fn compton_rhozzp_slice<F>(
    grid: &ComptonGrid,
    input: ComptonRhoZzpInput,
    mut density: F,
) -> Result<ComptonRhoZzpSlice, ComptonError>
where
    F: FnMut(Vector3, Vector3) -> Result<Real, ComptonError>,
{
    validate_grid_count("rhozzp_samples", input.sample_count)?;
    validate_finite("base_z", input.base_z)?;
    validate_grid_for_rhozzp(grid)?;

    let z_step = grid.zp[grid.nzp() - 1] / (input.sample_count as Real - 1.0);
    let rotation = grid.rotation_matrix.view();
    let mut z_prime = Array1::zeros(input.sample_count);
    let mut rho = Array1::zeros(input.sample_count);

    for index in 0..input.sample_count {
        let zp = input.base_z + z_step * index as Real;
        let mut r = [0.0, 0.0, input.base_z];
        let mut rp = [0.0, 0.0, zp];
        if grid.rotate {
            r = rotate_vector_checked_shape(rotation, r);
            rp = rotate_vector_checked_shape(rotation, rp);
        }

        let value = density(r, rp)?;
        if !value.is_finite() {
            return Err(ComptonError::NonFiniteDensity { value });
        }
        z_prime[index] = zp;
        rho[index] = value;
    }

    validate_real_vec("rhozzp_z_prime", &z_prime)?;
    validate_real_vec("rhozzp_rho", &rho)?;
    Ok(ComptonRhoZzpSlice { z_prime, rho })
}
