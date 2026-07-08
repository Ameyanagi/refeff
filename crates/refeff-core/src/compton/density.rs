use super::*;
use crate::rhorrp::{RhorrpPointPairDensityInput, rhorrp_point_pair_density};

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

/// Port of the FEFF COMPTON-to-RHORRP `jzzp.dat` callback flow.
///
/// FEFF computes each `rho(r,r')` sample by calling RHORRP's density-matrix
/// evaluator before integrating those values over the COMPTON `s,phi` plane.
/// This helper keeps that handoff in one typed operation once callers have
/// supplied the RHORRP wavefunction, phase, and FMS matrices.
pub fn compton_jzzp_from_rhorrp(
    grid: &ComptonGrid,
    density_input: ComptonRhorrpDensityInput<'_>,
) -> Result<RealMat, ComptonError> {
    compton_jzzp(grid, |first_point, second_point| {
        rhorrp_density_sample(density_input, first_point, second_point)
    })
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

/// Port of the FEFF COMPTON-to-RHORRP `rhozzp.dat` diagnostic callback flow.
///
/// This mirrors [`compton_rhozzp_slice`], but evaluates the diagnostic density
/// samples through RHORRP point-pair density inputs.
pub fn compton_rhozzp_slice_from_rhorrp(
    grid: &ComptonGrid,
    input: ComptonRhoZzpInput,
    density_input: ComptonRhorrpDensityInput<'_>,
) -> Result<ComptonRhoZzpSlice, ComptonError> {
    compton_rhozzp_slice(grid, input, |first_point, second_point| {
        rhorrp_density_sample(density_input, first_point, second_point)
    })
}

fn rhorrp_density_sample(
    input: ComptonRhorrpDensityInput<'_>,
    first_point: Vector3,
    second_point: Vector3,
) -> Result<Real, ComptonError> {
    rhorrp_point_pair_density(RhorrpPointPairDensityInput {
        first_point,
        second_point,
        atom_positions: input.atom_positions,
        atom_potentials: input.atom_potentials,
        fms_atom_count: input.fms_atom_count,
        restrict_first_point_to_central_voronoi: true,
        energies_hartree: input.energies_hartree,
        reference_energy_hartree: input.reference_energy_hartree,
        regular_large: input.regular_large,
        irregular_large: input.irregular_large,
        regular_small: input.regular_small,
        irregular_small: input.irregular_small,
        phase: input.phase,
        diagonal_scattering_matrices: input.diagonal_scattering_matrices,
        central_scattering_matrices: input.central_scattering_matrices,
        radial_x0: input.radial_x0,
        radial_dx: input.radial_dx,
        radial_count: input.radial_count,
        real_axis_count: input.real_axis_count,
        chemical_potential_hartree: input.chemical_potential_hartree,
        temperature_hartree: input.temperature_hartree,
        chemical_potential_override_hartree: input.chemical_potential_override_hartree,
    })
    .map_err(|source| ComptonError::RhorrpDensity { source })
}
