use ndarray::{ArrayView3, Axis};

use crate::interpolation::{locate_below, polynomial_interpolate};
use crate::{Complex, ComplexVec, Real};

use super::constants::{
    ATOMIC_DENSITY_CUTOFF_SQUARED, ATOMIC_DENSITY_INTERPOLATION_ORDER, ATOMIC_DENSITY_MIN_RADIUS,
};
use super::greens::{rhorrp_effective_temperature_hartree, rhorrp_pair_energy_density};
use super::integration::rhorrp_integrate_density;
use super::nearest::rhorrp_nearest_atom;
use super::types::{
    RhorrpAtomicDensityInput, RhorrpDensityIntegrationInput, RhorrpError, RhorrpNearestAtomInput,
    RhorrpPairEnergyDensityInput, RhorrpPointDensityFromTablesInput, RhorrpPointDensityInput,
    RhorrpPointEnergyDensityFromTablesInput, RhorrpPointEnergyDensityInput,
    RhorrpPointPairDensityFromTablesInput, RhorrpPointPairDensityInput,
    RhorrpPointPairEnergyDensityFromTablesInput, RhorrpPointPairEnergyDensityInput,
    RhorrpScatteringMatrixSelectionInput,
};
use super::validation::{
    validate_atomic_density_input, validate_point_density_input,
    validate_point_energy_density_input, validate_point_pair_density_input,
    validate_point_pair_energy_density_input, validate_scalar,
};

/// Port of FEFF `atomic_density`.
///
/// FEFF sums core radial densities from atoms within two Bohr of the requested
/// point. Each contributing atom uses quadratic `terp` interpolation on `ripot`
/// for the requested core-wavefunction column and returns the spherical
/// density `(p^2 + q^2) / (4*pi*r^2)`.
pub fn rhorrp_atomic_density(input: RhorrpAtomicDensityInput<'_>) -> Result<Real, RhorrpError> {
    validate_atomic_density_input(input)?;

    let orbital = input.orbital_index_1based - 1;
    let mut density = 0.0;
    for atom in 0..input.atom_positions.nrows() {
        let displacement = [
            input.atom_positions[(atom, 0)] - input.point[0],
            input.atom_positions[(atom, 1)] - input.point[1],
            input.atom_positions[(atom, 2)] - input.point[2],
        ];
        let distance_squared: Real = displacement.iter().map(|value| value * value).sum();
        if distance_squared > ATOMIC_DENSITY_CUTOFF_SQUARED {
            continue;
        }

        let radius = distance_squared.sqrt().max(ATOMIC_DENSITY_MIN_RADIUS);
        let potential = input.atom_potentials[atom];
        let large = interpolate_atomic_component(
            input.radii,
            input.large_components,
            orbital,
            potential,
            radius,
        )?;
        let small = interpolate_atomic_component(
            input.radii,
            input.small_components,
            orbital,
            potential,
            radius,
        )?;
        density += (large * large + small * small) / (4.0 * std::f64::consts::PI * radius * radius);
    }

    validate_scalar("atomic_density", 0, density)?;
    Ok(density)
}

/// Port of FEFF RHORRP point-density setup before `rhoerrp`.
///
/// This helper performs the same same-point setup used by RHORRP grid density
/// evaluation: find the nearest atom, select the potential-local wavefunction
/// and phase blocks, select the site-diagonal FMS matrix when available, then
/// call the ported `rhoerrp` plus contour-integration path.
pub fn rhorrp_point_density(input: RhorrpPointDensityInput<'_>) -> Result<Real, RhorrpError> {
    validate_point_density_input(input)?;
    let energy_density =
        rhorrp_point_energy_density_after_validation(input.energy_density_input())?;
    let temperature_hartree = rhorrp_effective_temperature_hartree(input.temperature_hartree)?;

    rhorrp_integrate_density(RhorrpDensityIntegrationInput {
        energies_hartree: input.energies_hartree,
        energy_density: energy_density.view(),
        real_axis_count: input.real_axis_count,
        chemical_potential_hartree: input.chemical_potential_hartree,
        temperature_hartree,
        chemical_potential_override_hartree: input.chemical_potential_override_hartree,
    })
}

/// Port of FEFF `rhoerrp(v, v, rhoe)` same-point setup from handoff tables.
///
/// This finds the nearest atom, selects potential-local wavefunction and phase
/// blocks, selects the site-diagonal FMS matrix when present, and evaluates the
/// energy-dependent density matrix without occupied-contour integration.
pub fn rhorrp_point_energy_density(
    input: RhorrpPointEnergyDensityInput<'_>,
) -> Result<ComplexVec, RhorrpError> {
    validate_point_energy_density_input(input)?;
    rhorrp_point_energy_density_after_validation(input)
}

/// Evaluate FEFF `rhoerrp(v, v, rhoe)` using `init_wavefunctions` tables.
pub fn rhorrp_point_energy_density_from_tables(
    input: RhorrpPointEnergyDensityFromTablesInput<'_>,
) -> Result<ComplexVec, RhorrpError> {
    rhorrp_point_energy_density(RhorrpPointEnergyDensityInput {
        point: input.point,
        atom_positions: input.atom_positions,
        atom_potentials: input.atom_potentials,
        fms_atom_count: input.fms_atom_count,
        energies_hartree: input.energies_hartree,
        reference_energy_hartree: input.reference_energy_hartree,
        regular_large: input.wavefunctions.regular_large.view(),
        irregular_large: input.wavefunctions.irregular_large.view(),
        regular_small: input.wavefunctions.regular_small.view(),
        irregular_small: input.wavefunctions.irregular_small.view(),
        phase: input.wavefunctions.phase_shifts.view(),
        diagonal_scattering_matrices: input.diagonal_scattering_matrices,
        radial_x0: input.radial_x0,
        radial_dx: input.radial_dx,
        radial_count: input.wavefunctions.radial_count(),
    })
}

/// Evaluate FEFF `rhorrp(v, v, rho)` using `init_wavefunctions` tables.
pub fn rhorrp_point_density_from_tables(
    input: RhorrpPointDensityFromTablesInput<'_>,
) -> Result<Real, RhorrpError> {
    let energy_density = rhorrp_point_energy_density_from_tables(input.energy_density_input())?;
    let temperature_hartree = rhorrp_effective_temperature_hartree(input.temperature_hartree)?;

    rhorrp_integrate_density(RhorrpDensityIntegrationInput {
        energies_hartree: input.energies_hartree,
        energy_density: energy_density.view(),
        real_axis_count: input.real_axis_count,
        chemical_potential_hartree: input.chemical_potential_hartree,
        temperature_hartree,
        chemical_potential_override_hartree: input.chemical_potential_override_hartree,
    })
}

fn rhorrp_point_energy_density_after_validation(
    input: RhorrpPointEnergyDensityInput<'_>,
) -> Result<ComplexVec, RhorrpError> {
    let nearest = rhorrp_nearest_atom(RhorrpNearestAtomInput {
        point: input.point,
        atom_positions: input.atom_positions,
        atom_potentials: input.atom_potentials,
        fms_atom_count: input.fms_atom_count,
    })?;
    let potential = nearest.potential_index;
    let regular_large = input.regular_large.index_axis(Axis(3), potential);
    let irregular_large = input.irregular_large.index_axis(Axis(3), potential);
    let regular_small = input.regular_small.index_axis(Axis(3), potential);
    let irregular_small = input.irregular_small.index_axis(Axis(3), potential);
    let phase = input.phase.index_axis(Axis(2), potential);
    let scattering_matrix =
        rhorrp_select_scattering_matrix(RhorrpScatteringMatrixSelectionInput {
            first_atom_index: nearest.atom_index,
            second_atom_index: nearest.atom_index,
            diagonal_scattering_matrices: input.diagonal_scattering_matrices,
            central_scattering_matrices: None,
        })?;

    rhorrp_pair_energy_density(RhorrpPairEnergyDensityInput {
        energies_hartree: input.energies_hartree,
        reference_energy_hartree: input.reference_energy_hartree,
        first_regular_large: regular_large,
        first_irregular_large: irregular_large,
        first_regular_small: regular_small,
        first_irregular_small: irregular_small,
        second_regular_large: regular_large,
        second_regular_small: regular_small,
        first_phase: phase,
        second_phase: phase,
        scattering_matrix,
        same_atom: true,
        first_displacement: nearest.displacement,
        second_displacement: nearest.displacement,
        radial_x0: input.radial_x0,
        radial_dx: input.radial_dx,
        radial_count: input.radial_count,
    })
}

/// Port of FEFF RHORRP point-pair setup before `rhoerrp`.
///
/// COMPTON and RHORRP density-matrix paths evaluate `rho(r,r')` for two
/// arbitrary points. This helper finds the nearest FEFF atom for each point,
/// selects potential-local wavefunction and phase blocks, selects the saved FMS
/// matrix when FEFF's `gg_diag.bin`/`gg_slice.bin` handoff can represent the
/// pair, then delegates to the ported pair-density kernel.
pub fn rhorrp_point_pair_density(
    input: RhorrpPointPairDensityInput<'_>,
) -> Result<Real, RhorrpError> {
    validate_point_pair_density_input(input)?;
    let energy_density =
        rhorrp_point_pair_energy_density_after_validation(input.energy_density_input())?;
    let temperature_hartree = rhorrp_effective_temperature_hartree(input.temperature_hartree)?;

    rhorrp_integrate_density(RhorrpDensityIntegrationInput {
        energies_hartree: input.energies_hartree,
        energy_density: energy_density.view(),
        real_axis_count: input.real_axis_count,
        chemical_potential_hartree: input.chemical_potential_hartree,
        temperature_hartree,
        chemical_potential_override_hartree: input.chemical_potential_override_hartree,
    })
}

/// Port of FEFF `rhoerrp(v, vp, rhoe)` from handoff tables.
///
/// This performs the point-pair nearest-atom setup, FEFF `gg_diag`/`gg_slice`
/// selection rules, optional central-Voronoi restriction, and then evaluates
/// the energy-dependent density matrix without occupied-contour integration.
pub fn rhorrp_point_pair_energy_density(
    input: RhorrpPointPairEnergyDensityInput<'_>,
) -> Result<ComplexVec, RhorrpError> {
    validate_point_pair_energy_density_input(input)?;
    rhorrp_point_pair_energy_density_after_validation(input)
}

/// Evaluate FEFF `rhoerrp(v, vp, rhoe)` using `init_wavefunctions` tables.
pub fn rhorrp_point_pair_energy_density_from_tables(
    input: RhorrpPointPairEnergyDensityFromTablesInput<'_>,
) -> Result<ComplexVec, RhorrpError> {
    rhorrp_point_pair_energy_density(RhorrpPointPairEnergyDensityInput {
        first_point: input.first_point,
        second_point: input.second_point,
        atom_positions: input.atom_positions,
        atom_potentials: input.atom_potentials,
        fms_atom_count: input.fms_atom_count,
        restrict_first_point_to_central_voronoi: input.restrict_first_point_to_central_voronoi,
        energies_hartree: input.energies_hartree,
        reference_energy_hartree: input.reference_energy_hartree,
        regular_large: input.wavefunctions.regular_large.view(),
        irregular_large: input.wavefunctions.irregular_large.view(),
        regular_small: input.wavefunctions.regular_small.view(),
        irregular_small: input.wavefunctions.irregular_small.view(),
        phase: input.wavefunctions.phase_shifts.view(),
        diagonal_scattering_matrices: input.diagonal_scattering_matrices,
        central_scattering_matrices: input.central_scattering_matrices,
        radial_x0: input.radial_x0,
        radial_dx: input.radial_dx,
        radial_count: input.wavefunctions.radial_count(),
    })
}

/// Evaluate FEFF `rhorrp(v, vp, rho)` using `init_wavefunctions` tables.
pub fn rhorrp_point_pair_density_from_tables(
    input: RhorrpPointPairDensityFromTablesInput<'_>,
) -> Result<Real, RhorrpError> {
    let energy_density =
        rhorrp_point_pair_energy_density_from_tables(input.energy_density_input())?;
    let temperature_hartree = rhorrp_effective_temperature_hartree(input.temperature_hartree)?;

    rhorrp_integrate_density(RhorrpDensityIntegrationInput {
        energies_hartree: input.energies_hartree,
        energy_density: energy_density.view(),
        real_axis_count: input.real_axis_count,
        chemical_potential_hartree: input.chemical_potential_hartree,
        temperature_hartree,
        chemical_potential_override_hartree: input.chemical_potential_override_hartree,
    })
}

fn rhorrp_point_pair_energy_density_after_validation(
    input: RhorrpPointPairEnergyDensityInput<'_>,
) -> Result<ComplexVec, RhorrpError> {
    let first = rhorrp_nearest_atom(RhorrpNearestAtomInput {
        point: input.first_point,
        atom_positions: input.atom_positions,
        atom_potentials: input.atom_potentials,
        fms_atom_count: input.fms_atom_count,
    })?;
    if input.restrict_first_point_to_central_voronoi && first.atom_index != 0 {
        return Ok(ComplexVec::zeros(input.energies_hartree.len()));
    }
    let second = rhorrp_nearest_atom(RhorrpNearestAtomInput {
        point: input.second_point,
        atom_positions: input.atom_positions,
        atom_potentials: input.atom_potentials,
        fms_atom_count: input.fms_atom_count,
    })?;

    let first_potential = first.potential_index;
    let second_potential = second.potential_index;
    let first_regular_large = input.regular_large.index_axis(Axis(3), first_potential);
    let first_irregular_large = input.irregular_large.index_axis(Axis(3), first_potential);
    let first_regular_small = input.regular_small.index_axis(Axis(3), first_potential);
    let first_irregular_small = input.irregular_small.index_axis(Axis(3), first_potential);
    let second_regular_large = input.regular_large.index_axis(Axis(3), second_potential);
    let second_regular_small = input.regular_small.index_axis(Axis(3), second_potential);
    let first_phase = input.phase.index_axis(Axis(2), first_potential);
    let second_phase = input.phase.index_axis(Axis(2), second_potential);
    let same_atom = first.atom_index == second.atom_index;
    let scattering_matrix =
        rhorrp_select_scattering_matrix(RhorrpScatteringMatrixSelectionInput {
            first_atom_index: first.atom_index,
            second_atom_index: second.atom_index,
            diagonal_scattering_matrices: input.diagonal_scattering_matrices,
            central_scattering_matrices: input.central_scattering_matrices,
        })?;

    rhorrp_pair_energy_density(RhorrpPairEnergyDensityInput {
        energies_hartree: input.energies_hartree,
        reference_energy_hartree: input.reference_energy_hartree,
        first_regular_large,
        first_irregular_large,
        first_regular_small,
        first_irregular_small,
        second_regular_large,
        second_regular_small,
        first_phase,
        second_phase,
        scattering_matrix,
        same_atom,
        first_displacement: first.displacement,
        second_displacement: second.displacement,
        radial_x0: input.radial_x0,
        radial_dx: input.radial_dx,
        radial_count: input.radial_count,
    })
}

/// Select the FEFF RHORRP scattering matrix for one nearest-atom pair.
///
/// This mirrors `rhoerrp`: same-site pairs use `gg_diag(:,:,iat,:)`, pairs
/// with `r` near atom 1 use the saved `gg_slice` central-row block, and
/// different-site pairs with `r` away from atom 1 have no saved full-matrix
/// block and therefore skip scattering.
pub fn rhorrp_select_scattering_matrix<'a>(
    input: RhorrpScatteringMatrixSelectionInput<'a>,
) -> Result<Option<ArrayView3<'a, Complex>>, RhorrpError> {
    if input.first_atom_index == input.second_atom_index {
        return input
            .diagonal_scattering_matrices
            .map(|matrices| {
                select_scattering_atom_block("diagonal", matrices, input.first_atom_index)
            })
            .transpose();
    }

    if input.first_atom_index == 0 {
        return input
            .central_scattering_matrices
            .map(|matrices| {
                select_scattering_atom_block("central", matrices, input.second_atom_index)
            })
            .transpose();
    }

    Ok(None)
}

fn select_scattering_atom_block<'a>(
    matrix: &'static str,
    matrices: ndarray::ArrayView4<'a, Complex>,
    atom_index: usize,
) -> Result<ArrayView3<'a, Complex>, RhorrpError> {
    let atom_count = matrices.len_of(Axis(1));
    if atom_index >= atom_count {
        return Err(RhorrpError::ScatteringMatrixAtomOutOfRange {
            matrix,
            atom_index,
            atom_count,
        });
    }
    Ok(matrices.index_axis_move(Axis(1), atom_index))
}

fn interpolate_atomic_component(
    radii: &[Real],
    components: ArrayView3<'_, Real>,
    orbital: usize,
    potential: usize,
    radius: Real,
) -> Result<Real, RhorrpError> {
    let located = locate_below(radius, radii);
    let start_1based = (located.saturating_sub(ATOMIC_DENSITY_INTERPOLATION_ORDER / 2))
        .clamp(1, radii.len() - ATOMIC_DENSITY_INTERPOLATION_ORDER);
    let start = start_1based - 1;
    let values = [
        components[(start, orbital, potential)],
        components[(start + 1, orbital, potential)],
        components[(start + 2, orbital, potential)],
    ];
    Ok(polynomial_interpolate(
        &radii[start..start + ATOMIC_DENSITY_INTERPOLATION_ORDER + 1],
        &values,
        radius,
    )?
    .value)
}
