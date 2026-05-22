use ndarray::{ArrayView2, ArrayView3};

use crate::{Complex, Real, Vector3};

use super::constants::ATOMIC_DENSITY_INTERPOLATION_ORDER;
use super::types::*;

pub(super) fn validate_density_grid_input(
    input: RhorrpDensityGridInput<'_>,
) -> Result<(), RhorrpError> {
    validate_dimension_count(input.points_per_axis.len())?;
    let (rows, columns) = input.axes.dim();
    if rows != 3 || columns != input.points_per_axis.len() {
        return Err(RhorrpError::InvalidAxesShape {
            rows,
            columns,
            expected_columns: input.points_per_axis.len(),
        });
    }
    validate_point_counts(input.points_per_axis)?;
    validate_vector("origin", input.origin)?;
    for (index, &value) in input.axes.iter().enumerate() {
        if !value.is_finite() {
            return Err(RhorrpError::NonFiniteValue {
                name: "axes",
                index,
                value,
            });
        }
    }
    Ok(())
}

pub(super) fn validate_fms_inclusion_input(
    input: RhorrpFmsInclusionInput<'_>,
) -> Result<(), RhorrpError> {
    let (rows, columns) = input.atom_positions.dim();
    if columns != 3 {
        return Err(RhorrpError::InvalidAtomPositionShape { rows, columns });
    }
    if rows == 0 {
        return Err(RhorrpError::NoAtoms);
    }
    validate_scalar("fms_radius", 0, input.fms_radius)?;
    for (index, &value) in input.atom_positions.iter().enumerate() {
        validate_scalar("atom_positions", index, value)?;
    }
    for (potential, &representative) in input.representative_atoms.iter().enumerate() {
        if representative >= rows {
            return Err(RhorrpError::InvalidRepresentativeAtom {
                potential,
                representative,
                atoms: rows,
            });
        }
    }
    Ok(())
}

pub(super) fn validate_nearest_atom_table_input(
    input: RhorrpNearestAtomTableInput<'_>,
) -> Result<usize, RhorrpError> {
    let (rows, columns) = input.points.dim();
    if rows != 3 {
        return Err(RhorrpError::InvalidPointTableShape { rows, columns });
    }
    for (index, &value) in input.points.iter().enumerate() {
        validate_scalar("nearest_atom_points", index, value)?;
    }
    validate_atom_search_input(
        input.atom_positions,
        input.atom_potentials,
        input.fms_atom_count,
    )
}

pub(super) fn validate_atom_search_input(
    atom_positions: ArrayView2<'_, Real>,
    atom_potentials: &[usize],
    fms_atom_count: Option<usize>,
) -> Result<usize, RhorrpError> {
    let (rows, columns) = atom_positions.dim();
    if columns != 3 {
        return Err(RhorrpError::InvalidAtomPositionShape { rows, columns });
    }
    if rows == 0 {
        return Err(RhorrpError::NoAtoms);
    }
    if atom_potentials.len() != rows {
        return Err(RhorrpError::AtomPotentialLengthMismatch {
            potentials: atom_potentials.len(),
            atoms: rows,
        });
    }
    if let Some(fms_atom_count) = fms_atom_count
        && (fms_atom_count == 0 || fms_atom_count > rows)
    {
        return Err(RhorrpError::InvalidFmsAtomCount {
            fms_atom_count,
            atoms: rows,
        });
    }
    for (index, &value) in atom_positions.iter().enumerate() {
        validate_scalar("atom_positions", index, value)?;
    }
    Ok(fms_atom_count.unwrap_or(rows))
}

pub(super) fn validate_vector(name: &'static str, vector: Vector3) -> Result<(), RhorrpError> {
    for (index, value) in vector.into_iter().enumerate() {
        validate_scalar(name, index, value)?;
    }
    Ok(())
}

pub(super) fn validate_scalar(
    name: &'static str,
    index: usize,
    value: Real,
) -> Result<(), RhorrpError> {
    if value.is_finite() {
        Ok(())
    } else {
        Err(RhorrpError::NonFiniteValue { name, index, value })
    }
}

pub(super) fn validate_radial_interpolation_input(
    input: RhorrpRadialInterpolationInput,
) -> Result<(), RhorrpError> {
    validate_scalar("radius", 0, input.radius)?;
    validate_scalar("x0", 0, input.x0)?;
    validate_scalar("dx", 0, input.dx)?;
    if input.radius < 0.0 {
        return Err(RhorrpError::InvalidRadius {
            value: input.radius,
        });
    }
    if input.dx <= 0.0 {
        return Err(RhorrpError::InvalidRadialStep { value: input.dx });
    }
    if input.radial_count == 0 || input.radial_count > isize::MAX as usize {
        return Err(RhorrpError::InvalidRadialCount {
            radial_count: input.radial_count,
        });
    }
    Ok(())
}

pub(super) fn validate_energy_prefactor_input(
    input: RhorrpEnergyPrefactorInput,
) -> Result<(), RhorrpError> {
    validate_scalar("energy_hartree.real", 0, input.energy_hartree.re)?;
    validate_scalar("energy_hartree.imag", 0, input.energy_hartree.im)?;
    validate_scalar(
        "reference_energy_hartree.real",
        0,
        input.reference_energy_hartree.re,
    )?;
    validate_scalar(
        "reference_energy_hartree.imag",
        0,
        input.reference_energy_hartree.im,
    )
}

pub(super) fn validate_energy_density_input(
    input: RhorrpEnergyDensityInput<'_>,
) -> Result<(), RhorrpError> {
    if input.energies_hartree.len() != input.green_function.len() {
        return Err(RhorrpError::EnergyDensityLengthMismatch {
            energies: input.energies_hartree.len(),
            green: input.green_function.len(),
        });
    }
    validate_scalar("radius", 0, input.radius)?;
    validate_scalar("prime_radius", 0, input.prime_radius)?;
    validate_scalar(
        "reference_energy_hartree.real",
        0,
        input.reference_energy_hartree.re,
    )?;
    validate_scalar(
        "reference_energy_hartree.imag",
        0,
        input.reference_energy_hartree.im,
    )?;
    if input.radius <= 0.0 {
        return Err(RhorrpError::InvalidPositiveRadius {
            name: "radius",
            value: input.radius,
        });
    }
    if input.prime_radius <= 0.0 {
        return Err(RhorrpError::InvalidPositiveRadius {
            name: "prime_radius",
            value: input.prime_radius,
        });
    }
    for (index, &energy) in input.energies_hartree.iter().enumerate() {
        validate_scalar("energies_hartree.real", index, energy.re)?;
        validate_scalar("energies_hartree.imag", index, energy.im)?;
    }
    for (index, &green) in input.green_function.iter().enumerate() {
        validate_scalar("green_function.real", index, green.re)?;
        validate_scalar("green_function.imag", index, green.im)?;
    }
    Ok(())
}

pub(super) fn validate_pair_energy_density_input(
    input: RhorrpPairEnergyDensityInput<'_>,
) -> Result<usize, RhorrpError> {
    let (energy, angular, radial) = input.first_regular_large.dim();
    if energy == 0 || angular == 0 || radial == 0 {
        return Err(RhorrpError::InvalidWavefunctionShape {
            energy,
            angular,
            radial,
        });
    }
    validate_wavefunction_component_shape(
        "first_irregular_large",
        input.first_regular_large,
        input.first_irregular_large,
    )?;
    validate_wavefunction_component_shape(
        "first_regular_small",
        input.first_regular_large,
        input.first_regular_small,
    )?;
    validate_wavefunction_component_shape(
        "first_irregular_small",
        input.first_regular_large,
        input.first_irregular_small,
    )?;
    validate_wavefunction_component_shape(
        "second_regular_large",
        input.first_regular_large,
        input.second_regular_large,
    )?;
    validate_wavefunction_component_shape(
        "second_regular_small",
        input.first_regular_large,
        input.second_regular_small,
    )?;
    validate_phase_shape("first_phase", input.first_phase, energy, angular)?;
    validate_phase_shape("second_phase", input.second_phase, energy, angular)?;
    if let Some(scattering_matrix) = input.scattering_matrix {
        let state_count = angular
            .checked_mul(angular)
            .ok_or(RhorrpError::PointCountOverflow)?;
        validate_scattering_matrix_shape(scattering_matrix, energy, state_count)?;
    }
    Ok(energy)
}

pub(super) fn validate_same_site_green_input(
    input: RhorrpSameSiteGreenInput<'_>,
) -> Result<(usize, usize, usize), RhorrpError> {
    validate_scalar("cosine_between", 0, input.cosine_between)?;
    let (energy, angular, radial) = input.regular_large.dim();
    if energy == 0 || angular == 0 || radial == 0 {
        return Err(RhorrpError::InvalidWavefunctionShape {
            energy,
            angular,
            radial,
        });
    }
    validate_wavefunction_component_shape(
        "irregular_large",
        input.regular_large,
        input.irregular_large,
    )?;
    validate_wavefunction_component_shape(
        "regular_small",
        input.regular_large,
        input.regular_small,
    )?;
    validate_wavefunction_component_shape(
        "irregular_small",
        input.regular_large,
        input.irregular_small,
    )?;
    validate_wavefunction_interpolation_input(RhorrpWavefunctionInterpolationInput {
        wavefunctions: input.regular_large,
        index_below_1based: input.first_location.index_below_1based,
        fraction: input.first_location.fraction,
    })?;
    validate_wavefunction_interpolation_input(RhorrpWavefunctionInterpolationInput {
        wavefunctions: input.regular_large,
        index_below_1based: input.second_location.index_below_1based,
        fraction: input.second_location.fraction,
    })?;
    Ok((energy, angular, radial))
}

pub(super) fn validate_scattering_green_input(
    input: RhorrpScatteringGreenInput<'_>,
) -> Result<(usize, usize, usize), RhorrpError> {
    validate_vector("first_displacement", input.first_displacement)?;
    validate_vector("second_displacement", input.second_displacement)?;
    let (energy, angular, radial) = input.first_regular_large.dim();
    if energy == 0 || angular == 0 || radial == 0 {
        return Err(RhorrpError::InvalidWavefunctionShape {
            energy,
            angular,
            radial,
        });
    }
    validate_wavefunction_component_shape(
        "first_regular_small",
        input.first_regular_large,
        input.first_regular_small,
    )?;
    validate_wavefunction_component_shape(
        "second_regular_large",
        input.first_regular_large,
        input.second_regular_large,
    )?;
    validate_wavefunction_component_shape(
        "second_regular_small",
        input.first_regular_large,
        input.second_regular_small,
    )?;
    validate_phase_shape("first_phase", input.first_phase, energy, angular)?;
    validate_phase_shape("second_phase", input.second_phase, energy, angular)?;
    let state_count = angular
        .checked_mul(angular)
        .ok_or(RhorrpError::PointCountOverflow)?;
    validate_scattering_matrix_shape(input.scattering_matrix, energy, state_count)?;
    validate_wavefunction_interpolation_input(RhorrpWavefunctionInterpolationInput {
        wavefunctions: input.first_regular_large,
        index_below_1based: input.first_location.index_below_1based,
        fraction: input.first_location.fraction,
    })?;
    validate_wavefunction_interpolation_input(RhorrpWavefunctionInterpolationInput {
        wavefunctions: input.second_regular_large,
        index_below_1based: input.second_location.index_below_1based,
        fraction: input.second_location.fraction,
    })?;
    for (index, value) in input.first_phase.iter().enumerate() {
        validate_scalar("first_phase.real", index, value.re)?;
        validate_scalar("first_phase.imag", index, value.im)?;
    }
    for (index, value) in input.second_phase.iter().enumerate() {
        validate_scalar("second_phase.real", index, value.re)?;
        validate_scalar("second_phase.imag", index, value.im)?;
    }
    for (index, value) in input.scattering_matrix.iter().enumerate() {
        validate_scalar("scattering_matrix.real", index, value.re)?;
        validate_scalar("scattering_matrix.imag", index, value.im)?;
    }
    Ok((energy, angular, state_count))
}

pub(super) fn validate_wavefunction_component_shape(
    component: &'static str,
    reference: ArrayView3<'_, Complex>,
    actual: ArrayView3<'_, Complex>,
) -> Result<(), RhorrpError> {
    let (expected_energy, expected_angular, expected_radial) = reference.dim();
    let (actual_energy, actual_angular, actual_radial) = actual.dim();
    if actual.dim() != reference.dim() {
        return Err(RhorrpError::WavefunctionComponentShapeMismatch {
            component,
            expected_energy,
            expected_angular,
            expected_radial,
            actual_energy,
            actual_angular,
            actual_radial,
        });
    }
    Ok(())
}

pub(super) fn validate_phase_shape(
    component: &'static str,
    actual: ArrayView2<'_, Complex>,
    expected_energy: usize,
    expected_angular: usize,
) -> Result<(), RhorrpError> {
    let (actual_energy, actual_angular) = actual.dim();
    if actual_energy != expected_energy || actual_angular != expected_angular {
        return Err(RhorrpError::PhaseShapeMismatch {
            component,
            expected_energy,
            expected_angular,
            actual_energy,
            actual_angular,
        });
    }
    Ok(())
}

pub(super) fn validate_scattering_matrix_shape(
    actual: ArrayView3<'_, Complex>,
    expected_energy: usize,
    expected_states: usize,
) -> Result<(), RhorrpError> {
    let (actual_energy, actual_rows, actual_columns) = actual.dim();
    if actual_energy != expected_energy
        || actual_rows != expected_states
        || actual_columns != expected_states
    {
        return Err(RhorrpError::ScatteringMatrixShapeMismatch {
            expected_energy,
            expected_states,
            actual_energy,
            actual_rows,
            actual_columns,
        });
    }
    Ok(())
}

pub(super) fn validate_wavefunction_interpolation_input(
    input: RhorrpWavefunctionInterpolationInput<'_>,
) -> Result<(), RhorrpError> {
    let (energy, angular, radial) = input.wavefunctions.dim();
    if energy == 0 || angular == 0 || radial == 0 {
        return Err(RhorrpError::InvalidWavefunctionShape {
            energy,
            angular,
            radial,
        });
    }
    validate_scalar("wavefunction_fraction", 0, input.fraction)?;
    if input.index_below_1based >= 0 {
        let upper = if input.index_below_1based == 0 {
            0
        } else {
            usize::try_from(input.index_below_1based).map_err(|_| {
                RhorrpError::InvalidWavefunctionIndex {
                    index: input.index_below_1based,
                    radial,
                }
            })?
        };
        if upper >= radial {
            return Err(RhorrpError::InvalidWavefunctionIndex {
                index: input.index_below_1based,
                radial,
            });
        }
    }
    Ok(())
}

pub(super) fn validate_irregular_fix_input(
    input: RhorrpIrregularFixInput<'_>,
) -> Result<(), RhorrpError> {
    if input.radii.len() != input.values.len() {
        return Err(RhorrpError::IrregularFixLengthMismatch {
            radii: input.radii.len(),
            values: input.values.len(),
        });
    }
    if input.radii.len() < 100 {
        return Err(RhorrpError::InsufficientIrregularFixPoints {
            points: input.radii.len(),
            required: 100,
        });
    }
    for (index, &radius) in input.radii.iter().enumerate() {
        validate_scalar("irregular_radii", index, radius)?;
    }
    for (index, value) in input.values.iter().enumerate() {
        validate_scalar("irregular_values.real", index, value.re)?;
        validate_scalar("irregular_values.imag", index, value.im)?;
    }
    Ok(())
}

pub(super) fn validate_atomic_density_input(
    input: RhorrpAtomicDensityInput<'_>,
) -> Result<(), RhorrpError> {
    validate_vector("atomic_density_point", input.point)?;
    let (atoms, columns) = input.atom_positions.dim();
    if columns != 3 {
        return Err(RhorrpError::InvalidAtomPositionShape {
            rows: atoms,
            columns,
        });
    }
    if atoms == 0 {
        return Err(RhorrpError::NoAtoms);
    }
    if input.atom_potentials.len() != atoms {
        return Err(RhorrpError::AtomPotentialLengthMismatch {
            potentials: input.atom_potentials.len(),
            atoms,
        });
    }

    let large_shape = input.large_components.dim();
    let small_shape = input.small_components.dim();
    if large_shape != small_shape {
        return Err(RhorrpError::AtomicDensityComponentShapeMismatch {
            large_radial: large_shape.0,
            large_orbital: large_shape.1,
            large_potential: large_shape.2,
            small_radial: small_shape.0,
            small_orbital: small_shape.1,
            small_potential: small_shape.2,
        });
    }
    let (radial, orbital, potential_count) = large_shape;
    if radial == 0 || orbital == 0 || potential_count == 0 {
        return Err(RhorrpError::InvalidAtomicDensityShape {
            table: "component",
            radial,
            orbital,
            potential: potential_count,
        });
    }
    if input.radii.len() != radial {
        return Err(RhorrpError::AtomicDensityRadialLengthMismatch {
            radii: input.radii.len(),
            components: radial,
        });
    }
    let required = ATOMIC_DENSITY_INTERPOLATION_ORDER + 1;
    if radial < required {
        return Err(RhorrpError::InsufficientAtomicDensityRadii {
            points: radial,
            required,
        });
    }
    if input.orbital_index_1based == 0 || input.orbital_index_1based > orbital {
        return Err(RhorrpError::InvalidAtomicDensityOrbital {
            orbital: input.orbital_index_1based,
            orbital_count: orbital,
        });
    }
    for (atom, &potential) in input.atom_potentials.iter().enumerate() {
        if potential >= potential_count {
            return Err(RhorrpError::InvalidAtomicDensityPotential {
                atom_index_1based: atom + 1,
                potential,
                max_potential: potential_count.saturating_sub(1),
            });
        }
    }
    for (index, &value) in input.atom_positions.iter().enumerate() {
        validate_scalar("atomic_density_atom_positions", index, value)?;
    }
    for (index, &radius) in input.radii.iter().enumerate() {
        validate_scalar("atomic_density_radii", index, radius)?;
    }
    for (index, &value) in input.large_components.iter().enumerate() {
        validate_scalar("atomic_density_large_components", index, value)?;
    }
    for (index, &value) in input.small_components.iter().enumerate() {
        validate_scalar("atomic_density_small_components", index, value)?;
    }
    Ok(())
}

pub(super) fn validate_density_integration_input(
    input: RhorrpDensityIntegrationInput<'_>,
) -> Result<(), RhorrpError> {
    if input.energies_hartree.len() != input.energy_density.len() {
        return Err(RhorrpError::DensityIntegrationLengthMismatch {
            energies: input.energies_hartree.len(),
            densities: input.energy_density.len(),
        });
    }
    if input.real_axis_count < 2 || input.real_axis_count > input.energies_hartree.len() {
        return Err(RhorrpError::InvalidDensityIntegrationRealAxisCount {
            real_axis_count: input.real_axis_count,
            energy_count: input.energies_hartree.len(),
        });
    }
    validate_scalar(
        "density_integration_chemical_potential_hartree",
        0,
        input.chemical_potential_hartree,
    )?;
    validate_scalar(
        "density_integration_temperature_hartree",
        0,
        input.temperature_hartree,
    )?;
    if let Some(override_mu) = input.chemical_potential_override_hartree {
        validate_scalar(
            "density_integration_chemical_potential_override_hartree",
            0,
            override_mu,
        )?;
    }
    for (index, energy) in input.energies_hartree.iter().enumerate() {
        validate_scalar("density_integration_energy.real", index, energy.re)?;
        validate_scalar("density_integration_energy.imag", index, energy.im)?;
    }
    for (index, density) in input.energy_density.iter().enumerate() {
        validate_scalar("density_integration_density.real", index, density.re)?;
        validate_scalar("density_integration_density.imag", index, density.im)?;
    }
    Ok(())
}

pub(super) fn validate_dimension_count(dimensions: usize) -> Result<(), RhorrpError> {
    if !(1..=3).contains(&dimensions) {
        return Err(RhorrpError::InvalidDimensionCount { dimensions });
    }
    Ok(())
}

pub(super) fn validate_point_counts(points_per_axis: &[usize]) -> Result<(), RhorrpError> {
    for (axis, &value) in points_per_axis.iter().enumerate() {
        if value < 2 {
            return Err(RhorrpError::InvalidPointCount { axis, value });
        }
    }
    Ok(())
}

pub(super) fn validate_grid_index(
    points_per_axis: &[usize],
    index_1based: &[usize],
) -> Result<(), RhorrpError> {
    if index_1based.len() != points_per_axis.len() {
        return Err(RhorrpError::IndexLengthMismatch {
            index_len: index_1based.len(),
            dimensions: points_per_axis.len(),
        });
    }
    for (axis, (&index, &limit)) in index_1based.iter().zip(points_per_axis.iter()).enumerate() {
        if index == 0 || index > limit {
            return Err(RhorrpError::InvalidGridIndex { axis, index, limit });
        }
    }
    Ok(())
}

pub(super) fn checked_total_points(points_per_axis: &[usize]) -> Result<usize, RhorrpError> {
    points_per_axis
        .iter()
        .try_fold(1usize, |total, &count| total.checked_mul(count))
        .ok_or(RhorrpError::PointCountOverflow)
}
