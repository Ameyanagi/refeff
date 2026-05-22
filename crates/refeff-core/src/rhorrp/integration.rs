use ndarray::ArrayView1;

use crate::interpolation::polynomial_interpolate_complex;
use crate::{Complex, Real};

use super::constants::{
    DENSITY_INTEGRATION_HORIZONTAL_EPSILON, DENSITY_INTEGRATION_INTERPOLATION_ORDER,
    DENSITY_INTEGRATION_SUBDIVISIONS,
};
use super::radial::rhorrp_fermi_distribution;
use super::types::{RhorrpDensityIntegrationInput, RhorrpError, RhorrpFermiDistributionInput};
use super::validation::{validate_density_integration_input, validate_scalar};

/// Port of FEFF `rhorrp` energy-contour integration.
///
/// This helper starts after `rhoerrp` has produced the energy-dependent density
/// matrix. It preserves FEFF's vertical trapezoid leg, hard-coded ten-way
/// horizontal subdivision with quadratic `terpc`, Matsubara pole sum, and final
/// imaginary-part extraction.
pub fn rhorrp_integrate_density(
    input: RhorrpDensityIntegrationInput<'_>,
) -> Result<Real, RhorrpError> {
    validate_density_integration_input(input)?;

    let mut fermi = rhorrp_fermi_distribution(RhorrpFermiDistributionInput {
        energy_hartree: input.energies_hartree[0],
        chemical_potential_hartree: input.chemical_potential_hartree,
        temperature_hartree: input.temperature_hartree,
        chemical_potential_override_hartree: input.chemical_potential_override_hartree,
    })?;
    let mut integrated = input.energies_hartree[0] * input.energy_density[0] * fermi;
    let mut previous_density = input.energy_density[0];
    let mut previous_fermi = fermi;
    let mut horizontal_start = None;

    for energy_index in 1..input.real_axis_count {
        let delta = input.energies_hartree[energy_index] - input.energies_hartree[energy_index - 1];
        if delta.re > DENSITY_INTEGRATION_HORIZONTAL_EPSILON {
            horizontal_start = Some(energy_index - 1);
            break;
        }

        fermi = rhorrp_fermi_distribution(RhorrpFermiDistributionInput {
            energy_hartree: input.energies_hartree[energy_index],
            chemical_potential_hartree: input.chemical_potential_hartree,
            temperature_hartree: input.temperature_hartree,
            chemical_potential_override_hartree: input.chemical_potential_override_hartree,
        })?;
        let density = input.energy_density[energy_index];
        integrated += (previous_density * previous_fermi + density * fermi) * 0.5 * delta;
        previous_density = density;
        previous_fermi = fermi;
    }

    let horizontal_start = horizontal_start.ok_or(RhorrpError::MissingDensityIntegrationCorner)?;
    let horizontal_points = input.real_axis_count - horizontal_start;
    let required = DENSITY_INTEGRATION_INTERPOLATION_ORDER + 1;
    if horizontal_points < required {
        return Err(RhorrpError::InsufficientDensityIntegrationPoints {
            points: horizontal_points,
            required,
        });
    }

    for energy_index in (horizontal_start + 1)..input.real_axis_count {
        let delta = (input.energies_hartree[energy_index]
            - input.energies_hartree[energy_index - 1])
            / DENSITY_INTEGRATION_SUBDIVISIONS as Real;
        for subdivision in 1..=DENSITY_INTEGRATION_SUBDIVISIONS {
            let energy = input.energies_hartree[energy_index - 1] + delta * subdivision as Real;
            fermi = rhorrp_fermi_distribution(RhorrpFermiDistributionInput {
                energy_hartree: energy,
                chemical_potential_hartree: input.chemical_potential_hartree,
                temperature_hartree: input.temperature_hartree,
                chemical_potential_override_hartree: input.chemical_potential_override_hartree,
            })?;
            let density = interpolate_density_contour(
                input.energies_hartree,
                input.energy_density,
                horizontal_start,
                input.real_axis_count,
                energy.re,
            )?;
            integrated += (previous_density * previous_fermi + density * fermi) * 0.5 * delta;
            previous_density = density;
            previous_fermi = fermi;
        }
    }

    for energy_index in input.real_axis_count..input.energies_hartree.len() {
        integrated += Complex::new(0.0, -2.0 * std::f64::consts::PI * input.temperature_hartree)
            * input.energy_density[energy_index];
    }

    validate_scalar("integrated_density", 0, integrated.im)?;
    Ok(integrated.im)
}

fn interpolate_density_contour(
    energies: ArrayView1<'_, Complex>,
    density: ArrayView1<'_, Complex>,
    horizontal_start: usize,
    real_axis_count: usize,
    energy: Real,
) -> Result<Complex, RhorrpError> {
    let located = locate_density_contour_below(energies, horizontal_start, real_axis_count, energy);
    let local_len = real_axis_count - horizontal_start;
    let start_1based = (located.saturating_sub(DENSITY_INTEGRATION_INTERPOLATION_ORDER / 2))
        .clamp(1, local_len - DENSITY_INTEGRATION_INTERPOLATION_ORDER);
    let start = horizontal_start + start_1based - 1;
    let interpolation_energies = [
        energies[start].re,
        energies[start + 1].re,
        energies[start + 2].re,
    ];
    let values = [density[start], density[start + 1], density[start + 2]];
    Ok(
        polynomial_interpolate_complex(&interpolation_energies, &values, energy)
            .map_err(|source| RhorrpError::DensityIntegrationInterpolation { source })?
            .value,
    )
}

fn locate_density_contour_below(
    energies: ArrayView1<'_, Complex>,
    start: usize,
    end: usize,
    energy: Real,
) -> usize {
    let mut lower = 0;
    let mut upper = end - start + 1;

    while upper - lower > 1 {
        let middle = (upper + lower) / 2;
        let middle_value = energies[start + middle - 1].re;
        if energy < middle_value {
            upper = middle;
        } else {
            lower = middle;
        }
    }

    lower
}
