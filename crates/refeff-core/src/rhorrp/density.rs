use ndarray::ArrayView3;

use crate::Real;
use crate::interpolation::{locate_below, polynomial_interpolate};

use super::constants::{
    ATOMIC_DENSITY_CUTOFF_SQUARED, ATOMIC_DENSITY_INTERPOLATION_ORDER, ATOMIC_DENSITY_MIN_RADIUS,
};
use super::types::{RhorrpAtomicDensityInput, RhorrpError};
use super::validation::{validate_atomic_density_input, validate_scalar};

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
