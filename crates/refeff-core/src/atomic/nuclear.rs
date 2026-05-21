use crate::Real;
use ndarray::Array1;

use super::{
    ATOM_NUCDEV_RADIUS_FACTOR, AtomMathError, AtomicNuclearPotential, AtomicNuclearPotentialInput,
    nuclear_mass, validate_finite_scalar, validate_nuclear_count,
    validate_positive_finite_nuclear_scalar,
};

/// Port of FEFF `ATOM/nucdev.f90`.
///
/// The point-nucleus branch returns the Coulomb potential `-dz/r`. Negative
/// `requested_nucleus_index` values select FEFF's finite uniform-nucleus branch
/// using the tabulated nuclear mass, matching the ATOM high-Z path.
pub fn atomic_nuclear_potential(
    input: AtomicNuclearPotentialInput,
) -> Result<AtomicNuclearPotential, AtomMathError> {
    validate_nuclear_potential_input(input)?;
    calculate_atomic_nuclear_potential(input)
}

fn calculate_atomic_nuclear_potential(
    input: AtomicNuclearPotentialInput,
) -> Result<AtomicNuclearPotential, AtomMathError> {
    let (nucleus_index, first_radius_times_charge) = atomic_nuclear_mesh_parameters(input)?;
    let first_radius = first_radius_times_charge / input.nuclear_charge;
    let radii = Array1::from_shape_fn(input.radial_count, |row| {
        first_radius * (input.step * row as Real).exp()
    });
    for radius in radii.iter().copied() {
        if radius <= 0.0 {
            return Err(AtomMathError::NonPositiveRadius { radius });
        }
        validate_finite_scalar("nucdev_radius", radius)?;
    }

    let mut development_coefficients = Array1::<Real>::zeros(input.coefficient_count);
    let mut potential = radii.mapv(|radius| -input.nuclear_charge / radius);
    if nucleus_index <= 1 {
        development_coefficients[0] = -input.nuclear_charge;
    } else {
        if nucleus_index > input.radial_count {
            return Err(AtomMathError::NuclearRadiusOutOfRange {
                nucleus_index,
                radial_count: input.radial_count,
            });
        }
        let nuclear_radius = radii[nucleus_index - 1];
        let quadratic = -3.0 * input.nuclear_charge / (nuclear_radius + nuclear_radius);
        let quartic = -quadratic / (3.0 * nuclear_radius * nuclear_radius);
        development_coefficients[1] = quadratic;
        development_coefficients[3] = quartic;
        for row in 0..(nucleus_index - 1) {
            potential[row] = quadratic + quartic * radii[row] * radii[row];
        }
    }
    for value in development_coefficients.iter().copied() {
        validate_finite_scalar("nucdev_coefficient", value)?;
    }
    for value in potential.iter().copied() {
        validate_finite_scalar("nucdev_potential", value)?;
    }

    Ok(AtomicNuclearPotential {
        development_coefficients,
        radii,
        potential,
        nucleus_index,
        first_radius_times_charge,
    })
}

fn atomic_nuclear_mesh_parameters(
    input: AtomicNuclearPotentialInput,
) -> Result<(usize, Real), AtomMathError> {
    let mut nucleus_index = requested_nucleus_index_abs(input.requested_nucleus_index)?;
    let mut first_radius_times_charge = input.first_radius_times_charge;
    let mut nuclear_mass_amu = 0.0;
    if input.requested_nucleus_index < 0 {
        let atomic_number = atomic_number_from_charge(input.nuclear_charge)?;
        nuclear_mass_amu = nuclear_mass(atomic_number)?;
    }

    if nuclear_mass_amu <= 0.1 {
        return Ok((1, first_radius_times_charge));
    }

    if nucleus_index == 0 || nucleus_index > input.radial_count {
        return Err(AtomMathError::NuclearRadiusOutOfRange {
            nucleus_index,
            radial_count: input.radial_count,
        });
    }
    let mass_exponent = Real::from(1.0_f32 / 3.0_f32);
    let scaled_radius =
        input.nuclear_charge * nuclear_mass_amu.powf(mass_exponent) * ATOM_NUCDEV_RADIUS_FACTOR;
    let requested_first_radius = scaled_radius / (input.step * (nucleus_index as Real - 1.0)).exp();
    if requested_first_radius <= first_radius_times_charge {
        first_radius_times_charge = requested_first_radius;
    } else {
        let radius_steps = (scaled_radius / first_radius_times_charge).ln() / input.step;
        let half_steps = (radius_steps / 2.0).trunc();
        nucleus_index = 3 + 2 * half_steps as usize;
        if nucleus_index >= input.radial_count {
            return Err(AtomMathError::NuclearRadiusOutOfRange {
                nucleus_index,
                radial_count: input.radial_count,
            });
        }
        first_radius_times_charge =
            scaled_radius * (-(nucleus_index as Real - 1.0) * input.step).exp();
    }
    validate_finite_scalar(
        "nucdev_first_radius_times_charge",
        first_radius_times_charge,
    )?;
    Ok((nucleus_index, first_radius_times_charge))
}

fn requested_nucleus_index_abs(requested: isize) -> Result<usize, AtomMathError> {
    if requested == isize::MIN {
        return Err(AtomMathError::NuclearRadiusOutOfRange {
            nucleus_index: usize::MAX,
            radial_count: 0,
        });
    }
    Ok(requested.unsigned_abs())
}

fn atomic_number_from_charge(nuclear_charge: Real) -> Result<usize, AtomMathError> {
    if nuclear_charge < 1.0 || nuclear_charge > usize::MAX as Real {
        return Err(AtomMathError::InvalidNuclearPotentialScalar {
            field: "nuclear_charge",
            value: nuclear_charge,
        });
    }
    Ok(nuclear_charge.trunc() as usize)
}

fn validate_nuclear_potential_input(
    input: AtomicNuclearPotentialInput,
) -> Result<(), AtomMathError> {
    validate_positive_finite_nuclear_scalar("nuclear_charge", input.nuclear_charge)?;
    validate_positive_finite_nuclear_scalar("step", input.step)?;
    validate_positive_finite_nuclear_scalar(
        "first_radius_times_charge",
        input.first_radius_times_charge,
    )?;
    validate_nuclear_count("radial_count", input.radial_count, 1)?;
    validate_nuclear_count("coefficient_count", input.coefficient_count, 5)?;
    Ok(())
}
