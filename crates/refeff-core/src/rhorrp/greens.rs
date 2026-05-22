use ndarray::ArrayView3;

use crate::angular::{legendre_polynomials_into, spherical_harmonics};
use crate::{Complex, ComplexMat, ComplexVec, Real, Vector3};

use super::constants::{FEFF_FINE_STRUCTURE_ALPHA, RHORRP_ORIGIN_EPSILON};
use super::integration::rhorrp_integrate_density;
use super::radial::{rhorrp_interpolate_wavefunction, rhorrp_radial_interpolation_location};
use super::types::{
    RhorrpDensityIntegrationInput, RhorrpEnergyDensityInput, RhorrpEnergyPrefactorInput,
    RhorrpError, RhorrpPairDensityInput, RhorrpPairEnergyDensityInput,
    RhorrpRadialInterpolationInput, RhorrpRadialInterpolationLocation, RhorrpSameSiteGreenInput,
    RhorrpScatteringGreenInput, RhorrpWavefunctionInterpolationInput,
};
use super::validation::{
    validate_energy_density_input, validate_energy_prefactor_input,
    validate_pair_energy_density_input, validate_same_site_green_input, validate_scalar,
    validate_scattering_green_input, validate_vector,
};

/// Port of FEFF `rhoerrp` final per-energy prefactor.
///
/// FEFF converts `p2 = E - eref0` to the relativistic wave number `ck`, derives
/// the small-component ratio `pu`, and multiplies the accumulated Green's
/// function by `4 * ck / (pi * (1 + pu^2))`.
pub fn rhorrp_energy_prefactor(input: RhorrpEnergyPrefactorInput) -> Result<Complex, RhorrpError> {
    validate_energy_prefactor_input(input)?;

    let one = Complex::new(1.0, 0.0);
    let p2 = input.energy_hartree - input.reference_energy_hartree;
    let alpha_p2 = p2 * FEFF_FINE_STRUCTURE_ALPHA;
    let ck = (p2 * 2.0 + alpha_p2 * alpha_p2).sqrt();
    let scaled_ck = ck * FEFF_FINE_STRUCTURE_ALPHA;
    let pu = -scaled_ck / (one + (one + scaled_ck * scaled_ck).sqrt());
    Ok(ck * (4.0 / std::f64::consts::PI) / (one + pu * pu))
}

/// Port of FEFF `rhoerrp` final energy-density scaling loop.
///
/// After local/scattering contributions are accumulated in `Ge`, FEFF applies
/// the relativistic per-energy prefactor and divides by `r * r'` to produce
/// `rhoe(ie)`.
pub fn rhorrp_finish_energy_density(
    input: RhorrpEnergyDensityInput<'_>,
) -> Result<ComplexVec, RhorrpError> {
    validate_energy_density_input(input)?;
    let radius_scale = input.radius * input.prime_radius;
    let mut density = ComplexVec::zeros(input.energies_hartree.len());
    for (index, (&energy, &green)) in input
        .energies_hartree
        .iter()
        .zip(input.green_function.iter())
        .enumerate()
    {
        let prefactor = rhorrp_energy_prefactor(RhorrpEnergyPrefactorInput {
            energy_hartree: energy,
            reference_energy_hartree: input.reference_energy_hartree,
        })?;
        density[index] = green * prefactor / radius_scale;
    }
    Ok(density)
}

/// Port of FEFF `rhoerrp` after atom and FMS-slice selection.
///
/// The caller supplies wavefunction/phase views for the selected potentials and
/// the already-selected scattering matrix for this point pair. The helper keeps
/// FEFF's near-origin displacement adjustment, logarithmic radial-grid lookup,
/// optional same-site local term, optional scattering term, and final
/// relativistic energy scaling in one composable operation.
pub fn rhorrp_pair_energy_density(
    input: RhorrpPairEnergyDensityInput<'_>,
) -> Result<ComplexVec, RhorrpError> {
    let energy_count = validate_pair_energy_density_input(input)?;
    let (first_displacement, first_radius) =
        regularize_density_displacement("first_displacement", input.first_displacement)?;
    let (second_displacement, second_radius) =
        regularize_density_displacement("second_displacement", input.second_displacement)?;
    let first_location = rhorrp_radial_interpolation_location(RhorrpRadialInterpolationInput {
        radius: first_radius,
        x0: input.radial_x0,
        dx: input.radial_dx,
        radial_count: input.radial_count,
    })?;
    let second_location = rhorrp_radial_interpolation_location(RhorrpRadialInterpolationInput {
        radius: second_radius,
        x0: input.radial_x0,
        dx: input.radial_dx,
        radial_count: input.radial_count,
    })?;

    let mut green = ComplexVec::zeros(energy_count);
    if input.same_atom {
        let same_site = rhorrp_same_site_green(RhorrpSameSiteGreenInput {
            regular_large: input.first_regular_large,
            irregular_large: input.first_irregular_large,
            regular_small: input.first_regular_small,
            irregular_small: input.first_irregular_small,
            first_location,
            second_location,
            cosine_between: cosine_between_vectors(first_displacement, second_displacement)?,
        })?;
        for (total, contribution) in green.iter_mut().zip(same_site.iter()) {
            *total += *contribution;
        }
    }
    if let Some(scattering_matrix) = input.scattering_matrix {
        let scattering = rhorrp_scattering_green(RhorrpScatteringGreenInput {
            first_regular_large: input.first_regular_large,
            first_regular_small: input.first_regular_small,
            second_regular_large: input.second_regular_large,
            second_regular_small: input.second_regular_small,
            first_phase: input.first_phase,
            second_phase: input.second_phase,
            scattering_matrix,
            first_location,
            second_location,
            first_displacement,
            second_displacement,
        })?;
        for (total, contribution) in green.iter_mut().zip(scattering.iter()) {
            *total += *contribution;
        }
    }

    rhorrp_finish_energy_density(RhorrpEnergyDensityInput {
        energies_hartree: input.energies_hartree,
        green_function: green.view(),
        reference_energy_hartree: input.reference_energy_hartree,
        radius: first_radius,
        prime_radius: second_radius,
    })
}

/// Port of FEFF `rhorrp` after point-pair setup.
///
/// This helper evaluates the energy-dependent density matrix with
/// [`rhorrp_pair_energy_density`] and immediately integrates it over the FEFF
/// occupied-state contour with [`rhorrp_integrate_density`].
pub fn rhorrp_pair_density(input: RhorrpPairDensityInput<'_>) -> Result<Real, RhorrpError> {
    let energy_density = rhorrp_pair_energy_density(input.pair_energy)?;
    rhorrp_integrate_density(RhorrpDensityIntegrationInput {
        energies_hartree: input.pair_energy.energies_hartree,
        energy_density: energy_density.view(),
        real_axis_count: input.real_axis_count,
        chemical_potential_hartree: input.chemical_potential_hartree,
        temperature_hartree: input.temperature_hartree,
        chemical_potential_override_hartree: input.chemical_potential_override_hartree,
    })
}

/// Port of FEFF `rhoerrp` same-site local Green's-function term.
///
/// This evaluates the branch used when `r` and `r'` are nearest to the same
/// atom. FEFF orders the two radial interpolation locations by lower radial
/// index, uses regular solutions at the lesser radius and irregular-minus-iR
/// solutions at the greater radius, then sums over `l` with `P_l(cos theta)`.
pub fn rhorrp_same_site_green(
    input: RhorrpSameSiteGreenInput<'_>,
) -> Result<ComplexVec, RhorrpError> {
    let (energy_count, angular_count, _) = validate_same_site_green_input(input)?;
    let (lesser, greater) = ordered_radial_locations(input.first_location, input.second_location);

    let regular_large_lesser = interpolate_component(input.regular_large, lesser)?;
    let regular_large_greater = interpolate_component(input.regular_large, greater)?;
    let irregular_large_greater = interpolate_component(input.irregular_large, greater)?;
    let regular_small_lesser = interpolate_component(input.regular_small, lesser)?;
    let regular_small_greater = interpolate_component(input.regular_small, greater)?;
    let irregular_small_greater = interpolate_component(input.irregular_small, greater)?;

    let mut legendre = vec![0.0; angular_count];
    legendre_polynomials_into(input.cosine_between, &mut legendre);

    let imaginary = Complex::new(0.0, 1.0);
    let mut green = ComplexVec::zeros(energy_count);
    for energy in 0..energy_count {
        for angular in 0..angular_count {
            let rho_l = -regular_large_lesser[(energy, angular)]
                * (irregular_large_greater[(energy, angular)]
                    - imaginary * regular_large_greater[(energy, angular)])
                - regular_small_lesser[(energy, angular)]
                    * (irregular_small_greater[(energy, angular)]
                        - imaginary * regular_small_greater[(energy, angular)]);
            let angular_factor =
                legendre[angular] * (2 * angular + 1) as Real / (4.0 * std::f64::consts::PI);
            green[energy] += rho_l * angular_factor;
        }
    }
    Ok(green)
}

/// Port of FEFF `rhoerrp` scattering Green's-function term.
///
/// This evaluates the branch below `call ylm` in FEFF. The `L`/`L'` state axes
/// use FEFF spherical-harmonic order, while the radial components are indexed
/// by their corresponding angular momentum `l`.
pub fn rhorrp_scattering_green(
    input: RhorrpScatteringGreenInput<'_>,
) -> Result<ComplexVec, RhorrpError> {
    let (energy_count, angular_count, state_count) = validate_scattering_green_input(input)?;
    let first_large = interpolate_component(input.first_regular_large, input.first_location)?;
    let first_small = interpolate_component(input.first_regular_small, input.first_location)?;
    let second_large = interpolate_component(input.second_regular_large, input.second_location)?;
    let second_small = interpolate_component(input.second_regular_small, input.second_location)?;
    let lmax = angular_count - 1;
    let first_harmonics = spherical_harmonics(input.first_displacement, lmax)?;
    let second_harmonics = spherical_harmonics(input.second_displacement, lmax)?;

    let imaginary = Complex::new(0.0, 1.0);
    let mut green = ComplexVec::zeros(energy_count);
    for first_state in 0..state_count {
        let first_l = angular_momentum_for_state_index(first_state);
        let first_factor = first_harmonics[first_state] * imaginary_power(first_l);
        for second_state in 0..state_count {
            let second_l = angular_momentum_for_state_index(second_state);
            let angular_factor = first_factor
                * second_harmonics[second_state].conj()
                * negative_imaginary_power(second_l);
            for energy in 0..energy_count {
                let radial = first_large[(energy, first_l)] * second_large[(energy, second_l)]
                    + first_small[(energy, first_l)] * second_small[(energy, second_l)];
                let phase = (imaginary
                    * (input.first_phase[(energy, first_l)]
                        + input.second_phase[(energy, second_l)]))
                    .exp();
                green[energy] += radial
                    * angular_factor
                    * phase
                    * input.scattering_matrix[(energy, first_state, second_state)];
            }
        }
    }
    Ok(green)
}

fn ordered_radial_locations(
    first: RhorrpRadialInterpolationLocation,
    second: RhorrpRadialInterpolationLocation,
) -> (
    RhorrpRadialInterpolationLocation,
    RhorrpRadialInterpolationLocation,
) {
    if first.index_below_1based > second.index_below_1based {
        (second, first)
    } else {
        (first, second)
    }
}

fn interpolate_component(
    wavefunctions: ArrayView3<'_, Complex>,
    location: RhorrpRadialInterpolationLocation,
) -> Result<ComplexMat, RhorrpError> {
    rhorrp_interpolate_wavefunction(RhorrpWavefunctionInterpolationInput {
        wavefunctions,
        index_below_1based: location.index_below_1based,
        fraction: location.fraction,
    })
}

fn regularize_density_displacement(
    name: &'static str,
    displacement: Vector3,
) -> Result<(Vector3, Real), RhorrpError> {
    validate_vector(name, displacement)?;
    let radius_squared: Real = displacement.iter().map(|value| value * value).sum();
    let radius = radius_squared.sqrt();
    if radius < RHORRP_ORIGIN_EPSILON {
        let mut adjusted = displacement;
        adjusted[2] += RHORRP_ORIGIN_EPSILON;
        Ok((adjusted, RHORRP_ORIGIN_EPSILON))
    } else {
        Ok((displacement, radius))
    }
}

fn cosine_between_vectors(first: Vector3, second: Vector3) -> Result<Real, RhorrpError> {
    let dot: Real = first
        .iter()
        .zip(second.iter())
        .map(|(left, right)| left * right)
        .sum();
    let first_norm = first.iter().map(|value| value * value).sum::<Real>().sqrt();
    let second_norm = second
        .iter()
        .map(|value| value * value)
        .sum::<Real>()
        .sqrt();
    let cosine = dot / (first_norm * second_norm);
    validate_scalar("cosine_between", 0, cosine)?;
    Ok(cosine)
}

fn angular_momentum_for_state_index(state: usize) -> usize {
    let mut angular = 0usize;
    while (angular + 1)
        .checked_mul(angular + 1)
        .is_some_and(|limit| limit <= state)
    {
        angular += 1;
    }
    angular
}

fn imaginary_power(exponent: usize) -> Complex {
    match exponent % 4 {
        0 => Complex::new(1.0, 0.0),
        1 => Complex::new(0.0, 1.0),
        2 => Complex::new(-1.0, 0.0),
        _ => Complex::new(0.0, -1.0),
    }
}

fn negative_imaginary_power(exponent: usize) -> Complex {
    match exponent % 4 {
        0 => Complex::new(1.0, 0.0),
        1 => Complex::new(0.0, -1.0),
        2 => Complex::new(-1.0, 0.0),
        _ => Complex::new(0.0, 1.0),
    }
}
